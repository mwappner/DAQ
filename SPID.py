# -*- coding: utf-8 -*-
"""
This script is to make measurements with a National Instruments DAQ.

This script holds the final version of our control loop, designed to 
raise an object at a constant given velocity. It is based on the 'SLoop' 
script, where a lot of previous test drives and versions were developed.

@author: GrupoFWP
"""

import queue
import threading
import fwp_analysis as fan
import fwp_daq as daq
import fwp_pid as fpid
from fwp_plot import add_style
import fwp_save as sav
from fwp_utils import clip_between
import matplotlib.pyplot as plt
import numpy as np
import os
import nidaqmx as nid
#from time import sleep

#%%

"""Control loop designed to raise an object at constant speed"""

########################## PARAMETERS ##########################

device = daq.devices()[0]

ai_pin = 15 # Literally the number of the DAQ pin
ai_conf = 'Diff' # Referenced mode (measure against GND)

pwm_pin = 38 # Literally the number of the DAQ pin
pwm_frequency = 100e3 # in Hz
pwm_initial_duty_cycle = 30 # in %
pwm_min_duty_cycle = 30 # in %

wheel_radius = 2.5 # in cm
chopper_sections = 100 # number of black spaces on photogate's chopper

samplerate = 5e3 # in Hz
nsamples_each = 1000 # number of samples taken between PID actions

setpoint = 15 # in cm/s
P = 7/3
kp = 45*.6
ki = kp*P/(8*nsamples_each/samplerate)
kd = 2*kp/(P*nsamples_each/samplerate)
integrator = 'windowed'
window_size = 5

log_data = True
filename_mask = 'Log_kp_{kp}_ki_{ki}_kd_{kd}'
filename_folder = 'PID_{}_setpoint'.format(setpoint)

#%%

########################## ALGORITHM ##########################

"""The user's desired PID works on linear velocity and duty cycle.
But the actual PID works on photogate period to avoid some calculus.

To do that, at the beginning of the control loop this code transforms 
the user's setpoint expressed on velocity (cm/s) to a virtual setpoint 
expressed on photogate frequency (u.a.). It runs 'real_to_virtual' 
function to do that.

Then, at the end of the control loop, this code transforms the PID 
measurements expressed on photogate frequency (u.a.) to the user's 
desired measurements expressed on velocity (cm/s). It runs 
'virtual_to_real' function to do that.
"""

# First, define how to calculate photogate frequency on each measurement
def calculate_photogate_frequency(read_data):
    """Returns frequency of photogate signal expressed on a.u."""
    # This function is used to analyze each measurement
    photogate_derivative = np.diff(read_data)
    photogate_period = fan.peak_separation(
            photogate_derivative,
            prominence=1, 
            height=2)
    return 1 / photogate_period 
    # Period isn't lineal with duty cycle
    # And this frequency isn't expressed on s^-1

# Then, define how to calculate a new duty cycle on each measurement
pwm_initial_duty_cycle = pwm_initial_duty_cycle / 100
pwm_min_duty_cycle = pwm_min_duty_cycle / 100
def calculate_duty_cycle(read_data):
    """Returns duty cycle expressed on u.a."""
    try:
        velocity = calculate_photogate_frequency(read_data)
    except ValueError: # No peaks found
#        velocity = pid.last_log.feedback_value # Asume vel=0
        velocity = 0
    new_duty_cycle = pid.calculate(velocity)
    new_duty_cycle = clip_between(new_duty_cycle, 
                                  pwm_min_duty_cycle, .99)    
    return new_duty_cycle

# Now define how to go from real to virtual variable and visceversa
circunference = 2 * np.pi * wheel_radius
dt = 1 / samplerate
dT = nsamples_each * dt
def real_to_virtual(velocity):
    """Calculates frequency of photogate expressed on a.u. from velocity"""
    # This function is used to calculate setpoint at the beginning
    photogate_frequency = velocity * chopper_sections / circunference
    photogate_frequency = dt * photogate_frequency
    return photogate_frequency
def virtual_to_real(photogate_frequency):
    """Calculates velocity from frequency of photogate expressed on a.u."""
    # This function is only used to calculate velocity at the end
    photogate_frequency = photogate_frequency / dt
    velocity = circunference * photogate_frequency / chopper_sections
    return velocity
                      
########################## CONFIGURATION* ##########################
"""Code sections marked with '*' shouldn't be modified by the user"""

# Initialize communication with DAQ
task = daq.DAQ(device)

# Choose PID
pid = fpid.PIDController(setpoint=setpoint, kp=kp, ki=ki, kd=kd, dt=dT, 
                         log_data=log_data, integrator=integrator)

#pid.set_integrator(window_length=500) # Set integration window

# Configure inputs
task.add_analog_inputs(ai_pin)
task.inputs.pins.configuration = ai_conf
task.inputs.samplerate = samplerate

# Configure outputs
task.add_pwm_outputs(pwm_pin)
task.outputs.pins.frequency = pwm_frequency
task.outputs.pins.duty_cycle = pwm_initial_duty_cycle

# Configure PID
pid.setpoint_transformer = real_to_virtual
pid.set_integrator(window_length=window_size)
pid.reset()
pid.clearlog()
pid.calculate(0) # Initialize with fake first measurement

#%%

########################## MEASUREMENT* ##########################

# Define one thread
q = queue.Queue()
def worker():
    while True:
        data = q.get() # Waits for data to be available
        new_duty_cycle = calculate_duty_cycle(data)
        task.outputs.pins.duty_cycle = new_duty_cycle
t = threading.Thread(target=worker)
             
# Measure on another thread
task.outputs.pins.status = True
t.start()
print("Turn on :D")
#sleep(1)
print("Running PID!")
while True:
    try:
        data = task.inputs.read(int(nsamples_each))
        q.put(data)
    except KeyboardInterrupt:
        break
task.outputs.pins.status = False
print("Turn off :/")

# Close communication
task.close()

########################## DATA MANAGEMENT ##########################

# Configure data log
header = ['Time (s)', 'Velocity (cm/s)', 'Duty cycle (%)', 
          'Proportional term (u.a.)', 'Integral term (u.a.)', 
          'Derivative term (u.a.)']
footer = dict(ai_conf=ai_conf,
              pwm_frequency=pwm_frequency,
              samplerate=samplerate,
              nsamples_each=nsamples_each,
              pwm_min_duty_cycle=pwm_min_duty_cycle*100,
              pwm_initial_duty_cycle=pwm_initial_duty_cycle*100,
              **pid.params)
filename_generator = sav.savefile_helper(filename_folder,
                                     filename_mask)

# Get data
v = virtual_to_real(np.array(pid.log.feedback_value))
dc = np.array(pid.log.new_value) * 100 # User's DC is % but DAQ's is 0-1
t = np.linspace(0, len(v)*dT, len(v))
p = np.array(pid.log.p_term)
i = np.array(pid.log.i_term)
d = np.array(pid.log.d_term)
e = virtual_to_real(p)

# Calculate PID data to plot
pterm = p * pid.kp / dc
iterm = i * pid.ki / dc
dterm = d * pid.kd / dc

# Start plotting
plt.figure()
            
# Velocity vs time
plt.subplot(3, 1, 1)
plt.plot(t, v, 'o-')
plt.plot(t, e, 'o-') # error
plt.hlines(setpoint, min(t), max(t), linestyles='dotted')
plt.ylabel("Velocity (cm/s)")
plt.ylim(ymax=1.1*max(max(v[1:]), max(e[1:])),
         ymin=1.1*min(min(v[1:]), min(e[1:])))
plt.legend(['Signal', 'Error', 'Setpoint'])

# Duty cycle vs time
plt.subplot(3, 1, 2)
plt.plot(t, dc, 'o-r', label='Signal')
plt.hlines(100, min(t), max(t), linestyles='dotted', label='Limits')
plt.hlines(pwm_min_duty_cycle * 100, min(t), max(t), 
           linestyles='dotted')
plt.ylabel("Duty Cycle (%)")
plt.ylim(ymax=200, ymin=-100)
plt.legend()

# PID parameters vs time
plt.subplot(3,1,3)
plot_styles = ['-o',':o', '--x']
for x, s in zip([iterm, pterm, dterm], plot_styles):
    plt.plot(t, x * 100, s)
plt.legend(['I term', 'P term', 'D term'])
plt.xlabel("Time (s)")
plt.ylabel("PID Parameter (%)")

# Show plot
add_style()
mng = plt.get_current_fig_manager()
mng.window.showMaximized()

# Save data
filename = filename_generator(**pid.params)
sav.savetxt(os.path.join(os.getcwd(), 'Measurements', 
                     'PID', filename),
            np.array([t, v, dc, p, i, d]).T, 
            header=header, footer=footer)

#%% Cohen_Coon

"""Makes a single measurement on two analog input."""

# PARAMETERS

device = daq.devices()[0] # Assuming you have only 1 connected NI device.
ai_pins = [15]#, 17] # Literally the number of the DAQ pin
ai_conf = 'Dif' # Differential mode

samplerate = 200e3
duration = 3
jump = [65, 40]

filename_mask = "Cohen_Coon_{}_{}_jump.txt"
filename_generator = sav.savefile_helper('Cohen_Coon_2',
                                         filename_mask)
header = ['Tiempo (s)', 'Señal (V)', 'Generador (V)']
footer = dict(samplerate=samplerate,
              ai_conf=ai_conf)

nsamples_plot_interval = 1000
nsamples_plot_max = 10000

# ACTIVE CODE

# Initialize communication in order to read
task = daq.DAQ(device, print_messages=True)
        
# Configure input
task.add_analog_inputs(*ai_pins)
task.inputs.pins.configuration = ai_conf
task.inputs.samplerate = samplerate

# Make a continuous measurement
signal = task.inputs.read(duration=duration, samplerate=samplerate)

# End communication
task.close()

# See data
print("{:.2f}, {:.2f}".format(min(data[:,1]), max(data[:,1])))

plt.figure()
#plt.subplot(2,1,1)
#plt.plot(signal[0,:nsamples_plot_max], 'b.-')
#plt.subplot(2,1,2)
#plt.plot(signal[1,:nsamples_plot_max], 'r.-')
plt.plot(signal, 'b.-')

# Save data
time = np.expand_dims(np.linspace(0, duration, duration*samplerate), 
                      axis=0)

data = np.array(signal).T
if data.ndim==1:
    data = np.expand_dims(data, axis=0).T
                         
data = np.concatenate((time.T, data), axis=1)
sav.savetxt(sav.new_name(filename_generator(*jump)),
            data,
            header=header,
            footer=footer)

#%% Cohen_Coon_SMeasure

# Main parameters
samplerate = 5e3
mode = nid.constants.TerminalConfiguration.DIFFERENTIAL

# Other parameters
duration = 3
samples_to_measure = int(samplerate * duration)
channels = ["Dev1/ai0", "Dev1/ai2"]
jump = [80, 50]

filename_mask = "Cohen_Coon_{}_{}_jump.txt"
filename_generator = sav.savefile_helper('Cohen_Coon_Diff',
                                         filename_mask)
header = ['Tiempo (s)', 'Señal (V)', 'Generador (V)']
footer = 'samplerate={:.0f}Hz, mode={}'.format(
        samplerate,
        str(mode).split('.')[-1])

plot_max = 10000
plot_step = 100

# ACTIVE CODE
with nid.Task() as task:

    for channel in channels:
        # Configure channel
        task.ai_channels.add_ai_voltage_chan(
                channel,
                terminal_config=mode)
#                ai_max=2,
#                ai_min=-2)
        
    # Set sampling_rate and samples per channel
    task.timing.cfg_samp_clk_timing(
            rate=samplerate,
            samps_per_chan=samples_to_measure)
        
    # Measure
    print("Started acquiring")
    signal = task.read(
            number_of_samples_per_channel=samples_to_measure)
    task.wait_until_done()
    print("Ended acquiring")
        
data = np.array(signal).T
if data.ndim==1:
    data = np.expand_dims(data, axis=0).T

print("{:.2f}, {:.2f}".format(min(data[:,1]), max(data[:,1])))

signal = np.array(signal)
time = np.linspace(0, duration, samples_to_measure)
plt.figure()
plt.subplot(2,1,1)
plt.plot(time[:plot_max], signal[0,:plot_max], 'b-')
plt.subplot(2,1,2)
plt.plot(time[:plot_max], signal[1,:plot_max], 'r-')
#plt.plot(time, signal)

plt.figure()
time = np.linspace(0, duration, samples_to_measure)
plt.subplot(2,1,1)
plt.plot(time[plot_max:], abs(signal[0,plot_max:]), 'b-')
#plt.plot(time[plot_max::plot_step], signal[0,plot_max::plot_step], 'b-')
plt.subplot(2,1,2)
plt.plot(time[plot_max:], abs(signal[1,plot_max:]), 'r-')
#plt.plot(time[plot_max::plot_step], signal[1,plot_max::plot_step], 'r-')

#%%

time = np.expand_dims(np.linspace(0, duration, samples_to_measure), axis=0)
data = np.concatenate((time.T, data), axis=1)
sav.savetxt(sav.new_name(filename_generator(*jump)),
            data,
            header=header,
            footer=footer)