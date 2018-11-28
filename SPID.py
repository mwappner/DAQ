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
ai_conf = 'Ref' # Referenced mode (measure against GND)

pwm_pin = 38 # Literally the number of the DAQ pin
pwm_frequency = 100e3
pwm_initial_duty_cycle = 0.3
pwm_min_duty_cycle = 0.2

wheel_radius = 0.025 # in meters
chopper_sections = 100 # amount of black spaces on photogate's chopper

samplerate = 5e3
nsamples_each = 1000

setpoint = 0.003
kp = 200
ki = 20
kd = 0.2
integrator = 'infinite'

log_data = True
filename_mask = 'Log_kp_{kp}_ki_{ki}_kd_{kd}'
filename_folder = 'PID_{}'.format(pwm_min_duty_cycle)
                      
########################## CONFIGURATION* ##########################
"""Code sections marked with '*' shouldn't be modified by the user"""

# Initialize communication with DAQ
task = daq.DAQ(device)

# Choose PID
dt = nsamples_each/samplerate
pid = fpid.PIDController(setpoint=setpoint, kp=kp, ki=ki, kd=kd, dt=dt, 
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
pid.reset()
pid.clearlog()
pid.calculate(0) # Initialize with fake first measurement
            
########################## ALGORITHM ##########################

# Define how to calculate velocity
circunference = 2 * np.pi * wheel_radius
def calculate_velocity(read_data):
    photogate_derivative = np.diff(read_data)
    one_section_period = fan.peak_separation(
            photogate_derivative, 
            dt, 
            prominence=1, 
            height=2)
    return circunference / (chopper_sections * one_section_period)

# Define how to calculate duty cycle
def calculate_duty_cycle(read_data):
    try:
        velocity = calculate_velocity(read_data)
    except ValueError: # No peaks found
        velocity = pid.last_log.feedback_value # Asume vel=0
    new_duty_cycle = pid.calculate(velocity)
    new_duty_cycle = clip_between(new_duty_cycle, 
                                  pwm_min_duty_cycle, .99)    
    return new_duty_cycle

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
        data = task.inputs__read__(int(nsamples_each))    
        q.put(data)
    except KeyboardInterrupt:
        break
task.outputs.pins.status = False
print("Turn off :/")

# Close communication
task.close()

########################## DATA MANAGEMENT ##########################

# Configure data log
header = ['Feedback value (m/s)', 'New value (a.u.)', 
          'Proportional term (u.a.)', 'Integral term (u.a.)', 
          'Derivative term (u.a.)']
footer = dict(ai_conf=ai_conf,
              pwm_frequency=pwm_frequency,
              samplerate=samplerate,
              nsamples_each=nsamples_each,
              **pid.params)
filename_generator = sav.savefile_helper(filename_folder,
                                     filename_mask)

# Save data
log = np.array(pid.log).T  # Categories by columns
filename = filename_generator(**pid.params)
sav.savetxt(os.path.join(os.getcwd(), 'Measurements', 
                     'PID', filename),
        log, header=header, footer=footer)

# Main data
v = np.array(pid.log.feedback_value)
dc = np.array(pid.log.new_value)
t = np.linspace(0, len(v)*dt, len(v))
e = np.array(pid.log.p_term)

# PID data
i = np.array(pid.log.i_term) * pid.ki / dc
p = np.array(pid.log.p_term) * pid.kp / dc
d = np.array(pid.log.d_term) * pid.kd / dc

# Start plotting
plt.figure()
            
# Velocity vs time

plt.subplot(3, 1, 1)
plt.plot(t, v, 'o-')
plt.plot(t, e, 'o-')
plt.hlines(pid.setpoint, min(t), max(t), linestyles='dotted')
plt.ylabel("Velocity (m/s)")
plt.legend(['Signal', 'Error', 'Setpoint'])

# Duty cycle vs time
plt.subplot(3, 1, 2)
plt.plot(t, 100 * dc, 'o-r', label='Signal')
plt.hlines(100, min(t), max(t), linestyles='dotted', label='Limits')
plt.hlines(pwm_min_duty_cycle * 100, min(t), max(t), 
           linestyles='dotted')
plt.ylabel("Duty Cycle (%)")
plt.legend()

# PID parameters vs time
plt.subplot(3,1,3)
plot_styles = ['-o',':o', '--x']
for x, s in zip([i, p, d], plot_styles):
    plt.plot(t, x * 100, s)
plt.legend(['I term', 'P term', 'D term'])
plt.xlabel("Time (s)")
plt.ylabel("PID Parameter (%)")

# Show plot
add_style()
mng = plt.get_current_fig_manager()
mng.window.showMaximized()

#%% Cohen_Coon

"""Makes a continuous measurement on two analog input."""

# PARAMETERS

device = daq.devices()[0] # Assuming you have only 1 connected NI device.
ai_pins = [15] # Literally the number of the DAQ pin
ai_conf = 'Ref' # Referenced mode (measure against GND)

samplerate = 5e3
nsamples_each = 1000

# ACTIVE CODE

# Initialize communication in order to read
task = daq.DAQ(device, print_messages=True)
        
# Configure input
task.add_analog_inputs(15)
task.inputs.pins.configuration = ai_conf
task.inputs.samplerate = samplerate

# Make a continuous measurement
signal = task.inputs.read(duration=5, # None means continuous
                   nsamples_each=nsamples_each,
                   use_stream=False)

# End communication
task.close()

# Make a plot
plt.figure()
plt.plot(signal)

#%% Cohen_Coon_SMeasure

# Main parameters
samplerate = 50e3
mode = nid.constants.TerminalConfiguration.RSE

# Other parameters
duration = 3
samples_to_measure = int(samplerate * duration)
channels = ["Dev1/ai0", "Dev1/ai1"]
jump = [65, 40]

filename_mask = "Cohen_Coon_{}_{}_jump.txt"
filename_generator = sav.savefile_helper('Cohen_Coon',
                                         filename_mask)
header = ['Tiempo (s)', 'Señal (V)', 'Generador (V)']
footer = 'samplerate={:.0f}Hz, mode={}'.format(
        samplerate,
        str(mode).split('.')[-1])

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
    signal = task.read(
            number_of_samples_per_channel=samples_to_measure)
    task.wait_until_done()
        

time = np.expand_dims(np.linspace(0, duration, samples_to_measure), axis=0)

data = np.array(signal).T
if data.ndim==1:
    data = np.expand_dims(data, axis=0).T

data = np.concatenate((time.T, data), axis=1)

sav.savetxt(sav.new_name(filename_generator(*jump)),
            data,
            header=header,
            footer=footer)

print("{:.2f}, {:.2f}".format(min(data[:,1]), max(data[:,1])))

#%%


