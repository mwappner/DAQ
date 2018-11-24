# -*- coding: utf-8 -*-
"""
This script is to measure and make a control loop with a NI USB6212 DAQ.

This script is based on our old script 'SOldLoop'. It works using the 
'fwp_daq' module we designed. This script's goal is to make a control 
loop that can raise an object at a constant given velocity.

@author: GrupoFWP
"""

import queue
import threading
import fwp_analysis as fan
import fwp_daq as daq
import fwp_daq_channels as fch
import fwp_pid as fpid
#import fwp_lab_instruments as ins
from fwp_plot import add_style
from fwp_save import savetxt, savefile_helper
from fwp_utils import clip_between
import matplotlib.pyplot as plt
import numpy as np
import os
from time import sleep

#%% PWM_Single

"""Sets a PWM output with constant duty cycle and therefore mean value"""

# PARAMETERS

device = daq.devices()[0] # Assuming you have only 1 connected NI device.
pwm_pin = 38 # Literally the number of the DAQ pin
pwm_frequency = 100e3
pwm_duty_cycle = np.linspace(.1,1,10)

# ACTIVE CODE

# Initialize communication
task = daq.Task(device, mode='w')

# Configure output
task.add_channels(fch.PWMOutputChannel, pwm_pin)
task.channels.frequency = pwm_frequency
task.channels.duty_cycle = pwm_duty_cycle[0]
"""Could do all this together:
task.add_channels(fch.PWMOutputChannel, pwm_pin,
                  frequency = pwm_frequency,
                  duty_cycle = pwm_duty_cycle)
"""    

# Turn on and off the output
task.channels.status = True # turn on
sleep(10)
task.channels.status = False # turn off

# End communication
task.close()

#%% PWM_Sweep

"""Makes a sweep on a PWM output's duty cycle and therefore mean value"""

# PARAMETERS

device = daq.devices()[0] # Assuming you have only 1 connected NI device.
pwm_pin = 38 # Literally the number of the DAQ pin
pwm_frequency = 100e3
pwm_duty_cycle = np.linspace(.1,.99,10)

# ACTIVE CODE

# Initialize communication
task = daq.Task(device, mode='w')

# Configure output
task.add_channels(fch.PWMOutputChannel, pwm_pin)
task.channels.frequency = pwm_frequency
task.channels.duty_cycle = pwm_duty_cycle[0]

# Make a sweep on output's duty cycle
task.channels.status = True # turn on
for dc in pwm_duty_cycle:
    task.channels.duty_cycle = dc # change duty cycle
    print("Changed to {:.2f}".format(dc))
    """ Could also call by channel:
    task.ctr0.duty_cycle = dc
    """
    sleep(2)
task.channels.status = False # turn off

# End communication
task.close()

#%% Read_Single

"""Makes a single measurement on an analog input."""

# PARAMETERS

device = daq.devices()[0] # Assuming you have only 1 connected NI device.
ai_pins = [15] # Literally the number of the DAQ pin
ai_conf = 'Ref' # Referenced mode (measure against GND)
nsamples = int(200e3)

# ACTIVE CODE

# Initialize communication
task = daq.Task(device, mode='r')
"""Could be initialized on test mode:
task = daq.Task('Dev1', mode='r', conection=False)
"""
        
# Configure input
task.add_channels(fch.AnalogInputChannel, *ai_pins)
task.channels.configuration = ai_conf

# Make a measurement
signal = task.read(nsamples=10000, 
                   samplerate=None) # None means maximum SR
"""Beware!
An daqError will be raised if not all the DAQ's acquired samples can be 
passed to the PC on time.
2 channels => nsamples = 29500 can be done but not 30000.
"""

# End communication
task.close()

# Make a plot
plt.figure()
plt.plot(signal)

#%% Read_Continuous

"""Makes a continuous measurement on an analog input."""

# PARAMETERS

device = daq.devices()[0] # Assuming you have only 1 connected NI device.
ai_pins = [15] # Literally the number of the DAQ pin
ai_conf = 'Ref' # Referenced mode (measure against GND)
nsamples = int(200e3)

# ACTIVE CODE

# Initialize communication in order to read
task = daq.Task(device, mode='r', print_messages=True)
        
# Configure input
task.add_channels(fch.AnalogInputChannel, *ai_pins)
task.channels.configuration = ai_conf

# Make a continuous measurement
signal = task.read(nsamples=None, # None means continuous
                   samplerate=50e3, # None means maximum SR
                   nsamples_each=500)

# End communication
task.close()

# Make a plot
plt.figure()
plt.plot(signal)

#%% Read_N_Write

"""Reads the PWM signal it writes"""

# PARAMETERS

device = daq.devices()[0]

ai_pin = 15 # Literally the number of the DAQ pin
ai_conf = 'Ref' # Referenced mode (measure against GND)

pwm_pin = 38 # Literally the number of the DAQ pin
pwm_frequency = 100e3
pwm_duty_cycle = .5

nsamples = 1000
samplerate = 400e3
nsamples_each = 200

# ACTIVE CODE

# Initialize communication
task = daq.DAQ(device)

# Configure input channel
task.add_analog_inputs(ai_pin)
task.inputs.ai0.configuration = ai_conf
task.inputs.samplerate = samplerate

# Configure output channel
task.add_pwm_outputs(pwm_pin)
task.outputs.ctr0.frequency = pwm_frequency
task.outputs.ctr0.duty_cycle = pwm_duty_cycle

# Measure
task.outputs.write(status=True) # Output on
signal = task.inputs.read(nsamples=nsamples,
                          samplerate=samplerate,
                          nsamples_each=nsamples_each,
                          use_stream=False)
signal = np.array(signal)
task.outputs.write(status=False) # Output off

# End communication                  
task.close()

# Get time
time = np.arange(0, len(signal)/samplerate, 1/samplerate)

# Plot
fig = plt.figure()
plt.xlabel('Time (s)')
plt.ylabel('Data (V)')
add_style()
plt.plot(time, signal, '.')

'''Check why c059fd5 'fwp_daq.Task.read' doesn't acquire continuously

import nidaqmx as nid
import nidaqmx.stream_readers as sr

# ACTIVE CODE

# Initialize communication in order to read
task = nid.Task()
streamer = sr.AnalogSingleChannelReader(
        task.in_stream)
        
# Configure input
ai_channel = task.ai_channels.add_ai_voltage_chan(
        physical_channel = 'Dev1/ai0',
        units = nid.constants.VoltageUnits.VOLTS
        )
ai_channel.ai_term_cfg = nid.constants.TerminalConfiguration.NRSE

do_return = True
signal = zeros((1, nsamples_each),
               dtype=np.float64)
each_signal = zeros((1,
                     nsamples_each),
                     dtype=np.float64)
message = "Number of {}-sized samples' arrays".format(
        nsamples_each)
message = message + " read: {}"
ntimes = 0
def no_callback(task_handle, 
            every_n_samples_event_type,
            number_of_samples, callback_data):

    """A nidaqmx callback that just reads"""
    
    global do_return, nsamples_each
    global ntimes, message
    global each_signal, signal
    
    if do_return:
        each_signal = streamer.read_many_sample(
            each_signal,
            number_of_samples_per_channel=nsamples_each,
            timeout=20)
        
        signal = multiappend(signal, each_signal)
    ntimes += 1
    print(message.format(ntimes))

    return 0


task.register_every_n_samples_acquired_into_buffer_event(
        nsamples_callback, # call callback every
        wrapper_callback)


# Make a continuous measurement
signal = task.read(nsamples=None, # None means continuous
                   samplerate=50e3, # None means maximum SR
                   nsamples_each=500)

# End communication
task.close()
'''
#%% Read_N_Write_Callback

"""Measure with a plotting callback

This didn't work with c059fd5 'fwp_daq.Task.read' :(
Apparently, callback cannot be used with streamers. 
Shame on you, nidaqmx!"""

# PARAMETERS

device = daq.devices()[0]

ai_pin = 15 # Literally the number of the DAQ pin
ai_conf = 'Ref' # Referenced mode (measure against GND)

pwm_pin = 38 # Literally the number of the DAQ pin

nsamples = 10000
samplerate = 100e3
nsamples_each = 10000
nsamples_callback = 2000

# ACTIVE CODE

# Initialize communication
task = daq.DAQ(device)

# Configure input channel
task.add_analog_inputs(ai_pin)
task.inputs.ai0.configuration = ai_conf
task.inputs.samplerate = samplerate

# Configure output channel
task.add_pwm_outputs(pwm_pin)

# Configure plot
fig = plt.figure()
ax = plt.axes()
line = ax.plot([])
add_style()

# Define plotting callback
def callback(read_data):
    global line
    line.set_data([])
    line.set_data(read_data)

## Define callback
#def callback():
#    print('Hey')

# Measure
task.outputs.write() # Output on
signal = task.inputs.read(nsamples=nsamples,
                          samplerate=samplerate,
                          nsamples_each=nsamples_each,
                          callback=callback,
                          nsamples_callback=nsamples_callback)
task.outputs.write(False) # Output off
task.close()

# Get time
time = np.arange(0, len(signal)/samplerate, 1/samplerate)

# Plot
fig = plt.figure()
plt.xlabel('Time (s)')
plt.ylabel('Data (V)')
add_style()
plt.plot(time, signal, '.')

"""Don't forget to also try this with nsamples=None!"""

#%% Velocity

"""Measures velocity for a given duty cycle on the function generator"""

# PARAMETERS

device = daq.devices()[0]

ai_pin = 15 # Literally the number of the DAQ pin
ai_conf = 'Ref' # Referenced mode (measure against GND)

pwm_frequency = 10e3
pwm_duty_cycle = .99
pwm_supply = 8


nsamples = 1000
samplerate = 5e3

wheel_radius = 0.025 # in meters

# ACTIVE CODE

# Initialize communication
task = daq.DAQ(device)

# Define how to calculate velocity
dt = nsamples/samplerate
def calculate_velocity(read_data):
    photogate_derivative = np.diff(read_data)
    rotation_period = fan.peak_separation(photogate_derivative, 
                                           dt, prominence=1, 
                                           height=2)
    velocity = wheel_radius/rotation_period
    return velocity

# Configure input channel
task.add_analog_inputs(ai_pin)
task.inputs.ai0.configuration = ai_conf
task.inputs.samplerate = samplerate

# Measure
signal = task.inputs.__read__(nsamples,
                              np.zeros(nsamples))
signal = np.array(signal)

# End communication                  
task.close()

# Calculate some stuff
velocity = calculate_velocity(signal)
time = np.arange(0, nsamples/samplerate, 1/samplerate)
print(velocity)

# Configure saving
header = ['Time (s)', 'Voltage (V)']
footer = dict(ai_conf=ai_conf,
              pwm_frequency=pwm_frequency,
              pwm_duty_cycle=pwm_duty_cycle,
              pwm_supply=pwm_supply,
              samplerate=samplerate,
              velocity=velocity)

# Save
function = savefile_helper('Velocity', 'Duty_{:.2f}.txt')
savetxt(function(pwm_duty_cycle),
        np.array([time, signal]).T, header=header, footer=footer)

#%% Velocity_With_DAQ_PWM

"""Measures velocity for a given duty cycle on the DAQ's PWM"""

# PARAMETERS

device = daq.devices()[0]

ai_pin = 15 # Literally the number of the DAQ pin
ai_conf = 'Ref' # Referenced mode (measure against GND)

pwm_pin = 38
pwm_frequency = 10e3
pwm_duty_cycle = .3
pwm_supply = 8

nsamples = 1000
samplerate = 5e3

wheel_radius = 0.025 # in meters

# ACTIVE CODE

# Initialize communication
task = daq.DAQ(device)

# Define how to calculate velocity
dt = nsamples/samplerate
def calculate_velocity(read_data):
    photogate_derivative = np.diff(read_data)
    rotation_period = fan.peak_separation(photogate_derivative, 
                                           dt, prominence=1, 
                                           height=2)
    velocity = wheel_radius/rotation_period
    return velocity

# Configure input channel
task.add_analog_inputs(ai_pin)
task.inputs.ai0.configuration = ai_conf
task.inputs.samplerate = samplerate

# Configure output channel
task.add_pwm_outputs(pwm_pin)
task.outputs.ctr0.frequency = pwm_frequency
task.outputs.ctr0.duty_cycle = pwm_duty_cycle

# Measure
task.outputs.write(status=True) # Output on
signal = task.inputs.__read__(nsamples)
signal = np.array(signal)
task.outputs.write(status=False) # Output off

# End communication                  
task.close()

# Calculate some stuff
velocity = calculate_velocity(signal)
time = np.arange(0, nsamples/samplerate, 1/samplerate)
print(velocity)

# Configure saving
header = ['Time (s)', 'Voltage (V)']
footer = dict(ai_conf=ai_conf,
              pwm_frequency=pwm_frequency,
              pwm_duty_cycle=pwm_duty_cycle,
              pwm_supply=pwm_supply,
              samplerate=samplerate,
              velocity=velocity)

# Save
function = savefile_helper('Velocity', 'Duty_{:.2f}.txt')
savetxt(function(pwm_duty_cycle),
        np.array([time, signal]).T, header=header, footer=footer)

#%% Control_Loop

"""Control loop designed to raise an object at constant speed.

This was our original idea using nidaqmx callback. Didn't work on 
c059fd5 'fwp_daq' because callback can't use stream_readers"""

# PARAMETERS

device = daq.devices()[0]

ai_pin = 15 # Literally the number of the DAQ pin
ai_conf = 'Ref' # Referenced mode (measure against GND)

pwm_pin = 38 # Literally the number of the DAQ pin
pwm_frequency = 100e3
pwm_duty_cycle = np.linspace(.1,1,10)

wheel_radius = 0.025 # in meters

samplerate = 100e3
nsamples_each = 1000

# ACTIVE CODE

# Initialize communication for writing and reading at the same time
task = daq.DAQ(device)

# Define how to calculate velocity
dt = nsamples/samplerate
def calculate_velocity(read_data):
    photogate_derivative = np.diff(read_data)
    rotation_period = fan.peak_separation(photogate_derivative, 
                                           dt, prominence=1, 
                                           height=2)
    velocity = wheel_radius/rotation_period
    return velocity
pid = fan.PIDController(setpoint=1, kp=10, ki=5, kd=7, 
                        dt=dt, log_data=True)

# Configure inputs
task.add_analog_inputs(ai_pin)
task.inputs.pins.configuration = ai_conf

# Configure outputs
task.add_pwm_outputs(pwm_pin)
task.outputs.pins.frequency = pwm_frequency
task.outputs.pins.duty_cycle = pwm_duty_cycle[0]

# Define callback
def calculate_duty_cycle(read_data):
    
    # Now I apply PID
    velocity = calculate_velocity(read_data)
    new_dc = pid.calculate(velocity)
      
    # And finally I change duty cycle
    task.ouputs.duty_cycle = clip_between(new_dc, *(0,100))
  
# Measure
task.outputs.pins.status = True
signal = task.inputs.read(nsamples=None,
                          nsamples_each=500,
                          samplerate=samplerate,
                          nsamples_callback=nsamples_callback,
                          callback=calculate_duty_cycle)
task.outputs.pins.status = False

# Close communication
task.close()

# Configure log
log = np.array(pid.log).T  # Categories by columns
header = ['Feedback value (m/s)', 'New value (a.u.)', 
          'Proportional term (u.a.)', 'Integral term (u.a.)', 
          'Derivative term (u.a.)']
footer = dict(ai_conf=ai_conf,
              pwm_frequency=pwm_frequency,
              pid=pid, # PID parameters
              samplerate=samplerate,
              nsamples_each=nsamples_each,
              nsamples_callback=nsamples_callback)

# Save log
savetxt(os.path.join(os.getcwd(), 'Measurements', 'Log.txt'),
        log, header=header, footer=footer)

#%% Control_Loop_No_stream

"""Another control loop designed to raise an object at constant speed.

This script doesn't use c059fd5 'fwp_daq.Task.read' on continuous 
acquisition mode because we didn't want any more trouble with 'callback'. 
It doesn't use it on single acquisition mode either because it seemed 
simpler not to use 'streamers' at all. In fact, just in case it holds 
commented commands that allow to use a PWM made with a function 
generator instead of a DAQ."""

# PARAMETERS

device = daq.devices()[0]
#gen_port = 'USB0::0x0699::0x0346::C034198::INSTR'

ai_pin = 15 # Literally the number of the DAQ pin
ai_conf = 'Ref' # Referenced mode (measure against GND)

pwm_pin = 38 # Literally the number of the DAQ pin
pwm_frequency = 100e3
pwm_initial_duty = 0.01

wheel_radius = 0.025 # in meters

samplerate = 100e3
nsamples_each = 100

# ACTIVE CODE

# Initialize communication for writing and reading at the same time
task = daq.DAQ(device)
#gen = ins.Gen(gen_port, nchannels=1)

# Define how to calculate velocity
dt = nsamples_each/samplerate
def calculate_velocity(read_data):
    photogate_derivative = np.diff(read_data)
    rotation_period = fan.peak_separation(photogate_derivative, 
                                           dt, prominence=1, 
                                           height=2)
    velocity = wheel_radius/rotation_period
    return velocity
pid = fpid.PIDController(setpoint=1, kp=10, ki=5, kd=7, 
                         dt=dt, log_data=True)

# Configure inputs
task.add_analog_inputs(ai_pin)
task.inputs.pins.configuration = ai_conf
task.inputs.samplerate = samplerate

# Configure outputs
task.add_pwm_outputs(pwm_pin)
task.outputs.pins.frequency = pwm_frequency
task.outputs.pins.duty_cycle = pwm_initial_duty
#gen.output(False, waveform='squ', duty_cycle=pwm_initial_duty)
#gen.output(False, offset=2.5, amplitude=5, frequency=10e3)

# Just in case
pid.reset()
pid.clearlog()

# Define callback's replacement
def calculate_duty(read_data):
    
    try:
        velocity = calculate_velocity(read_data)
    except ValueError: # No peaks found
        velocity = pid.last_log.feedback_value # Return last value
    new_dc = pid.calculate(velocity)
    new_dc = clip_between(new_dc, *(.01,.99))
    
    return new_dc

# Define one thread
q = queue.Queue()
def worker():
    while True:
        data = q.get() #waits for data to be available
        new_dc = calculate_duty(data)
        task.outputs.pins.duty_cycle = new_dc
#        gen.output(duty_cycle=new_dc)
t = threading.Thread(target=worker)

# Measurement per se
task.outputs.pins.status = True
#gen.output(True)
t.start()
while True:
    try:
        # Here goes the other thread
        data = task.inputs._Task__task.read(
            number_of_samples_per_channel=int(nsamples_each))    
        q.put(np.array(data))
    except KeyboardInterrupt:
        pass
task.outputs.pins.status = False
#gen.output(False)

# Close communication
task.close()

# Configure log
log = np.array(pid.log).T  # Categories by columns
header = ['Feedback value (m/s)', 'New value (a.u.)', 
          'Proportional term (u.a.)', 'Integral term (u.a.)', 
          'Derivative term (u.a.)']
footer = dict(ai_conf=ai_conf,
              pwm_frequency=pwm_frequency,
              pid=pid, # PID parameters
              samplerate=samplerate,
              nsamples_each=nsamples_each,
              nsamples_callback=nsamples_callback)

# Save log
savetxt(os.path.join(os.getcwd(), 'Measurements', 'Log.txt'),
        log, header=header, footer=footer)