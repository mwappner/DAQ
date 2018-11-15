# -*- coding: utf-8 -*-
"""
This script is to make measurements with a National Instruments DAQ.

This script is based on our old script 'SOldLoop'. It works using the 
'fwp_daq' module we designed. This script's goal goal is to make a 
control loop that can raise an object at a constant given velocity.

@author: GrupoFWP
"""

import queue
import threading
import fwp_analysis as fan
import fwp_daq as daq
import fwp_daq_channels as fch
from fwp_plot import add_style
from fwp_save import savetxt
from fwp_utils import clip_between
import matplotlib.pyplot as plt
import numpy as np
import os
from time import sleep
#conf = nid.constants.TerminalConfiguration

#%% PWM_Single

"""Setes a PWM output with constant duty cycle and therefore mean value"""

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
pwm_duty_cycle = np.linspace(.1,1,10)

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
pwm_duty_cycle = np.linspace(.1,1,10)

nsamples = 10000
samplerate = 400e3
nsamples_each = 1000

# ACTIVE CODE

# Initialize communication
task = daq.DAQ(device)

# Configure input channel
task.add_analog_inputs(ai_pin)
task.inputs.ai0.configuration = ai_conf
task.inputs.samplerate = samplerate

# Configure output channel
task.add_pwm_outputs(pwm_pin)

# Measure
task.outputs.write(status=True) # Output on
signal = task.inputs.read(nsamples=nsamples,
                          samplerate=samplerate,
                          nsamples_each=nsamples_each)
task.outputs.write(status=False) # Output off

# End communication                  
task.close()

# Get time
samplerate = task.inputs.samplerate
time = np.arange(0, len(signal)/samplerate, 1/samplerate)

# Plot
fig = plt.figure()
plt.xlabel('Time (s)')
plt.ylabel('Data (V)')
add_style()
plt.plot(time, signal, '.')

'''Check why current 'fwp_daq.Task.read' doesn't acquire continuously

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

This doesn't work with current 'fwp_daq.Task.read' :(
Apparently, callback cannot be used with streamers. 
Shame on you, nidaqmx!"""

# PARAMETERS

device = daq.devices()[0]

ai_pin = 15 # Literally the number of the DAQ pin
ai_conf = 'Ref' # Referenced mode (measure against GND)

pwm_pin = 38 # Literally the number of the DAQ pin

nsamples = 10000
samplerate = 100e3
nsamples_each = 1000
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
samplerate = task.inputs.samplerate
time = np.arange(0, len(signal)/samplerate, 1/samplerate)

# Plot
fig = plt.figure()
plt.xlabel('Time (s)')
plt.ylabel('Data (V)')
add_style()
plt.plot(time, signal, '.')

"""Don't forget to also try this with nsamples=None!"""

#%% Control_Loop

"""Control loop designed to raise an object at constant speed.

Doesn't work with current 'fwp_daq' :("""

# PARAMETERS

device = daq.devices()[0]

ai_pin = 15 # Literally the number of the DAQ pin
ai_conf = 'Ref' # Referenced mode (measure against GND)

pwm_pin = 38 # Literally the number of the DAQ pin
pwm_frequency = 100e3
pwm_duty_cycle = np.linspace(.1,1,10)

wheel_radius = 0.025 # in meters

nsamples_callback = 20
samplingrate = 100e3
nsamples_each = 1000

# NO VALE dt=1/samplingrate, depende de cada cuanto corremos el PID
pid = fan.PIDController(setpoint=1, kp=10, ki=5, kd=7, 
                        dt=nsamples_callback/samplingrate, 
                        log_data=True)

# ACTIVE CODE

# Initialize communication for writing and reading at the same time
task = daq.DAQ(device)

# Configure inputs
task.add_analog_inputs(ai_pin)
task.inputs.configuration = ai_conf

# Configure outputs
task.add_pwm_outputs(pwm_pin)
task.outputs.frequency = pwm_frequency
task.outputs.duty_cycle = pwm_duty_cycle[0]

# Define callback
def callback(read_data):
    
    # Now I apply PID
    photogate_derivative = np.diff(read_data)
    angular_velocity = fan.peak_separation(photogate_derivative, 
                                           pid.dt, prominence=1, 
                                           height=2)
    velocity = angular_velocity * wheel_radius
    new_dc = pid.calculate(velocity)
      
    # And finally I change duty cycle
    task.ouputs.duty_cycle = clip_between(new_dc, *(0,100))
  
# Measure
task.outputs.status = True
signal = task.inputs.read(nsamples=None,
                          nsamples_each=500,
                          samplerate=samplingrate,
                          nsamples_callback=nsamples_callback,
                          callback=callback)
task.outputs.status = False

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
              samplerate=samplingrate,
              nsamples_each=nsamples_each,
              nsamples_callback=nsamples_callback)

# Save log
savetxt(os.path.join(os.getcwd(), 'Measurements', 'Log.txt'),
        log, header=header, footer=footer)

#%% Control_Loop_No_stream

"""Another control loop designed to raise an object at constant speed.

This script doesn't use the current 'fwp_daq.Task.read' on continuous 
acquisition mode because we didn't want any more trouble with 'callback'. 
It doesn't use it on single acquisition mode either because it seemed 
simpler not to use 'streamers' at all.

Beware! It's currently divided into three pieces so that we don't have 
to 'Task.close()' and reconfigure every time"""

# PARAMETERS

device = daq.devices()[0]

ai_pin = 15 # Literally the number of the DAQ pin
ai_conf = 'Ref' # Referenced mode (measure against GND)

pwm_pin = 38 # Literally the number of the DAQ pin
pwm_frequency = 100e3
pwm_initial_duty = 0.01

wheel_radius = 0.025 # in meters

samplingrate = 100e3
nsamples_each = 100

pid = fan.PIDController(setpoint=1, kp=10, ki=5, kd=7, 
                        dt=nsamples_each/samplingrate, 
                        log_data=True)

# ACTIVE CODE

# Initialize communication for writing and reading at the same time
task = daq.DAQ(device)

# Configure inputs
task.add_analog_inputs(ai_pin)
task.inputs.pins.configuration = ai_conf
task.inputs._Task__task.timing.cfg_samp_clk_timing(
        rate=samplingrate,
        sample_mode=daq.single)

# Configure outputs
task.add_pwm_outputs(pwm_pin)
task.outputs.pins.frequency = pwm_frequency
task.outputs.pins.duty_cycle = pwm_initial_duty

#%%

pid.reset()
pid.clearlog()

# Define callback's replacement
def calculate_duty(read_data):
    
    global last_velocity
    
    # Now I apply PID
    photogate_derivative = np.diff(read_data)
    try:
        angular_velocity = fan.peak_separation(photogate_derivative, 
                                               pid.dt, prominence=1, 
                                               height=2)
        velocity = angular_velocity * wheel_radius

    except ValueError: # No peaks found
        velocity = pid.last_log.feedback_value # Return last value

    new_dc = pid.calculate(velocity)
      
    # And finally I change duty cycle
    task.outputs.pins.duty_cycle = clip_between(new_dc, *(.01,.99))

q = queue.Queue()
#data = task.inputs._Task__task.read(
#        number_of_samples_per_channel=int(nsamples_each))
#q.put(data)

def worker():
    while True:
        data = q.get() #waits for data to be available
        print(1)
        new_duty = calculate_duty(data)
        print(2)
        task.outputs.channels.duty_cycle = new_duty
        print(3)
        
t = threading.Thread(target=worker)
task.outputs.status = True
t.start()

while True:
    data = task.inputs._Task__task.read(
        number_of_samples_per_channel=int(nsamples_each))    
    q.put(np.array(data))
    sleep(.1)

#%%

task.outputs.status = False
task.close()