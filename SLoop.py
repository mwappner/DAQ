# -*- coding: utf-8 -*-
"""
This script is to make measurements with a National Instruments DAQ.

This script is based on our old script 'SOldLoop'. It works using the 
'fwp_daq' module we designed. This script's goal goal is to make a 
control loop that can raise an object at a constant given velocity.

@author: GrupoFWP
"""

import fwp_analysis as fan
import fwp_daq as daq
import fwp_daq_channels as fch
import matplotlib.pyplot as plt
import nidaqmx as nid
import numpy as np
import os
from time import sleep
conf = nid.constants.TerminalConfiguration

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
task.all.frequency = pwm_frequency
task.all.duty_cycle = pwm_duty_cycle[0]
"""Could do all this together:
task.add_channels(fch.PWMOutputChannel, pwm_pin,
                  frequency = pwm_frequency,
                  duty_cycle = pwm_duty_cycle)
"""    

# Turn on and off the output
task.all.status = True # turn on
sleep(10)
task.all.status = False # turn off

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
task.all.frequency = pwm_frequency
task.all.duty_cycle = pwm_duty_cycle[0]

# Make a sweep on output's duty cycle
task.all.status = True # turn on
for dc in pwm_duty_cycle:
    task.all.duty_cycle = dc # change duty cycle
    """ Could also call by channel:
    task.ctr0.duty_cycle = dc
    """
    sleep(2)
task.all.status = False # turn off

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
        
# Configure input
task.add_channels(fch.AnalogInputChannel, *ai_pins)

# Make a measurement
signal = task.read(nsamples_total=10000, 
                   samplerate=None) # None means maximum SR
"""Beware!
An daqError will be raised if not all the DAQ's acquired samples can be 
passed to the PC on time.
2 channels => nsamples_total = 29500 can be done but not 30000.
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
task = daq.Task(device, mode='r')
        
# Configure input
task.add_channels(fch.AnalogInputChannel, *ai_pins)

# Make a continuous measurement
signal = task.read(nsamples_total=None, # None means continuous
                   samplerate=50e3, # None means maximum SR
                   nsamples_each=500,
                   )

# End communication
task.close()

# Make a plot
plt.figure()
plt.plot(signal)

#%% Control_Loop

# PARAMETERS

device = daq.devices()[0]

ai_pin = 15 # Literally the number of the DAQ pin
ai_conf = 'Ref' # Referenced mode (measure against GND)

pwm_pin = 38 # Literally the number of the DAQ pin
pwm_frequency = 100e3
pwm_default_duty_cycle = np.linspace(.1,1,10)

wheel_radius = 0.025 # in meters

nsamples_callback = 20
samplingrate = 100e3
nsamples_each = 1000

pid = fan.PIDController(setpoint=1, kp=10, ki=5, kd=7, 
                        dt=1/samplingrate)

# ACTIVE CODE

# Initialize communication for writing and reading at the same time
task = daq.DAQ(device)

# Configure inputs
task.add_analog_inputs(ai_pin)
#task.inputs.configuration = ai_conf
task.inputs.ai0._AnalogInputChannel__channel.ai_term_cfg = conf.NRSE

# Configure outputs
task.add_pwm_outputs(pwm_pin)
task.outputs.frequency = pwm_frequency
task.outputs.duty_cycle = pwm_default_duty_cycle

# Control loop's saving mechanism
with open(os.path.join('Measurements','log.txt'), 'w') as file:
    file.write('Vel\t duty_cycle\t P\t I\t D') # First line
    
    # Control loop's callback
    def callback(task_handle, every_n_samples_event_type,
                 number_of_samples, callback_data):
            
        # First I read a few values
        samples = task.read(nsamples_total=nsamples_callback,
                            samplerate=samplingrate)
        
        # Now I apply PID
        d = np.diff(samples)
        vel = fan.peak_separation(d, pid.dt, prominence=1, height=2)
        vel = vel * wheel_radius
        new_dc = pid.calculate(vel)
        
        # Then I save some data
        data = fan.append_data_to_string(vel, new_dc, 
                                         pid.p_term, pid.i_term, 
                                         pid.d_term)
        file.write(data)
        
        # And finally I change duty cycle
        task.ouputs.duty_cycle = fan.set_bewtween(new_dc, *(0,100))
    
    # Turn on outputs
    task.outputs.status = True
    
    # Start measurement
    signal = task.inputs.read(nsamples_total=None,
                              nsamples_each=500,
                              samplerate=samplingrate,
                              nsamples_callback=nsamples_callback,
                              callback=callback)
    
    task.outputs.status = False
    task.close()