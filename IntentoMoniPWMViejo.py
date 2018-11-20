# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 19:50:49 2018

@author: mfar
"""
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
from nidaqmx import stream_writers as sw
import nidaqmx.stream_readers as sread
import nidaqmx as nid
import daq








#%% Control_Loop_No_stream

# PARAMETERS

device = daq.devices()[0]

ai_pin = "Dev20/ai0" # Literally the number of the DAQ pin
ai_conf = 'Ref' # Referenced mode (measure against GND)

pwm_channels = 'Dev1/ctr0' # Clock output
pwm_frequency = 100
pwm_duty_cycle = np.linspace(.1,1,10)

wheel_radius = 0.025 # in meters

samplingrate = 100e3
nsamples_each = 100

pid = fan.PIDController(setpoint=1, kp=10, ki=5, kd=7, 
                        dt=nsamples_each/samplingrate, 
                        log_data=True)

# ACTIVE CODE

# Initialize communication for writing and reading at the same time
with nid.Task() as read_task, nid.Task() as task_co:

    # Channels configuration
    read_task.ai_channels.add_ai_voltage_chan(ai_pin,
        max_val=10, min_val=-10)

    # Measurement configuration
    reader = sread.AnalogMultiChannelReader(read_task.in_stream)
    values_read = np.zeros((1, nsamples_each), 
                  dtype=np.float64)

    # Configure clock output
    channels_co = daq.pwm_output_channels(
            task_co,
            physical_channels = pwm_channels,
            frequency = pwm_frequency,
            duty_cycle = pwm_duty_cycle[0]
            )
   
    # Set contiuous PWM signal
    task_co.timing.cfg_implicit_timing(
            sample_mode = nid.constants.AcquisitionType.CONTINUOUS)
   
    # Create a PWM stream
    stream_co = sw.CounterWriter(task_co.out_stream)

    # Play   
    task_co.start()    
     
    # Make measurement
    read_task.start()
    

    #np.testing.assert_allclose(values_read, rtol=0.05, atol=0.005)

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
    
    return new_dc
    # And finally I change duty cycle
    #task.outputs.pins.duty_cycle  = clip_between(new_dc, *(.01,.99))

q = queue.Queue()
#data = task.inputs._Task__task.read(
#        number_of_samples_per_channel=int(nsamples_each))
#q.put(data)

def worker():
    while True:
        data = q.get() #waits for data to be available
        print(1)
        new_duty = clip_between(calculate_duty(data), *(.01,.99))
        channels_co.co_pulse_duty_cyc = new_duty
        print(2)
        stream_co.write_one_sample_pulse_frequency(
                frequency = channels_co.co_pulse_freq,
                duty_cycle = channels_co.co_pulse_duty_cyc
                )
        print(3)
        
t = threading.Thread(target=worker)
t.start()

while True:
    data = values_read
    q.put(np.array(data))
    sleep(.1)
    
    