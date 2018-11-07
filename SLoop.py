# -*- coding: utf-8 -*-
"""
This script is to make measurements with a National Instruments DAQ.

@author: GrupoFWP
"""

import fwp_daq as daq
import fwp_daq_channels as fch
import numpy as np

#%%

import fwp_save as sav
import fwp_analysis as fan
import nidaqmx as nid
from nidaqmx import stream_readers as sr
from nidaqmx import stream_writers as sw
from nidaqmx.utils import flatten_channel_string
from time import sleep
import fwp_wavemaker as wm



#%% Just turn on a PWM signal

pwm_channels = 'Dev1/ctr0' # Clock output
pwm_frequency = 100
pwm_duty_cycle = .5

with nid.Task() as task_co:
    
    # Configure clock output
    channels_co = daq.pwm_outputs(
            task_co,
            physical_channels = pwm_channels,
            frequency = pwm_frequency,
            duty_cycle = pwm_duty_cycle
            )
    
    # Set contiuous PWM signal
    task_co.timing.cfg_implicit_timing(
            sample_mode = nid.constants.AcquisitionType.CONTINUOUS)
    
    # Create a PWM stream
    stream_co = sw.CounterWriter(task_co.out_stream)

    # Play    
    task_co.start()
    sleep(10)

#%% Change PWM mean value

pwm_channels = 'Dev1/ctr0' # Clock output
pwm_frequency = 100
pwm_duty_cycle = np.linspace(.1,1,10)

with nid.Task() as task_co:
    
    # Configure clock output
    channels_co = daq.pwm_outputs(
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
    
    for dc in pwm_duty_cycle:
        sleep(3)
        channels_co.co_pulse_duty_cyc = dc
        stream_co.write_one_sample_pulse_frequency(
                frequency = channels_co.co_pulse_freq,
                duty_cycle = channels_co.co_pulse_duty_cyc
                )
        print("Hope I changed duty cycle to {:.2f} x'D".format(dc))
        sleep(3)
    task_co.stop()

#%% Moni's Voltage Control Loop --> streamers (pag 57)

name = 'V_Control_Loop'

# DAQ Configuration
samplerate = 400e3
mode = nid.constants.TerminalConfiguration.NRSE
number_of_channels=2
channels_to_read = ["Dev1/ai0", "Dev1/ai2"]
channels_to_write = ["Dev1/ao0", "Dev1/ao1"]

# Signal's Configuration
signal_frequency = 10
signal_pk_amplitude = 2
periods_to_measure = 50

# PID's Configuration
pidvalue=1
pidconstant=0.1

# ACTIVE CODE

# Other configuration
duration = periods_to_measure/signal_frequency
samples_to_measure = int(samplerate * duration/1000)
filename = sav.savefile_helper(dirname = name, 
                               filename_template = 'NChannels_{}.txt')
header = 'Time [s]\tData [V]'

# First I make a ramp
waveform= wm.Wave('triangular', frequency=10, amplitude=1)
output_array = waveform.evaluate_sr(sr=samplerate, duration=duration)

# Now I define a callback function
def callback(task_handle, every_n_samples_event_type,
             number_of_samples, callback_data):
    
    print('Every N Samples callback invoked.')

    samples = reader.read_many_sample(
            values_read, 
            number_of_samples_per_channel=number_of_samples,
            timeout=2)
    
    global output_array
    non_local_var['samples'].extend(samples)
    
    if max(samples) > (pidvalue+0.1):
        delta = max(samples) - pidvalue
        output_array -= pidconstant * delta
        
    elif max(samples) < (pidvalue-0.1):
         delta = pidvalue - max(samples)
         output_array += pidconstant * delta
         
    return 0

# Now I start the actual PID loop        
with nid.Task() as write_task, nid.Task() as read_task:

    # First I configure the reading
    read_task.ai_channels.add_ai_voltage_chan(
        flatten_channel_string(channels_to_read),
        max_val=10, min_val=-10)
    reader = sr.AnalogMultiChannelReader(read_task.in_stream)
    
    # Now I configure the writing
    write_task.ao_channels.add_ao_voltage_chan(
            flatten_channel_string(channels_to_write),
            max_val=10, min_val=-10)
    writer = sw.AnalogMultiChannelWriter(write_task.out_stream)

    # source task.
    # Start the read and write tasks before starting the sample clock
    # source task.

    read_task.start()
    read_task.register_every_n_samples_acquired_into_buffer_event(
            20, # Every 20 samples, call callback function
            callback) 
    
    write_task.start()
    writer.write_many_sample(output_array)

    values_read = np.zeros((number_of_channels, samples_to_measure),
                           dtype=np.float64)
    reader.read_many_sample(
        values_read, number_of_samples_per_channel=samples_to_measure,
        timeout=2)
        
    non_local_var = {'samples': []}     
    
#    np.testing.assert_allclose(values_read, rtol=0.05, atol=0.005)

## Save measurement
#
#nchannels = nchannels + 1
#print("For {} channels, signal has size {}".format(
#        nchannels,
#        np.size(signal)))
#time = np.linspace(0, duration, samples_to_measure)
#try:
#    data = np.zeros((values_read[0,:], values_read[:,0]+1))
#    data[:,0] = time
#    data[:,1:] = values_read[0,:]
#    data[:,2:] = values_read[1,:]
#    data[:,3:] = values_read[2,:]
#except IndexError:
#    data = np.array([time, signal]).T
#np.savetxt(filename(nchannels), data, header=header)

#%% With new module!!! Change PWM mean value

#pwm_pin = 1 # Clock output
#pwm_frequency = 100e3
#pwm_duty_cycle = np.linspace(.1,1,10)
#
#device = daq.devices()[0]
#
#with daq.OldTask(device, mode='w') as task:
#    
#    # Configure clock output
#    task.add_channels(fch.PWMOutputChannel, pwm_pin)
#    task.pins[pwm_pin].frequency = pwm_frequency
#    task.pins[pwm_pin].duty_cycle = pwm_duty_cycle[0]
#    """Could do all this together:
#    task.add_channels(fch.PWMOutputChannel, pwm_pin,
#                      frequency = pwm_frequency,
#                      duty_cycle = pwm_duty_cycle)
#    """    
#    
#    task.pins(pwm_pin).status = True
#    for dc in pwm_duty_cycle:
#        task.pins[pwm_pin].duty_cycle = dc
#        print("Hope I changed duty cycle to {:.2f} x'D".format(dc))
#        sleep(3)
#    task.pins(pwm_pin).status = False

#%% With newer module!

"""Makes loop changing PWM output's duty cycle and therefore mean value"""

pwm_pin = 1 # Literally the number of the DAQ pin
pwm_frequency = 100e3
pwm_duty_cycle = np.linspace(.1,1,10)

device = daq.devices()[0]

with daq.Task(device, mode='w') as task:
    
    # Configure clock output
    task.add_channels(fch.PWMOutputChannel, pwm_pin)
    task.all.frequency = pwm_frequency
    task.all.duty_cycle = pwm_duty_cycle[0]
    """Could do all this together:
    task.add_channels(fch.PWMOutputChannel, pwm_pin,
                      frequency = pwm_frequency,
                      duty_cycle = pwm_duty_cycle)
    """    
    
    task.all.status = True
    for dc in pwm_duty_cycle:
        task.all.duty_cycle = dc
        """ Could also call by channel:
        task.ctr0.duty_cycle = dc
        """
        sleep(3)
    task.all.status = False
    task.close()

#%% Whith newer module too!

"""Makes a single measurement on analog input/s."""

ai_pins = [15]#, 16]#17]
ai_conf = 'Ref'

device = daq.devices()[0]

nsamples = int(200e3)

task = daq.Task(device, mode='r')

#with daq.Task(device, mode='r') as task:
        
# Configure clock output
task.add_channels(fch.AnalogInputChannel, *ai_pins)

signal = task.read(nsamples_total=10000, # 2 ch => 29500 sí, 30000 no
                   samplerate=None) # None means maximum SR

#task.close()
                   
#%% Whith newer module too!

"""Makes a continuous measurement on analog input/s."""

ai_pins = [15, 17]
ai_conf = 'Dif' # measures in differential mode

device = daq.devices()[0]

with daq.Task(device, mode='r') as task:
    
    # Configure clock output
    task.add_channels(fch.AnalogInputChannel, *ai_pins)
    task.all.configuration = ai_conf
    
    signal = task.read(nsamples_total=None, # None means continuous
                       samplerate=None) # None means maximum SR
    
    task.close()

#%% Prototype of control loop with new module!

ai_pin = 15 # default mode is non referenced

pwm_pin = 1 # literally the number of the DAQ pin
pwm_frequency = 100e3
pwm_default_duty_cycle = np.linspace(.1,1,10)
wheel_radius = 0.025 #in meters

device = daq.devices()[0]

nsamples_callback = 20
samplingrate = 100e3

pid = fan.PIDController(setpoint=1, kp=10, ki=5, kd=7, dt=1/samplingrate)

with daq.DAQ(device) as task, open('log.txt', 'w') as file:

    file.write('Vel\t duty_cycle\t P\t I\t D')
    def callback(task_handle, every_n_samples_event_type,
                 number_of_samples, callback_data):
            
        samples = task.read(nsamples_total=200,
                            samplerate=samplingrate) # None means maximum
        vel = fan.peak_separation(samples, pid.dt, prominence=.5) * wheel_radius
        new_dc = pid.calculate(vel)
        
        data = fan.append_data_to_string(vel, new_dc, pid.p_term, pid.i_term, pid.d_term)
        file.write(data)
        
        task.ouputs.duty_cycle = fan.set_bewtween(new_dc, *(0,100))    
    
    # Add channels
    task.add_analog_inputs(ai_pin)
    task.add_pwm_outputs(pwm_pin)
    
    # Configure all outputs    
    task.outputs.frequency = pwm_frequency
    task.outputs.duty_cycle = pwm_default_duty_cycle
    
    # Turn on outputs
    task.outputs.status = True
    
    # Start measurement
    signal = task.inputs.read(nsamples_total=None,
                              nsamples_callback=nsamples_callback,
                              callback=callback)
    
    task.output.status = False
    task.close()