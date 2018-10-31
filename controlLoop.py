# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:57:40 2018

@author: mfar
"""

import fwp_analysis as anly
import fwp_lab_instruments as ins
import fwp_save as sav
import matplotlib.pyplot as plt
import os
import nidaqmx as nid
import numpy as np
from time import sleep

from nidaqmx.utils import flatten_channel_string
from nidaqmx.stream_readers import (AnalogSingleChannelReader, AnalogMultiChannelReader)
from nidaqmx.stream_writers import (AnalogSingleChannelWriter, AnalogMultiChannelWriter)


#%% streamers (pag 57)
samplerate = 400e3
mode = nid.constants.TerminalConfiguration.NRSE

signal_frequency = 10
signal_pk_amplitude = 2
periods_to_measure = 50
gen_port = 'ASRL1::INSTR'
gen_totalchannels = 2 # Ojo que no siempre hay dos canales

name = 'Multichannel_Settling_Time'

# Other parameters
duration = periods_to_measure/signal_frequency
samples_to_measure = int(samplerate * duration/1000)

number_of_channels=3
channels_to_test = [
                    "Dev20/ai0",
                    "Dev20/ai1",
                    "Dev20/ai2",
                    ]

gen = ins.Gen(port=gen_port, nchannels=gen_totalchannels)
signal_slope = signal_pk_amplitude * signal_frequency

#folder = os.path.join(os.getcwd(),
#                      'Measurements',
#                      name)
#folder = sav.new_dir(folder)
#filename = lambda nchannels : os.path.join(
#        folder, 
#        'NChannels_{}.txt'.format(nchannels))
filename = sav.savefile_helper(dirname = name, 
                               filename_template = 'NChannels_{}.txt')

header = 'Time [s]\tData [V]'

# ACTIVE CODE
pidvalue=20
pidconstant=2
gen.output(True, waveform='ramp100', 
           frequency=signal_frequency,
           amplitude=2)

values_to_test = np.array(
            [[random.uniform(-10, 10) for _ in range(number_of_samples)]
             for _ in range(number_of_channels)], dtype=np.float64):
                 
                 
with nid.Task() as write_task, nid.Task() as read_task:
    def callback(task_handle, every_n_samples_event_type,
                 number_of_samples, callback_data):
        print('Every N Samples callback invoked.')

        samples = read_task.read(number_of_samples_per_channel=20)
        non_local_var['samples'].extend(samples)
        if max(samples)>(pidvalue+2):
            delta=max(samples)-pidvalue
            values_to_test = values_to_test+pidconstant*delta
        elif max(samples)<(pidvalue-2):
             delta=pidvalue-max(samples)
            values_to_test = values_to_test-pidconstant*delta
        return 0
    read_task.ai_channels.add_ai_voltage_chan(
        flatten_channel_string(channels_to_test),
        max_val=10, min_val=-10)
    reader = AnalogMultiChannelReader(read_task.in_stream)


    write_task.ao_channels.add_ao_voltage_chan(
            flatten_channel_string(channels_to_test),
            max_val=10, min_val=-10)
    writer = AnalogMultiChannelWriter(write_task.out_stream)

    writer = AnalogMultiChannelWriter(write_task.out_stream)

    # source task.
    # Start the read and write tasks before starting the sample clock
    # source task.

    read_task.start()
    read_task.register_every_n_samples_acquired_into_buffer_event(
    20, callback) 
    write_task.start()
    values_read = np.zeros(
        (number_of_channels, samples_to_measure), dtype=np.float64)
    reader.read_many_sample(
        values_read, number_of_samples_per_channel=samples_to_measure,
        timeout=2)
        
    non_local_var = {'samples': []}



       
        

    #np.testing.assert_allclose(values_read, rtol=0.05, atol=0.005)
# Save measurement

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
#
#gen.output(False)
#gen.gen.close()