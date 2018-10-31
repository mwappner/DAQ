# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import nidaqmx
import collections
from matplotlib import pyplot as plt
import nidaqmx.stream_readers
import nidaqmx.system
import time
#system = nidaqmx.system.System.local()
#
#for device in system.devices:
#    print(device)

def check_device():
    
    import nidaqmx.system
    
    system = nidaqmx.system.System.local()
    
    for dev in system.devices:
        print(dev)

def configure_ai(
        task,
        physical_channels,
        voltage_range,
        terminal_configuration,
        acquisition_units = nidaqmx.constants.VoltageUnits.VOLTS,
        ):

    if not isinstance(voltage_range, tuple):
        voltage_range = [voltage_range]
        
    if not isinstance(terminal_configuration, tuple):
        terminal_configuration = [terminal_configuration]
    
    if not isinstance(physical_channels, tuple):
        physical_channels = [physical_channels]
    
    chan = []
    
    for i in range(len(physical_channels)):
        chan.append(
                task.ai_channels.add_ai_voltage_chan(
                    physical_channel = physical_channels[i],
                    min_val = voltage_range[i][0],
                    max_val = voltage_range[i][1],
                    units = acquisition_units
                    )
                )
                
        chan[i].ai_term_cfg = terminal_configuration[i]
    
    return chan

def configure_pwm(
        task,
        physical_channels,
        frequency,
        duty_cycle,
        frequency_units = nidaqmx.constants.FrequencyUnits.HZ,
        ):

    if not isinstance(frequency, tuple):
        frequency = [frequency]
    
    if not isinstance(duty_cycle, tuple):
        duty_cycle = [duty_cycle]
    
    if not isinstance(physical_channels, tuple):
        physical_channels = [physical_channels]
    
    chan = []
    
    for i in range(len(physical_channels)):
        chan.append(
            task.co_channels.add_co_pulse_chan_freq(
                    counter = physical_channels[i],
                    freq = frequency[i],
                    duty_cycle = duty_cycle[i],
                    units = frequency_units
                    )
                )
    
    task.timing.cfg_implicit_timing(
            sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS
            )
    
    return chan


def acquire(
        task,
        n_samples,
        sample_frequency,
        ):
    
    task.timing.cfg_samp_clk_timing(sample_frequency)
    
    data = np.zeros((task.number_of_channels, n_samples))
    reader = nidaqmx.stream_readers.AnalogMultiChannelReader(task.in_stream)
    reader.read_many_sample(data, -1)
    
    return (data, task.timing.samp_clk_rate)





def continuous_acquire(
        task,
        n_samples,
        sample_frequency,
        task_co = None,
        chan_co = None,
        stream_co = None
        ):
    
    print('Acquire')
    
    data_count = 0
    n_channels = task.number_of_channels
    
    task.timing.cfg_samp_clk_timing(
            rate = sample_frequency,
            sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS
            )
    task.in_stream.input_buf_size = 5 * n_samples
    
    data = np.zeros((n_channels, n_samples))
    
    try:
        print('Start')
        task.start()
        
        while True:
#            
#            data[0:n_channels, 0:n_samples] = np.array(task.read(n_samples))
#            data_count += n_samples
            
            time.sleep(0.5)
#            stream_co.write_one_sample_pulse_frequency(
#                    frequency = chan_co.co_pulse_freq,
#                    duty_cycle = 0.9
#                    )
#
#            time.sleep(0.5)
#            stream_co.write_one_sample_pulse_frequency(
#                    frequency = chan_co.co_pulse_freq,
#                    duty_cycle = 0.1
#                    )
            
#            duty = 0.1
#            if stream_co != None:
#                time.sleep(1)
#                
#                if duty < 0.5:
#                    stream_co.write_one_sample_pulse_frequency(
#                            frequency = chan_co.co_pulse_freq,
#                            duty_cycle = 0.9
#                            )
#                    duty = 0.9
#                elif duty >= 0.5:
#                    stream_co.write_one_sample_pulse_frequency(
#                            frequency = chan_co.co_pulse_freq,
#                            duty_cycle = 0.1
#                            )
#                    duty = 0.1
#                print(duty)
        
    except KeyboardInterrupt:
        
        task.stop()
        if task_co != None: task_co.stop()
        
        return (data, task.timing.samp_clk_rate)


if __name__ is '__main__':
    
    print(check_device())
    
    chan_col = nidaqmx.system._collections.physical_channel_collection.COPhysicalChannelCollection('Dev1')
    print(chan_col.channel_names)
    
#
#SampleNumber = 2**16
#SampleRate = 10000
#
#with nidaqmx.Task() as task:
#    
#    
#    
#    
#    
#      
##data = data.transpose()    
#data = data[0,:]
#data_fft = 2*np.fft.fft(data)/len(data)
#
#data_fft = np.abs(data_fft[0:len(data_fft)//2])
#freq = np.linspace(0,SampleRate/2,len(data_fft))
#f_medida = freq[np.argmax(data_fft)]
#
#print(f_medida)
#
#
#plt.plot(freq,data_fft)
#
#
#
#    
#    
#    
#    
#    
#    