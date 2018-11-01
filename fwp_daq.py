# -*- coding: utf-8 -*-
"""
The 'fwp_daq' module is designed to measure with NI USB 6212.

Based on https://github.com/fotonicaOrg/daq.git from commit 387c7c3

"""

#import numpy as np
import nidaqmx as nid
#import nidaqmx.stream_readers as sr
#import nidaqmx.stream_writers as sw
import nidaqmx.system as sys
#from matplotlib import pyplot as plt
#import time

#%%

def devices():
    
    """Returns list of NI Devices.
    
    Parameters
    ----------
    nothing
    
    Returns
    -------
    devices : list
        List of NI Devices' names as string.
    """
    
    devices = []
    for dev in sys.System.local().devices:
        dev = str(dev)
        dev = dev.split('=')[-1].split(')')[0]
        devices.append(dev)
        print(dev)
    
    return devices

#%%

def analog_inputs(
        task,
        physical_channels,
        voltage_range=[-10, 10],
        configuration=nid.constants.TerminalConfiguration.NRSE
        ):
    
    """Initializes analog input channel/s.
    
    Parameters
    ----------
    task : nid.Task()
        Task where to initialize analog input channel/s to.
    physical_channels : str, list
        Pyhsical channel/s to be initialized as analog input channel/s.
    voltage_range=[-10,10] : list
        Range of the analog input channel/s. Each of them should be a 
        list or tuple that contains minimum and maximum in V.
    configuration=NRSE : nid.constants.TerminalConfiguration, optional
        Analog input channel/s terminal configuration.
    
    Returns
    -------
    ai_channels : list, nid.task.ai_channels.add_ai_voltage_chan
        Analog input channel/s object/s.
    """    

    if not isinstance(physical_channels, tuple):
        physical_channels = [physical_channels]

    if not isinstance(voltage_range, list):
        voltage_range = [voltage_range for ch in physical_channels]
        
    if not isinstance(configuration, list):
        configuration = [configuration 
                         for ch in physical_channels]
    
    ai_channels = []
    for i, ch in enumerate(physical_channels):
        ai_channels.append(
            task.ai_channels.add_ai_voltage_chan(
                physical_channel = ch,
                min_val = voltage_range[i][0],
                max_val = voltage_range[i][1],
                units = nid.constants.VoltageUnits.VOLTS
                )
            )
        ai_channels[i].ai_term_cfg = configuration[i]
    
    if len(ai_channels) > 1:
        return ai_channels
    else:
        return ai_channels[0]
    

#%%

def pwm_outputs(
        task,
        physical_channels,
        duty_cycle=.5,
        frequency=10e3
        ):

    """Initializes digital PWM output channel/s.
    
    Parameters
    ----------
    task : nid.Task()
        Task where to initialize analog input channel/s to.
    physical_channels : str, list
        Pyhsical channel/s to be initialized as analog input channel/s.
    duty_cycle=.5 : int, float {0<=duty_cycle<=1}
        Duty cycle to be set on digital PWM output channel/s at start.
    frequency=10e3 : int, float
        Frequency in Hz to be set on digital PWM output channel/s.
    
    Returns
    -------
    pwm_channels : list, nid.task.ai_channels.add_ai_voltage_chan
        Digital PWM output channel/s object/s.
    """    
    
    if not isinstance(physical_channels, list):
        physical_channels = [physical_channels]
    
    if not isinstance(frequency, list):
        frequency = list(frequency for ch in physical_channels)
    
    if not isinstance(duty_cycle, list):
        duty_cycle = list(duty_cycle for ch in physical_channels)
    
    pwm_channels = []
    for i in range(len(physical_channels)):
        pwm_channels.append(
            task.co_channels.add_co_pulse_chan_freq(
                    counter = physical_channels[i],
                    freq = frequency[i],
                    duty_cycle = duty_cycle[i],
                    units = nid.constants.FrequencyUnits.HZ
                    )
                )
    
    task.timing.cfg_implicit_timing(
            sample_mode = nid.constants.AcquisitionType.CONTINUOUS
            )
    
    if len(pwm_channels) > 1:
        return pwm_channels
    else:
        return pwm_channels[0]
