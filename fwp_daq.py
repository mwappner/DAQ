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
import nidaqmx.constants.TerminalConfiguration as conf
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

def dynamic_list(inlist):
    
    if inlist is None:
        return(None)
    if len(inlist) == 0:
        raise TypeError('Empty list!')
    if len(inlist) == 1:
        return inlist[0]
    else:
        return inlist

#%%

class Task:
    
    def __init__(self, device):
        
        self.__device = device
        self.__task = nid.Task()
        self.__analog_inputs = None
        self.__pwm_outputs = None
    
    @property
    def analog_inputs(self):
        return self.__analog_inputs
    
    @analog_inputs.setter
    def analog_inputs(self, pins):
        
        if self.__analog_inputs is None:
            self.__analog_inputs = AnalogInputs(
                    pins=pins,
                    device=self.__device,
                    task=self.__task,
                    )
        else:
            voltage_range = self.__analog_inputs.range
            configuration = self.__analog_inputs.configuration
            self.__analog_inputs = AnalogInputs(
                    pins,
                    voltage_range=voltage_range,
                    configuration=configuration,
                    device=self.__device,
                    task=self.__task,
                    )
        
    @property
    def pwm_outputs(self):
        return self.__pwm_outputs

    
#%%

class AnalogInputs:

    def init(self, pins, device=None, task=None, 
             voltage_range=[-10, 10], 
             configuration=conf.NRSE):

        """Initializes analog input channel/s.
        
        Parameters
        ----------
        device : str
            NI device's name where analog input channel/s should be put.
        pins : int, list
            Device's pins to initialize as analog input channel/s.
        voltage_range=[-10,10] : list
            Range of the analog input channel/s. Each of them should be 
            a list or tuple that contains minimum and maximum in V.
        configuration=NRSE : nid.constants, optional
            Analog input channel/s terminal configuration.
        
        Returns
        -------
        ai_channels : list, nid.task.ai_channels.add_ai_voltage_chan
            Analog input channel/s object/s.
        
        """          
        
        self.__device = device
        self.__task = task
        self.pins = pins
        # This attribute sets also...
        # ...self.channels,
        # ...self.__ai_channels, 
        # ...self.nchannels
        
        self.configuration = configuration
        self.input_range = voltage_range
    
    @property
    def pins(self):
        return dynamic_list(self.__pins)
    
    @pins.setter
    def pins(self, pins):
        
        self.__pins = pins
        self.__channels = self.__channels__(pins)
        if pins is None:
            self.__nchannels = 0
        else:
            self.__nchannels = len(self.channels)
        
        ai_channels = []
        for i, ch in enumerate(self.channels):
            ai_channels.append(
                self.__task.ai_channels.add_ai_voltage_chan(
                    physical_channel = ch,
                    units = nid.constants.VoltageUnits.VOLTS
                    )
                )
        self.__ai_channels = ai_channels
    
    @property
    def channels(self):
        return dynamic_list(self.__channels)
    
    @channels.setter
    def channels(self, channels):
        message = "Can't be manually set!"
        message = message + "Must set AnalogInputs.pins"
        raise AttributeError(message)

    @property
    def nchannels(self):
        return self.__nchannels
    
    @nchannels.setter
    def nchannels(self, channels):
        message = "Can't be manually set!"
        message = message + "Must set AnalogInputs.pins"
        raise AttributeError(message)
    
    @property
    def configuration(self):
        return dynamic_list(self.__configuration)
    
    @configuration.setter
    def configuration(self, mode):
        
        if not isinstance(mode, list):
            mode = [mode for ch in self.channels]
        elif len(mode) != self.nchannels:
            raise ValueError("Not enough elements in configuration")
        
        if self.nchannels == 1:
            self.__channels.ai_term_cfg = mode[0]
        else:
            for i, m in enumerate(mode):
                if self.__configuration[i] != m:
                    self.__channels[i].ai_term_cfg = m
        self.__configuration = mode
    
    @property
    def input_range(self):
        return self.__range
    
    @input_range.setter
    def input_range(self, voltage_range):
        
        if not isinstance(voltage_range, list):
            voltage_range = [voltage_range]
        if len(voltage_range) != self.nchannels:
            raise ValueError("Not enough elements in input_range")
        for vr in voltage_range:
            if len(vr) != 2:
                print("{} should be tuple or list of lenght 2")
        
        for i, v in enumerate(voltage_range):
            if self.__range[i] != v:
                if self.__range[i][0] != v[0]:
                    self.__ai_channels[i].ai_min = v[0]
                    self.__input_min[i] = v[0]
                if self.__range[i][1] != v[1]:
                    self.__ai_channels[i].ai_max = v[1]
                    self.__input_max[i] = v[1]
        self.__configuration = voltage_range
        
    @property
    def input_min(self):
        return self.__input_min
    
    @input_min.setter
    def input_min(self, voltage):
        
        if not isinstance(voltage, list):
            voltage = [voltage]
        if len(voltage) != self.nchannels:
            raise ValueError("Not enough elements in input_min")
        
        for i in range(self.nchannels):
            if self.__input_min[i] != voltage[i]:
                self.__ai_channels[i].ai_min = voltage[i]
        self.__input_min = voltage
                
    @property
    def input_max(self):
        return self.__input_max
    
    @input_max.setter
    def input_max(self, voltage):
        
        if not isinstance(voltage, list):
            voltage = [voltage]
        if len(voltage) != self.nchannels:
            raise ValueError("Not enough elements in input_min")
        
        for i in range(self.nchannels):
            if self.__input_max[i] != voltage[i]:
                self.__ai_channels[i].ai_max = voltage[i]
        self.__input_max = voltage
    
    
    def __channels__(self, pins):
        
        """Transforms from pin to analog input channel.
        
        Parameters
        ----------
        pins : int, list
            Pins to be initialized. Each of them should be an int from 
            0 to 15.
        
        Returns
        -------
        channels : str, list
            Channels
        """
        
        if pins is None:
            return None
        
        reference = [15, 17, 19, 21, 24, 26, 29, 31, 
                     16, 18, 20, 22, 25, 27, 30, 32]
        
        channels = []
        for p in pins:
            try:
                p = reference[p]
                p = '{}/ai{}'.format(self.device, p)
            except IndexError:
                message = "Wrong pin {} for analog input"
                raise ValueError(message.format(p))
            channels.append(p)
        
        return channels

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
