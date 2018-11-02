# -*- coding: utf-8 -*-
"""
The 'fwp_daq' module is designed to measure with NI USB 6212.

Based on https://github.com/fotonicaOrg/daq.git from commit 387c7c3

"""

import fwp_string as fst
#import numpy as np
import nidaqmx as nid
#import nidaqmx.stream_readers as sr
import nidaqmx.stream_writers as sw
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

class DynamicDic:
    
    def __init__(self, dic):
        self.__dic = dic
    
    def __call__(self, *key):

        if len(key) == 1:
            return self.__dic[key[0]]
        
        else:
            return [self.__dic[k] for k in key]
    
    def update(self, dic):
        
        self.__dic.update(dic)
    
    def is_empty(self, key):
        
        if key in self.__dic.keys():
            return False
        else:
            return True

class DynamicList:
    
    def __init__(self, l):
        self.__list = l
    
    def __call__(self, *index):
        
        if len(index) == 1:
            return self.__list[index[0]]
        
        else:
            return [self.__list[i] for i in index]
    
    def update(self, index, element):
        
        try:
            self.__list[index] = element
        except IndexError:
            self.__list.append(element)

#%%

class Task:
    
    def __init__(self, device):
        
        self.__device = device
        self.__task = nid.Task()
        self.__pins = DynamicDic()
        
        self.__analog_inputs = DynamicDic()
        self.__pwm_outputs = DynamicDic()
        
        self.__pwm_stream = None
        
    @property
    def pins(self):
        return self.__pins
    
    @pins.setter
    def pins(self, value):
        raise ValueError("You shouldn't modify this manually!")
    
    @property
    def analog_inputs(self):
        return self.__analog_inputs
    
    @analog_inputs.setter
    def analog_inputs(self, *pins, **kwargs):
        
        new_channels = []
        for p in pins:
            if self.__analog_inputs.is_empty(p):
                channel = self.__add_channels__(
                    self.__device,
                    self.__task,
                    AnalogInputChannel,
                    p,
                    kwargs)
                new_channels[p] = channel
        
        self.pins.update({p : 'AnalogInputChannel' for p in pins})
        self.__analog_inputs.update(new_channels)
    
    @property
    def pwm_outputs(self):
        return self.__pwm_outputs
    
    @pwm_outputs.setter
    def pwm_outputs(self, *pins, **kwargs):
        
        new_channels = []
        for p in pins:
            if self.__analog_inputs.is_empty(p):
                channel = self.__add_channels__(
                    self.__device,
                    self.__task,
                    PWMOutputChannel,
                    p,
                    kwargs)
                new_channels[p] = channel
        
        self.pins.update({p : 'AnalogInputChannel' for p in pins})
        self.__pwm_outputs.update(new_channels)
    
    def __add_channel__(self, device, task, ChannelClass, pin, **kwargs):
        
        channel = ChannelClass(device, task, pin, **kwargs)
        
        return channel

#%%

class AnalogInputChannel:

    def __init__(self, device, task, pin,
                 voltage_range=[-10, 10], 
                 configuration='NRSE'):

        """Initializes analog input channel.
        
        Parameters
        ----------
        device : str
            NI device's name where analog input channel/s should be put.
        task : nidaqmx.Task()
            NIDAQMX's task where this channels should be added.
        pin : int
            Device's pin to initialize as analog input channel/s.
        voltage_range=[-10,10] : list, tuple, optional
            Range of the analog input channel/s. Each of them should be 
            a list or tuple of length 2 that contains minimum and maximum 
            in V.
        configuration='NRSE' : {'NR', 'R', 'DIFF', 'PSEUDODIFF'}, optional
            Analog input channel/s terminal configuration.
        """          
        
        self.__device = device
        self.__task = task
        
        self.pin = pin
        self.channel = self.__channel__(pin)
        self.gnd_pin = self.__gnd_pin__()

        ai_channel = self.__task.ai_channels.add_ai_voltage_chan(
                physical_channel = self.channel,
                units = nid.constants.VoltageUnits.VOLTS
                )
        self.__channel = ai_channel
        
        self.configuration = configuration
        self.input_range = voltage_range
    
    @property
    def configuration(self):
        return self.__configuration
    
    @configuration.setter
    def configuration(self, mode):
        
        partial_keys = {
             ('&', 'ps') : conf.PSEUDODIFFERENTIAL,
             'dif' : conf.DIFFERENTIAL,
             ('&', 'n') : conf.NRSE,
             'r' : conf.RSE,
             }
        mode = fst.string_recognizer(mode, partial_keys)
        
        if self.__configuration != mode:
            self.__channel.ai_term_cfg = mode
            self.__configuration = mode
    
    @property
    def input_range(self):
        return self.__range
    
    @input_range.setter
    def input_range(self, voltage_range):
        
        voltage_range = list(voltage_range)
        if self.__range != voltage_range:
            if self.__range[0] != voltage_range[0]:
                self.input_min = voltage_range[0]
            if self.__range[1] != voltage_range[1]:
                self.input_max = voltage_range[1]
            self.__configuration = voltage_range
    
    @property
    def input_min(self):
        return self.__input_min
    
    @input_min.setter
    def input_min(self, voltage):
        
        if self.__input_min != voltage:
            self.__channel.ai_min = voltage
            self.__input_min = voltage
            self.__range[0] = voltage
                
    @property
    def input_max(self):
        return self.__input_max
    
    @input_max.setter
    def input_max(self, voltage):
        
        if self.__input_max != voltage:
            self.__channel.ai_max = voltage
            self.__input_max = voltage
            self.__range[1] = voltage
    
    def __channel__(self, pin):
        
        """Transforms from pin to analog input channel.
        
        Parameters
        ----------
        pin : int
            Pin (physical channel). Should be an int.
        
        Returns
        -------
        channel : str
            Channel. A string "Dev{}/ai{}" formatted by two int.
        """
        
        reference = [15, 17, 19, 21, 24, 26, 29, 31, 
                     16, 18, 20, 22, 25, 27, 30, 32]
        
        try:
            channel = reference[pin]
            channel = '{}/ai{}'.format(self.__device, pin)
            return channel
        except IndexError:
            message = "Wrong pin {} for analog input"
            raise ValueError(message.format(pin))
    
    def __diff_gnd_pin__(self, ai_pin):
        
        """Returns differential GND for an analog input pin.
        
        Parameters
        ----------
        ai_pin : int
            Pin (physical channel) of analog input. Should be int.
        
        Returns
        -------
        diff_gnd_pin : int
            Pin (physical channel) of analog input's GND.
        """
        
        reference = [16, 18, 20, 22, 25, 27, 30, 32]
        
        try:
            diff_gnd_pin = reference[ai_pin]
            return diff_gnd_pin
        except IndexError:
            message = "Wrong pin {} for analog input"
            raise ValueError(message.format(ai_pin))

    def __gnd_pin__(self):
        if self.configuration == conf.DIFFERENTIAL:
            return self.__diff_gnd_pin__(self.pin)
        else:
            return 28
            
#%%

class PWMOutputChannel:

    def __init__(self, device, task, pin,
                 stream = None,
                 frequency=100e3, 
                 duty_cycle=0.5):

        """Initializes PWM digital output channel.
        
        Parameters
        ----------
        device : str
            NI device's name where PWM output channel/s should be put.
        task : nidaqmx.Task()
            NIDAQMX's task where this channels should be added.
        pin : int
            Device's pin to initialize as PWM output channel/s.
        frequency=100e3 : int, float, optional
            PWM's main frequency.
        duty_cycle=.5 : int, float, {0 <= duty_cycle <= 1}, optional
            PWM's duty cycle, which defines mean value of signal as 
            'duty_cycle*max' where 'max' is the '+5' voltage.        
        """          
        
        self.__device = device
        self.__task = task
        self.__stream = stream
        
        self.pin = pin
        self.channel = self.__channel__(pin)
        self.low = [5, 11, 37, 43]
        self.high = [10, 42]
        
        ai_channel = self.__task.ai_channels.add_ai_voltage_chan(
                physical_channel = self.channel,
                units = nid.constants.VoltageUnits.VOLTS
                )
        self.__channel = ai_channel
        self.__task.timing.cfg_implicit_timing(
            sample_mode = nid.constants.AcquisitionType.CONTINUOUS)
        
        self.frequency = frequency
        self.duty_cycle = duty_cycle
    
    @property
    def duty_cycle(self):
        return self.__duty_cycle
    
    @duty_cycle.setter
    def duty_cycle(self, value):
        
        if self.__duty_cycle != value:
            self.__channel.co_pulse_duty_cyc = value
            self.__duty_cycle = value
            if self.status:
                self.status = 'R'
                
    @property
    def frequency(self):
        return self.__frequency
    
    @frequency.setter
    def frequency(self, value):
        
        if self.__frequency != value:
            self.__channel.co_pulse_freq = value
            self.__frequency = value
            if self.status:
                self.status = 'R'
    
    @property
    def status(self):
        return self.__status

    @status.setter
    def status(self, key):
        
        if isinstance(key, str):
            if self.__stream is None:
                self.__stream = sw.CounterWriter(self.__task.out_stream)
            self.__stream.write_one_sample_pulse_frequency(
                frequency = self.frequency,
                duty_cycle = self.duty_cycle,
                )
        elif isinstance(key, bool):
            if self.__status != key:
                if key:
                    if self.__stream is None:
                        self.__stream = sw.CounterWriter(
                                self.__task.out_stream)
                    self.__stream.write_one_sample_pulse_frequency(
                        frequency = self.frequency,
                        duty_cycle = self.duty_cycle,
                        )
                self.__status = key
    
    def __channel__(self, pin):
        
        """Transforms from pin to PWM digital output channel.
        
        Parameters
        ----------
        pin : int, list
            Pin (physical channel). Should be an int.
        
        Returns
        -------
        channel : str, list
            Channel. A string "Dev{}/ctr{}" formatted by two int.
        """
        
        reference = [1, 2, 3, 4, 6, 7, 8, 9, 33, 
                     34, 35, 36, 38, 39, 40, 41]
        
        try:
            channel = reference[pin]
            channel = '{}/ctr{}'.format(self.__device, pin)
            return channel
        except IndexError:
            message = "Wrong pin {} for digital input"
            raise ValueError(message.format(pin))

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
