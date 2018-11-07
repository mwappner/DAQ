# -*- coding: utf-8 -*-
"""
The 'fwp_daq_channels' module contains base classes for 'fwp_daq'.

@author: Usuario
"""

import fwp_string as fst
import nidaqmx as nid

conf = nid.constants.TerminalConfiguration
continuous = nid.constants.AcquisitionType.CONTINUOUS

#%%

class AnalogInputChannel:

    def __init__(self, device, task, streamer, pin,
                 voltage_range=[-10, 10], 
                 configuration='NonReferenced',
                 print_messages=False,
                 conection=True):

        """Initializes analog input channel.
        
        Parameters
        ----------
        device : str
            NI device's name where analog input channel/s should be put.
        task : nidaqmx.Task()
            NIDAQMX's task where this channels should be added.
        streamer : nidaqmx.stream_readers class
            NIDAMX's stream reader for this type of channel.
        pin : int
            Device's pin to initialize as analog input channel/s.
        voltage_range=[-10,10] : list, tuple, optional
            Range of the analog input channel. Should be a list or tuple 
            of length 2 that contains minimum and maximum in V.
        configuration='NRSE' : {'NR', 'R', 'DIFF', 'PSEUDODIFF'}, optional
            Analog input channel terminal configuration.
        print_messages=False : bool, optional
            Whether to print messages or not.
        conection=True : bool, optional
            Whether you have a real conection with a NI USB 6212 or not. 
            Allows to test classes.
        """          
        
        self.__device = device
        self.__task = task
        self.__streamer = streamer

        self.conection = conection
        if not conection:
            self.print = True
        else:
            self.print = print_messages
        
        self.pin = pin
        self.channel = self.__channel__(pin)

        if conection:
            ai_channel = self.__task.ai_channels.add_ai_voltage_chan(
                    physical_channel = self.channel,
                    units = nid.constants.VoltageUnits.VOLTS
                    )
            self.__channel = ai_channel
        else:
            self.__channel = None
            self.__print__("Should 'add_ai_voltage...'")
            
        self.configuration = configuration
        self.input_range = voltage_range
        
        self.gnd_pin = self.__gnd_pin__()
    
    @property
    def streamer(self):
        return self.__streamer
    
    @streamer.setter
    def streamer(self, streamer):
        self.__streamer = streamer
    
    @property
    def configuration(self):
        return self.__configuration
    
    @configuration.setter
    def configuration(self, mode):
        
        partial_keys = {
             ('&', 'ps') : conf.PSEUDODIFFERENTIAL,
             'dif' : conf.DIFFERENTIAL,
             ('&', 'no') : conf.NRSE,
             'ref' : conf.RSE,
             }
        mode = fst.string_recognizer(mode, partial_keys)
        
        try:
            self.__configuration
        except:
            self.__configuration = None
        
        if self.__configuration != mode:
            if self.conection:
                self.__channel.ai_term_cfg = mode
            else:
                self.__print__("Should 'ai_term_cfg'")
            self.__configuration = mode
    
    @property
    def input_range(self):
        return self.__range
    
    @input_range.setter
    def input_range(self, voltage_range):
        
        voltage_range = list(voltage_range)
        
        try:
            self.__range
        except:
            self.__range = [None, None]
        
        if self.__range != voltage_range:
            if self.__range[0] != voltage_range[0]:
                self.input_min = voltage_range[0]
            if self.__range[1] != voltage_range[1]:
                self.input_max = voltage_range[1]
            self.__range = voltage_range
    
    @property
    def input_min(self):
        return self.__input_min
    
    @input_min.setter
    def input_min(self, voltage):
        
        try:
            self.__input_min
        except:
            self.__input_min = None
        
        if self.__input_min != voltage:
            if self.conection:
                self.__channel.ai_min = voltage
            else:
                self.__print__("Should 'ai_min'")
            self.__input_min = voltage
            self.__range[0] = voltage
                
    @property
    def input_max(self):
        return self.__input_max
    
    @input_max.setter
    def input_max(self, voltage):

        try:
            self.__input_max
        except:
            self.__input_max = None
        
        if self.__input_max != voltage:
            if self.conection:
                self.__channel.ai_max = voltage
            else:
                self.__print__("Should 'ai_max'")
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
            channel = reference.index(pin)
            channel = '{}/ai{}'.format(self.__device, channel)
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
        
        reference = {15:16, 17:18, 19:20, 21:22, 
                     24:25, 26:27, 29:30, 31:32}
        
        try:
            diff_gnd_pin = reference[ai_pin]
            return diff_gnd_pin
        except ValueError:
            message = "Wrong pin {} for analog input"
            raise ValueError(message.format(ai_pin))

    def __gnd_pin__(self):
        
        if self.configuration == conf.DIFFERENTIAL:
            return self.__diff_gnd_pin__(self.pin)
        else:
            return 28
    
    def __print__(self, message):
        
        if self.print:
            print(message)
            
#%%

class PWMOutputChannel:

    def __init__(self, device, task, streamer, pin,
                 frequency=100e3, duty_cycle=0.5,
                 print_messages=False, conection=True):

        """Initializes PWM digital output channel.
        
        Parameters
        ----------
        device : str
            NI device's name where PWM output channel/s should be put.
        task : nidaqmx.Task()
            NIDAQMX's task where this channels should be added.
        streamer : nidaqmx.stream_writers class
            NIDAMX's stream writer for this type of channel.
        pin : int
            Device's pin to initialize as PWM output channel/s.
        frequency=100e3 : int, float, optional
            PWM's main frequency.
        duty_cycle=.5 : int, float, {0 <= duty_cycle <= 1}, optional
            PWM's duty cycle, which defines mean value of signal as 
            'duty_cycle*max' where 'max' is the '+5' voltage.
        print_messages=False : bool, optional
            Whether to print messages or not.
        conection=True : bool, optional
            Whether you have a real conection with a NI USB 6212 or not. 
            Allows to test classes.
        """          
        
        self.__device = device
        self.__task = task
        self.__streamer = streamer

        self.conection = conection
        if not conection:
            self.print = True
        else:
            self.print = print_messages
        
        self.pin = pin
        self.channel = self.__channel__(pin)
        self.low = 37#[5, 11, 37, 43]
        self.high = 42#[10, 42]
        
        if conection:
            ai_channel = self.__task.co_channels.add_co_pulse_chan_freq(
                counter = self.channel,
                units = nid.constants.FrequencyUnits.HZ
                )
            self.__channel = ai_channel
            self.__task.timing.cfg_implicit_timing(
                sample_mode = continuous)
        else:
            self.__channel = None
            print("Should 'add_co_pulse_chan...'+'timing.cfg_impli...'")
        
        self.__frequency = frequency
        self.__duty_cycle = duty_cycle
        self.__status = False
    
    @property
    def streamer(self):
        return self.__streamer
    
    @streamer.setter
    def streamer(self, streamer):
        self.__streamer = streamer
    
    @property
    def duty_cycle(self):
        return self.__duty_cycle
    
    @duty_cycle.setter
    def duty_cycle(self, value):
        
        if self.__duty_cycle != value:
            if self.conection:
                self.__channel.co_pulse_duty_cyc = value
            else:
                self.__print__("Should 'co_pulse_duty...'")
            self.__duty_cycle = value
            if self.status:
                self.status = 'R' # reconfigures
                
    @property
    def frequency(self):
        return self.__frequency
    
    @frequency.setter
    def frequency(self, value):
        
        if self.__frequency != value:
            if self.conection:
                self.__channel.co_pulse_freq = value
            else:
                self.__print__("Should 'co_pulse_freq'")
            self.__frequency = value
            if self.status:
                self.status = 'R'
    
    @property
    def status(self):
        return self.__status

    @status.setter
    def status(self, key):
        
        if self.__status != key:
            if self.conection:
                if key:
                    self.__task.start()
                    self.__stream.write_one_sample_pulse_frequency(
                            frequency = self.frequency,
                            duty_cycle = self.duty_cycle,
                            )
                else:
                    self.__task.end()
            else:
                self.__print__("Should 'start' or 'stop'")
            if isinstance(key, str):
                self.__status = True
            else:
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
        
        reference = [38, 39]
#        reference = [1, 2, 3, 4, 6, 7, 8, 9, 33, 
#                     34, 35, 36, 38, 39, 40, 41]
        
        try:
            channel = reference.index(pin)
            channel = '{}/ctr{}'.format(self.__device, channel)
            return channel
        except ValueError:
            message = "Wrong pin {} for digital input"
            raise ValueError(message.format(pin))
    
    def __print__(self, message):
        
        if self.print:
            print(message)