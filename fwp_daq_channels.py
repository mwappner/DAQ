# -*- coding: utf-8 -*-
"""
The 'fwp_daq_channels' module contains base classes for 'fwp_daq'.

@author: Vall
"""

import fwp_string as fst
import nidaqmx as nid

conf = nid.constants.TerminalConfiguration
continuous = nid.constants.AcquisitionType.CONTINUOUS

#%%

class AnalogInputChannel:

    """Manages an analog input channel for a NI USB 6212 DAQ.
    
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

    Other Parameters
    ----------------
    voltage_range=[-10,10] : list, tuple, optional
        Range of the analog input channel. Should be a list or tuple 
        of length 2 that contains minimum and maximum in V.
    configuration='NonReferenced' : optional
        Analog input channel terminal configuration.
    print_messages=False : bool, optional
        Whether to print messages or not.
    test_mode=False : bool, optional
        Allows to test classes even if you don't have a real 
        conection with a NI USB 6212.
    
    Attributes
    ----------
    pin : int
        Number of DAQ pin.
    channel : str
        Name of DAQ channel.
    streamer : nidaqmx.stream_readers
        DAQ's reading manager.
    configuration : nidaqmx.constants.TerminalConfiguration
        Analog channel's terminal configuration. Could be: referenced 
        (measures against GND), non-referenced (measures against SENSE),
        differential (measureas against another analog input pin).
    range : list
        Analog channel's input range in V. Must be a list with 2 
        elements (first one is minimum voltage ande second one, maximum).
    input_min : int, float
        Analog channel's input minimum voltage in V.
    input_max : int, float
        Analog channel's input maximum voltage in V.
    gnd_pin : int
        Number of DAQ pin that holds the other terminal of this channel. 
        Could be an analog pin (differential mode), SENSE pin 
        (non-referenced mode) or GND pin (referenced mode)
    
    Other Attributes
    ----------------
    write_mode : bool
        Whether this is a writing class or a reading one.
    test_mode : bool
        Whether it's test mode (no real connection) or not.
    print : bool
        Whether to print inner messages or not.
        
    """   
    
    def __init__(self, device, task, streamer, pin,
                 voltage_range=[-10, 10], 
                 configuration='NonReferenced',
                 print_messages=False,
                 test_mode=False):       
        
        # First, set general DAQ attributes
        self.__device = device
        self.__task = task
        self.streamer = streamer

        # Is there a real DAQ connected or are we just testing?
        self.test_mode = test_mode
        if test_mode:
            self.print = True
        else:
            self.print = print_messages

        # Some general attributes of this channel        
        self.pin = pin
        self.channel = self.__channel__(pin)

        # Add this channel to the DAQ
        if not self.test_mode:
            ai_channel = self.__task.ai_channels.add_ai_voltage_chan(
                    physical_channel = self.channel,
                    units = nid.constants.VoltageUnits.VOLTS
                    )
            self.__channel = ai_channel
        else:
            self.__channel = None
            self.__print__("Should 'add_ai_voltage...'")

        # Configure this channel
        self.configuration = configuration
        self.input_range = voltage_range
    
    @property
    def configuration(self):
        return self.__configuration
    
    @configuration.setter
    def configuration(self, mode):
        
        # Recognize mode
        partial_keys = {
             ('&', 'ps') : conf.PSEUDODIFFERENTIAL,
             'dif' : conf.DIFFERENTIAL,
             ('&', 'no') : conf.NRSE,
             'ref' : conf.RSE,
             }
        mode = fst.string_recognizer(mode, partial_keys)
        
        # Check if I need to reconfigure
        try:
            needs_reconfiguration = self.__configuration != mode
        except:
            needs_reconfiguration = True
            
        # Reconfigure if needed
        if needs_reconfiguration:
            if not self.test_mode:
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
        if len(voltage_range) != 2:
            raise AttributeError("Range must have length 2")
        
        # Check if I need to reconfigure
        try:
            needs_reconfiguration = self.__range != voltage_range
        except:
            needs_reconfiguration = True
            
        # Reconfigure if needed
        if needs_reconfiguration:
            self.__range = voltage_range
            self.input_min = voltage_range[0]
            self.input_max = voltage_range[1]
    
    @property
    def input_min(self):
        return self.__input_min
    
    @input_min.setter
    def input_min(self, voltage):
        
        # Check if I need to reconfigure
        try:
            needs_reconfiguration = self.__input_min != voltage
        except:
            needs_reconfiguration = True
            
        # Reconfigure if needed
        if needs_reconfiguration:
            if not self.test_mode:
                self.__channel.ai_min = voltage
            else:
                self.__print__("Should 'ai_min'")
            self.__input_min = voltage
            self.__range = (voltage, self.__range[1])
                
    @property
    def input_max(self):
        return self.__input_max
    
    @input_max.setter
    def input_max(self, voltage):

        # Check if I need to reconfigure
        try:
            needs_reconfiguration = self.__input_max != voltage
        except:
            needs_reconfiguration = True
            
        # Reconfigure if needed
        if needs_reconfiguration:
            if not self.test_mode:
                self.__channel.ai_max = voltage
            else:
                self.__print__("Should 'ai_max'")
            self.__input_max = voltage
            self.__range = (self.__range[0], voltage)
    
    @property
    def gnd_pin(self):   
        if self.configuration == conf.DIFFERENTIAL:
            return self.__diff_gnd_pin__(self.pin)
        elif self.configuration == conf.NRSE:
            return 23
        else:
            return 28
    
    @gnd_pin.setter
    def gnd_pin(self, value):
        raise AttributeError("Can't modify this manually!")
    
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
    
    def __print__(self, message):
        
        """Only prints if self.print is True.
        
        Parameters
        ----------
        message : str
            Message to print
        
        Returns
        -------
        nothing
        
        Raises
        ------
        print
        
        """
        
        if self.print:
            print(message)
            
#%%

class PWMOutputChannel:

    """Manages a PWM digital output channel for a NI USB 6212 DAQ.
    
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
        
    Other Parameters
    ----------------
    frequency=100e3 : int, float, optional
        PWM's main frequency.
    duty_cycle=.5 : int, float, {0 <= duty_cycle <= 1}, optional
        PWM's duty cycle, which defines mean value of signal as 
        'duty_cycle*max' where 'max' is the '+5' voltage.
    print_messages=False : bool, optional
        Whether to print messages or not.
    test_mode=False : bool, optional
        Allows to test classes even if you don't have a real 
        conection with a NI USB 6212.
        
    Attributes
    ----------
    pin : int
        Number of DAQ pin.
    channel : str
        Name of DAQ channel.
    streamer : nidaqmx.stream_writers
        DAQ's writing manager.
    frequency : int, float
        Digital PWM channel's main frequency.
    duty_cycle : int, float {0<=duty_cycle<=1}
        Digital PWM channel's duty cycle and therefore mean value's 
        coefficient.
    status : bool
        Whether this digital PWM channel is on or off.
    low_pin : int
        Number of DAQ pin that holds digital GND (low value).
    high_pin : int
        Number of DAQ pin that holds digital +5 V (high value).
    
    Other Attributes
    ----------------
    write_mode : bool
        Whether this is a writing class or a reading one.
    test_mode : bool
        Whether it's test mode (no real connection) or not.
    print : bool
        Whether to print inner messages or not.
        
    """      
    
    def __init__(self, device, task, streamer, pin,
                 frequency=100e3, duty_cycle=0.5,
                 print_messages=False, test_mode=False):    

        # First, set general DAQ attributes        
        self.__device = device
        self.__task = task
        self.streamer = streamer

        # Is there a real DAQ connected or are we just testing?
        self.test_mode = test_mode
        if test_mode:
            self.print = True
        else:
            self.print = print_messages
        
        # Some general attributes of this channel
        self.pin = pin
        self.channel = self.__channel__(pin)
        
        # Add this channel to the DAQ
        if not test_mode:
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
        
        # Configure this channel
        self.__status = False
        self.frequency = frequency
        self.duty_cycle = duty_cycle

    @property
    def low_pin(self):
        return 37
    
    @low_pin.setter
    def low_pin(self, value):
        raise AttributeError("Can't modify this!")
    
    @property
    def high_pin(self, value):
        return 42

    @high_pin.setter
    def high_pin(self, value):
        raise AttributeError("Can't modify this!")    

    @property
    def duty_cycle(self):
        return self.__duty_cycle
    
    @duty_cycle.setter
    def duty_cycle(self, value):
        
        # Check if I need to reconfigure
        try:
            needs_reconfiguration = self.__duty_cycle != value
        except:
            needs_reconfiguration = True
            
        # Reconfigure if needed
        if needs_reconfiguration:
            if not self.test_mode:
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
        
        # Check if I need to reconfigure
        try:
            needs_reconfiguration = self.__frequency != value
        except:
            needs_reconfiguration = True
            
        # Reconfigure if needed
        if needs_reconfiguration:
            if not self.test_mode:
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
        
        # Check if I need to reconfigure
        try:
            needs_reconfiguration = self.__status != key
        except:
            needs_reconfiguration = True
            
        # Reconfigure if needed
        if needs_reconfiguration:
            if isinstance(key, str):
                key = True
            if not self.test_mode:
                if key:
                    self.__task.start()
                    self.streamer.write_one_sample_pulse_frequency(
                            frequency = self.frequency,
                            duty_cycle = self.duty_cycle,
                            )
                else:
                    self.__task.stop()
            else:
                self.__print__("Should 'start' or 'stop'")
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
        
        try:
            channel = reference.index(pin)
            channel = '{}/ctr{}'.format(self.__device, channel)
            return channel
        except ValueError:
            message = "Wrong pin {} for digital input"
            raise ValueError(message.format(pin))
    
    def __print__(self, message):
        
        """Only prints if self.print is True.
        
        Parameters
        ----------
        message : str
            Message to print
        
        Returns
        -------
        nothing
        
        Raises
        ------
        print
        
        """
        
        if self.print:
            print(message)