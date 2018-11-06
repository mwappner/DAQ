# -*- coding: utf-8 -*-
"""
The 'fwp_daq' module is designed to measure with NI USB 6212.

Based on https://github.com/fotonicaOrg/daq.git from commit 387c7c3

"""

from fwp_classes import DynamicDic
import fwp_string as fst
import numpy as np
import nidaqmx as nid
import nidaqmx.stream_readers as sr
import nidaqmx.stream_writers as sw
import nidaqmx.system as sys
#from matplotlib import pyplot as plt
#import time

continuous = nid.constants.AcquisitionType.CONTINUOUS
conf = nid.constants.TerminalConfiguration

def multiappend(nparray, new_nparray, fast_speed=True):
    """Analog to np.append but with 2D np.arrays"""
    
    nrows = len(nparray[:,0])
    if not fast_speed:
        if len(np.new_nparray[:,0]) != nrows:
            raise IndexError("Different number of rows.")
        elif len(np.new_nparray[0,:]) != len(np.nparray[0,:]):
            raise IndexError("Different number of columns.")
    
    construct = []
    for i in range(nrows):
        construct.append(np.append(nparray[i,:], new_nparray[i,:]))
    return np.array(construct)

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

class DAQ:
    
    def __init__(self, device, print_messages=False):
        
        self.__device = device
        self.__pins = DynamicDic()
        self.writer()
        self.reader()
        
        self.__rtask = Task(self.device,
                            self.reader,
                            False,
                            print_messages)
        self.__wtask = Task(self.device,
                            self.writer,
                            True,
                            print_messages)
        
        self.__analog_inputs = DynamicDic()
        self.__pwm_outputs = DynamicDic()
                
        self.print = print_messages
        
    @property
    def pins(self):
        return self.__pins
    
    @pins.setter
    def pins(self, value):
        raise ValueError("You shouldn't modify this manually!")
    
    @property
    def ninputs(self):
        return self.__ninputs
    
    @ninputs.setter
    def ninputs(self, value):
        self.__print__("Can't modify this manually. Auto-updating...")
        channels = [k for k,v in self.pins.items() if 'in' in v.lower()]
        self.__ninputs = len(channels)

    @property
    def noutputs(self):
        return self.__noutputs
    
    @noutputs.setter
    def noutputs(self, value):
        self.__print__("Can't modify this manually. Auto-updating...")
        channels = [k for k,v in self.pins.items() if 'out' in v.lower()]
        self.__noutputs = len(channels)
    
    @property
    def analog_inputs(self):
        return self.__analog_inputs
    
    @analog_inputs.setter
    def analog_inputs(self, values):
        raise AttributeError("Must use 'add_analog_inputs'!")

    def add_analog_inputs(self, *pins, **kwargs):
        new_channels = self.__rtask.add_channels(AnalogInputChannel, 
                                                 *pins, **kwargs)
        self.reader()
        self.__pins.update(new_channels)
        self.__analog_inputs.update(new_channels)
    
    @property
    def pwm_outputs(self, *pins, **kwargs):
        return self.__pwm_outputs
    
    @pwm_outputs.setter
    def pwm_outputs(self, values):
        raise AttributeError("Must use 'add_pwm_outputs'!")

    def add_pwm_outputs(self, *pins, **kwargs):
        new_channels = self.__wtask.add_channels(PWMOutputChannel, 
                                                 *pins, **kwargs)
        self.writer()
        self.__pins.update(new_channels)
        self.__pwm_outputs.update(new_channels)
        
    @property
    def reader(self):
        return self.__reader
    
    @reader.setter
    def reader(self, *args):
        self.__print__("Can't change this manually. Auto-updating...")
        if self.n_inputs > 1:
            reader = sr.AnalogMultiChannelReader(self.__rtask.in_stream)
        else:
            reader = sr.AnalogSingleChannelReader(self.__rtask.in_stream)
        try:
            self.__rtask.streamer = reader
            self.__reader = reader
        except:
            self.__reader = reader
    
    @property
    def writer(self):
        return self.__writer
    
    @writer.setter
    def writer(self, *args):
        self.__print__("Can't change this manually. Auto-updating...")
        writer = sw.CounterWriter(self.__task.out_stream)
        try:
            self.__wtask.streamer = writer
            self.__writer = writer
        except:
            self.__writer = writer
    
    def write(self, status, **kwargs):
        self.__wtask.write(status, **kwargs)
    
    def read(self, nsamples_total=None, samplerate=None, 
             callback=None, **kwargs):
        self.__rtask.read(nsamples_total=None,
                          samplerate=None,
                          callback=None,
                          **kwargs)
    
    def __print__(self, message):
        
        if self.print:
            print(message)

#%%

class Task:
    
    def __init__(self, device, streamer=None, 
                 mode='r', print_messages=False):
        
        self.__device = device
        self.__task = nid.Task()
        self.__streamer = streamer
        
        self.__pins = DynamicDic()
        
        if 'r' in mode.lower():
            self.write_mode = False
        elif 'w' in mode.lower():
            self.write_mode = True
        
        self.print = print_messages
        
    @property
    def pins(self):
        return self.__pins
    
    @pins.setter
    def pins(self, value):
        raise ValueError("You shouldn't modify this manually!")
    
    @property
    def nchannels(self):
        return self.__nchannels
    
    @nchannels.setter
    def nchannels(self, value):
        self.__print__("Can't modify this manually. Auto-updating...")
        self.__nchannels = len(self.pins)
    
    def add_channels(self, ChannelClass, *pins, **kwargs):
               
        new_channels = {}
        for p in pins:
            if self.attribute.is_empty(p):
                channel = ChannelClass(
                    self.__device,
                    self.__task,
                    self.__streamer,
                    p,
                    kwargs)
                new_channels[p] = channel
        
        self.__pins.update({p : ChannelClass.__name__ for p in pins})
        return new_channels
    
    @property
    def streamer(self):
        return self.__streamer
    
    @streamer.setter
    def streamer(self, streamer=None):
        if self.__streamer is None:
            if not self.write_mode:
                if self.nchannels > 1:
                    reader = sr.AnalogMultiChannelReader(
                            self.__task.in_stream)
                else:
                    reader = sr.AnalogSingleChannelReader(
                            self.__task.in_stream)
                self.__streamer = reader
            else:
                self.__streamer = sw.CounterWriter(
                        self.__task.out_stream)
        else:
            self.__streamer = streamer
        try:
            for p in self.pins.keys():
                self.pins[p].streamer = streamer
        except:
            self.pins
    
    def read(self, nsamples_total=None, samplerate=None,
             nsamples_callback=None, callback=None,
             nsamples_each=20):
        
        if self.write_mode:
            raise TypeError("This task is meant to write!")
        
        if callback is not None:
    
            
            self.__task.register_every_n_samples_acquired_into_buffer_event(
                nsamples_callback, # call every nsamples_callback
                callback)
            nsamples_each = nsamples_callback
            self.__print__("Each time, I take nsamples_callback")


        if nsamples_total is None:
        
            self.__task.timing.cfg_samp_clk_timing(
                    rate = samplerate,
                    sample_mode = continuous
                    )
            
            signal = np.array([])
            self.__task.start()
            print("Acquiring... Press Ctrl+C to stop.")
            while True:
                
                try:
                    each_signal = np.zeros((self.n_inputs,
                                            nsamples_each))
                    self.__streamer.read_many_sample(
                        each_signal, 
                        number_of_samples_per_channel=nsamples_each,
                        timeout=2)
                    signal = multiappend(signal, each_signal)
                except KeyboardInterrupt:
                    self.__task.stop()
                    return signal

        else:

            signal = np.zeros((self.n_inputs, nsamples_total),
                               dtype=np.float64)
            self.__task.start()
            self.__streamer.read_many_sample(
                signal, 
                number_of_samples_per_channel=nsamples_total,
                timeout=2)
            return signal
    
    
    def write(self, status=True, frequency=None, duty_cycle=None):
    
        if not self.write_mode:
            raise TypeError("This task is meant to read!")
        elif self.nchannels>1:
            raise IndexError("This method is only available for 1 PWM Output")
        
        # Look for pin
        pin = list(self.pins.keys())[0]
        
        # Reconfigure if needed
        if frequency is not None:
            self.pins[pin].frequency = frequency
        if duty_cycle is not None:
            self.pins[pin].duty_cycle = duty_cycle
        self.pins[pin].status = status
    
    def __print__(self, message):
        
        if self.print:
            print(message)

#%%

class AnalogInputChannel:

    def __init__(self, device, task, streamer, pin,
                 voltage_range=[-10, 10], 
                 configuration='NRSE'):

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
            Range of the analog input channel/s. Each of them should be 
            a list or tuple of length 2 that contains minimum and maximum 
            in V.
        configuration='NRSE' : {'NR', 'R', 'DIFF', 'PSEUDODIFF'}, optional
            Analog input channel/s terminal configuration.
        """          
        
        self.__device = device
        self.__task = task
        self.__streamer = streamer
        
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

    def __init__(self, device, task, streamer, pin,
                 frequency=100e3, 
                 duty_cycle=0.5):

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
        """          
        
        self.__device = device
        self.__task = task
        self.__streamer = streamer
        
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
            self.__streamer.write_one_sample_pulse_frequency(
                frequency = self.frequency,
                duty_cycle = self.duty_cycle,
                )
        elif isinstance(key, bool):
            if self.__status != key:
                if key:
                    self.__task.start()
                    self.__stream.write_one_sample_pulse_frequency(
                        frequency = self.frequency,
                        duty_cycle = self.duty_cycle,
                        )
                else:
                    self.__task.end()
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
