# -*- coding: utf-8 -*-
"""
The 'fwp_daq' module is designed to measure with NI USB 6212.

Based on https://github.com/fotonicaOrg/daq.git from commit 387c7c3

"""

from fwp_classes import DynamicDict, WrapperDict
import fwp_daq_channels as fch
import numpy as np
import nidaqmx as nid
import nidaqmx.stream_readers as sr
import nidaqmx.stream_writers as sw
import nidaqmx.system as sys

continuous = nid.constants.AcquisitionType.CONTINUOUS

def zeros(size, dtype=np.float64):
    
    """Analog to np.zeros but reshapes to N if size=(1, N)"""
    
    try:
        len(size)
        size = tuple(size)
    except TypeError:
        pass
    
    if isinstance(size, tuple):
        if size[0] == 1:
            size = size[1]
    
    return np.zeros(size, dtype=dtype)

def multiappend(nparray, new_nparray, fast_speed=True):
    
    """Analog to np.append but with 2D np.arrays"""
    
    try:
        nrows = len(new_nparray[:,0])
    except IndexError:
        nrows = 1
    if not fast_speed:
        try:
            nrows0 = len(np.nparray[:,0])
        except IndexError:
            nrows = 1
        if nrows0 != nrows:
            raise IndexError("Different number of rows.")
    
    if len(nparray) == 0:
        return new_nparray
    
    elif nrows == 1:
        return np.append(nparray, new_nparray)
    
    else:
        construct = []
        for i in range(nrows):
            row = nparray[i,:]
            row = np.append(row, new_nparray[i,:])
            construct.append(row)
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

"""Here starts the newer version"""

#%%

class DAQ:
    
    """Allows to read and write with callback on NI USB 6212"""
    
    def __init__(self, device, print_messages=False,
                 conection=True):
        
        """Initializes a DAQ class that allows to read and write.
        
        Parameters
        ----------
        device : str
            NI device's name where analog input channel/s should be put.
        print_messages=False : bool, optional
            Whether to print messages or not.
        conection=True : bool, optional
            Whether you have a real conection with a NI USB 6212 or not. 
            Allows to test classes.
        """  
        
        self.__device = device
        
        self.conection = conection
        if not conection:
            self.print = True
        else:
            self.print = print_messages
        
        self.tasks = WrapperDict(
                
            __inputs = Task(self.device,
                            mode='r',
                            print_messages=print_messages,
                            conection=conection),
            __outputs = Task(self.device,
                             mode='w',
                             print_messages=print_messages,
                             conection=conection)
            )
    
    @property
    def inputs(self):
        return self.__inputs
    
    @inputs.setter
    def inputs(self, value):
        msg = "Why don't you chek 'add_analog_inputs' instead?"
        raise AttributeError(msg)

    @property
    def outputs(self):
        return self.__outputs
    
    @outputs.setter
    def outputs(self, value):
        msg = "Why don't you chek 'add_pwm_outputs' instead?"
        raise AttributeError(msg)
    
    @property
    def ninputs(self):
        return self.inputs.nchannels
    
    @ninputs.setter
    def ninputs(self, value):
        raise AttributeError("Can't modify this manually!")

    @property
    def noutputs(self):
        return self.outputs.nchannels
    
    @noutputs.setter
    def noutputs(self, value):
        raise AttributeError("Can't modify this manually!")

    @property
    def reader(self):
        return self.inputs.streamer
    
    @reader.setter
    def reader(self, value):
        raise AttributeError("Can't modify this manually!")

    @property
    def writer(self):
        return self.outputs.streamer
    
    @writer.setter
    def writer(self, value):
        raise AttributeError("Can't modify this manually!")

    def add_analog_inputs(self, *pins, **kwargs):
        self.inputs.add_channels(fch.AnalogInputChannel, 
                                 *pins, **kwargs)

    def add_pwm_outputs(self, *pins, **kwargs):
        self.outputs.add_channels(fch.PWMOutputChannel, *pins, **kwargs)
        
    def close(self):
        self.outputs.close()
        self.inputs.close()
        
    def __print__(self, message):
        if self.print:
            print(message)

#%%

class Task:
    
    def __init__(self, device, mode='r', 
                 print_messages=False, conection=True):
        
        """Initializes class that allows whether to read or to write.
        
        Parameters
        ----------
        device : str
            NI device's name where analog input channel/s should be put.
        mode : str, optional {'r', 'w'}
            Says whether this class is to read or to write.
        print_messages=False : bool, optional
            Whether to print messages or not.
        conection=True : bool, optional
            Whether you have a real conection with a NI USB 6212 or not. 
            Allows to test classes.
        """  
        
        self.__device = device
        self.__task = nid.Task()

        if 'w' in mode.lower():
            self.write_mode = True        
        elif 'r' in mode.lower():
            self.write_mode = False

        self.conection = conection
        if not conection:
            self.print = True
        else:
            self.print = print_messages

        self.all = WrapperDict()
        self.pins = WrapperDict()
        self.__nchannels = 0
        
        self.streamer = True
    
    @property
    def nchannels(self):
        return self.__nchannels
    
    @nchannels.setter
    def nchannels(self, value):
        raise AttributeError("Can't modify this manually!")
    
    def add_channels(self, ChannelClass, *pins, **kwargs):
        
        new_pins = {}
        new_channels = {}
        
        if isinstance(pins[0], dict):
            
            for name, p in pins[0].items():
                ch = ChannelClass(
                    self.__device,
                    self.__task,
                    self.__streamer,
                    p,
                    **kwargs,
                    print_messages=self.print,
                    conection=self.conection)
                new_pins[p] = ch
                new_channels[name] = ch
            
        else:
            
            for p in pins:
                ch = ChannelClass(
                    self.__device,
                    self.__task,
                    self.__streamer,
                    p,
                    **kwargs,
                    print_messages=self.print,
                    conection=self.conection)
                new_pins[p] = ch
                name = ch.channel.split('/')[1]
                new_channels[name] = ch
        
        self.pins.update(new_pins)
        self.all.update(new_channels)
        
        self.__dict__.update(new_channels)
        self.__nchannels = self.nchannels + len(new_channels)
        self.streamer = True
    
    @property
    def streamer(self):
        return self.__streamer
    
    @streamer.setter
    def streamer(self, boolean):
        if not isinstance(boolean, bool):
            self.__print__("Hey! This value should be a bool.")
        if boolean:
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
            raise AttributeError("Can't set this manually!")
        if self.nchannels > 0:
            try:
                self.all.streamer = self.__streamer
            except:
                self.__print__("Coudn't set streamer to channels")
    
    def read(self, nsamples_total=None, samplerate=None,
             nsamples_callback=None, callback=None,
             nsamples_each=20):
        
        if self.write_mode:
            raise TypeError("This task is meant to write!")
        
        if samplerate is None:
            
            samplerate = int(400e3/self.nchannels)
        
        if callback is not None:
    
            if self.conection:
                self.__task.register_every_n_samples_acquired_into_buffer_event(
                    nsamples_callback, # call every nsamples_callback
                    callback)
            else:
                self.__print__("Should 'task.register_every...'")
            nsamples_each = nsamples_callback
            self.__print__("Each time, I take nsamples_callback")

        if nsamples_total is None:
        
            if self.conection:
                self.__task.timing.cfg_samp_clk_timing(
                        rate = samplerate,
                        sample_mode = continuous
                        )
            else:
                self.__print__("Should 'task.timing.cfg...'")
            
            signal = np.array([])
            if self.conection:
                self.__task.start()
            else:
                self.__print__("Should run 'task.start'")
            print("Acquiring... Press Ctrl+C to stop.")
            
            nbuffers = 0
            while True:
                
                try:
                    each_signal = zeros((self.nchannels,
                                         nsamples_each),
                                         dtype=np.float64)
                    if self.conection:
                        self.__streamer.read_many_sample(
                            each_signal, 
                            number_of_samples_per_channel=nsamples_each,
                            timeout=2)
                    signal = multiappend(signal, each_signal)
                    nbuffers = nbuffers + 1
                    self.__print__("Number of buffers: " + nbuffers)
                except KeyboardInterrupt:
                    self.__task.stop()
                    return signal

        else:

            signal = zeros((self.nchannels, nsamples_total),
                            dtype=np.float64)
            if self.conection:
                self.__task.start()
                self.__streamer.read_many_sample(
                    signal, 
                    number_of_samples_per_channel=nsamples_total,
                    timeout=2)
                self.__task.stop()
            else:
                self.__print__("Should 'start'+'read_many...'+'stop'")
            return signal
    
    def write(self, status=True, frequency=None, duty_cycle=None):
    
        if not self.write_mode:
            raise TypeError("This task is meant to read!")
        elif self.nchannels>1:
            msg = "This method is only available for 1 PWM Output"
            raise IndexError(msg)
                
        # Reconfigure if needed
        if frequency is not None:
            self.all.frequency = frequency
        if duty_cycle is not None:
            self.all.duty_cycle = duty_cycle
        self.all.status = status
    
    def close(self):
        self.__task.close()
    
    def __print__(self, message):
        
        if self.print:
            print(message)

#%%

"""Here starts an older version of the same classes"""

#%%

class OldDAQ:
    
    def __init__(self, device, print_messages=False):
        
        self.__device = device
        self.writer()
        self.reader()
        
        self.__rtask = OldTask(self.device,
                               self.reader,
                               False,
                               print_messages)
        self.__wtask = OldTask(self.device,
                               self.writer,
                               True,
                               print_messages)
        
        self.__analog_inputs = DynamicDict()
        self.__pwm_outputs = DynamicDict()
                
        self.print = print_messages
    
    @property
    def ninputs(self):
        return self.inputs.nchannels
    
    @ninputs.setter
    def ninputs(self, value):
        raise AttributeError("Can't modify this manually")

    @property
    def noutputs(self):
        return self.outputs.nchannels
    
    @noutputs.setter
    def noutputs(self, value):
        raise AttributeError("Can't modify this manually")
    
    @property
    def analog_inputs(self):
        return self.__analog_inputs
    
    @analog_inputs.setter
    def analog_inputs(self, values):
        raise AttributeError("Must use 'add_analog_inputs'!")

    def add_analog_inputs(self, *pins, **kwargs):
        new_channels = self.__rtask.add_channels(fch.AnalogInputChannel, 
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
        new_channels = self.__wtask.add_channels(fch.PWMOutputChannel, 
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
             callback=None, nsamples_each=20, **kwargs):
        self.__rtask.read(nsamples_total=nsamples_total,
                          samplerate=samplerate,
                          callback=callback,
                          nsamples_each=20,
                          **kwargs)
    
    def __print__(self, message):
        
        if self.print:
            print(message)

#%%

class OldTask:
    
    def __init__(self, device, streamer=True, 
                 mode='r', print_messages=False):
        
        self.__device = device
        self.__task = nid.Task()
        self.__streamer = streamer
        
        self.__pins = DynamicDict()
        
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
            if self.pins.is_empty(p):
                channel = ChannelClass(
                    self.__device,
                    self.__task,
                    self.__streamer,
                    p,
                    kwargs)
                new_channels[p] = channel
        
        self.__pins.update({p : ChannelClass.__name__ for p in pins})
        self.nchannels = True
        return new_channels
    
    @property
    def streamer(self):
        return self.__streamer
    
    @streamer.setter
    def streamer(self, streamer):
        if streamer is True:
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
        for p in self.pins.keys():
            try:
                self.pins[p].streamer = streamer
            except:
                pass
    
    def read(self, nsamples_total=None, samplerate=None,
             nsamples_callback=None, callback=None,
             nsamples_each=20):
        
        if self.write_mode:
            raise TypeError("This task is meant to write!")
        
        if samplerate is None:
            
            samplerate = 400e3/self.nchannels
        
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

"""Here starts a really old version made of functions"""

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
