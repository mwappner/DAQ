# -*- coding: utf-8 -*-
"""
The 'fwp_daq' module is designed to measure with NI USB 6212.

Based on 'fwp_old_daq'.

@author: Vall
"""

from fwp_classes import WrapperDict
import fwp_daq_channels as fch
import fwp_utils as utl
import inspect as spec
import numpy as np
import nidaqmx as nid
import nidaqmx.stream_readers as sr
import nidaqmx.stream_writers as sw
import nidaqmx.system as sys

task_states = nid.constants.TaskMode
continuous = nid.constants.AcquisitionType.CONTINUOUS
single = nid.constants.AcquisitionType.FINITE

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
    
    return devices

#%%

class DAQ:
    
    """Class that allows to read and write with a NI USB 6212 DAQ.
    
    Parameters
    ----------
    device : str
        NI device's name where analog input channel/s should be put.
    print_messages=False : bool, optional
        Whether to print messages or not.
    test_mode=False : bool, optional
        Allows to test classes even if you don't have a real 
        conection with a NI USB 6212.
    
    Attributes
    ----------
    tasks : fwp_classes.WrapperDict
        DAQ channels' manager (contains inputs and outputs' Task object)
    inputs : Task
        DAQ inputs' manager.
    outputs : Task
        DAQ outputs' manager.
    ninputs : int
        Number of DAQ's inputs.
    noutputs : int
        Number of DAQ's outputs.
    reader : nidaqmx.stream_reader
        DAQ's reading manager.
    writer : nidaqmx.stream_writer
        DAQ's writing manager.
    
    Methods
    -------
    add_analog_inputs(*pins, **kwargs)
        Adds analog input channel/s.
    add_pwm_outputs(*pins, **kwargs)
        Adds digital PWM output channel/s.
    inputs.read(nsamples=None, samplerate=None, callback=None)
        Reads from input channel/s if this task is meant to read.
    outputs.write(status=True, frequency=None, duty_cycle=None)
        Writes on output channel/s if this task is meant to write. Up to 
        now, it's only possible to write on a single PWM channel.
    close()
        Ends communication with DAQ.
    
    Other Attributes
    ----------------
    test_mode : bool
        Whether it's test mode (no real connection) or not.
    print : bool
        Whether to print inner messages or not.
    
    """  
    
    def __init__(self, device, print_messages=False,
                 test_mode=False):
        
        # DAQ's general attributes
        self.__device = device
        
        # Is there a real DAQ connected or are we just testing?
        self.test_mode = test_mode
        if test_mode:
            self.print = True
        else:
            self.print = print_messages
        
        # DAQ channels' manager
        self.__inputs = Task(self.__device,
                             mode='r',
                             print_messages=print_messages,
                             test_mode=test_mode)
        self.__outputs = Task(self.__device,
                              mode='w',
                              print_messages=print_messages,
                              test_mode=test_mode)
        self.tasks = WrapperDict(inputs = self.inputs,
                                 outputs = self.outputs)
    
    def __enter__(self):
        return self
#        return self.__task

    def __exit__(self, type, value, traceback):
        self.close()
#        self.__task.close()
    
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

        """Adds analog input channel/s.
        
        Parameters
        ----------
        pins : int, optional
            Literally the number/s of DAQ's pins.
        
        Other Parameters
        ----------------
        voltage_range=[-10,10] : list, tuple, optional
            Range of the analog input channel. Should be a list or tuple 
            of length 2 that contains minimum and maximum in V.
        configuration='NonRef' : str, optional
            Analog input channel terminal configuration.

        Returns
        -------
        nothing
        
        See Also
        --------
        Task.add_channels
        fwp_daq_channels.AnalogInputChannel
        
        """
        
        self.inputs.add_channels(fch.AnalogInputChannel, 
                                 *pins, **kwargs)

    def add_pwm_outputs(self, *pins, **kwargs):
        
        """Adds PWM output channel/s.
        
        Parameters
        ----------
        pins : int, optional
            Literally the number/s of DAQ's pins.
        
        Other Parameters
        ----------------
        frequency=100e3 : int, float, optional
            PWM's main frequency.
        duty_cycle=.5 : int, float, {0 <= duty_cycle <= 1}, optional
            PWM's duty cycle, which defines mean value of signal as 
            'duty_cycle*max' where 'max' is the '+5' voltage.
        
        Returns
        -------
        nothing
        
        See Also
        --------
        Task.add_channels
        fwp_daq_channels.PWMOutputChannel
        
        """
        
        self.outputs.add_channels(fch.PWMOutputChannel, *pins, **kwargs)
        
    def close(self):
        
        """Closes the tasks.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        nothing
        
        See Also
        --------
        nidaqmx.Task.close
        
        """
        
        self.outputs.close()
        self.inputs.close()
        
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

class Task:

    """Allows whether to read or to write with a NI USB 6212 DAQ.
    
    Parameters
    ----------
    device : str
        NI device's name.
    mode : str, optional {'r', 'w'}
        Says whether this class is meant to read or to write.
    print_messages=False : bool, optional
        Whether to print messages or not.
    test_mode=False : bool, optional
        Allows to test classes even if you don't have a real 
        conection with a NI USB 6212.
    
    Attributes
    ----------
    streamer : nidaqmx.stream_writers or nidaqmx.stream_readers
        DAQ task's manager that either writes or reads (but not both).
    channels : fwp_classes.WrapperDict
        DAQ task's channels, callable by name.
    pins : fwp_classes.WrapperDict
        DAQ task's channels, callable by pin.
    nchannels : int
        DAQ task's number of channels.
    
    Methods
    -------
    add_channels(ChannelClass, *pins, **kwargs)
        Adds channel/s of a certain class.
    read(nsamples=None, samplerate=None, callback=None)
        Reads from input channel/s if this task is meant to read.
    write(status=True, frequency=None, duty_cycle=None)
        Writes on output channel/s if this task is meant to write. Up to 
        now, it's only possible to write on a single PWM channel.
    stop()
        Stops the current task.
    close()
        Closes communication.
    
    Other Attributes
    ----------------
    write_mode : bool
        Whether this is a writing class or a reading one.
    test_mode : bool
        Whether it's test mode (no real connection) or not.
    print : bool
        Whether to print inner messages or not.
    
    """  
    
    def __init__(self, device, mode='r', 
                 print_messages=False, test_mode=False):
        
        # DAQ's general attributes
        self.__device = device
        self.__task = nid.Task()

        # Is this task meant to write or to read? Can't do both!
        self.__write_mode = 'w' in mode.lower()

        # Is there a real DAQ connected or are we just testing?
        self.test_mode = test_mode
        if test_mode:
            self.print = test_mode
        else:
            self.print = print_messages

        # DAQ channels' managers
        self.channels = WrapperDict()
        self.pins = WrapperDict()
        self.__nchannels = 0
        
        # DAQ's reading or writing manager and configuration
        self.streamer = None
    
    def __enter__(self):
        return self
#        return self.__task

    def __exit__(self, type, value, traceback):
        self.close()
#        self.__task.close()
    
    @property
    def write_mode(self):
        return self.__write_mode

    @write_mode.setter
    def write_mode(self, value):
        msg = "Can't modify this! Must close and open new task"
        raise AttributeError(msg)

    @property
    def nchannels(self):
        return self.__nchannels
    
    @nchannels.setter
    def nchannels(self, value):
        raise AttributeError("Can't modify this manually!")
    
    def add_channels(self, ChannelClass, *pins, **kwargs):

        """Adds channel/s of a certain class.
        
        Parameters
        ----------
        ChannelClass : fwp_daq_channels class
            New channel/s' class.
        pins : int, optional
            Literally DAQ pin/s' number. Must put at least one.
        **kwargs : optional
            Optional parameters of new channel/s' class.
        
        Returns
        -------
        nothing
        
        """
        
        if not bool(pins):
            raise ValueError("Must pass at least one pin number")
        
        # Add channels        
        new_pins = {}
        new_channels = {}
        for p in pins:
            ch = ChannelClass(
                self.__device,
                self.__task,
                self.__streamer,
                p,
                **kwargs,
                print_messages=self.print,
                test_mode=self.test_mode)
            new_pins[p] = ch
            name = ch.channel.split('/')[1]
            new_channels[name] = ch

        # Update channels and pins        
        self.pins.update(new_pins) # holds them by pin
        self.channels.update(new_channels) # holds them by channel
        self.__dict__.update(new_channels) # attributes by channel name
        
        # Reconfigure
        self.__nchannels = self.nchannels + len(new_channels)
        self.streamer = None
        self.samplerate = None
        self.buffersize = None
    
    @property
    def streamer(self):
        return self.__streamer
    
    @streamer.setter
    def streamer(self, value):
        if value is not None:
            raise AttributeError("Hey! You can't set this manually!")
        
        # If read mode, StreamReader.
        if not self.write_mode:
            if self.nchannels > 1:
                reader = sr.AnalogMultiChannelReader(
                        self.__task.in_stream)
            else:
                reader = sr.AnalogSingleChannelReader(
                        self.__task.in_stream)
            self.__streamer = reader
        # Else, StreamWriter for PWM output channel
        else:
            self.__streamer = sw.CounterWriter(
                    self.__task.out_stream)
        
        # Reconfigure channels' streamer
        if self.nchannels > 0:
            try:
                self.pins.streamer = self.__streamer
            except:
                raise AttributeError("Coudn't set streamer to channels")
    
    @property
    def samplerate(self):
        if self.write_mode:
            return TypeError("This task is meant to write!")
        else:
            return self.__samplerate
    
    @samplerate.setter
    def samplerate(self, value):
        if self.write_mode:
            return TypeError("This task is meant to write!")
        
        else:
            
            # Check if I need to reconfigure
            try:
                needs_reconfiguration = self.__samplerate != value
            except:
                needs_reconfiguration = True
                
            # Reconfigure if needed
            if needs_reconfiguration:
                if value is None: # Default value is maximum value
                    try:
                        value = int(400e3/self.nchannels)
                    except ZeroDivisionError:
                        value = 400e3
                self.__check_samplerate__(value)
                if not self.test_mode:
                    self.__task.timing.cfg_samp_clk_timing(
                            rate = value)
                else:
                    self.__print__("Should 'task.timing.cgf_samp...'")
                self.__samplerate = value

    @property
    def buffersize(self):
        if self.write_mode:
            return TypeError("This task is meant to write!")
        else:
            return self.__buffersize
    
    """Some things we would have liked to do research on:
    task.streamer._in_stream.channels_to_read
    task.streamer._in_stream.logging_file_path
    task.streamer._in_stream.configure_logging
    task.streamer._in_stream.curr_read_pos
    task.streamer._in_stream.read_all_avail_samp
    task.streamer._in_stream.input_onbrd_buf_size
    """
    
    @buffersize.setter
    def buffersize(self, value):
        if self.write_mode:
            return TypeError("This task is meant to write!")
        else:
            
            # Check if I need to reconfigure
            try:
                needs_reconfiguration = self.__buffersize != value
            except:
                needs_reconfiguration = True
                
            # Reconfigure if needed
            if needs_reconfiguration:
                if not self.test_mode:
                    if value is None: # Default value is DAQ's one.
                        value = self.streamer._in_stream.input_buf_size
                    else:
                        self.streamer._in_stream.input_buf_size = value
                else:
                    self.__print__("Should 'streamer._in_stream.in...'")
                self.__buffersize = value
    
    def read(self, nsamples=None, samplerate=None, duration=None,
             callback=None, nsamples_each=200, use_stream=True, 
             do_return=True):
        
        """Reads from the input channels.
        
        Parameters
        ----------
        nsamples=None : int, optional
            Total number of samples to be acquired from each channel. If 
            None, the acquisition is continuous and must be stopped by a 
            KeyboardInterrupt.
        samplerate=None : int, float, optional
            Samplerate in Hz by channel. If None, samplerate attribute 
            is used, which is maximum samplerate by default.
        duration=None : int, optional
            Total duration of the measurement. If nsamples is not 
            specified but duration is, a new nsamples will be determined 
            as 'nsamples=duration*samplerate'.
        callback=None : function, optional
            Callback function. Mustn't return anything. And must either 
            take in no parameters or either take in only one parameter, 
            which will be filled with acquired data.
        nsamples_each=200 : int, optional
            Number of samples acquired by the DAQ before they are passed 
            to the PC. Same number of samples are collected before 
            running callback function.
        use_stream=True : bool, optional
            Whether to use a 'nidaqmx.stream_readers' method if possible 
            or not.
        do_return=True : bool, optional
            Whether to return the acquired signal or not.
        
        Returns
        -------
        signal : np.array
            Measured data. If nsamples is not None, it has shape 
            (self.nchannels, nsamples). If nsamples is None, it has 
            shape (self.nchannels, i*nsamples_each) where i is an 
            integer number.
        
        See Also
        --------
        https://nidaqmx-python.readthedocs.io/en/latest/task.html#
        nidaqmx.task.Task.register_every_n_samples_acquired_into_buffer_
        event
            
        """

        # INITIAL PARAMETERS
        
        # First, some general configuration
        if self.write_mode:
            raise TypeError("This task is meant to write!")

        if samplerate is None:
            samplerate = self.samplerate
        else:
            self.__check_samplerate__(samplerate)
            
        if nsamples is None and duration is not None:
            nsamples = duration * samplerate

        # Is callback required?
        if nsamples is not None and nsamples < nsamples_each:
            callback=None
            self.__print__("Callback won't be played because \
                           'nsamples_each' is smaller than 'nsamples'")

        # Does callback take data as a parameter?
        if callback is None:
            callback_parameters = False             
        else:            
            callback_parameters = spec.getfullargspec(callback)[0]
            if len(callback_parameters)>1:
                raise ValueError("Callback must have only 1 variable")
            callback_parameters = bool(callback_parameters)

        # Then, get a wrapper callback to wrap the user's
        wrapper_callback = self.__choose_wrapper_callback__(
            nsamples,
            callback,
            callback_parameters,
            use_stream)
        """There, 'parameters' indicates whether the user's callback 
        takes in a parameter or not"""
        
        # If wrapper callback needed, configure it.
        if wrapper_callback is not None:
            if not self.test_mode:
                self.__task.register_every_n_samples_acquired_into_buffer_event(
                        nsamples_each, # call callback every
                        wrapper_callback)
            else:
                self.__print__("Should 'task.register_every...'")
    
        # If necessary, set array for the total acquired samples
        if do_return and nsamples is not None: # single
            signal = utl.zeros((self.nchannels, nsamples),
                           dtype=np.float64)
        elif do_return: # continuous
            signal = np.array(dtype=np.float64)
            
        # Just in case, be ready for measuring in tiny pieces
        global each_signal, message, ntimes
        each_signal = utl.zeros((self.nchannels,
                             nsamples_each),
                             dtype=np.float64)
        message = "Number of {}-sized samples' arrays".format(
                nsamples_each)
        message = message + " read: {}"
        ntimes = 0
    
        # SINGLE ACQUISITION
        if nsamples is not None:
    
            # Set single reading mode
            if not self.test_mode:
                self.__task.timing.cfg_samp_clk_timing(
                        rate=samplerate,
                        sample_mode=single)
            else:
                self.__print__("Should 'task.timing.cfg...'")
            
            # According to wrapper callback...
            if wrapper_callback is None or not callback_parameters:
                
                if do_return:                    
                    # Just measure
                    if not self.test_mode:
                        self.__task.start()
                        if use_stream:
                            signal = self.__stream_read__(
                                    nsamples,
                                    signal)
                        else:
                            signal = self.__read__(nsamples)
                        self.__task.stop()
                    else:
                        self.__print__("Should 'start'+'read_ma...'+'stop'")
                    return signal
                else:
                    # No need to measure
                    return
            
            else:
                
                self.__task.start()
#                while True:
#                    try:
#                        'a'
#                    except KeyboardInterrupt:
#                        if do_return:
#                            return signal
#                        else:
#                            return
                self.__task.wait_until_done()
                self.__task.stop()
                if do_return:
                    return signal 
                else:
                    return
        
        # CONTINUOUS ACQUISITION
        else:
            
            # Set continuous reading mode
            if not self.test_mode:
                self.__task.timing.cfg_samp_clk_timing(
                        rate = samplerate,
                        sample_mode = continuous
                        )
            else:
                self.__print__("Should 'task.timing.cfg...'")
            
            # Start the task
            if not self.test_mode:
                self.__task.start()
            else:
                self.__print__("Should run 'task.start'")
            print("Acquiring... Press Ctrl+C to stop.")
#            while True:
#                try:
#                    'a'
#                except KeyboardInterrupt:
#                    self.__task.stop()
#                    if do_return:
#                        return signal
#                    else:
#                        return
            try:
                self.__task.wait_until_done()
            except KeyboardInterrupt:
                pass
            self.__task.stop()
            if do_return:
                return signal 
            else:
                return
    
    def write(self, status=True, frequency=None, duty_cycle=None):
    
        """Sets the output channels to write.
        
        Up to now, it only allows to turn on/off one digital PWM output.
        
        Parameters
        ----------
        status=True : bool, optional
            Whether to turn on or off.
        frequency=None : bool, int, float, optional
            PWM's pulse frequency. If None, it uses the pre-configured 
            frequency.
        duty_cycle=None : bool, int, float, {0<=duty_cycle<=1}, optional
            PWM's pulse duty cycle and therefore mean value. If None, it 
            uses the pre-configured duty cycle.
        
        Returns
        -------
        nothing
        
        See Also
        --------
        fwp_daq_channels.DigitalPWMOutput.status
        
        """ 
        
        if not self.write_mode:
            raise TypeError("This task is meant to read!")
        elif self.nchannels>1:
            msg = "This method is only available for 1 PWM Output"
            raise IndexError(msg)
                
        # Reconfigure if needed
        if frequency is not None:
            self.pins.frequency = frequency
        if duty_cycle is not None:
            self.pins.duty_cycle = duty_cycle
        self.pins.status = status
    
    def stop(self):
        
        """Stops the task.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        nothing
        
        See Also
        --------
        nidaqmx.Task.stop
        
        """
        
        self.__task.stop()
    
    def close(self):
        
        """Closes the task.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        nothing
        
        See Also
        --------
        nidaqmx.Task.close
        
        """
        
        self.__task.close()
    
    def __read__(self, nsamples):
        
        data = self.__task.read(
            number_of_samples_per_channel=int(nsamples))
        self.__task.wait_until_done()
        
#        # IS ALL THIS REALLY NECESSARY?
#        try:
#            len(data[0])
#        except:
#            return np.array(data)
#        
#        data = np.array(data).T
#        if self.nchannels == 1:
#            data = np.expand_dims(data, axis=0).T
#        
        return np.array(data)

    def __stream_read__(self, nsamples, signal):
        
        self.__streamer.read_many_sample(
            signal,
            number_of_samples_per_channel=int(nsamples),
            timeout=20)
        
        return signal

    def __choose_wrapper_callback__(self, nsamples, callback, 
                                    callback_parameters,
                                    use_stream):

        # Now choose the right callback wrapper
        if nsamples is not None: # SINGLE ACQUISITION
            if callback is None:
                return self.__get_wrapper_callback__(0)
            elif not callback_parameters:
                return self.__get_wrapper_callback__(1)
            else:
                return self.__get_wrapper_callback__(2)
        else: # CONTINUOUS ACQUISITION
            if callback is None:
                if not use_stream:
                    return self.__get_wrapper_callback__(3)
                else:
                    return self.__get_wrapper_callback__(4)
            elif not callback_parameters:
                return self.__get_wrapper_callback__(5)
            else:
                return self.__get_wrapper_callback__(6)
    
    def __get_wrapper_callback__(self, option):
        
        # These are the possible wrapper callbacks
        def no_callback(task_handle, 
                        every_n_samples_event_type,
                        number_of_samples, callback_data):
            
            """A nidaqmx callback that just reads"""
            
            global do_return, nsamples_each
            global ntimes, message
            global each_signal, signal
            
            if do_return:
                each_signal = self.__read__(nsamples_each)
                signal = utl.multiappend(signal, each_signal)
                
            ntimes += 1
            self.__print__(message.format(ntimes))
            
            return 0

        def no_callback_w_stream(task_handle, 
                                 every_n_samples_event_type,
                                 number_of_samples, callback_data):
            
            """A nidaqmx callback that just reads with read_streamers"""
            
            global do_return, nsamples_each
            global ntimes, message
            global each_signal, signal
            
            if do_return:
                each_signal = self.__stream_read__(nsamples_each,
                                                   each_signal)
                signal = utl.multiappend(signal, each_signal)
            ntimes += 1
            self.__print__(message.format(ntimes))
            
            return 0

        def wrap_callback(task_handle, 
                          every_n_samples_event_type,
                          number_of_samples, callback_data):
            
            """A nidaqmx callback that just wrapps"""

            global callback
            
            callback()
                
            return 0
        
        def noarg_callback(task_handle, 
                           every_n_samples_event_type,
                           number_of_samples, callback_data):
            
            """A nidaqmx callback that just wrapps"""
            
            global callback
            global do_return, nsamples_each
            global ntimes, message
            global each_signal, signal
            
            callback()
            
            if do_return:            
                each_signal = self.__read__(nsamples_each)
                signal = utl.multiappend(signal, each_signal)
            ntimes += 1
            self.__print__(message.format(ntimes))            
            
            return 0
        
        def arg_callback(task_handle, 
                          every_n_samples_event_type,
                          number_of_samples, callback_data):
            
            """A nidaqmx callback that wrapps and reads"""
            
            global callback
            global do_return, nsamples_each
            global ntimes, message
            global each_signal, signal
            
            each_signal = self.__read__(nsamples_each)            
            callback(each_signal)
            
            if do_return:
                signal = utl.multiappend(signal, each_signal)
            ntimes += 1
            self.__print__(message.format(ntimes))
            
            return 0
        
        def stop_callback(task_handle, 
                          every_n_samples_event_type,
                          number_of_samples, callback_data):
            
            """A nidaqmx callback that wrapps, reads and stops"""
            
            global callback, nsamples
            global do_return, nsamples_each, nsamples_acquired
            global ntimes, message
            global each_signal, signal
            
            nsamples_acquired = ntimes * nsamples_each
            if nsamples_acquired <= nsamples:
                
                each_signal = self.__read__(nsamples_each)
                callback(each_signal)
                
                if do_return:
                    signal = utl.multiappend(signal, each_signal)
                ntimes += 1
                self.__print__(message.format(ntimes))
                
            else:
#                raise KeyboardInterrupt
                self.__task.control(task_states.TASK_STOP)
            
            return 0
        
        # This is the algorithm to choose 
        # Option must be an int from 0 to 6
        wrapper_callback = [None,
                            wrap_callback,
                            stop_callback,
                            no_callback,
                            no_callback_w_stream,
                            noarg_callback,
                            arg_callback,
                            ]
        
        try:
            try:
                self.__print__("Chose '{}' as wrapper".format(
                    wrapper_callback[option].__name__))
            except AttributeError:
                self.__print__("Chose 'None' as wrapper")
            return wrapper_callback[option]
        except IndexError:
            raise KeyError("No callback wrapper found")
          
    def __check_samplerate__(self, samplerate):
        
        if samplerate > 400e3:
            raise ValueError("Must be <= 400 kHz")
        if samplerate * self.nchannels > 400e3:
            msg = "Must be <= {:.0f} Hz".format(
                    400e3/self.nchannels)
            raise ValueError(msg)
        
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
