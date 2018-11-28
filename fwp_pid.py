# -*- coding: utf-8 -*-
"""
This module includes a number of classes and helper functions to implement 
a PID controller with logging capabilities and including three different
integration methods. Can be extendable.

The main class of this module is PIDController which uses the secondary 
classes Logger and one of the Integrator classes, depending on the integration
mode selected: InfiniteIntegrator, WindowedIntegrator and WeightedIntegrator.
The function Integral_switcher selects the appropiate class.

The helper class InOut is an implementation of a FIFO type buffer subclassing 
collections.deque, used by other secondary classes. The helper function 
set_props is used by set_integrator and set_logger, methods of PIDController.
The helper class PIDLog is a collections.NamedTuple whis fields given by 
stuff_to_log.

All Integrator classes have methods integrate(value) and reset() that calculate 
the integral for the next step and resets the instance's attributes, 
respectively.

@author: Marcos
"""

from collections import deque, namedtuple
from fwp_save import new_name, savetxt

#%%
class InOut(deque):
    '''Subclass of collections.deque with one added method to provide
    kind of FIFO-like behaviour.
    
    Methods:
    --------
    
    put(vaue):
        Appends a value on the right and returns value on the left.'''
        
    def __init__(self, size, iterable=[]):
        super().__init__(iterable, maxlen=int(size))
        
    def put(self, val):
        out = self.popleft()
        self.append(val)
        return out
    
    @property
    def size(self):
        return self.maxlen #read-only
    
def set_props(obj, **props):
    for propname, propval in props.items():
        setattr(obj, propname, propval)
        
#%% Itegrator classes

class InfiniteIntegrator:
    '''Infinite integrator class. Integrates over all history.
    
    Parameters
    ----------
    dt : float
        Time interval used for discrete integration.
    integral_so_far : float, optional
        Used in case the user wants to initialize the integral with some
        specific history.
        
    Methods
    -------
    integrate(value)
        Adds value to history and integrates over all history.
    reset(integral_so_far=0)
        Resets history to given integral. Defaults to 0.
    '''
    
    def __init__(self, dt, integral_so_far=0):
        self.dt = dt
        self.reset(integral_so_far)

    def __str__(self):
        return 'infinite'
        
    def integrate(self, value):
        self.integral += value * self.dt
        return self.integral
        
    def reset(self, integral_so_far=0):
        self.integral = integral_so_far


class WindowIntegrator:
    '''Window integrator class. Integrates inside a given window.
    
    Parameters
    ----------
    dt : float
        Time interval used for discrete integration.
    integral_so_far : float, optional
        Used in case the user wants to initialize the integral with some
        specific history.
    window_length : int, floatm optional
        Length of window to integrate in. Defaults to 1000.
        
    Methods
    -------
    integrate(value)
        Adds value to history and integrates in given window.
    reset(integral_so_far=0)
        Resets history to given integral. Defaults to 0.
    '''    
    
    def __init__(self, dt, window_length=1000, integral_so_far=0):
        self.dt = dt
        self._window_length = window_length
        self.reset(integral_so_far)
        
    def __str__(self):
        return 'windowed'
    
    def integrate(self, value):
        self.integral += value * self.dt - self.window.put(value)
        return self.integral
        
    def reset(self, integral_so_far=0):
        self.integral = integral_so_far
        self.window = InOut(size=self.window_length, iterable=[0])
    
    @property
    def window_length(self):
        return self._window_length
    @window_length.setter
    def window_length(self, value):
        self._window_length = int(value)
        self.window = InOut(size=self.window_length, iterable=self.window)
        self.integral = sum(self.window)
        
class WeightedIntegrator:
    '''Weighted integrator class. Integrates over all history with a
    wieight factor that decreases over time. This way, olver values
    wieigh less.
    
    Parameters
    ----------
    dt : float
        Time interval used for discrete integration.
    integral_so_far : float, optional
        Used in case the user wants to initialize the integral with some
        specific history.
    alpha : float, optional
        Factor to calculate weight. The bigger alpha, the faster the weight 
        decreases, rendering older values less important.
        
    Methods
    -------
    integrate(value)
        Adds value to history and integrates over all history with weight.
    reset(integral_so_far=0)
        Resets history to given integral. Defaults to 0.
    '''
    
    def __init__(self, dt, alpha=5, interal_so_far=0):
        self.dt = dt
        self.alpha = alpha
        self.reset(interal_so_far)
        
    def __str__(self):
        return 'weighted'
    
    def integrate(self, value):
        self.integral *= self.alpha
        self.integral += value
        self.integral /= 1 + self.alpha
        return self.integral
        
    def reset(self, integral_so_far=0):
        self.integral = integral_so_far

integral_types = {
        'infinite': InfiniteIntegrator,
        'windowed': WindowIntegrator,
        'weighted': WeightedIntegrator,        
        }    
    
def integral_switcher(integral_type):
    '''Selects integral type'''
    if not isinstance(integral_type, str):
        s = 'Integral type should be of class str, not {}.'
        raise TypeError(s.format(type(integral_type)))
    
    if integral_type.lower() not in integral_types.keys():
        s = 'Integral type should be one of {}'
        raise ValueError(s.format(list(integral_types.keys())))
    return integral_types[integral_type.lower()]
        
#%%

#when modifying this, modify PIDController.calculate accordingly
stuff_to_log_list = ('feedback_value',
                     'new_value',
                     'p_term',
                     'i_term',
                     'd_term')
       
# Create a named tuple with default value for all fields an empty list
#PIDlog = namedtuple('PIDLog', stuff_to_log, defaults=[[]]*len(stuff_to_log)) #for Python 3.7 nd up
PIDlog = namedtuple('PIDLog', stuff_to_log_list)

class Logger:
    """A class implementing some logging capabilites both to memory, through
    the Logger.log attribute, and to disk by writing to logger.file. 
    
    Parameters
    ----------
    log_data : bool
        Decides if data should be logged to memory in Logger.log
    maxlen : int, float, optional
        Decides if data should be logged to disk in Logger.file
    write : bool, optional
        Value of the PID integral term's constant. Default=False.
    file : str,  optional
        File to which data should be logged if write=True. Default='log.txt'.
    
    Other parameters
    ----------------
    log_format : str, optional
        Decides format the logged data should have when writing to file. By
        default, it uses exponential notation with four significant digits.
        Default: '{:.3e}\t'.
    log_time : bool, optional
        Decides if Logger should write time of log to file. Not implemented. 
        Default: False.

    Methods
    -------
    input_log(stuff_to_log)
        calculates next step of the PID with internal parameters and state 
        using the feedback_value
    clearlog()
        clears instance log and creates a new log file
    
    Attributes
    ----------
    Other than the given in parameters:
    log : collections.deque
        Contains logged data to a maximum length of Logger.maxlen
    """
    
    def __init__(self, log_data, maxlen=10000, write=False, file='log.txt',
                 log_format='{:.3e}\t', log_time=False):
       
        #initialize stuff
        self._original_file = file
        self.log = []
        self.maxlen = maxlen
        self.log_format = log_format
        self.clearlog()
        
        #boolean values
        self.log_data = log_data
        self.write = write
#        self.log_time = log_time 

    @property
    def file(self):
        return self._file
    @file.setter
    def file(self, value):
        # Get unique file name
        self._file = new_name(value)
        self._original_file = value
        # States whether file is initialized with header        
        self.file_initialized = False
        
    @property
    def maxlen(self):
        return self.log.maxlen
    @maxlen.setter
    def maxlen(self, value):
        # Redefine log to new length
        self.log = deque(self.log, maxlen=value)
        
    @property
    def log_data(self):
        return self._log_data
    @log_data.setter
    def log_data(self, value):
        if not isinstance(value, bool):
            raise TypeError('log_data must be bool.')
        self._log_data = value

    @property
    def write(self):
        return self._write
    @write.setter
    def write(self, value):
        if not isinstance(value, bool):
            raise TypeError('write must be bool.')
            
        self._write = value
        # If write is True and file is not initialized, do it
        if self.write:
            self.__initialize_file__()            
            
    def __initialize_file__(self, file=None, force_init=False):
        '''Initialize file with header'''
        #only if file is not initialized or forece is not on
        if not self.file_initialized or force_init:
            if file is None:
                file = self.file
            
            # Initialize file with categories as header
            s = '#' + '{}\t' * len(stuff_to_log_list) + '\n'
            with open(file, 'a') as f:
                f.write(s.format(*stuff_to_log_list))
            
            self.file_initialized = True
            
    def write_now(self, file=None, force=False, footer=None):
        ''' Write log to file given file. If none is given, Logger.file
        will be used. File won't be written if Logger.write = True, unless
        user inputs force=True.'''
        # If write mode is on, do nothing, unless force=True
        if self.write and not force:
            print('File was written while logging. Use force=True to write anyway.')
            return
        
        # If using new given file, initialize it       
        if file is not None:
            file = new_name(file)
            self.__initialize_file__(file, force_init=True)
        else: 
            file = self.file
        
        with open(file, 'a') as f:
            for line in self.log:
                s = self._log_format_complete.format(*line)
                f.write(s)
            if footer is not None:
                f.write('# ' + footer)

    @property
    def log_format(self):
        return self._log_format
    @log_format.setter
    def log_format(self, value):
        if not isinstance(value, str):
            raise TypeError("log_format should be string with format '{:4e} '.")
        self._log_format = value
        self._log_format_complete = value * len(stuff_to_log_list) + '\n'
        
    @property
    def log_time(self):
        return self._log_time
    @log_time.setter
    def log_time(self, value):
        raise Exception('log_time not yet implemented')
        
    def input_log(self, stuff_to_log):
        '''Logs given data using log_format. Input stuff_to_log should
        match stuff_to_log_list.'''
        
        #if data should be logged
        if self.log_data:
            self.log.append(stuff_to_log)
            
        #if data should be writren
        if self.write:
            s = self._log_format_complete.format(*stuff_to_log)
            with open(self.file, 'a') as f:
                f.write(s)

    def clearlog(self):
        '''Creates new unique file and resets log.'''
        self.file = self._original_file
        self.log = deque(maxlen=self.maxlen)

    
        
#%% PID class

class PIDController:
    """A simple class implementing a PID contoller that keeps a log.
    
    Based on https://gist.github.com/hgrecco/16edd24989c63b6fc2eeb829c6d6b7ea
    
    Parameters
    ----------
    setpoint : int, float
        Value the PID is suposed to achieve and keep constant.
    kp : int, float, optional
        Value of the PID proportional term's constant. Default=1.
    ki : int, float, optional
        Value of the PID integral term's constant. Default=0.
    kd : int, float, optional
        Value of the PID derivative term's constant. Default=0.
    dt : int, float, optional
        Value of the time interval. Default=1.
    
    Other parameters
    ----------------
    log_data : bool, optional
        Decides whether to keep a log of every calculation or not. 
        Initializes PIDController.logger with a Logger inscantce. User
        can later declare more specific logger properties through set_logger
        or by directly modifying PIDController.logger attribute. Default=False.
    integrator : string {'infinite', 'windowed', 'weighted'}, optional
        Selects integration mode. Initializes PIDController.integrator with 
        one of the Integrator classes inscantce. User can later declare more 
        specific logger properties through set_integrator or by directly 
        modifying PIDController.integrator attribute. Default=False.

    Methods
    -------
    calculate(feedback_value)
        calculates next step of the PID with internal parameters and state
        using the feedback_value
    reset()
        resets PID internal parameters (i.e. integral, derivative and 
        proportional terms, last computed error and so on)
    clearlog()
        clears instance log, last log and creates a new log file
    set_logger(**props)
        sets logger attributes and properties
    set_integrator(**props)
        sets integrator attributes and properties    
    
    Attributes
    ----------
    p_term : float
        last recorded proportional term without proportional constant kp
    i_term : float
        last recorded proportional term without proportional constant kp
    d_term : float
        last recorded proportional term without proportional constant kp
    last_log : PIDLog
        last recorded log containing all properties as described in 
        fwp_pid.stuff_to_log
    log : PIDLog
        complete log since las reset   
    logger : Logger class instance
        Logger class instance to 
    integrator : Integrator class instance
        Instance of one of the integrator classes: WeightedIntegrator, 
        WindowedIntegrator or InfiniteIntegrator.
    integrator_type : str
        Type of integrator indicating which Integrator class instance is
        being used.
    
    Example
    -------
    >>> pid = PIDController(42, 3, 2, 1, log_data=True, integrator='weighted')
    >>> pid.set_integrator(alpha=1.2)
    >>> pid.set_logger(write=True, file='Measurements/PIDlogs/log.txt')
    >>> while True:
    >>>     signal = read()
            actuator = pid.calculate(signal)
            write(actuator)
            
    >>> the_log = pid.log
    >>> pid.reset()
    >>> pid.clearlog()
    """
    
    def __init__(self, setpoint, kp=1.0, ki=0.0, kd=0.0, dt=1, 
                 log_data=False, integrator='windowed'):

        #setpoint transformer defaults to nothing
        self.__setpoint_transformer = lambda val: val

        #pid parameters
        self.setpoint = setpoint
        self.kp = kp
        self.ki = ki
        self.kd = kd

        #integrator and logger class instances
        self.integrator = self.__create_integrator__(integrator, dt)
        self.logger = log_data
        
        #fresh p, i and d terms, and clear log
        self.reset()
        self.clearlog()
                
    def __repr__(self):
        string = 'PIDController with parameters: kp={}, ki={}, kd={}'
        return string.format(self.kp, self.ki, self.kd)

    def __str__(self):
        string = 'kp={}, ki={}, kd={}'
        return string.format(self.kp, self.ki, self.kd)
    
    def calculate(self, feedback_value):
        '''Calculates the next step of the PID with given feedback value.'''
#        self.last_feedback = feedback_value
        error = self.actual_setpoint - feedback_value

        delta_error = error - self.last_error

        self.p_term = error
        self.integrator.integrate(error) #i_term
        self.d_term = delta_error / self.dt

        self.last_error = error

        new_value = self.kp * self.p_term
        new_value += self.ki * self.i_term
        new_value += self.kd * self.d_term
        
        self.last_log = PIDlog._make([feedback_value, new_value,
                                  self.p_term, self.i_term, self.d_term])
        
        self.logger.input_log(self.last_log) #only if needed

        return new_value

    def clearlog(self):
        '''Clear PID log.'''
        self.logger.clearlog()
        self.last_log = PIDlog._make([[]]*len(stuff_to_log_list))

    def reset(self):
        '''Reset stored PID values, not parameters or log.'''
        self.last_error = 0
        self.p_term = 0
        self.d_term = 0
        self.integrator.reset()
#        self.last_feedback = 0

    @property
    def i_term(self):
        return self.integrator.integral
    
    @property
    def params(self):
        return {param:getattr(self, param) for param in
                ('kp', 'ki', 'kd', 'dt', 'setpoint')}
    
    # ##############
    # Setpoint stuff
    
    @property
    def setpoint_transformer(self):
        '''A function to transform the setpoint with, to make calcualtions
        marginaly faster. It should be used to trsanform from units the user
        inputs as setpoint to the units of the feedback_value for the 
        PIDController.calulate()method. By default, it does nothing.'''
        return self.__setpoint_transformer
    @setpoint_transformer.setter
    def setpoint_transformer(self, fun):
        try:
            fun(1)
        except TypeError as e:
            msg = ('Value passed to setpoint_setter must be a callable',
                   'object that trsanforms user-input setpoint to the'
                   'units the PID uses.')
            raise TypeError(''.join(msg))
        self.__setpoint_transformer = fun
        self.actual_setpoint = fun(self.setpoint)
        
    @property
    def setpoint(self):
        return self._setpoint
    @setpoint.setter
    def setpoint (self, value):
        self._setpoint = value
        self.actual_setpoint = self.setpoint_transformer(value) 
        
    # #########
    # Log stuff
    
    @property
    def log(self):
        #read-only
        if self.logger.log: #if it has logged data
            return self.__makelog__()
        else:
            raise ValueError('No logged data.')
            
    def __makelog__(self):
        '''Make a PIDlog nuamedtuple containing the list of each
        
        value in each field.'''
        log = []
        for i in range(len(stuff_to_log_list)):
            log.append([prop[i] for prop in self.logger.log])
        return PIDlog._make(log)

    # ############
    # Logger stuff

    @property
    def logger(self):
        return self._logger
    @logger.setter
    def logger(self, value):
        if isinstance(value, bool):
            self._logger = Logger(value)
        elif hasattr(value, 'calculate'): 
            self._logger = value
        else: 
            s = ('Logger should be a boolean reperesenting whether',
                 ' to log data or not, or a Logger class instance.')
            raise ValueError(''.join(s))
        
    def set_logger(self, **props):
        '''Sets logger properties given in props to given value.'''
        set_props(self.logger, **props)
            
    @property
    def log_data(self):
        return self.logger.log_data
    @log_data.setter
    def log_data(self, value):
        self.logger.log_data = value
       
    # ################
    # Integrator stuff
    
    @property
    def integrator(self):
        return self._integrator
    @integrator.setter
    def integrator(self, value):
        #if it's a string staiting integrator type
        if isinstance(value, str):
            self._integrator = self.__create_integrator__(value, self.dt) 
        #if it's an integrator class instance
        elif hasattr(value, 'integrate'): 
            self._integrator = value #integrator instance
        else:
            s = ('Integrator should be a strig stating integrator ',
                  'type or an integrator class instance.')
            raise ValueError(''.join(s))
        
    @property
    def integrator_type(self):
        return str(self._integrator)
    @integrator_type.setter
    def integrator_type(self, value):
        self.integrator = value
                
    def set_integrator(self, **props):
        '''Sets integrator properties given in props to given value.
        Each integratin mode has different properties. It does not
        set integrator type. For that, see integratoy_type and
        integrator.'''
        set_props(self.integrator, **props)
            
    @property
    def dt(self):
        return self.integrator.dt
    @dt.setter
    def dt(self, value):
        value = float(value)
        self.integrator.dt = value
        
    def __create_integrator__(self, integrator_str, dt):
        intcls = integral_switcher(integrator_str)
        return intcls(dt)
            
    # #########################################
    # Stuff to take into account actuator range
    
    @property
    def control_range(self):
        return (self.lower, self.upper)
    @control_range.setter
    def control_range(self, value):
        if not isinstance(value, (tuple, list)):
            raise TypeError('Value must be tuple or list.')
        if len(value)!=2:
            raise ValueError('Value must be of lenght 2.')
        self.lower, self.upper = value
    
    @property
    def lower(self):
        return self._lower
    @lower.setter
    def lower(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError('Value must be a number.')
        self._lower = value
            
    @property
    def upper(self):
        return self._upper
    @upper.setter
    def upper(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError('Value must be a number.')
        self._upper = value
                
    def calc_with_range(self, feedback_value):
        pass