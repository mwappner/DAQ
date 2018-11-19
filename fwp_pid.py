# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 13:25:14 2018

@author: Marcos
"""

from collections import deque, namedtuple
from fwp_save import new_name

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
#a += b - f.put(b)
        
#%% Itegrator classes

class InfiniteIntegrator:
    
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
        self._window_length = value
        self.window = InOut(size=self.window_length, iterable=self.window)
        self.integral = sum(self.window)
        
class WeightedIntegrator:
    
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
    if integral_type not in integral_types.keys():
        s = 'integral type should be one of {}'
        raise ValueError(s.format(list(integral_types.keys())))
    return integral_types[integral_type]
        
#%%

stuff_to_log_list = ('feedback_value',
                'new_value',
                'p_term',
                'i_term',
                'd_term')
       
# Create a named tuple with default value for all fields an empty list
#PIDlog = namedtuple('PIDLog', stuff_to_log, defaults=[[]]*len(stuff_to_log)) #for Python 3.7 nd up
PIDlog = namedtuple('PIDLog', stuff_to_log_list)

class Logger:
    
    def __init__(self, log_data, maxlen=10000, write=False, file='log.txt',
                 log_format='{:.3e}\t', log_time=False):
       
        #initialize stuff
        self._original_file = file
        self._maxlen = maxlen
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
        self._file = new_name(value)
        self._original_file = value
        #States whether file is initialized with header        
        self.file_initialized = False
        
    @property
    def maxlen(self):
        return self._maxlen
    @maxlen.setter
    def maxlen(self, value):
        self._maxlen = value #redefine log to new length
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
        if self.write and not self.file_initialized:
            # Initialize file with categories as header
            s = '#' + '{}\t' * len(stuff_to_log_list) + '\n'
            with open(self.file, 'a') as f:
                f.write(s.format(*stuff_to_log_list))
            
            self.file_inicialized = True
        
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

    
    Example
    -------
    >>> pid = PIDController(42, 3, 2, 1, log_data=True, integrator='weighted')
    >>> pid.set_integrator(alpha=1.2)
    >>> while True:
    >>>     signal = read()
            actuator = pid.calculate(signal)
            write(actuator)
            
        the_log = pid.log
        pid.reset()
        pid.clearlog()
    """
    def __init__(self, setpoint, kp=1.0, ki=0.0, kd=0.0, dt=1, 
                 log_data=False, integrator='windowed'):

        #pid parameters
        self.setpoint = setpoint
        self.kp = kp
        self.ki = ki
        self.kd = kd

        #integrator and logger class instances
        self.integrator = self.create_integrator(integrator, dt)
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
        '''Calculates the PID next step.'''
#        self.last_feedback = feedback_value
        error = self.setpoint - feedback_value

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

    #log stuff
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

    #logger stuff
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
        for propname, propval in props.items():
            setattr(self.logger, propname, propval)
            
    @property
    def log_data(self):
        return self.logger.log_data
    @log_data.setter
    def log_data(self, value):
        self.logger.log_data = value
        
    #integrator stuff
    @property
    def integrator(self):
        return self._integrator
    @integrator.setter
    def integrator(self, value):
        #if it's a string staiting integrator type
        if isinstance(value, str):
            self._integrator = self.create_integrator(value, self.dt) 
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
        for propname, propval in props.items():
            setattr(self.integrator, propname, propval)
            
    @property
    def dt(self):
        return self.integrator.dt
    @dt.setter
    def dt(self, value):
        value = float(value)
        self.integrator.dt = value
        
    def create_integrator(self, integrator_str, dt):
        intcls = integral_switcher(integrator_str)
        return intcls(dt)
            
