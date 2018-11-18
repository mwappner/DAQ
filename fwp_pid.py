# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 13:25:14 2018

@author: Marcos
"""

from collections import deque, namedtuple
from fwp_save import new_name

#%%
class InOut(deque):
    
    def put(self, val):
        out = self.pop()
        self.appendleft(val)
        return out
    
#a += b - f.put(b)
        
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
        self.file = file
        self._log = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.log_format = log_format
        
        #boolean values
        self.log_data = log_data
        self.write = write
        self.log_time = log_time

    @property
    def file(self):
        return self._file
    @file.setter
    def file(self, value):
        self._file = new_name(value)
        #States whether file is initialized with header        
        self.file_initialized = False
        
    @property
    def maxlen(self):
        return self._maxlen
    @maxlen.setter
    def maxlen(self, value):
        self._maxlen = value #redefine _log to new length
        self._log = deque(self._log, maxlen=value)
        
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
        
    def log(self, stuff_to_log):
        '''Logs given data using log_format. Input stuff_to_log should
        match stuff_to_log_list.'''
        
        #if data should be logged
        if self.log_data:
            self._log.append(stuff_to_log)
            
        #if data should be writren
        if self.write:
            s = self._log_format_complete.format(*stuff_to_log)
            with open(self.file, 'a') as f:
                f.write(s)
    
        
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
    log_data : bool, optional
        Devides whether to keep a log of every calculation or not. Default=False.
    
    Other parameters
    ----------------
    max_log_lengh : int, optional
        Maximum allowed log length. Default 1e7
    on_log_overflow : str {'del', 'delall', 'write'}, optional
        Decides action to take when max_log_length is reached. 'del' deletes oldest
        entry and adds new one (like in a fixed-size buffer), 'delall' resets the
        log to an empty list, 'write' writes the log to a log.txt file and clears
        the log. Won't overwrite existing log.txt files.
    integration_mode : str {'full', 'weighted', 'fixed'}, optional
        Sets integral term mode. 'full' integrates over all time, i.e. since the PID
        was last reset. 'weighted' gives a wight to each value that shrinks with each
        ieration, making older samples less important. 'fixed' integrates inside a 
        window of fixed length.
    integration_params : dict, optional
        Parameters for chosen integration mode.
    
    Example
    -------
    >>> pid = PIDController(42, 3, 2, 1, log_data=True)
    >>> while True:
    >>>     signal = read()
            actuator = pid.calculate(signal)
            write(actuator)
            
        the_log = pid.log
        pid.reset()
        pid.clearlog()
    """
    def __init__(self, setpoint, kp=1.0, ki=0.0, kd=0.0, dt=1, 
                 log_data=False, max_log_length=1e6, on_log_overflow='del',
                 integration_mode='full', integration_params={}):

        self.setpoint = setpoint
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        
        #fresh p, i and d terms
        self.reset()
        
        # start with a fresh log
        self.log_data = log_data
        self.clearlog()
        
    def __repr__(self):
        string = 'PIDController with parameters: kp={}, ki={}, kd={}'
        return string.format(self.kp, self.ki, self.kd)

    def __str__(self):
        string = 'kp={}, ki={}, kd={}'
        return string.format(self.kp, self.ki, self.kd)
    
    def calculate(self, feedback_value):
        
#        self.last_feedback = feedback_value
        error = self.setpoint - feedback_value

        delta_error = error - self.last_error

        self.p_term = error
        self.i_term += error * self.dt
        self.d_term = delta_error / self.dt

        self.last_error = error

        new_value = self.kp * self.p_term
        new_value += self.ki * self.i_term
        new_value += self.kd * self.d_term
        
        self.last_log = PIDlog._make([feedback_value, new_value,
                                  self.p_term, self.i_term, self.d_term])
        
        self.logger.log(self.last_log)

        return new_value

    def clearlog(self):
        self.logger.log = deque(maxlen=10000)
        self.last_log = PIDlog._make([[]]*len(stuff_to_log))

    def reset(self):
        self.last_error = 0
        self.p_term = 0
        self.i_term = 0
        self.d_term = 0
#        self.last_feedback = 0

    @property
    def log(self):
        #read-only
        if self._log:
            return self.__makelog__()
        else:
            raise ValueError('No logged data.')

            
    def __makelog__(self):
        '''Make a PIDlog nuamedtuple containing the list of each
        value in each field.'''
        log = []
        for i in range(len(stuff_to_log)):
            log.append([prop[i] for prop in self.logger._log])
        return PIDlog._make(log)
