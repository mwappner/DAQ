# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 13:25:14 2018

@author: Marcos
"""

from collections import deque

#%%
class InOut(deque):
    
    def put(self, val):
        out = self.pop()
        self.appendleft(val)
        return out
    
#a += b - f.put(b)
#%% PID class

stuff_to_log = ('feedback_value',
                'new_value',
                'p_term',
                'i_term',
                'd_term')

# Create a named tuple with default value for all fields an empty list
#PIDlog = namedtuple('PIDLog', stuff_to_log, defaults=[[]]*len(stuff_to_log)) #for Python 3.7 nd up
PIDlog = namedtuple('PIDLog', stuff_to_log)

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
        
        # Only log data if I ask it to
        if self.log_data:
            self.__log.append(self.last_log)

        return new_value

    def clearlog(self):
        self.__log = []
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
        if self.__log:
            return self.__makelog__()
        else:
            raise ValueError('No logged data.')

    @property
    def log_data(self):
        return self.__log_data
    @log_data.setter
    def log_data(self, value):
        if not isinstance(value, bool):
            raise TypeError('log_data must be bool.')
        self.__log_data = value
            
    def __makelog__(self):
        '''Make a PIDlog nuamedtuple containing the list of each
        value in each field.'''
        log = []
        for i in range(len(self.last_log)):
            log.append([prop[i] for prop in self.__log])
        return PIDlog._make(log)
