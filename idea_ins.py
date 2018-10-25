# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import nummpy as np

def from_visa(port):
    # Do stuff to open communication with instrument
    pass

def check_isnumber(num):
    try:
        num = float(num)
        num + 0
    except (ValueError, TypeError) as e:
        e.args = 'Value must be a number or number string.',
        raise
    else:
        return num

#%%
class Instrument:
    def __init__(self, port, autoupdate=True):
        self.port = port
        self.autoupdate = autoupdate
        self.vi = from_visa(port)
        
        #set attributes from configuration
        config_dict = self.get_config_output()
        for key, value in config_dict.items():
            self.__setattr__('_' + key, value)
            
        self.to_update = dict()
            
    def get_config_output(self, channel=1):
        # Ask for curren configuration. Return a dictionary.
        # Configuration has a known list of keys
        pass
    
    def tell_gen(self, prop_to_change, value):
        # Communicate with func. gen. with visa
        # Probably use a dictionary to relate attribute name (i.e., frequency,
        # amplitude, waveform) to propper visa command
        pass
    
    def run_if_updatable(self, func):
        '''Decorator. If autoupdate is True, automaticaly update value. Else, 
        place into list to update.'''   
        prop = func.__name__
        def modify(value):
            if self.autoupdate:
                #Update value
                self.tell_gen(prop, value)
            else:
                self.to_update[prop] = value
            return func(value)
        
    # There HAS to be a better way to do the following. 
    # Decorating __setattr__, maybe?
    # Attributes will depend on type of instrument. Might be necessary to
    # define properties programaticaly.
    
    @property
    def frequency(self):
        return self._frequency
    @frequency.setter
    @run_if_updatable
    def frequency(self, value):
        value = check_isnumber(value)
        if value < 0:
            raise ValueError('Frequency should be positive.')
        self._frequency = value
        
    @property
    def amplitude(self):
        return self._amplitude
    @amplitude.setter
    @run_if_updatable
    def amplitude(self, value):
        value = check_isnumber(value)
        if value < 0:
            raise ValueError('Amplitude should be positive.')
        self._amplitude = value
        
    @property
    def waveform(self):
        return self._waveform
    @waveform.setter
    @run_if_updatable
    def waveform(self, value):
        #chech if value is an appropiate string
        self._waveform = value
       
    @property
    def output(self):
        return self._output
    @output.setter
    @run_if_updatable
    def output(self, value):
        self._output = value
        
    # And so on for other properties
 
    def update_many(self, **props_and_values):
        # Concatenate many commands to send to generator
        # Props_and_values will be a dictionary containing 
        # property names and values to update
        pass
    
    def update_pending(self):
        ''' Update all attributes whose values where changed but
        not sent to the func. gen. If autoupdate is False, this does
        nothing.'''
        
        if self.autoupdate:
            self.update_many(self.to_update)
            
#%%
            
#Some dictionary containing the properties available to a function generator
gen_props = dict() 

class Gen(Instrument):
    ''' A function generator class. 
    Missing multi channel support. Could be implemented adding functionality
    to base classes methods, adding attributes for channels as instruments. 
    Instruments could also be rewritten to have a channel attribute to pass to
    the calls through visa. In that case, the visa itself should be in a 
    different object.'''
        def __init__(self, port, nchannels=1, autoupdate=True):
        
        self.nchannels = nchannels
        self.available_props = gen_props
        super().__init__(port, autoupdate)
        
class Osc(Instrument):
    ''' An osciloscope class. 
    Missing multi channel support. Could be implemented adding functionality
    to base classes methods, adding attributes for channels as instruments. 
    Instruments could also be rewritten to have a channel attribute to pass to
    the calls through visa. In that case, the visa itself should be in a 
    different object.'''
    
    def __init__(self, port, nchannels=1, autoupdate=True):
        
        self.nchannels = nchannels
        self.available_props = gen_props
        super().__init__(port, autoupdate)    
        
    def get_screen(self):
        pass
    
    def measure(self):
        pass
    
    def whatever_else_you_need(self):
        pass