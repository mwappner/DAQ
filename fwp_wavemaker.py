# -*- coding: utf-8 -*-
""" This module works as a function generator

It includes:
Defined functions for several waveforms incorporating a switcher to make choosing easier.
A class for evaluating the multiple waveforms
A class for calculating fourier partial sums and evaluating it.
"""

import numpy as np
from scipy.signal import sawtooth, square
    
def create_sine(time, freq, *args):
    """ Creates sine wave 
    
    Parameters
    ----------
    time : array
        time vector in which to evaluate the funcion
    
    freq : int or float
        expected frequency of sine wave
    
    args : dummy 
        used to give compatibility with other functions
    
    Returns
    -------
    
    Evaluated sine wave of given frequency
    """
    
    wave =np.sin(2 * np.pi * time * freq)
    return wave        
    
def create_ramps(time, freq, type_of_ramp=1):
    """ Creates ascending and descending sawtooth wave,
    or a tringle wave, depending on the value of type_of_ramp,
    using the function 'sawtooth' from scypy signal module.
    Used by create_sawtooth_up, create_sawtooth_down and 
    create_triangular.
    
    Parameters
    ----------
    time : array
        time vector in which to evaluate the funcion
    
    freq : int or float
        expected frequency of created wave
    
    type_of_ramp : {0, 1, 2}
        0 returns a sawtooth waveform with positive slope
        1 returns a sawtooth waveform with negative slope
        0 returns a triangle waveform
    
    Returns
    -------
    
    Evaluated sawtooth or triangle wave of given frequency
    """
    
    wave = sawtooth(2 * np.pi * time * freq, type_of_ramp)
    return wave
    
def create_sawtooth_up(time, freq, *args):
    """ Creates sawtooth waveform with positive slope
   
    Parameters
    ----------
    time : array
        time vector in which to evaluate the funcion
    
    freq : int or float
        expected frequency of sawtooth wave

    args : dummy 
        used to give compatibility with other functions

    Returns
    -------
    
    Evaluated sawtooth waveform with positive slope  and given frequency
    """
    
    wave = create_ramps(time ,freq, 1)
    return wave        

def create_sawtooth_down(time, freq, *args):
    """ Creates sawtooth waveform with negative slope
   
    Parameters
    ----------
    time : array
        time vector in which to evaluate the funcion
    
    freq : int or float
        expected frequency of sawtooth wave
        
    args : dummy 
        used to give compatibility with other functions

    Returns
    -------
    
    Evaluated sawtooth waveform with negative slope and given frequency
    """

    wave = create_ramps(time, freq, 0)
    return wave        

def create_triangular(time, freq, *args):
    """ Creates a triangular wave with symmetric ramps
   
    Parameters
    ----------
    time : array
        time vector in which to evaluate the funcion
    
    freq : int or float
        expected frequency of triangular wave
        
    args : dummy 
        used to give compatibility with other functions

    Returns
    -------
    
    Evaluated triangular waveform with given frequency
    """
    

    wave = create_ramps(time, freq, .5)
    return wave        
     
def create_square(time, freq, dutycycle = .5, *args):
    """ Creates a square wave. Uses square function from
    scypy signal module

    Parameters
    ----------
    time : array
        time vector in which to evaluate the funcion
    
    freq : int or float
        expected frequency of square wave
        
    dutycycle=.5 : scalar or numpy array
        Duty cycle. Default is 0.5 (50% duty cycle). If 
        an array, causes wave shape to change over time,
        and must be the same length as time.
        
    args : dummy 
        used to give compatibility with other functions

    Returns
    -------
    
    Evaluated square waveform with given frequency
    """
    #dutycycle not implemented due to bug
    wave = square(2 * np.pi * time * freq)
    return wave
    
def create_custom(time, freq, *args):
    """ Creates a wave from given custom function. 
    
    Useful to get compatibility between the custom function provided and other
    modules like PyAudioWave.

    Parameters
    ----------
    time : array
        time vector in which to evaluate the funcion
    
    freq : int or float
        expected frequency of custom wave
        
    args : (*params, custom_func)
        *params should contain the parameters that will be passed to the custom
        function provided
        
    Returns
    -------
    
    Evaluated square waveform with given frequency
    """
    #last argument is the function, the rest are parameters
    *params, custom_func = args
    wave = custom_func(time, freq, *params)
    return wave

def create_sum(time, freq, amp, *args):
    """ Creates an arbitraty sum of sine waves.

    It uses the frequencies in freq and either uniform
    amplitude if amp is None, or the given amplitudes if
    amp is array-like. Output comes out normalized.
    
    Parameters
    ----------
    time : array
        time vector in which to evaluate the funcion
    
    freq : array-like
        expected frequency of sine wave
       
    amp : None or array-like 
        if None, amplitude of all summed waves is equal. If
        array-like, it should be same length as freq.
        
    args : dummy 
        used to give compatibility with other functions

    Returns
    -------
    
    Evaluated square waveform with given frequency
    """
    
    if len(amp)==0:
        #If am wasn't given, it is an empty tuple
        amp = np.ones(len(freq))
    
    if len(freq) != len(amp):
        raise ValueError('Amplitud and frequency arrays should e the same leght!')
    
    #to be able to handle time vectors and scalars
    if hasattr(time, '__len__'):
        time= np.array(time)
        wave = np.zeros(time.shape)
    else:
        wave = 0
        
    for f, a in zip(freq, amp):
        wave += create_sine(time, f) * a
    #Normalize it:
    wave /= sum(amp) 

    return wave
      
def given_waveform(input_waveform):
    """ Switcher to easily choose waveform.
    
    If the given waveform is not in the list, it raises a ValueError and a list
    containing the accepted inputs.
    
    Parameters
    ----------
    input_waveform : string
        name of desired function to generate
   
    Returns
    -------
    Chosen waveform function
    """
    
    switcher = {
        'sine': create_sine,
        'sawtoothup': create_sawtooth_up,
        'sawtoothdown': create_sawtooth_down  ,          
        'ramp': create_sawtooth_up, #redirects to sawtoothup
        'sawtooth': create_sawtooth_up, #redirects to sawtoothup
        'triangular': create_triangular,
        'square': create_square,
        'custom': create_custom,
        'sum': create_sum
    }

    func = switcher.get(input_waveform, wrong_input_build(list(switcher.keys())))
    return func

def wrong_input_build(input_list):
    def wrong_input(*args):
        msg = 'Given waveform is invalid. Choose from following list:{}'.format(input_list)
        raise ValueError(msg)
    return wrong_input

#%% Clase que genera ondas

class Wave:
    '''A class for generating and evaluating different waveforms.
  
    Attributes
    ----------
    waveform : str {'sine', 'sawtoothup', 'sawtoothdown', 'ramp', 'triangular', 'square', 'custom'} optional
        waveform type. If 'custom', function should acept inputs
        (time, frequency, *args). Default = 'sine'
    frequency : float (optional)
        wave frequency
    amplitude : float (optional)
        wave amplitud
        
    Methods
    ----------
    evaluate(time)
        returns evaluated function type
    evaluate_sr(sr, duration, nsamples)
        returns evaluated function type

    '''
    
    def __init__(self, waveform='sine', frequency=400, amplitude=1, *args):
        ''' See class atributes.
        
        If wave is 'custom', the custom function should be passed to *args.
        '''
        
        self._frequency = frequency
        self.amplitude = amplitude
        self.waveform = waveform
        self.extra_args = args
        
    def __str__(self):
        return '{} Wave instance.'.format(self.waveform)

    @property
    def frequency(self):
        '''Frequency getter: returns frequency of wave. 
        
        If frequency is an iterable, as it be in a sum or a 
        custom function, returns first value. Used to have 
        backwards compatibility wen sum and custom were added.'''
        
        if isinstance(self._frequency, (list, tuple, np.ndarray)):
            return self._frequency[0]
        else:
            return self._frequency
        
    @frequency.setter
    def frequency(self, value):
        '''Frequency setter: sets value as self._frequency.'''
        self._frequency = value    
        
    @property
    def waveform(self):
        '''Waveform getter'''
        return self._waveform_type

    @waveform.setter
    def waveform(self, value):
        '''Wavefor setter'''
        self._waveform_func = given_waveform(value)
        self._waveform_type = value


    def evaluate(self, time, *args):
        '''Takes in an array-like object to evaluate the funcion in.
        
        Parameters
        ----------
        time : array
            time vector in which to evaluate the funcion
        args : tuple (optional)
            extra arguments to be passed to evaluated function
            
        Returns
        -------
        
        Evaluated waveform 
        '''         

        if isinstance(self.amplitude, (list, tuple, np.ndarray)):
            #for sums 
            wave = self._waveform_func(time, self._frequency, self.amplitude)
        else:
            wave = self._waveform_func(time, self._frequency, *args, self.extra_args) * self.amplitude
        return wave

    def evaluate_sr(self, sampling_rate, duration=None, nsamples=None, return_time=False, custom_args=()):
        '''Evaluates the function in a time vector with the given sampling rate
        for given duration or ampunt of samples.
        
        User must specify either duration or nsamples, but not both.
        
        Parameters
        ----------
        sampling_rate : int
            time vector in which to evaluate the funcion
        duration : float (optional)
            duration of signal. Default = None
        nsamples : int (optional)
            amount of samples tu return. Default = None
        return_time : bool (optional)
            decides if time vector is returned or not
        custom_args : tuple (optional)
            extra arguments to be passed to evaluated function
            
        Returns
        -------
        
        Evaluated waveform or tuple containing time and evaluated waveform
        '''
        
        if sampling_rate < 1:
            raise ValueError('Sampling rate must be postive integer.')
            
        if duration is None:
            if nsamples is None:
                raise ValueError('Must specify either duration or nsamples.')
            else:
                if nsamples < 1:
                    raise ValueError('nsamples must be positive integer.')
                time = np.linspace(0, nsamples / sampling_rate, nsamples)
        else:
            if nsamples is not None:
               raise ValueError("Can't specify both duration and nsamples. One must be None (dafault).")
            else:
                if not duration > 0:
                    raise ValueError('duration must be positive.')
                time = np.linspace(0, duration, int(sampling_rate * duration))
                
        if return_time:
            return time, self.evaluate(time, *custom_args)
        else:
            return self.evaluate(time, *custom_args)

#%% Waves for many channels
class MultichannelWave:
    '''A class for generating and evaluating different waveforms. Supports many
    waves in a single instance, hence 'multichannel'.
  
    Attributes (read only)
    ----------
    waveform : str {'sine', 'sawtoothup', 'sawtoothdown', 'ramp', 'triangular', 'square', 'custom'}
    frequency : float
        wave frequency
    amplitude : float
        wave amplitud
    nchannels : int

        number current channels
        
    Methods
    ----------
    add_channel(waveform, frequency, amplitude)
        return nothing, adds Wave instance to self.waves
    evaluate(time)
        returns evaluated function type
    evaluate_sr(sr, duration, nsamples)
        returns evaluated function type

    '''
    def __init__(self):
        self.waves = []

    def __str__(self):
        return 'MultichannelWave instance with {} channels containing the following waveforms: {}'.format(self.nchannels, self.waveform)
        
    def add_channel(self, *args, **kwargs):
        ''' Adds a channel to the MultichannelWave instance by calling
        insantiating Wave with the given parameters. See Wave.
        '''

        self.waves.append(Wave(*args, **kwargs))

    @property
    def frequency(self):
        return [w.frequency for w in self.waves]

    @frequency.setter
    def frequency(self, value):
        raise AttributeError('Frequency should be set for each wave individually. Use self.waves.frequency.')
    
    @property
    def amplitude(self):
        return [w.amplitude for w in self.waves]
    
    @amplitude.setter
    def amplitude(self, value):
        raise AttributeError('Amplitude should be set for each wave individually. Use self.waves.amplitude.')

    @property
    def waveform(self):
        return [w.waveform for w in self.waves]
        
    @waveform.setter
    def waveform(self, value):
        raise AttributeError('Waveform should be set for each wave individually. Use self.waves.waveform.')

    @property
    def nchannels(self):
        return len(self.waves)

    @nchannels.setter
    def nchannels(self, value):
        raise AttributeError('nchannels can not be set.')

    def evaluate(self, *args, **kwargs):
        '''Takes in an array-like object to evaluate the funcion in.
        
        The returned array has a channel in each column.
        
        Parameters
        ----------
        time : array
            time vector in which to evaluate the funcion
        args : tuple (optional)
            extra arguments to be passed to evaluated function
            
        Returns
        -------
        
        Array of evaluated waveform 
        '''   
        
        signal = [w.evaluate(*args, **kwargs) for w in self.waves]
        return np.array(signal).T
    
    def evaluate_sr(self, *args, **kwargs):
        '''Evaluates the functions in a time vector with the given sampling rate
        for given duration or ampunt of samples.
        
        User must specify either duration or nsamples, but not both. The 
        returned array has a channel in each column.
        
        Parameters
        ----------
        sampling_rate : int
            time vector in which to evaluate the funcion
        duration : float (optional)
            duration of signal. Default = None
        nsamples : int (optional)
            amount of samples tu return. Default = None
        return_time : bool (optional)
            decides if time vector is returned or not
        custom_args : tuple (optional)
            extra arguments to be passed to evaluated function
            
        Returns
        -------
        
        Array of evaluated waveforms or tuple containing time and 
        array of evaluated waveforms
        '''
        
        # Tries to get return_time from kwargs. If it wasn't passed, set default false
        return_time = kwargs.get('return_time', False)
        
        if return_time:
            time, signal = self.waves[0].evaluate_sr(*args, **kwargs)
            
            if len(self.waves) > 1:
                signal = [signal]
                kwargs['return_time'] = False
                signal.extend([w.evaluate_sr(*args, **kwargs) for w in self.waves[1:]])
                
                return time, np.array(signal).T
            
            else:
                return time, signal
            
        else:
            signal = [w.evaluate_sr(*args, **kwargs) for w in self.waves]
            return np.array(signal).T
    
'''Example:
    
    mw = MultichannelWave()
    waves = ('sine', 'sine', 'square')
    frequencies = (2, 3, 4)
    amplitudes = (1, .7, .8)
    for w, f, a in zip(waves, frequencies, amplitudes):
        mw.add_channel(w, f, a)
    
    time = np.linspace(0, 1, 400)
    signal = mw.evaluate(time)
    plt.plot(time, signal)
    '''
#%% Fourier series class for wave generator

def fourier_switcher(input_waveform):
    """ Switcher to easily choose waveform.
    
    If the given waveform is not in the list, it raises a ValueError and a list
    containing the accepted inputs.
    
    Parameters
    ----------
    input_waveform : string
        name of desired function to generate
   
    Returns
    -------
    Chosen waveform function
    """
    
    switcher = {
            'square': square_series,
            'triangular': triangular_series,
            'sawtooth': sawtooth_series,
            'custom': custom_series}
    func = switcher.get(input_waveform, wrong_input_build(list(switcher.keys())))
    return func

def square_series(order, freq, *args):
    """ Creates parameters for a square series
    
    If the given waveform is not in the list, it raises a ValueError and a list
    containing the accepted inputs.
    
    Parameters
    ----------
    order : int
        order up to which to calculate fourier partial sum 
    frequency : float
        fundamental frequency of generated fourier wave
   
    Returns
    -------
    amps, freqs
        amplitude and frequency vectors used in calculation of partial sum
    """
    
    amps = [1/n for n in range(1, 2*order+1, 2)]
    freqs = np.arange(1, 2*order+1, 2) * freq
    return amps, freqs
        
def sawtooth_series(order, freq, *args):
    """ Creates parameters for a sawtooth series
    
    If the given waveform is not in the list, it raises a ValueError and a list
    containing the accepted inputs.
    
    Parameters
    ----------
    order : int
        order up to which to calculate fourier partial sum 
    frequency : float
        fundamental frequency of generated fourier wave
   
    Returns
    -------
    amps, freqs
        amplitude and frequency vectors used in calculation of partial sum
    """
    
    amps = [1/n for n in range(1, order+1)]
    freqs = np.arange(1, order+1) * freq
    return amps, freqs
    
def triangular_series(order, freq, *args):
    """ Creates parameters for a triangluar series
    
    If the given waveform is not in the list, it raises a ValueError and a list
    containing the accepted inputs.
    
    Parameters
    ----------
    order : int
        order up to which to calculate fourier partial sum 
    frequency : float
        fundamental frequency of generated fourier wave
   
    Returns
    -------
    amps, freqs
        amplitude and frequency vectors used in calculation of partial sum
    """
    
    amps = [(-1)**((n-1)*.5)/n**2 for n in range(1, 2*order+1, 2)]
    freqs = np.arange(1, 2*order+1, 2) * freq
    return amps, freqs
    
def custom_series(order, freq, amp, *args):
    """ Creates parameters for a custom fourier series
    
    If the given waveform is not in the list, it raises a ValueError and a list
    containing the accepted inputs.
    
    Parameters
    ----------
    order : dummy
        is redefined inside implementatoin. Kept for compatibility.
    frequency : float
        fundamental frequency of generated fourier wave
    amp: tuple
        tuple containing amplitude vectors of cosine and sine terms for the
        custom fourier series
   
    Returns
    -------
    amps, freqs
        amplitude tple (passed directly from input) and frequency vector used
        in calculation of partial sum
    """
    
    order = len(amp[0])
    amps = amp
    freqs = np.arange(1, order+1) * freq
    return amps, freqs
    
class Fourier:
    '''Generates an object with a single method: evaluate(time).
  
    Attributes
    ----------
    waveform : str {'sawtooth',  'triangular', 'square', 'custom'} 
        waveform type.
    wave : Wave object 
        Wave instance containgng a sum object that implements the fourier
        series up to given order.
    custom : bool
        desides wether user has requested custom series or not
        
    Methods
    ----------
    evaluate(time)
        returns evaluated fourier partial sum

    '''
    def __init__(self, waveform='square', frequency=400, order=5, *args):
        """Initializes class instance. 
               
        Parameters
        ----------
        waveform : str {'sawtooth',  'triangular', 'square', 'custom'} (Optional)
            waveform type. Default: 'square'
        frequency : float (Optional)
            fundamental frequency of the constructed wave in Hz. Default: 400
        order : int (optional)
            order of the constructed fourier series, i.e. the series will
            be calculated up to the nth non zero term, with n=order.
        args : tuple (optional)
            if waveform is 'custom', a tuple of length 2, each element 
            containing the amplitudes of the cosine and sine terms, 
            respectively. Order will be ignored and will be assumed to be
            equal to len(amplitudes[0]).
            
        Returns
        -------
        
        Evaluated fourier partial sum 
        """          
        
        self.waveform = waveform
        self._order = order #doesn't call setup_props because there's no frequency defined yet
        self.setup_props(frequency)
        self.extra_args = args
        
        self.custom = waveform=='custom'
    
    
    def setup_props(self, freq):
        '''Sets up frequencyes, amplitudes and wave attributes for given freq.'''
        
        self.amplitudes, self._frequencies =  self._waveform_maker(self.order, freq)
        self.wave = Wave('sum', self._frequencies, self.amplitudes)

        
    @property
    def frequency(self):
        '''Frequency getter: returns fundamental frequency of wave.'''
        
        return self._frequencies[0]
        
    @frequency.setter
    def frequency(self, value):
        '''Frequency setter: calculates the frequency vector for given
        fundamental frequency and order. Redefine Wave accordingly.'''
        
        self.setup_props(value)
        
    @property
    def order(self):
        '''Order getter: returns order of the last nonzero term in partial sum.'''
        
        return self._order
        
    @order.setter
    def order(self, value):
        '''Order setter: Calculates new appropiate frequency and amplitude
        vectors for given order value. Redefine Wave accordingly.'''
        
        self._order = value
        self.setup_props(self.frequency)
    
    @property
    def waveform(self):
        '''Waveform getter: returns waveform string.'''

        return self._waveform
    
    @waveform.setter
    def waveform(self, value):
        '''Wavefrorm setter: sets the appropiate waveform_maker and refreshes
        the amplitude vector.'''

        self._waveform = value
        self._waveform_maker = fourier_switcher(value)
        self.setup_props(self.frequency)

    def evaluate(self, time):
        """Takes in an array-like object to evaluate the funcion in.
        
        Parameters
        ----------
        time : array
            time vector in which to evaluate the funcion
            
        Returns
        -------
        
        Evaluated waveform 
        """          
        
        if self.custom:
            #missing support for custom phases
            
            #cosine series:
            self.wave.amplitude = self.amplitudes[0]
            wave = self.wave.evaluate(time + np.pi *.5) * .5
            
            #sine series:
            self.wave.amplitude = self.amplitudes[1]
            wave += self.wave.evaluate(time) * .5
            
            return wave
            
        else:
            return self.wave.evaluate(time)

    def evaluate_sr(self, *args, **kwargs):
        """Evaluates the function in a time vector with the given sampling rate
        for given duration or ampunt of samples.
        
        User must specify either duration or nsamples, but not both.
        
        Parameters
        ----------
        sampling_rate : int
            time vector in which to evaluate the funcion
        duration : float (optional)
            duration of signal. Default = None
        nsamples : int (optional)
            amount of samples tu return. Default = None
        return_time : bool (optional)
            decides if time vector is returned or not
        custom_args : tuple (optional)
            extra arguments to be passed to evaluated function
            
        Returns
        -------
        
        Evaluated waveform or tuple containing time and evaluated waveform
        """
        if self.custom:
            raise ValueError('No support for custom waves with this method.')
            
        else:
            return self.wave.evaluate_sr(*args, **kwargs)