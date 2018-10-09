# -*- coding: utf-8 -*-
"""The 'fwp_save' module saves data, dealing with overwriting.

It could be divided into 2 sections:
    (1) making new directories and free files to avoid overwriting 
    ('new_dir', 'free_file')
    (2) saving data into files with the option of not overwriting 
    ('saveplot', 'savetext', 'savewav')

new_dir : function
    Makes and returns a new related directory to avoid overwriting.
free_file : function
    Returns a name for a new file to avoid overwriting.
saveplot : function
    Saves a matplotlib.pyplot plot on an image file (i.e: 'png').
savetext : function
    Saves some np.array like data on a '.txt' file.
savewav : function
    Saves a PyAudio encoded audio on a '.wav' file.

@author: Vall
@date: 09-17-2018
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pyaudio
import wave

#%%

def new_dir(my_dir, newformat='{}_{}'):
    
    """Makes and returns a new directory to avoid overwriting.
    
    Takes a directory name 'my_dir' and checks whether it already 
    exists. If it doesn't, it returns 'dirname'. If it does, it 
    returns a related unoccupied directory name. In both cases, 
    the returned directory is initialized.
    
    Parameters
    ----------
    my_dir : str
        Desired directory (should also contain full path).
    
    Returns
    -------
    new_dir : str
        New directory (contains full path)
    
    Yields
    ------
    new_dir : directory
    
    """
    
    sepformat = newformat.split('{}')
    base = os.path.split(my_dir)[0]
    
    new_dir = my_dir
    while os.path.isdir(new_dir):
        new_dir = os.path.basename(new_dir)
        new_dir = new_dir.split(sepformat[-2])[-1]
        try:
            new_dir = new_dir.split(sepformat[-1])[0]
        except ValueError:
            new_dir = new_dir
        try:
            new_dir = newformat.format(my_dir, str(int(new_dir)+1))
        except ValueError:
            new_dir = newformat.format(my_dir, 2)
        new_dir = os.path.join(base, new_dir)
    os.makedirs(new_dir)
        
    return new_dir

#%%

def free_file(my_file, newformat='{}_{}'):
    
    """Returns a name for a new file to avoid overwriting.
        
    Takes a file name 'my_file'. It returns a related unnocupied 
    file name 'free_file'. If necessary, it makes a new 
    directory to agree with 'my_file' path.
        
    Parameters
    ----------
    my_file : str
        Tentative file name (must contain full path and extension).
    newformat='{}_{}' : str
        Format string that indicates how to make new names.
    
    Returns
    -------
    new_fname : str
        Unoccupied file name (also contains full path and extension).
        
    """
    
    base = os.path.split(my_file)[0]
    extension = os.path.splitext(my_file)[-1]
    
    if not os.path.isdir(base):
        os.makedirs(base)
        free_file = my_file
    
    else:
        sepformat = newformat.split('{}')
        free_file = my_file
        while os.path.isfile(free_file):
            free_file = os.path.splitext(free_file)[0]
            free_file = free_file.split(sepformat[-2])[-1]
            try:
                free_file = free_file.split(sepformat[-1])[0]
            except ValueError:
                free_file = free_file
            try:
                free_file = newformat.format(
                        os.path.splitext(my_file)[1],
                        str(int(free_file)+1),
                        )
            except ValueError:
                free_file = newformat.format(free_file, 2)
            free_file = os.path.join(base, free_file+extension)
    
    return free_file

#%%

def saveplot(file, overwrite=False):
    
    """Saves a plot on an image file.
    
    This function saves the current matplotlib.pyplot plot on a file. 
    If 'overwrite=False', it checks whether 'file' exists or not; if it 
    already exists, it defines a new file in order to not allow 
    overwritting. If overwrite=True, it saves the plot on 'file' even if 
    it already exists.
    
    Variables
    ---------
    file : string
        The name you wish (must include full path and extension)
    overwrite=False : bool
        Indicates whether to overwrite or not.
    
    Returns
    -------
    nothing
    
    Yields
    ------
    an image file
    
    See Also
    --------
    free_file()
    
    """
    
    if not os.path.isdir(os.path.split(file)[0]):
        os.makedirs(os.path.split(file)[0])
    
    if not overwrite:
        file = free_file(file)

    plt.savefig(file, bbox_inches='tight')
    
    print('Archivo guardado en {}'.format(file))
    

#%%

def savetext(datanumpylike, file, overwrite=False):
    
    """Takes some array-like data and saves it on a '.txt' file.
    
    This function takes some data and saves it on a '.txt' file.
    If 'overwrite=False', it checks whether 'file' exists or not; if it 
    already exists, it defines a new file in order to not allow 
    overwritting. If overwrite=True, it saves the plot on 'file' even if 
    it already exists.
    
    Variables
    ---------
    datanumpylike : array, list
        The data to be saved.
    file : string
        The name you wish (must include full path and extension)
    overwrite=False : bool
        Indicates whether to overwrite or not.
    
    Return
    ------
    nothing
    
    Yield
    -----
    '.txt' file
    
    See Also
    --------
    free_file()
    
    """
    
    base = os.path.split(file)[0]
    if not os.path.isdir(base):
        os.makedirs(base)

    file = os.path.join(
            base,
            (os.path.splitext(os.path.basename(file))[0] + '.txt'),
            )
    
    if not overwrite:
        file = free_file(file)
        
    np.savetxt(file, np.array(datanumpylike), 
               delimiter='\t', newline='\n')
    
    print('Archivo guardado en {}'.format(file))
    
    return

#%%

def savewav(datapyaudio,
            file,
            data_nchannels=1,
            data_format=pyaudio.paFloat32,
            data_samplerate=44100,
            overwrite=False):
    
    """Takes a PyAudio byte stream and saves it on a '.wav' file.
    
    Takes a PyAudio byte stream and saves it on a '.wav' file. It 
    specifies some parameters: 'datanchannels' (number of audio 
    channels), 'dataformat' (format of the audio data), and 'samplerate' 
    (sampling rate of the data). If 'overwrite=False', it checks whether 
    'file' exists or not; if it already exists, it defines a new file in 
    order to not allow overwritting. If overwrite=True, it saves the 
    plot on 'file' even if it already exists.
    
    Variables
    ---------
    datapyaudio : str
        PyAudio byte stream.
    file : str
        Desired file (must include full path and extension)
    data_nchannels=1 : int
        Data's number of audio channels.
    data_format : int
        Data's PyAudio format.
    overwrite=False : bool
        Indicates wheter to overwrite or not.
    
    Returns
    -------
    nothing
    
    Yields
    ------
    '.wav' file
    
    See Also
    --------
    free_file()
    
    """
    
    base = os.path.split(file)[0]
    if not os.path.isdir(base):
        os.makedirs(base)

    file = os.path.join(
            base,
            (os.path.splitext(os.path.basename(file))[0] + '.wav'),
            )
    
    if not overwrite:
        file = free_file(file)
    
    datalist = []
    datalist.append(datapyaudio)
    
    p = pyaudio.PyAudio()
    wf = wave.open(file, 'wb')
    
    wf.setnchannels(data_nchannels)
    wf.setsampwidth(p.get_sample_size(data_format))
    wf.setframerate(data_samplerate)
    wf.writeframes(b''.join(datalist))
    
    wf.close()
    
    print('Archivo guardado en {}'.format(file))
    
    return