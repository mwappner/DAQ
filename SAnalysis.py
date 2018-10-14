# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 15:28:58 2018

@author: Usuario
"""

import fwp_save as sav
import os
import numpy as np

#%% Samplerate_Sweep

# PARAMETERS

# Main parameters
periods_to_meassure = 100
signal_frequency = 1.2e3

samplerate_min = 100
samplerate_max = 10e3
samplerate_n = 200

name = 'Samplerate_Sweep'

# Other parameters
samplerate = np.linspace(samplerate_min,
                         samplerate_max,
                         samplerate_n)
duration = periods_to_meassure / signal_frequency

folder = os.path.join(os.getcwd(),
                      'Measurements',
                      name)
filename = lambda samplerate : os.path.join(
        folder, 
        '{:.0f}_Hz.txt'.format(samplerate))

# ACTIVE CODE

data = {}
for sr in samplerate:
    data.update({sr: np.loadtxt(filename(samplerate))})
    

