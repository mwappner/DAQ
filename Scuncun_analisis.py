# -*- coding: utf-8 -*-
"""
Analisis de Cohen-Coon

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import fwp_string as fst
from fwp_save import retrieve_footer

parent = os.path.join(os.getcwd(), 'Measurements')
carpetas = 'Cohen_Coon', 'Cohen_Coon_2', 'PID_0.37'
carpetas = [os.path.join(parent, file) for file in carpetas]

#%% Ver qué onda las mediciones

cual_carpeta = 1
contenidos = os.listdir(carpetas[cual_carpeta])
contenidos_completos = [os.path.join(carpetas[cual_carpeta], file) for file in contenidos]
#%%

#Select file and extract stuff
gen_freq = 10e3 #in Hz
file = contenidos_completos[1]
salto = fst.find_numbers(file)[-2: -1]
samplerate = fst.find_1st_number(retrieve_footer(file)) #in Hz
points_per_gen_period = samplerate / gen_freq

#load data
time, signal, gen = np.loadtxt(file, unpack=True)
gen /= np.mean(gen[gen>4]) # gen now goes between 0 and 1, approx

cada = 200
#integral = np.array([np.trapz(gen[i:i+cada]) for i in range(len(gen)-cada)])
#duty_cycle = np.array([np.mean(gen[i:i+cada]) for i in range(len(gen)-cada)])


#%% For all file

duty_cycles = []
jumps = []
i = 1
for file in contenidos_completos:
    
    #retrive DC jump
    jumps.append(np.array(fst.find_numbers(file)[-2:])) #last two numbers
    
    #load and normalize
    time, signal, gen = np.loadtxt(file, unpack=True)
    gen /= np.mean(gen[gen>4]) # gen now goes between 0 and 1, approx
    
    cada = 500
    #integral = np.array([np.trapz(gen[i:i+cada]) for i in range(len(gen)-cada)])
    this_duty = np.array([np.mean(gen[i:i+cada]) for i in range(len(gen)-cada)])
    duty_cycles.append(this_duty)
    print('Done with {}/{}'.format(i, len(contenidos_completos)))
    i += 1
    
#%%

fig, axarr = plt.subplots(2,2)
for dc, j, ax in zip(duty_cycles, jumps, axarr.flat):
    ax.plot(dc)
    xrange = ax.get_xlim()
    ax.hlines(j/100, *xrange)
    ax.set_title(str(j))
    ax.grid(True)
    
plt.tight_layout()