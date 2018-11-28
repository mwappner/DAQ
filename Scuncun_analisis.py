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
import fwp_analysis as fan

parent = os.path.join(os.getcwd(), 'Measurements')
carpetas = 'Cohen_Coon', 'Cohen_Coon_2', 'PID_0.37'
carpetas = [os.path.join(parent, file) for file in carpetas]

#%% Ver qué onda las mediciones

cual_carpeta = 0
contenidos = os.listdir(carpetas[cual_carpeta])
contenidos_completos = [os.path.join(carpetas[cual_carpeta], file) for file in contenidos]

def normalize(data, threshold=None):
    
    dataout = data.copy()
    if threshold is None:
        dataout /= np.max(dataout)
    else:
        dataout /= np.mean(data[data>threshold])
    
    return dataout

def calc_vel(time, singal, **kwargs):
    ds = np.diff(signal)
    picos = find_peaks(ds, **kwargs)[0]
    vel = np.diff(picos)
    t = time[picos[:-1]]
    return t, vel.astype(float)

#def fix_vel(vel, threshold):
    
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

#ds = normalize(np.diff(signal), 3)
#signal = normalize(signal, 4)
#picos = find_peaks(ds, height=.6, prominence=.4)[0]
#vel = np.diff(picos)
t, vel = calc_vel(time, signal, height=3, prominence=2)

plt.plot(time, signal)
#plt.plot(time[picos], ds[picos],'x')
plt.plot(t, vel)


#%% For all file

duty_cycles = []
jumps = []
vels = []
vt = []
signals = []

i = 1
cada = 200

for file in contenidos_completos:
    
    #retrieve DC jump
    jumps.append(np.array(fst.find_numbers(file)[-2:])) #last two numbers
    
    #load and normalize
    time, signal, gen = np.loadtxt(file, unpack=True)
    gen = normalize(gen, 4) # gen now goes between 0 and 1, approx
    signals.append(normalize(signal, 4))
    
    #calculate duty cycles
    cada = 500
    #integral = np.array([np.trapz(gen[i:i+cada]) for i in range(len(gen)-cada)])
    this_duty = np.array([np.mean(gen[i:i+cada]) for i in range(len(gen)-cada)])
    duty_cycles.append(this_duty)
    
    #calculate velocity
    t, vel = calc_vel(time, signal, height=3, prominence=2)
    t = t[vel<1000]
    vel = vel[vel<1000]
    try:
        vel = fan.smooth(vel)
    except ValueError:
        pass
    vel = normalize(vel)
    
    vels.append(vel)
    vt.append(t)
    
    print('Done with {}/{}'.format(i, len(contenidos_completos)))
    i += 1
    
#%%
dt = np.diff(time[:2])
maketime = lambda arr, dt: np.linspace(0, dt*len(arr), len(arr))

fig, axarr = plt.subplots(4,2)

stuff_to_loop = duty_cycles, jumps, axarr.flat, vt, vels, signals

for dc, j, ax, t, v, s in zip(*stuff_to_loop):
    
    #plot signal
    ax.plot(maketime(s, dt), s)
    
    #plot dutycycle
    ax.plot(maketime(dc, dt), dc)
    xrange = ax.get_xlim()
    ax.hlines(j/100, *xrange)
    
    #plot vel
    ax.plot(t, v, '-o')
    ax.set_title(str(j))
    ax.grid(True)
    
plt.tight_layout()

#%%
dt = np.diff(time[:2])
maketime = lambda arr, dt: np.linspace(0, dt*len(arr), len(arr))

for s, t, v in zip(signals, vt, vels):
    plt.figure()
    plt.plot(maketime(s, dt), s)
    plt.plot(t, v, '-o')
    
#%% single datapoint

which = 5

dt = np.diff(time[:2])
maketime = lambda arr, dt: np.linspace(0, dt*len(arr), len(arr))

fig, ax = plt.subplots()
   
#plot signal
ax.plot(maketime(signals[which], dt), signals[which])

#plot dutycycle
ax.plot(maketime(duty_cycles[which], dt), duty_cycles[which])
xrange = ax.get_xlim()
ax.hlines(jumps[which]/100, *xrange)

#plot vel
ax.plot(vt[which], vels[which], '-o')
ax.set_title(str(jumps[which]))
ax.grid(True)
    
plt.tight_layout()
