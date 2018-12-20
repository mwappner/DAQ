# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:55:59 2018

@author: Marcos
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from fwp_analysis import smooth

base = os.path.join(os.getcwd(), 'Measurements')
original = 'Cohen_Coon_Diff'
original = os.path.join(base, original)
filtrados = 'filtrado750', 'filtrados500', 'filtrados1k'
filtrados = [os.path.join(base, c) for c in filtrados]


def calc_vel(time, signal, **kwargs):
    ds = np.diff(signal)
    picos = find_peaks(ds, **kwargs)[0]
    period = np.diff(picos).astype(float)
    t = time[picos[:-1]]
    return t, 1/period


#%%

cual_carpeta = 0

carpeta = filtrados[cual_carpeta]
contenido = os.listdir(carpeta)
contenido_completo = [os.path.join(carpeta, f) for f in contenido]

originales = os.listdir(original)
originales_completo = [os.path.join(carpeta, f) for f in originales]

#%%

#este = 1
for este in (0,1,3):
    cada = 60
    time, data, gen, filt = np.loadtxt(contenido_completo[este], unpack=True)
    this_duty = np.array([np.mean(gen[i:i+cada]) for i in range(len(gen)-cada)])
    this_duty /= gen.max()
    
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.set_size_inches([13, 10])
    ax1.plot(time, data/data.max())
    ax1.plot(time[:-cada], this_duty, 'k')
    
    filt /= filt.max()
    ax2.plot(time, filt)
    sf = smooth(filt, 11)
    ax2.plot(time, sf,'--o')
    
    ds = np.diff(sf)
    ds /= ds.max()
    picos = find_peaks(ds, height=.65)[0]
    mean_distance = np.mean(np.diff(picos))
    picos = find_peaks(ds, height=.65, distance=mean_distance/4 )[0]
    t = time[picos]
    ax2.plot(time[:-1], ds)
    #picos2 = find_peaks(filt, height=3, prominence=.3)[0]
    #t2 = time[picos2]
    ax2.plot(t, ds[picos],'ro')
    #ax2.plot(t2, filt[picos2],'go')
    
    periods = np.diff(picos).astype(float)
    frequencies = 1/periods
    frequencies /= frequencies.max()
    ax1.plot(time[picos[:-1]], frequencies)

#%%

cada = 60
lims = [[500000, 750000], [100000, 195000], ['por los indices'], [180000, 260000]]
for este in (1, ): #saco el 3 por ahora

    #cargo y normalizo datos
    time, data, gen, filt = np.loadtxt(contenido_completo[este], unpack=True)
    this_duty = np.array([np.mean(gen[i:i+cada]) for i in range(len(gen)-cada)])
    this_duty /= gen.max()

    filt /= filt.max()
    sf = smooth(filt, 11)
    
    #calculo picos
    ds = np.diff(sf)
    ds /= ds.max()
    picos = find_peaks(ds, height=.65)[0]
    mean_distance = np.mean(np.diff(picos))
    picos = find_peaks(ds, height=.65, distance=mean_distance/4 )[0]

    #calculo frecuencia
    periods = np.diff(picos).astype(float)
    frequencies = 1/periods
    frequencies /= frequencies.max()

    #ploteo
    f, ax = plt.subplots()
    f.set_size_inches([7.63, 2.43])
    color1 = 'tab:orange'
    ax.plot(time[:-cada], this_duty*100, 
            color=color1, label='Duty cycle', linewidth=3)
    ax.set_ylim(0,80)
    ax.set_xlim(time[lims[este]])
    ax.hlines((30, 50), *ax.get_xlim())

    ax.set_xlabel('Tiempo [s]', fontsize=15)
    ax.set_ylabel('Duty Cycle [%]', color=color1, fontsize=15)
    ax.tick_params('y', colors=color1)
    ax.tick_params('both', labelsize=15)

    ax.grid(True)
#    ax.legend()

    ax2 = ax.twinx()
    color2 = 'tab:green'

#    ax2.plot(time[picos[:-1]], frequencies, 'g-o', label='Velocidad')
    ax2.plot(time[picos[:-1]], smooth(frequencies, 7), 
             '-*', color=color2, label='Velocidad', linewidth=3)
#    ax2.plot(frequencies, 'g-o', label='Velocidad')

    ax2.set_ylabel('Velocidad  [u.a.]', color=color2, fontsize=15)
    ax2.tick_params('y', colors=color2, labelsize=15)
    ax2.set_ylim(0,.55)
   
#    ax.set_title(contenido[este])
#    ax2.legend()
    f.tight_layout()

#%%

desde = 200, 65, 0, 120
datos = []
for este in (0,1,3): #saco el 3 por ahora

    #cargo y normalizo datos
    _, data, _, filt = np.loadtxt(contenido_completo[este], unpack=True)
    filt /= filt.max()
    sf = smooth(filt, 11)
    
    #calculo picos
    ds = np.diff(sf)
    ds /= ds.max()
    picos = find_peaks(ds, height=.65)[0]
    mean_distance = np.mean(np.diff(picos))
    picos = find_peaks(ds, height=.65, distance=mean_distance/4 )[0]

    #calculo frecuencia
    periods = np.diff(picos).astype(float) / 200e3
#    frequencies = 1/periods
    datos.append(periods[desde[este]:])
    print('Lesto el pollo!')

#%% Generate random data

data = np.concatenate(datos)
#plt.hist(data,20, range=(None,.0009))
#
#data = np.random.normal(size=1000)
hist, bins = np.histogram(data, bins=20, range=(.0065, data.max()))

bin_centers = bins[:-1] + np.diff(bins)/2
cdf = np.cumsum(hist)
cdf = cdf / cdf[-1]

def cut_on_duration(data, lim):
    cs = np.cumsum(data)
    try:
        i = np.where(cs>lim)[0][0]
    except IndexError:
        return data
    else:
        return data[:i]
    
def get_periods(size, lim):
    #devuelve períodos aleatorios con duración total lim segundos
    values = np.random.rand(size)
    value_bins = np.searchsorted(cdf, values)
    periods = bin_centers[value_bins]
    return cut_on_duration(periods, lim)
#plt.subplot(121)
#plt.hist(data, bins=20, range=(.0065, data.max()))
#plt.subplot(122)
#plt.hist(random_from_hist, bins=20)

wheel_radius = 2.5 # in cm
chopper_sections = 100 # amount of black spaces on photogate's chopper
circumference = 2 * np.pi * wheel_radius

calculate_velocity = lambda periods: circumference / (chopper_sections * periods)

#%%
number_of_repetitions = 100
number_of_steps = 20
increase_vel = []
increase_dv = []
increase_all = []

dt = data[0][1,0]
step_size = len(data[0]) / number_of_steps
durations_points = [int((k+1)*step_size) for k in range(number_of_steps)] #in points
durations = np.array(durations_points) * dt #in seconds

for m, d in enumerate(data):
    
    this_vel = np.zeros(number_of_steps) # mean values for all repetitions
    this_dv = np.zeros(number_of_steps) # std values for all repetitions
    all_vels = [] # all values with different starting points

    
    for k, duration in enumerate(durations_points): 
        
        # Calculate velocity for an increasing length of time:
        v = []
        
        # Start from a random point in the dataset
        if k < number_of_steps-2:
            for _ in range(number_of_repetitions):
                
                start = np.random.randint(len(d)-duration)
                stop = start + duration
                
                try:
                    v.append(calculate_velocity_error(d[start:stop, :])[0])
                except ValueError:
                    pass
            
        # Except when doing the complete run
        else:
            v = [calculate_velocity_error(d)[0]]
        
        # Store values
        this_dv[k] = np.std(v)
        this_vel[k] = np.mean(v)
        all_vels.append(v)
            
    increase_vel.append(this_vel)
    increase_dv.append(this_dv)
    increase_all.append(all_vels)
    
    print('Done doing {}/{}'.format(m+1, len(data)))