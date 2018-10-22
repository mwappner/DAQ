# -*- coding: utf-8 -*-
"""
This script is to analyse measurements made with NI DAQ.

@author: GrupoFWP
"""

import fwp_analysis as anly
import fwp_plot as fplt
import fwp_save as sav
import matplotlib.pyplot as plt
import numpy as np
import os
import fwp_string as fstr
from scipy.signal import find_peaks

class Struct:
    def __init__(self):
        pass
    
    def __repr__(self):
        return str(vars(self))
#%% Samplerate_Sweep (by Val)
"""This script analyses a samplerate sweep for a fixed signal.

It makes an animation showing voltage vs time graphs for different 
samplerates. It plots main frequency and its Fourier amplitude as a 
function of samplerate.
"""

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
        '{}_Hz.txt'.format(samplerate))

# ACTIVE CODE

all_time = []
all_voltage = []
for sr in samplerate:
    time, voltage = np.loadtxt(filename(sr), unpack=True)
    all_time.append(list(time))
    all_voltage.append(list(voltage))
del voltage, time

fig = plt.figure()
ax = plt.axes()
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Voltaje (V)")
fplt.add_style(fig.number, linewidth=1)
animation = fplt.animation_2D(
        all_time, 
        all_voltage,
        label_function=lambda i : "Frec. muestreo {:.1f} Hz".format(
                samplerate[i]),
        frames_number=30,
        fps=10,
        new_figure=False)

sav.saveanimation(animation,
                  os.path.join(folder, 'Video.gif'))
# This doesn't work and I'm not sure why. It does work saving it as mp4.

samplerate, frequencies, fourier_peak = np.loadtxt(
        os.path.join(folder, 'Data.txt'), 
        unpack=True)

plt.figure()
plt.subplot(211)
plt.plot(samplerate, frequencies, '.')
plt.title("Fourier para frecuencia fija {} Hz".format(signal_frequency))
plt.ylabel('Frecuencia (Hz)')
plt.subplot(212)
plt.plot(samplerate, fourier_peak, '.', label='Amplitud de Fourier')
plt.ylabel("Amplitud (u.a.)")
plt.xlabel("Frecuencia de muestreo (Hz)")
fplt.add_style(linewidth=1)

#%% By Val

# PARAMETERS

# Main parameters
samplerate = 4e3

signal_frequency = 10
signal_pk_amplitude = 2
periods_to_measure = 10
#gen_port = 'ASRL1::INSTR'
#gen_totalchannels = 2

name = 'Multichannel_Settling_Time'
folder = os.path.join(os.getcwd(),
                      'Measurements',
                      name)
filename = lambda nchannels : os.path.join(
        folder, 
        'NChannels_{}.txt'.format(nchannels))

# Other parameters
channels = ["Dev20/ai0",
            "Dev20/ai1",
            "Dev20/ai9",
            "Dev20/ai3",
            "Dev20/ai8",
#            "Dev20/ai5",
#            "Dev20/ai6",
            "Dev20/ai11"]

signal_slope = signal_pk_amplitude * signal_frequency

all_data = {}
for nchannels in range(len(channels)):
    all_data.update({nchannels+1: np.loadtxt(filename(nchannels+1))})
    
plt.figure()
plt.plot(all_data[3][:,0], all_data[3][:,1:])



#%% Sample rate + frequency sweep (by Moni)
name = 'Frequency_Sweep'
folder = os.path.join(os.getcwd(),
                      'Measurements',
                      name)

#get only flies corresponding to raw data, not their Fourirer transform:
rawdata=[os.path.join(
        folder,f) for f in os.listdir(folder) if not f.endswith('Fourier.txt')].sort()
        
maxt= []

sr = [f.split('_')[2] for f in rawdata]
freqgen = [f.split('_')[5] for f in rawdata]

for f in rawdata:
    time,data=np.loadtxt(f, unpack=True)
#    np.append(sr,f.split('_')[2])
#    np.append(freqgen,f.split('_')[5])
    for i in range(2, len(data)-2):
        if data[i]>data[i-1] and data[i]>data[i+1] and data[i]>data[i-2] and data[i]>data[i+2]:
            np.append(maxt, time[i])


deltatau=np.zeros(len(maxt)-1)
for j in range(len(maxt)-1):
    deltatau=maxt[j]-maxt[j+1]

freqm=np.mean(deltatau)/2*np.pi


plt.figure()
plt.plot(freq, frequencies, '.')
plt.title('Frecuencias')
plt.ylabel('Frecuencia a mano (Hz)')
plt.xlabel('Frecuencia gen (Hz)')
plt.grid()
plt.show()

#%% by Marcos

name = 'Frequency_Sweep'
folder = os.path.join(os.getcwd(),
                      'Measurements',
                      name)

# Por Fourier:
fourierfiles = sorted([os.path.join(
        folder,f) for f in os.listdir(folder) if f.endswith('Fourier.txt')])
    
f = fourierfiles[10]
actual_freq, fourier_freq, fourier_power = np.loadtxt(f, unpack=True)
#plt.stem(actual_freq, np.ones_like(actual_freq))
#plt.stem(fourier_freq, fourier_power)

plt.plot(actual_freq, fourier_freq,'-o')
plt.plot(actual_freq, actual_freq)
plt.xlabel('Actual frequency [Hz]')
plt.ylabel('Fourier calculated frequency [Hz]')
plt.legend(('Datos', 'Pendiente 1'))
plt.ylim((0, max(fourier_freq) * 1.1))
plt.title(os.path.basename(f))

#%% Por picos:
rawfiles = sorted([os.path.join(
        folder,f) for f in os.listdir(folder) if not f.endswith('Fourier.txt')])

# divide all file names by 
rawfiles_by_sr = {}
current_sr = None
temp = []

for f in rawfiles:
    this_sr = fstr.find_1st_number(os.path.basename(f))
    if current_sr == this_sr:
        temp.append(f)
    else:
        if len(temp)!=0:
            rawfiles_by_sr[current_sr] = temp
        current_sr = this_sr
        temp = [f]

# sorted list of sampling rates
samplingrates = sorted(list(rawfiles_by_sr.keys()))

cual = 10
archivos = rawfiles_by_sr[samplingrates[cual]]

signal_freqs = {}
#time, data = np.loadtxt(f, unpack=True)
for f in archivos:
    freq = fstr.find_numbers(f)[1]
    signal_freqs[freq] = Struct()
    signal_freqs[freq].file = f
    
    time, data = np.loadtxt(f, unpack=True)
    signal_freqs[freq].duration = time[-1]
    signal_freqs[freq].nperiods = len(find_peaks(data)[0])
    signal_freqs[freq].maybe_freq = len(find_peaks(data)[0])/time[-1]

#to be able to plot with lines, keys must be sorted
actual_freq = sorted(list(signal_freqs.keys()))

plt.plot(actual_freq, [signal_freqs[f].maybe_freq for f in actual_freq], '-o')