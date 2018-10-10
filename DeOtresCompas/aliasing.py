# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:14:13 2018
@author: Publico
"""
import nidaqmx
import matplotlib.pyplot as plt
import numpy as np
import os

# %%
tiempo_medicion = 1
sample_rates = np.arange(10, 500, 10)
frecuencia_picos = []
mediciones = []
folder = 'aliasing cuadrada input 50Hz'
for sample_rate in sample_rates:
    samples_per_channel = int(sample_rate*tiempo_medicion)
    time_vec = np.arange(0, tiempo_medicion, 1/sample_rate)
    med = np.nan
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("Dev6/ai1")
        task.timing.cfg_samp_clk_timing(sample_rate,
                                        samps_per_chan=samples_per_channel)
        med = task.read(number_of_samples_per_channel=samples_per_channel)
        task.wait_until_done()
    mediciones.append(med)
    fourier = np.abs(np.fft.rfft(med))
    fourier_freqs = np.fft.rfftfreq(len(med), d=1./sample_rate)
    frecuencia = fourier_freqs[np.argmax(fourier)]
    frecuencia_picos.append(frecuencia)
    
    fig, ax = plt.subplots(2)
    ax[0].plot(time_vec, med)
    ax[0].set_xlabel('tiempo (s)')
    ax[0].set_ylabel('tension (V)')
    ax[1].plot(fourier_freqs, fourier / max(fourier), '.-')
    ax[1].set_xlabel('frecuencia (Hz)')
    ax[1].set_ylabel('abs(fourier) (ua)')
    ax[1].set_xlim(0, 251)
    fig.tight_layout()
    fig.savefig('{}/{}Hz.png'.format(folder, sample_rate))
    
    fname = '{}/freqadq{}Hz.dat'.format(folder, sample_rate)
    coment = 'Medimos aliasing. Entrada = cuadrada 2Vpp 50Hz. Variando frecuencia de sampleo de a 10 Hz entre 10 Hz y 500 Hz.'
    if not os.path.isfile(fname):
        np.savetxt(fname, np.transpose([time_vec, med]), delimiter = ',', header = 'tiempo (s), tension (V)', footer=coment)
    else:
        print('NO GUARDO NADA')
        print('Ya existe guachin')
# %%
#plt.figure()
#plt.plot(sample_rates, frecuencia_picos, '.-')
#plt.xlabel('frecuencia de sampleo (Hz)')
#plt.ylabel('frecuencia pico de fft (Hz)')
#plt.savefig('aliasing.png')