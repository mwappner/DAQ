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
sample_rate = 70
samples_per_channel = sample_rate*tiempo_medicion
time_vec = np.arange(0, tiempo_medicion, 1/sample_rate)
med = np.nan
with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev6/ai1")
    task.timing.cfg_samp_clk_timing(sample_rate,
                                    samps_per_chan=samples_per_channel)
    med = task.read(number_of_samples_per_channel=samples_per_channel)
    task.wait_until_done()

# %%
fourier = np.abs(np.fft.rfft(med))
fourier_freqs = np.fft.rfftfreq(len(med), d=1./sample_rate)
frecuencia = fourier_freqs[np.argmax(fourier)]
plt.plot(time_vec, med)
plt.xlabel('tiempo (s)')
plt.ylabel('tension (V)')
plt.figure()
plt.plot(fourier_freqs, fourier)
plt.xlabel('frecuencia (Hz)')
plt.ylabel('abs(fourier) (ua)')
plt.title('{}'.format(frecuencia))
# %%
fname = 'alias1.dat'
coment = 'Medimos aliasing. Entrada = senoidal 2Vpp 100Hz'
if not os.path.isfile(fname):
    np.savetxt(fname, np.transpose([time_vec, med]), delimiter = ',', header = 'tiempo (s), tension (V)', footer=coment)
else:
    print('NO GUARDO NADA')
    print('Ya existe guachin')