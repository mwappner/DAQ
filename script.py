# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:55:37 2018

@author: Publico
"""

import nidaqmx as nid
import numpy as np
import matplotlib.pyplot as plt
import fwp_save as sav
import os
import fwp_lab_instruments as ins

def main_freq(data, sr):
    #Calculates main frequency in signal and its power
    fourier = np.abs(np.fft.rfft(data))
    fourier_freqs = np.fft.rfftfreq(len(data), d=1./sr)
    max_freq = fourier_freqs[np.argmax(fourier)]
    pot = max(fourier)
    return sr, max_freq, pot

#%% Juan

duration = 1
samplerate = 5e3

samples_per_channel = samplerate * duration
t = np.arange(0, duration, 1/samplerate)
#med = np.nan
mode = nid.constants.TerminalConfiguration.NRSE # Sets terminal configuration
# DIFFERENTIAL, NRSE, RSE, PSEUDODIFFERENTIAL
t = np.arange(0, duration, 1/samplerate)

with nid.Task() as task:
    task.ai_channels.add_ai_voltage_chan(
            "Dev20/ai1",
            terminal_config=mode)
    task.timing.cfg_samp_clk_timing(samplerate,
                                    samps_per_chan=int(samples_per_channel))
    med = task.read(number_of_samples_per_channel=int(samples_per_channel))
    task.wait_until_done()

plt.plot(med)

fourier = np.abs(np.fft.rfft(med))
fourier_freqs = np.fft.rfftfreq(len(med), d=1./samplerate)
frecuencia = fourier_freqs[np.argmax(fourier)]
plt.plot(t, med)
plt.xlabel('tiempo (s)')
plt.ylabel('tension (V)')
plt.figure()
plt.plot(fourier_freqs, fourier)
plt.xlabel('frecuencia (Hz)')
plt.ylabel('abs(fourier) (ua)')
plt.title('{}'.format(frecuencia))

#%% Samplerate_Sweep

periods_to_meassure = 100
signal_frequency = 1.2e3

samplerate_min = 100
samplerate_max = 10e3
samplerate_n = 200
mode = nid.constants.TerminalConfiguration.NRSE

name = 'Samplerate_Sweep'

samplerate = np.linspace(samplerate_min,
                         samplerate_max,
                         samplerate_n)


duration = periods_to_meassure / signal_frequency
folder = os.path.join(os.getcwd(),
                      'Measurements',
                      name)
folder = sav.new_dir(folder)
filename = lambda samplerate : os.path.join(folder, 
                                            '{:.0f}_Hz.txt'.format(samplerate))

header = 'time [s]  data [V]'

fourier_data = [] # contains sampling rate, main frequency and its power


with nid.Task() as task:
    task.ai_channels.add_ai_voltage_chan(
            "Dev20/ai1",
            terminal_config=mode)
    
    for sr in samplerate:
        print('Doing {}Hz'.format(sr))
        
        samples_to_meassure = int(sr * duration)
        task.timing.cfg_samp_clk_timing(rate=sr,
                                        samps_per_chan=samples_to_meassure)
        med = task.read(number_of_samples_per_channel=samples_to_meassure)
        task.wait_until_done()
#        len(med )
        t = np.linspace(0, duration, samples_to_meassure)

        np.savetxt(filename(sr), np.array([t,med]).T, header=header)
        fourier_data.append(main_freq(med, sr))

sav.savetext(np.array(fourier_data), os.path.join(folder, 'Data.txt'), 
             header=['Sampling rate (Hz)', 
                     'Main frequency (Hz)',
                     'Intensity'])

#%% Frequency_Sweep

# PARAMETERS

signal_frequency = 1.2e3

samplerate_min = 10
samplerate_max = 400e3
samplerate_n = 5#100
mode = nid.constants.TerminalConfiguration.NRSE

signal_frequency_n = 10#500

name = 'Frequency_Sweep'

port = 'ASRL1::INSTR'

# OTHER PARAMETERS

gen = ins.Gen(port=port, nchannels=2)

samplerate = np.linspace(samplerate_min,
                         samplerate_max,
                         samplerate_n)

folder = os.path.join(os.getcwd(),
                      'Measurements',
                      name)
folder = sav.new_dir(folder)
filename = lambda samplerate, signal_frequency : os.path.join(
        folder, 
        'Sr_{:.0f}_Hz_Freq_{:.0f}_Hz.txt'.format(signal_frequency, samplerate))
filename_fourier = lambda samplerate : os.path.join(
        folder,
        'Sr_{:.0f}_Hz_Fourier.txt'.format(samplerate))
header = 'time [s]\tdata [V]'
header_fourier = 'Sampling rate (Hz)\tMain frequency (Hz)\tIntensity'

fourier_data = [] # contains sampling rate, main frequency and its power


with nid.Task() as task:
    
    task.ai_channels.add_ai_voltage_chan(
            "Dev20/ai1",
            terminal_config=mode)
        
    for sr in samplerate:
        
        print('Doing SR {}Hz'.format(sr))
        
        signal_frequency_min = sr/10
        signal_frequency_max = 5*sr
        signal_frequency = np.linspace(signal_frequency_min,
                                       signal_frequency_max,
                                       signal_frequency_n)
        
        duration = 10/signal_frequency_min

        samples_to_meassure = int(sr * duration)
        task.timing.cfg_samp_clk_timing(rate=sr,
                                        samps_per_chan=samples_to_meassure)
        
        for freq in signal_frequency:
            
            gen.output(True, channel=1, print_changes=False,
                       frequency=freq, amplitude=2)

            med = task.read(number_of_samples_per_channel=samples_to_meassure)
            task.wait_until_done()

            t = np.linspace(0, duration, samples_to_meassure)
    
            np.savetxt(filename(sr), np.array([t,med]).T, header=header)
            fourier_data.append(main_freq(med, sr))

        np.savetext(np.array(fourier_data), 
                    os.path.join(folder, 
                                 filename_fourier), 
                    header=header_fourier)
gen.gen.close()

#%% Moni_Freq

with nid.Task() as task:
    task.ai_channels.add_ai_freq_voltage_chan("Dev20/ai1",max_val=20000)
    med = task.read()
    task.wait_until_done()


