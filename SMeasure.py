# -*- coding: utf-8 -*-
"""
This script is to make measurements with a National Instruments DAQ.

@author: GrupoFWP
"""

import fwp_analysis as anly
import fwp_lab_instruments as ins
import fwp_save as sav
import matplotlib.pyplot as plt
import os
import nidaqmx as nid
import numpy as np

#%% Only_One_Measure
"""
This script makes a voltage measurement.

It measures at differential mode. Then, it applies Fourier 
Transformation to get the main frequency and its Fourier intensity. It 
only plots the data.

Warnings
--------
Taken from 'github.com/DopplerPavlovTichno/Placa-adquisicion' f35f2c2
Subtly modified by FWP Group
"""

# PARAMETERS

# Main parameters
duration = 1
samplerate = 5e3

# Other parameters
samples_per_channel = samplerate * duration
time = np.arange(0, duration, 1/samplerate)
mode = nid.constants.TerminalConfiguration.NRSE # Terminal configuration
# DIFFERENTIAL, NRSE, RSE, PSEUDODIFFERENTIAL

# ACTIVE CODE

# Make a measurement using NI DAQ.
with nid.Task() as task:
    task.ai_channels.add_ai_voltage_chan(
            "Dev20/ai1", # analog input AI 1
            terminal_config=mode)
    task.timing.cfg_samp_clk_timing(
            samplerate,
            samps_per_chan=int(samples_per_channel))
    data = task.read(
            number_of_samples_per_channel=int(samples_per_channel))
    task.wait_until_done()

# Make Fourier transformation and get main frequency
fourier = np.abs(np.fft.rfft(data)) # Fourier transformation
fourier_frequencies = np.fft.rfftfreq(len(data), d=1./samplerate)
max_frequency = fourier_frequencies[np.argmax(fourier)]

# Plot data
plt.figure()
plt.plot(time, data)
plt.xlabel('Tiempo (s)')
plt.ylabel('Tensión (V)')

# Plot Fourier
plt.figure()
plt.plot(fourier_frequencies, fourier)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Intensidad de Fourier (ua)')
plt.title('{}'.format(max_frequency))

#%% Samplerate_Sweep
"""This scripts makes a sampling rate sweep for a fixed signal.

This code saves all measurements. It also applies Fourier Transformation 
to them, to get the main frequency and its Fourier amplitude. Then, it 
saves main frequency and its Fourier amplitude along with samplig rate.
"""

# PARAMETERS

# Main parameters
periods_to_meassure = 100
signal_frequency = 1.2e3

samplerate_min = 100
samplerate_max = 10e3
samplerate_n = 200
mode = nid.constants.TerminalConfiguration.NRSE

name = 'Samplerate_Sweep'

# Other parameters
samplerate = np.linspace(samplerate_min,
                         samplerate_max,
                         samplerate_n)
duration = periods_to_meassure / signal_frequency

folder = os.path.join(os.getcwd(),
                      'Measurements',
                      name)
folder = sav.new_dir(folder)
filename = lambda samplerate : os.path.join(
        folder, 
        '{:.0f}_Hz.txt'.format(samplerate))
header = 'time [s]  data [V]'

# ACTUAL CODE
               
fourier_data = [] # contains sampling rate, main frequency and its power
with nid.Task() as task:
    
    task.ai_channels.add_ai_voltage_chan(
            "Dev20/ai1",
            terminal_config=mode)
    
    for sr in samplerate:
        print('Doing {}Hz'.format(sr))
        
        samples_to_meassure = int(sr * duration)
        task.timing.cfg_samp_clk_timing(
                rate=sr,
                samps_per_chan=samples_to_meassure)
        
        signal = task.read(
                number_of_samples_per_channel=samples_to_meassure)
        task.wait_until_done()
        
        time = np.linspace(0, duration, samples_to_meassure)
        np.savetxt(filename(sr), np.array([time, signal]).T, 
                   header=header)
        
        max_frequency, fourier_peak = anly.main_frequency(signal, sr)
        fourier_data.append((sr, max_frequency, fourier_peak))

sav.savetext(np.array(fourier_data), os.path.join(folder, 'Data.txt'), 
             header=['Sampling rate (Hz)', 
                     'Main frequency (Hz)',
                     'Intensity'])

#%% Frequency_Sweep
"""This script makes a double sweep on frequency and samplerate.

For each frequency, it sweeps on samplerate. It saves all measurements. 
It also applies Fourier Transformation to get main frequency and its 
Fourier amplitude. For each frequency, it saves main frequency and 
Fourier amplitude along with samplerate.
"""

# PARAMETERS

# Main parameters
samplerate_min = 10
samplerate_max = 400e3
samplerate_n = 5#100
mode = nid.constants.TerminalConfiguration.NRSE

signal_frequency_n = 10#500
port = 'ASRL1::INSTR'

name = 'Frequency_Sweep'

# Other parameters
samplerate = np.linspace(samplerate_min,
                         samplerate_max,
                         samplerate_n)

gen = ins.Gen(port=port, nchannels=2)

folder = os.path.join(os.getcwd(),
                      'Measurements',
                      name)
folder = sav.new_dir(folder)
filename = lambda samplerate, signal_frequency : os.path.join(
        folder, 
        'Sr_{:.0f}_Hz_Freq_{:.0f}_Hz.txt'.format(signal_frequency, 
                                                 samplerate))
filename_fourier = lambda samplerate : os.path.join(
        folder,
        'Sr_{:.0f}_Hz_Fourier.txt'.format(samplerate))
header = ['Time [s]', 'Data [V]']
header_fourier = ['Sampling rate (Hz)',
                  'Main frequency (Hz)',
                  'Intensity (u.a.)']

# ACTIVE CODE

with nid.Task() as task:
    
    # Configure channel
    task.ai_channels.add_ai_voltage_chan(
            "Dev20/ai1",
            terminal_config=mode)
        
    for sr in samplerate:
        print('Doing SR {} Hz'.format(sr))
        
        # Make frequency array for a given samplerate
        signal_frequency_min = sr/10
        signal_frequency_max = 5*sr
        signal_frequency = np.linspace(signal_frequency_min,
                                       signal_frequency_max,
                                       signal_frequency_n)
        
        # Configure clock and measurement
        duration = 10/signal_frequency_min
        samples_to_meassure = int(sr * duration)
        task.timing.cfg_samp_clk_timing(
                rate=sr,
                samps_per_chan=samples_to_meassure)

        fourier_data = []
        for freq in signal_frequency:
            
            # Reconfigure function generator
            gen.output(True, channel=1, print_changes=False,
                       frequency=freq, amplitude=2)

            # Measure with DAQ
            signal = task.read(
                    number_of_samples_per_channel=samples_to_meassure)
            task.wait_until_done()

            # Save measurement
            time = np.linspace(0, duration, samples_to_meassure)
            np.savetxt(filename(sr), np.array([time, signal]).T, 
                       header=header)
            
            # Get main frequency and Fourier amplitude
            max_freq, fourier_peak = anly.main_frequency(signal, sr)
            fourier_data.append((sr, max_freq, fourier_peak))

        # Save samplerate, main frequency and Fourier amplitude
        sav.savetext(np.array(fourier_data), 
                     os.path.join(folder, 
                                  filename_fourier), 
                     header=header_fourier)

gen.gen.close()

#%% Moni_Freq

with nid.Task() as task:
    task.ai_channels.add_ai_freq_voltage_chan("Dev20/ai1", 
                                              max_val=20000)
    med = task.read()
    task.wait_until_done()


