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
periods_to_measure = 100
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
duration = periods_to_measure / signal_frequency

folder = os.path.join(os.getcwd(),
                      'Measurements',
                      name)
folder = sav.new_dir(folder)
filename = lambda samplerate : os.path.join(
        folder, 
        '{:.0f}_Hz.txt'.format(samplerate))
header = 'Time [s]\tData [V]'

# ACTUAL CODE
               
fourier_data = [] # contains sampling rate, main frequency and its power
with nid.Task() as task:
    
    task.ai_channels.add_ai_voltage_chan(
            "Dev20/ai1",
            terminal_config=mode)
    
    for sr in samplerate:
        print('Doing {}Hz'.format(sr))
        
        samples_to_measure = int(sr * duration)
        task.timing.cfg_samp_clk_timing(
                rate=sr,
                samps_per_chan=samples_to_measure)
        
        signal = task.read(
                number_of_samples_per_channel=samples_to_measure)
        task.wait_until_done()
        
        time = np.linspace(0, duration, samples_to_measure)
        np.savetxt(filename(sr), np.array([time, signal]).T, 
                   header=header)
        
        max_frequency, fourier_peak = anly.main_frequency(signal, sr)
        fourier_data.append((sr, max_frequency, fourier_peak))

sav.savetext(os.path.join(folder, 'Data.txt'), 
             np.array(fourier_data),
             header=['Sampling rate (Hz)', 
                     'Main frequency (Hz)',
                     'Intensity'])

#%% Frequency_Sweep
"""This script makes a double sweep on frequency and samplerate.

For each samplerate, it sweeps on frequency. It saves all measurements. 
It also applies Fourier Transformation to get main frequency and its 
Fourier amplitude. For each samplerate, it saves main frequency and 
Fourier amplitude.
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
        samples_to_measure = int(sr * duration)
        task.timing.cfg_samp_clk_timing(
                rate=sr,
                samps_per_chan=samples_to_measure)

        fourier_data = []
        for freq in signal_frequency:
            
            # Reconfigure function generator
            gen.output(True, channel=1, print_changes=False,
                       frequency=freq, amplitude=2)

            # Measure with DAQ
            signal = task.read(
                    number_of_samples_per_channel=samples_to_measure)
            task.wait_until_done()

            # Save measurement
            time = np.linspace(0, duration, samples_to_measure)
            sav.savetext(filename(sr),
                         np.array([time, signal]).T,
                         header=header)
            
            # Get main frequency and Fourier amplitude
            max_freq, fourier_peak = anly.main_frequency(signal, sr)
            fourier_data.append((sr, max_freq, fourier_peak))

        # Save samplerate, main frequency and Fourier amplitude
        sav.savetext(os.path.join(folder, 
                                  filename_fourier),
                     np.array(fourier_data), 
                     header=header_fourier)
gen.output(False)

gen.gen.close()

#%% Settling_Time
"""This script is designed to measure settling time for fixed conditions.

It generates a low-frequency square wave. And this wave is measured at 
maximum samplerate in order to watch the time it takes for the NI DAQ 
to fixe to minimum and maximum voltage. This script saves time and 
voltage.

Warnings
--------
Must first check I can generate a fwp_lab_instruments square wave.
>>> gen.output(True, waveform='square')
Should also check I can change duty cycle.
>>> gen.output(True, waveform='square75')
Could also ask what voltage the DAQ is used to measure:
>>> with nid.Task() as task:
>>>     task.ai_channels.add_ai_voltage_chan(
>>>             "Dev20/ai1",
>>>             terminal_config=mode)
>>>     print("Vmax = ", task.ai_channels.ai_voltage_chan.ai_max)
>>>     print("Vmin = ", task.ai_channels.ai_voltage_chan.ai_min)
"""

# PARAMETERS

# Main parameters
samplerate = 400e3
mode = nid.constants.TerminalConfiguration.NRSE

signal_frequency = 10#500
periods_to_measure = 50
gen_port = 'ASRL1::INSTR'
gen_totalchannels = 2

name = 'Settling_Time'

# Other parameters
duration = periods_to_measure/signal_frequency
samples_to_measure = int(samplerate * duration)

gen = ins.Gen(port=gen_port, nchannels=gen_totalchannels)

folder = os.path.join(os.getcwd(),
                      'Measurements',
                      name)
folder = sav.new_dir(folder)
header = ['Time [s]', 'Data [V]']

# ACTIVE CODE

gen.output(True, waveform='square', 
           frequency=signal_frequency,
           amplitude=2)
with nid.Task() as task:
    
    # Configure channel
    task.ai_channels.add_ai_voltage_chan(
            "Dev20/ai1",
            terminal_config=mode)
    
    task.timing.cfg_samp_clk_timing(
            rate=sr,
            samps_per_chan=samples_to_measure)

    signal = task.read(
            number_of_samples_per_channel=samples_to_measure)
    task.wait_until_done()
gen.output(False)

# Save measurement
time = np.linspace(0, duration, samples_to_measure)
sav.savetext(filename(sr),
             np.array([time, signal]).T,
             header=header)

gen.gen.close()

#%% Multichannel_Settling_Time
"""This script is designed to measure multichannel settling time.

It generates a low-frequency ramp wave (sawtooth like). And this wave is 
measured at maximum samplerate in order to watch the time it takes for 
the NI DAQ to fix to another channel.

This is repeated on a sweep for different number of channels. For each 
of them, it saves time and voltage.

Warnings
--------
Must first check I can generate a ramp with fwp_lab_instruments.
>>> gen.output(True, 'ramp')
Should also check I can change ramp simmetry.
>>> gen.output(True, 'ramp100')
Is 100% always positive slope or always negative slope?
"""

# PARAMETERS

# Main parameters
samplerate = 400e3
mode = nid.constants.TerminalConfiguration.NRSE

signal_frequency = 10
signal_pk_amplitude = 2
periods_to_measure = 50
gen_port = 'ASRL1::INSTR'
gen_totalchannels = 2

name = 'Multichannel_Settling_Time'

# Other parameters
duration = periods_to_measure/signal_frequency
samples_to_measure = int(samplerate * duration)
channels = ["Dev20/ai0",
            "Dev20/ai1",
            "Dev20/ai2",
            "Dev20/ai3",
            "Dev20/ai4",
            "Dev20/ai5",
            "Dev20/ai6",
            "Dev20/ai7"]

gen = ins.Gen(port=gen_port, nchannels=gen_totalchannels)
signal_slope = signal_pk_amplitude * signal_frequency

folder = os.path.join(os.getcwd(),
                      'Measurements',
                      name)
folder = sav.new_dir(folder)
filename = lambda nchannels : os.path.join(
        folder, 
        'NChannels_{}.txt'.format(nchannels))
header = 'Time [s]\tData [V]'

# ACTIVE CODE

gen.output(True, waveform='ramp100', 
           frequency=signal_frequency,
           amplitude=2)
with nid.Task() as task:

    task.timing.cfg_samp_clk_timing(
            rate=sr,
            samps_per_chan=samples_to_measure)
    
    for nchannels, channel in enumerate(channels):
        
        # Configure channel
        task.ai_channels.add_ai_voltage_chan(
                channel,
                terminal_config=mode)
#                ai_max=2,
#                ai_min=-2)

        # Measure
        signal = task.read(
                number_of_samples_per_channel=samples_to_measure)
        task.wait_until_done()
        
        # Save measurement
        nchannels = nchannels + 1
        print("For {} channels, signal has size {}".format(
                nchannels,
                np.size(signal)))
        time = np.linspace(0, duration, samples_to_measure)
        try:
            data = np.zeros((signal[:,0], signal[0,:]+1))
            data[:,0] = time
            data[:,1:] = signal
        except IndexError:
            data = np.array([time, signal]).T
        np.savetxt(filename(nchannels), data, header=header)

gen.output(False)
gen.gen.close()


#%% Moni_Freq

with nid.Task() as task:
    task.ai_channels.add_ai_freq_voltage_chan("Dev20/ai1", 
                                              max_val=20000)
    med = task.read()
    task.wait_until_done()


