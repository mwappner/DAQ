# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:55:37 2018

@author: Publico
"""

import nidaqmx as nid
import numpy as np

#%% Juan

duration = 1
samplerate = 70


samples_per_channel = samplerate * duration
t = np.arange(0, duration, 1/samplerate)
med = np.nan

with nid.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev20/ai1")
    task.timing.cfg_samp_clk_timing(samplerate,
                                    samps_per_chan=samples_per_channel)
    med = task.read(number_of_samples_per_channel=samples_per_channel)
    task.wait_until_done()

#%% Freq_Sweep

periods_per_samplerate = 10
frequency = 1.2e3

samplerate_min = 10
samplerate_max = 5e3
samplerate_n = 300

samplerate = np.linspace(samplerate_min,
                         samplerate_max,
                         samplerate_n)

duration = periods_per_samplerate / frequency

#%% Moni_Freq

with nid.Task() as task:
    task.ai_channels.add_ai_freq_voltage_chan("Dev20/ai1",max_val=20000)
    med = task.read()
    task.wait_until_done()


