# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 21:44:44 2018

@author: Usuario
"""

import nidaqmx as nid
import numpy as np

#%%

duration = 1
samplerate = 70


samples_per_channel = samplerate * duration
t = np.arange(0, duration, 1/samplerate)
med = np.nan

#%%

with nid.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev6/ai1")
    task.timing.cfg_samp_clk_timing(samplerate,
                                    samps_per_chan=samples_per_channel)
    med = task.read(number_of_samples_per_channel=samples_per_channel)
    task.wait_until_done()
    
#%%

task = nid.Task()
