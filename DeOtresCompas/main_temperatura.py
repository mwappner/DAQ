# -*- coding: utf-8 -*-
"""
Taken from https://github.com/fotonicaOrg/daq commit 387c7c3
"""

import sys
if "daq" not in sys.modules:
    import daq
else:
    import importlib.reload
    importlib.reload(daq)

import nidaqmx
import nidaqmx.stream_writers
import numpy as np
from matplotlib import pyplot as plt
import time

CAL = 100

ai_channels = ('Dev1/ai0')
co_channels = ('Dev1/ctr0')
pwm_freq = 100
pwm_duty_cycle = 0.5

voltage_range = ([-10,10])
n_samples = 100
freq = 100000

with nidaqmx.Task() as task_ai, nidaqmx.Task() as task_co:
    
    daq.configure_ai(
            task_ai,
            physical_channels = ai_channels,
            voltage_range = voltage_range,
            terminal_configuration = nidaqmx.constants.TerminalConfiguration.RSE
            )
    
    chan_co = daq.configure_pwm(
            task_co,
            physical_channels = co_channels,
            frequency = pwm_freq,
            duty_cycle = pwm_duty_cycle
            )
    
    task_co.timing.cfg_implicit_timing(sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS)
    
    stream_co = nidaqmx.stream_writers.CounterWriter(task_co.out_stream)
    task_co.start()
    
    (data, real_freq) = daq.continuous_acquire(
                task = task_ai,
                n_samples = n_samples,
                sample_frequency = freq,
                task_co = task_co,
                stream_co = stream_co,
                chan_co = chan_co[0]
                )

time = np.arange(data.size) / real_freq
#
#plt.plot(time, data[0,:])
#plt.xlabel('Tiempo (s)')
#plt.ylabel('Tensión registrada (V)')
#plt.grid()







