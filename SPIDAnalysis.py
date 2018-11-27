# -*- coding: utf-8 -*-
"""
This script analizes data read with our SPID and SLoop scripts.

@author: Vall
"""

from fwp_analysis import linear_fit#, peak_separation
from fwp_plot import add_style
from fwp_save import retrieve_footer#, retrieve_header
#from fwp_string import find_1st_number
import os
import matplotlib.pyplot as plt
import numpy as np

#%% Velocity

############## PARAMETERS ##############

folder = 'Velocity'
wheel_radius = 0.025 # in meters

############## ACTIVE CODE ##############

# Get filenames and footers
folder = os.path.join(os.getcwd(), 'Measurements', folder)
files = [os.path.join(folder, f) 
         for f in os.listdir(folder) if f.startswith('Duty')]
footers = [retrieve_footer(f) for f in files]
#data = [np.loadtxt(f) for f in files]

# Get data hidden on the footer
"""If I wanted them all....
parameter_names = [n for n in footers[0].keys()]
parameters = {}
for k in parameter_names:
    parameters.update(eval("{" + "'{}' : []".format(k) + "}"))
for f in footers:
    for k in parameter_names:
        eval("parameters['{}'].append(f[k])".format(k))
duty_cycle = np.array(parameters.pop('pwm_duty_cycle') for f in footers)
velocity = np.array([peak_separation(d[:,1], 
                                     np.mean(np.diff(d[:,0])))
                     for d in data])
"""

duty_cycle = np.array([f['pwm_duty_cycle'] for f in footers])
velocity = np.array([f['velocity'] for f in footers])

# Finally, plot it :D
plt.plot(duty_cycle, velocity, '.')
linear_fit(duty_cycle * 100, velocity, text_position=(.53, 'down'))
plt.xlabel("Duty Cycle (%)")
plt.ylabel("Velocity (m/s)")
add_style()


#%%

home = os.getcwd()

folders = [os.path.join(home,f) for f in os.listdir(home) 
            if f.startswith('Duty')]

#get only flies corresponding to raw data, not their Fourirer transform:
rawdata=[os.path.join(folder,f) 
        for f in os.listdir(folder) if not f.endswith('Fourier.txt')].sort()
  