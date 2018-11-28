# -*- coding: utf-8 -*-
"""
This script analizes data read with our SPID and SLoop scripts.

@author: Vall
"""

from fwp_analysis import linear_fit, peak_separation
from fwp_plot import add_style
import fwp_save as sav
import os
import matplotlib.pyplot as plt
import numpy as np

#%% Velocity

############## PARAMETERS ##############

folder = 'Velocity'
wheel_radius = 0.025 # in meters
chopper_sections = 100 # amount of black spaces on photogate's chopper

############## ACTIVE CODE ##############

# Get filenames and footers
folder = os.path.join(os.getcwd(), 'Measurements', folder)
files = [os.path.join(folder, f) 
         for f in os.listdir(folder) if f.startswith('Duty')]
footers = [sav.retrieve_footer(f) for f in files]
data = [np.loadtxt(f) for f in files]

# Get data hidden on the footer
"""If I wanted them all...
parameter_names = [n for n in footers[0].keys()]
parameters = {}
for k in parameter_names:
    parameters.update(eval("{" + "'{}' : []".format(k) + "}"))
for f in footers:
    for k in parameter_names:
        eval("parameters['{}'].append(f[k])".format(k))
"""
duty_cycle = np.array([f['pwm_duty_cycle'] for f in footers])*100

# Now calculate velocity and its error
circunference = 2 * np.pi * wheel_radius
def calculate_velocity_error(data):
    time = data[:,0]
    read_data = data[:,1]
    photogate_derivative = np.diff(read_data)
    one_section_period, error = peak_separation(
            photogate_derivative, 
            np.mean(np.diff(time)), 
            prominence=1, 
            height=2,
            return_error=True)
    velocity = circunference / (chopper_sections * one_section_period)
    error = circunference*error/(chopper_sections*one_section_period**2)
    return velocity, error
velocity = []
velocity_error = []
for d in data:
    v, dv = calculate_velocity_error(d)
    velocity.append(v*1000)
    velocity_error.append(dv*1000)
velocity = np.array(velocity)
velocity_error = np.array(velocity_error)

# Finally, plot it :D
#plt.plot(duty_cycle, velocity, '.')
m, b, r = linear_fit(duty_cycle, velocity, velocity_error, 
                     text_position=(.4, 'down'), mb_error_digits=(1, 1),
                     mb_units=(r'$\frac{mm}{s}$', r'$\frac{mm}{s}$'))
plt.xlabel("Duty Cycle (%)")
plt.ylabel("Velocity (mm/s)")
add_style(markersize=12, fontsize=16)
sav.saveplot(os.path.join(folder, 'Velocity.pdf'))