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
from scipy.signal import find_peaks

#%% Velocity

############## PARAMETERS ##############

folder = 'Velocity'
wheel_radius = 2.5 # in cm
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
    velocity.append(v)
    velocity_error.append(dv)
velocity = np.array(velocity)
velocity_error = np.array(velocity_error)

# Finally, plot it :D
#plt.plot(duty_cycle, velocity, '.')
m, b, r = linear_fit(duty_cycle, velocity, velocity_error, 
                     text_position=(.4, 'down'), mb_error_digits=(1, 1),
                     mb_units=(r'$\frac{cm}{s}$', r'$\frac{cm}{s}$'))
plt.xlabel("Duty Cycle (%)")
plt.ylabel("Velocity (cm/s)")
add_style(markersize=12, fontsize=16)
#sav.saveplot(os.path.join(folder, 'Velocity.pdf'))

#%% Revisiting Velocity

# I'll be trying three things: 

#%% 1. How constant is the period algon a single measurement

widths = []
for k, d in enumerate(data):
    dt = d[1,0]
    read_data = d[:,1]
    
    # Calculate and store widths (periods)
    peaks = find_peaks(np.diff(read_data), prominence=1, height=2)[0]
    peaks = peaks.astype(float) * dt
    w = np.diff(peaks)
    widths.append(w)

    # Remove outliers    
    mean = np.mean(w)
    std = np.std(w)
    w = w[w < mean + 4*std]
    
    # Create, plot and format histogram
    f, ax = plt.subplots()
    ax.hist(w) #8 bins
    ax.legend({'Count = {}'.format(len(w))})
    ax.set_title('{}/{}'.format(k+1, len(data)))
    
    ax.set_xlabel('Duration [s]')
    ax.set_ylabel('Count')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax.grid(True)
    
    # Save and close plot
    name = os.path.join('Measurements', 'Velfigs', 'Histograms',
                    'vel{}.png'.format(k+1))
    plt.tight_layout()
    f.savefig(name)
    plt.close(f)
    
##%% Create one histogram to check formating
    
#cual = 27
#this = widths[cual]
#
#f, ax = plt.subplots()
#ax.hist(this)
#ax.legend({'Count = {}'.format(len(this))})
#ax.set_title('{}/{}'.format(cual+1, len(data)))
#
#ax.set_xlabel('Duration [s]')
#ax.set_ylabel('Count')
#ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
#
#plt.tight_layout()

#%% 2. How the velocity calculation stabilizes as duration of 
# measurement is increased.

number_of_steps = 10
increase_vel = []
increase_dv = []

for d in data:
    
    this_vel = np.zeros(number_of_steps)
    this_dv = np.zeros(number_of_steps)
    durations = np.zeros(number_of_steps)

    step_size = len(d) / number_of_steps
    
    for k in range(number_of_steps): 
        
        # Calculate velocity for an increasing length of time:
        duration = int((k+1)*step_size) #in points, not seconds
        dt = data[0][1,0]
        durations[k] = duration * dt
        
        #Start from a random point in the dataset, except when doing the complete run
        if k + 1 == number_of_steps:
            start = 0
        else:
            start = np.random.randint(len(d)-duration)
        stop = start + duration
        
        #Calculate velocity and error
        try:
            v, dv = calculate_velocity_error(d[start:stop, :])
            this_vel[k] = v
            this_dv[k] = dv
        except ValueError:
            this_vel[k] = np.nan
            this_dv[k] = np.nan
            
    increase_vel.append(this_vel)
    increase_dv.append(this_dv)
        
#%% Create all figures and save them

for k, (v, iv, idv) in enumerate(zip(velocity, increase_vel, increase_dv)):

    # Create subplots
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.suptitle('{}/{}'.format(k+1, len(velocity)))
    
    # Plot Durations
    ax1.plot(durations, iv, '-x')
    ax1.hlines(v, durations[0], durations[-1])
    ax1.set_ylabel('Vel. [cm/s]')
    ax1.grid(True)
    
    # Plot Errors
    ax2.plot(durations, idv, '-x')
    ax2.set_ylabel('Err. [cm/s]')
    ax2.set_xlabel('Duration [s]')
    ax2.grid(True)
    
    # Format, save and close
    plt.tight_layout()
    f.subplots_adjust(hspace=0)

    name = os.path.join('Measurements', 'Velfigs', 'Increasing',
                        'vel{}.png'.format(k+1))
    f.savefig(name)
    plt.close(f)
    
##%% Create one figure to ckeck formating
    
#cual = 0
#v = velocity[cual]
#iv = increase_vel[cual]
#idv = increase_dv[cual]
#
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#f.suptitle('{}/{}'.format(k, len(velocity)))
#
##ax1.set_title('Velocity')
#ax1.plot(durations, iv, '-x')
#ax1.hlines(v, durations[0], durations[-1])
#ax1.set_ylabel('Vel. [cm/s]')
#ax1.grid(True)
#
##ax2.set_title('Error')
#ax2.plot(durations, idv, '-x')
#ax2.set_ylabel('Err. [cm/s]')
#ax2.set_xlabel('Duration [s]')
#ax2.grid(True)
#
#plt.tight_layout()
#f.subplots_adjust(hspace=0)
#
#name = os.path.join('Measurements', 'Velfigs', 'vel{}.png'.format(cual))
##f.savefig(name)

#%% 3. For a given duration or interval size, starting at 
# different points in time, see variability

number_of_repetitions = 100
number_of_steps = 20
increase_vel = []
increase_dv = []
increase_all = []

dt = data[0][1,0]
step_size = len(data[0]) / number_of_steps
durations_points = [int((k+1)*step_size) for k in range(number_of_steps)] #in points
durations = np.array(durations_points) * dt #in seconds

for m, d in enumerate(data):
    
    this_vel = np.zeros(number_of_steps) # mean values for all repetitions
    this_dv = np.zeros(number_of_steps) # std values for all repetitions
    all_vels = [] # all values with different starting points

    
    for k, duration in enumerate(durations_points): 
        
        # Calculate velocity for an increasing length of time:
        v = []
        
        # Start from a random point in the dataset
        if k < number_of_steps-2:
            for _ in range(number_of_repetitions):
                
                start = np.random.randint(len(d)-duration)
                stop = start + duration
                
                try:
                    v.append(calculate_velocity_error(d[start:stop, :])[0])
                except ValueError:
                    pass
            
        # Except when doing the complete run
        else:
            v = [calculate_velocity_error(d)[0]]
        
        # Store values
        this_dv[k] = np.std(v)
        this_vel[k] = np.mean(v)
        all_vels.append(v)
            
    increase_vel.append(this_vel)
    increase_dv.append(this_dv)
    increase_all.append(all_vels)
    
    print('Done doing {}/{}'.format(m+1, len(data)))
    
#%% Plots:
    
# 1: All points for a given duration with mean value

for k, (v, means) in enumerate(zip(increase_all, increase_vel)):

    # Create, plot and format histogram
    f, ax = plt.subplots()
    
    for d, vel in zip(durations, v):
        ax.plot([d] * len(vel), vel, '.r')
  
    ax.plot(durations, means, 'x-', label='Mean value')      
    ax.hlines(means[-1], durations[0], durations[-1], label='Final value')
    
    ax.set_title('{}/{}'.format(k+1, len(data)))
    ax.legend()
    
    ax.set_xlabel('Duration [s]')
    ax.set_ylabel('Vel. [cm/s]')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax.grid(True)
    
    # Save and close plot
    plt.tight_layout()
    name = os.path.join('Measurements', 'Velfigs', 'Variability',
                        'All', 'vel{}.png'.format(k+1))
    f.savefig(name)
    plt.close(f)

# 2: Std and deviation of the mean from real value

for k, (v, iv, idv) in enumerate(zip(velocity, increase_vel, increase_dv)):

    # Create subplots
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.suptitle('{}/{}'.format(k+1, len(velocity)))
    
    # Plot diff of mean and real value
    ax1.plot(durations, np.abs(iv - iv[-1]), 'o-')
    ax1.set_ylabel('Err. [cm/s]')
    ax1.grid(True)
    
    # Plot stds
    ax2.plot(durations, idv, 'o-')
    ax2.set_ylabel('Std. [cm/s]')
    ax2.set_xlabel('Duration [s]')
    ax2.grid(True)
    
    # Format, save and close
    plt.tight_layout()
    f.subplots_adjust(hspace=0)

    name = os.path.join('Measurements', 'Velfigs', 'Variability',
                        'Std', 'vel{}.png'.format(k+1))
    f.savefig(name)
    plt.close(f)
    

