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

#%% PID All Runs

#chopper_sections = 100
#wheel_radius = 2.5 # in cm
#circunference = 2 * np.pi * wheel_radius
#def virtual_to_real(photogate_frequency, samplerate):
#    """Calculates velocity from frequency of photogate expressed on a.u."""
#    # This function is only used to calculate velocity at the end
#    dt = 1 / samplerate
#    photogate_frequency = photogate_frequency / dt
#    velocity = circunference * photogate_frequency / chopper_sections
#    return velocity

foldernames = ['PID_15_setpoint', 'PID_1.5_setpoint',
              'PID_1_setpoint', 'PID_4_setpoint']

for foldername in foldernames:#['PID_1_setpoint']:

    # Get filenames and footers
    folder = os.path.join(os.getcwd(), 'Measurements',foldername)
    files = [os.path.join(folder, f)
             for f in os.listdir(folder) if f.startswith('Log')]
    saveplot_mask = '{:.0f}_setpoint_{:.2f}_kp_{:.2f}_ki_{:.2f}_kd_{:.2f}.pdf'
    saveplot_filename = sav.savefile_helper(foldername+' figs', saveplot_mask)
                 
    for z in range(len(files)):#range(1):
        num=z
        fileinuse=files[num]
        footers=sav.retrieve_footer(fileinuse)
        t, v, dc, pantes, iantes, dantes = np.loadtxt(fileinuse, unpack=True)
        i = np.array(iantes) * footers['ki'] / dc
        p = np.array(pantes) * footers['kp'] / dc
        d = np.array(dantes) * footers['kd'] / dc
        sp = footers['setpoint']
#        sp = virtual_to_real(footers['setpoint'], footers['samplerate'])
        #t=np.linspace(0, len(v)*footers['dt'], len(v))
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
              
        # Start plotting
        plt.figure()
        font = {'family' : 'sans-serif',
        'weight' : 'medium',
        'size'   : 13}
        
        plt.rc('font', **font)
        # Velocity vs time
        
        fig, axs = plt.subplots(2, 1, sharex=True)
        fig.subplots_adjust(hspace=0)
        fig.set_size_inches(10,7)
        axs[0].hlines(sp,0,t[len(t)-1],linestyles='-', linewidth=3, label='Setpoint')
        axs[0].plot(t, v, 'co-', label='V')
        #plt.hlines(pid.setpoint, min(t), max(t), linestyles='dotted')
        axs[0].set_ylabel("Velocidad [cm/s]")
        #axs[0].set_ylim(-0.2, 5.2)
        axs[0].grid(color='silver', linestyle='--', linewidth=1)
        axs[0].set_axisbelow(True) 
        axs[0].legend(loc='upper center')
        ax2 = axs[0].twinx() 
        ax2.plot(t, dc, 'mo-', label='DC')
        #ax2.grid(color='silver', linestyle='--', linewidth=2)
        #plt.hlines(pid.setpoint, min(t), max(t), linestyles='dotted')
        ax2.set_ylabel("Duty cycle [%]")
        ax2.set_axisbelow(True) 
        ax2.legend(loc='upper left')
        
        # Duty cycle vs time
        axs[1].plot(t,p, 'ro-', label='P')
        axs[1].plot(t,i, 'go-', label='I')
        axs[1].plot(t,d, 'bo-', label='D')
        #plt.plot(t, 100 * dc, 'o-r', label='Signal')
        #plt.hlines(pwm_min_duty_cycle * 100, min(t), max(t),
        #           linestyles='dotted')
        axs[1].set_ylabel("Términos de respuesta del PID [u.a.]")
        axs[1].legend()
        axs[1].grid(color='silver', linestyle='--', linewidth=1)
        axs[1].set_axisbelow(True) 
        axs[1].set_xlabel("Tiempo[s]")
        plt.tight_layout()
#        sav.saveplot()
#        plt.savefig(np.str(num)+'setpoint='+ np.str(footers['setpoint']) +'kp=' + np.str(footers['kp'])+'+ki=' + np.str(footers['ki']) +'+kd=' + np.str(footers['kd'])+'.pdf')
        sav.saveplot(saveplot_filename(num, sp, footers['kp'], 
                                       footers['ki'], footers['kd']))
        
        #plt.savefig(footers['pid']+'.pdf')
        #plt.savefig(np.str(asd)+'.pdf')
        
        ## PID parameters vs time
        #plt.subplot(3,1,3)
        #plot_styles = ['-o',':o', '--x']
        #for x, s in zip([i, p, d], plot_styles):
        #    plt.plot(t, x * 100, s)
        #plt.legend(['I term', 'P term', 'D term'])
        #plt.xlabel("Time (s)")
        #plt.ylabel("PID Parameter (%)")
        # Show plot
#        add_style()
#        plt.show()
#        mng = plt.get_current_fig_manager()
        #mng.window.showMaximized()
        plt.close('all')