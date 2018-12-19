# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 19:30:03 2018

@author: mfar
"""


import matplotlib.pyplot as plt
import numpy as np
import os
import fwp_save as sav
import codecs

# Start plotting
plt.figure()
font = {'family' : 'sans-serif',
'weight' : 'medium',
'size'   : 15}

plt.rc('font', **font)
# Velocity vs time
    
#%%

foldername = 'Cohen_Coon_Diff'
# Get filenames and footers
folder = os.path.join(os.getcwd(),foldername)
files = [os.path.join(folder, f)
         for f in os.listdir(folder) if f.startswith('Cohen')]
             

f=1000
RC=1/(2*np.pi*f)
header = 'Tiempo (s)\tSeñal (V)\tGenerador (V)\tFiltrado'
        

#%%

for z in range(12):
    num=z
    fileinuse=files[num]
    filecp = codecs.open(fileinuse, encoding = 'iso8859_14')
    #footers=sav.retrieve_footer(filecp)
    t, signal,gen = np.loadtxt(filecp, unpack=True,)
    highpass=np.zeros(len(t))
    highpass[0]=signal[0]
    lowpass=np.zeros(len(t))
    lowpass[0]=signal[0]
    dt= t[3]-t[2]
    a=dt/(RC+dt)
    b=RC/(RC+dt)
    
    for i in range(1,len(t)):
        lowpass[i]=signal[i]*a+lowpass[i-1]*b
        
#    for i in range(1,len(t)):
#        highpass[i]=(t[i]-t[i-1])*b+signal[i-1]*b
#        

    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.set_size_inches(10,7)
    axs[0].plot(t[:1000], signal[0:1000], 'co-', label='signal')
    #axs[0].plot(t[:5000], gen[:5000], 'ko-', label='gen')
    #plt.hlines(pid.setpoint, min(t), max(t), linestyles='dotted')
    axs[0].set_ylabel("Velocidad [u.a.]")
    #axs[0].set_ylim(-0.2, 5.2)
    axs[0].grid(color='silver', linestyle='--', linewidth=1)
    axs[0].set_axisbelow(True) 
    axs[0].legend(loc='upper center')
        
    # Duty cycle vs time
    axs[1].plot(t[:1000],lowpass[:1000], 'ro-', label='pasabajos')
    #plt.plot(t, 100 * dc, 'o-r', label='Signal')
    #plt.hlines(pwm_min_duty_cycle * 100, min(t), max(t),
    #           linestyles='dotted')
    axs[1].set_ylabel("Filtrado")
    axs[1].legend()
    axs[1].grid(color='silver', linestyle='--', linewidth=1)
    axs[1].set_axisbelow(True) 
    axs[1].set_xlabel("Tiempo[s]")
    plt.tight_layout()
    plt.savefig(fileinuse[79:-4]+np.str(z)+np.str(f)+'.pdf')
    
    data = np.array([t, signal,gen,lowpass]).T
    np.savetxt(fileinuse[79:-4]+np.str(z)+np.str(f),data, header=header)
    
    