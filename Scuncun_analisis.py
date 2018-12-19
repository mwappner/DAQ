# -*- coding: utf-8 -*-
"""
Analisis de Cohen-Coon

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks
import os
import fwp_string as fst
from fwp_save import retrieve_footer
import fwp_analysis as fan
import fwp_save as sav

parent = os.path.join(os.getcwd(), 'Measurements')
carpetas = 'Cohen_Coon', 'Cohen_Coon_2', 'Cohen_Coon_Diff'
carpetas = [os.path.join(parent, file) for file in carpetas]

#%% Ver qué onda las mediciones

cual_carpeta = 2
contenidos = os.listdir(carpetas[cual_carpeta])
contenidos_completos = [os.path.join(carpetas[cual_carpeta], file) for file in contenidos]
footers = [sav.retrieve_footer(f) for f in contenidos_completos]

def normalize(data, threshold=None):
    
    dataout = data.copy()
    if threshold is None:
        dataout /= np.max(dataout)
    else:
        dataout /= np.mean(data[data>threshold])
    
    return dataout

def calc_vel(time, singal, **kwargs):
    ds = np.diff(signal)
    picos = find_peaks(ds, **kwargs)[0]
    period = np.diff(picos).astype(float)
    t = time[picos[:-1]]
    return t, 1/period

#def fix_vel(vel, threshold):
    
#%%

#which file?
cual = 1

#Select file and extract stuff
gen_freq = 10e3 #in Hz
file = contenidos_completos[cual]
salto = fst.find_numbers(file)[0:1]
samplerate = fst.find_1st_number(retrieve_footer(file)) #in Hz
points_per_gen_period = samplerate / gen_freq

#load data
time, signal, gen = np.loadtxt(file, unpack=True)
signal = np.abs(signal)
gen /= np.mean(gen[gen>4]) # gen now goes between 0 and 1, approx

cada = 200
#integral = np.array([np.trapz(gen[i:i+cada]) for i in range(len(gen)-cada)])
#duty_cycle = np.array([np.mean(gen[i:i+cada]) for i in range(len(gen)-cada)])

ds = normalize(np.diff(signal), 3)
signal = normalize(signal, 4)
#picos = find_peaks(ds, height=.6, prominence=.4)[0]
#vel = np.diff(picos)
t, vel = calc_vel(time, signal, height=.6, prominence=.4)

plt.plot(time, signal)
#plt.plot(time[picos], ds[picos],'o')
plt.plot(t, vel)


#%% For all file

duty_cycles = []
jumps = []
vels = []
vt = []
signals = []

i = 1
cada = 200

for file in contenidos_completos:
    
    #retrieve DC jump
    jumps.append(np.array(fst.find_numbers(file)[:2])) #last two numbers
    
    #load and normalize
    time, signal, gen = np.loadtxt(file, unpack=True)
    signal = np.abs(signal)
    gen = normalize(gen, 4) # gen now goes between 0 and 1, approx
    signals.append(normalize(signal, 4))
    
    #calculate duty cycles
    this_duty = np.array([np.mean(gen[i:i+cada]) for i in range(len(gen)-cada)])
    duty_cycles.append(this_duty)
    
    #calculate velocity
    t, vel = calc_vel(time, signal, height=3, prominence=2)
#    t = t[vel<1000]
#    vel = vel[vel<1000]
    try:
        vel = fan.smooth(vel, window_len=21)
    except ValueError:
        pass
    vel = normalize(vel)
    
    vels.append(vel)
    vt.append(t)
    
    print('Done with {}/{}'.format(i, len(contenidos_completos)))
    i += 1
    
#%%
dt = np.diff(time[:2])
maketime = lambda arr, dt: np.linspace(0, dt*len(arr), len(arr))

fig, axarr = plt.subplots(5,2)

stuff_to_loop = duty_cycles, jumps, axarr.flat, vt, vels, signals

for dc, j, ax, t, v, s in zip(*stuff_to_loop):
    
    #plot signal
    ax.plot(maketime(s, dt), s)
    
    #plot dutycycle
    ax.plot(maketime(dc, dt), dc)
    xrange = ax.get_xlim()
    ax.hlines(j/100, *xrange)
    
    #plot vel
    ax.plot(t, v)
    ax.set_title(str(j))
    ax.grid(True)
    
plt.tight_layout()

#%%
dt = np.diff(time[:2])
maketime = lambda arr, dt: np.linspace(0, dt*len(arr), len(arr))

for s, t, v in zip(signals, vt, vels):
    plt.figure()
    plt.plot(maketime(s, dt), s)
    plt.plot(t, v, '-o')
    
#%% single datapoint


which = 2
jumps[2] = jumps[1]

dt = np.diff(time[:2])
maketime = lambda arr, dt: np.linspace(0, dt*len(arr), len(arr))

fig, ax = plt.subplots()
   
#plot signal
ax.plot(maketime(signals[which], dt), signals[which])

#plot dutycycle
ax.plot(maketime(duty_cycles[which], dt), duty_cycles[which])
xrange = ax.get_xlim()
ax.hlines(jumps[which]/100, *xrange)

#plot vel
ax.plot(vt[which], vels[which], '-o')
ax.set_title(str(jumps[which]))
ax.grid(True)
    
plt.tight_layout()


#%% Triple figu

completa = 2
dc = 6
v = 5

fig = plt.figure(constrained_layout=True)
fig.set_size_inches([10.24,  4.8 ])

gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[-1, -1])

### Complete ###
dt = np.diff(time[:2])
maketime = lambda arr, dt: np.linspace(0, dt*len(arr), len(arr))
 
#plot signal
ax1.plot(maketime(signals[which], dt), signals[which], linewidth=3, label='Señal PG')

#plot dutycycle
ax1.plot(maketime(duty_cycles[which], dt), duty_cycles[which], linewidth=2, label='Duty cycle')
xrange = ax.get_xlim()
ax1.hlines(jumps[which]/100, *xrange)

#plot vel
ax1.plot(vt[which], vels[which], '-o', linewidth=3, label='Velocidad')
ax1.grid(True)
ax1.set_xlim((0, 3))

#Format
ax1.set_xlabel('Tiempo [s]', fontsize=15)
ax1.set_ylabel('Varios [u.arb.]', fontsize=15)
ax1.legend(fontsize=15)
ax1.set_title('A: Medición completa', fontsize=20)
ax1.tick_params(labelsize=15)

### Velocidad ###

vel = fan.smooth(vels[v], 11)
color = ax1.lines[2].get_color()
ax2.plot(vt[v], vel, '-o', linewidth=3, label='Velocidad', color=color)
ax2.grid(True)
ax2.set_xlim((0, 3))
ax2.set_title('B: Buena curva de vel.', fontsize=20)
ax2.set_xlabel('Tiempo [s]', fontsize=15)
ax2.set_ylabel('Velocidad [u.arb.]', fontsize=15)
ax2.tick_params(labelsize=15)

### Duty Cycle ###

color = ax1.lines[1].get_color()

ax3.plot(maketime(duty_cycles[dc], dt), duty_cycles[dc] * 100, linewidth=2, 
         label='Duty cycle', color=color)
xrange = ax.get_xlim()
ax3.hlines(jumps[dc], *xrange)
ax3.set_xlim((0, 3))
ax3.set_title('C: Buena curva de d.c.', fontsize=20)
ax3.grid(True)
ax3.set_xlabel('Tiempo [s]', fontsize=15)
ax3.set_ylabel('Duty Cycle [%]', fontsize=15)
ax3.tick_params(labelsize=15)

fig.savefig('cuncun_feo.pdf')


#%% Histogramas de  duración de los períodos:

#Select file and extract stuff
for k, (file, name) in enumerate(zip(contenidos_completos, contenidos)):
    gen_freq = 10e3 #in Hz
#    file = contenidos_completos[cual]
    salto = fst.find_numbers(file)[0:2]
    samplerate = fst.find_1st_number(retrieve_footer(file)) #in Hz
    points_per_gen_period = samplerate / gen_freq
    
    #load data
    time, signal, gen = np.loadtxt(file, unpack=True)
    dt = time[1]
    
    # Calculate and store widths (periods)
    peaks = find_peaks(np.diff(signal), prominence=1, height=2)[0]
    peaks = peaks.astype(float) * dt
    w = np.diff(peaks)
    #widths.append(w)
    plt.figure()
    plt.hist(w, 30, range=(0.03, 1))
    plt.title('Duty {} a {}'.format(*salto))
    
    savename = os.path.splitext(name)[0] + '.jpg'
    plt.savefig(os.path.join('Measurements', 'CCfigs', 'Histograms', savename))
    plt.close()
    
    print('Done doing {}/{}'.format(k+1, len(contenidos_completos)))

#%%

#Dos intentos más: 
# usar calculate_velocity por secciones
    
#%% Calular frecuencia con transformade de fourier

for k, (file, name) in enumerate(zip(contenidos_completos, contenidos)):

    salto = fst.find_numbers(file)[0:2]
    time, signal, gen = np.loadtxt(file, unpack=True)

    f, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    f.set_size_inches([9.79, 5.26])
    
    ax1.plot(time, signal)
    ax1.set_xlim((time[0], time[-1]))
    ax1.set_ylabel('Señal [V]')
    ax1.grid(True)
    plt.title('Duty {} a {}'.format(*salto))

    ax2.specgram(signal, Fs=200000)
    ax2.set_xlabel('Tiempo [s]')
    ax2.set_ylabel('Frecuencia [Hz]')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    f.subplots_adjust(hspace=0)
    savename = os.path.splitext(name)[0] + '.jpg'
    f.savefig(os.path.join('Measurements', 'CCfigs', 'Sonograms', savename))
    plt.close(f)
    
    print('Done doing {}/{}'.format(k+1, len(contenidos_completos)))

