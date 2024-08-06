import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import *

def plot_response(fs, w, h, title="Low-pass Filter",label='', ax=None, ylim=-70):
    "Utility function to plot response functions"
    if not ax:
      fig = plt.figure()
      ax = fig.add_subplot(111)
    ax.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)), label=label)
    ax.set_ylim(ylim, 5)
    ax.set_xlim(0, 0.5*fs)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title(title)
    ax.legend()
    return ax

fpass = 370
fstop = 430
astop = 30
apass = 0
fs = 2000
fc = fpass + (fstop-fpass)/2

filtord_kaiser, beta = kaiserord(astop, 2*(fstop-fpass)/fs)
print(filtord_kaiser)
taps = firwin(filtord_kaiser, fc, window=('kaiser', beta), 
              fs=fs)
w, h = freqz(taps, [1], worN=2000)
ax = plot_response(fs, w, h, label='Kaiser', title="Low-pass Filter")

taps = remez(filtord_kaiser, [0, fpass, fstop, 1000], [1, 0], fs = 2000)
w, h = freqz(taps, [1], worN=2000)
plot_response(fs, w, h, label='Equiripple', title="Low-pass Filter", ax=ax)