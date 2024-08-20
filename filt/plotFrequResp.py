# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 01:23:56 2024

@author: mayerflo
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sigP
from matplotlib.ticker import FuncFormatter, MultipleLocator

import filt

def dB20(array):
    with np.errstate(divide='ignore'):
        return 20 * np.log10(array)

def plotFrequResp(vw, vH, sFs, sFpb, sFsb, sHpbMin, sHpbMax, sHsbMax):
    fig, ax1 = plt.subplots()
    ax1.set_title("Frequency Response")
    ax1.grid(True)
    
    # Normalize frequencies to the range 0 to π
    sFpb_normalized = sFpb / (sFs / 2) * np.pi
    sFsb_normalized = sFsb / (sFs / 2) * np.pi
    
    ax1.plot(vw, dB20(np.abs(vH)), "r", label='Frequency Response')
    ax1.plot([0, sFpb_normalized], [dB20(sHpbMax), dB20(sHpbMax)], "b--", linewidth=1.0, label='Passband Max')
    ax1.plot([0, sFpb_normalized], [dB20(sHpbMin), dB20(sHpbMin)], "b--", linewidth=1.0, label='Passband Min')
    ax1.plot([sFsb_normalized, np.pi], [dB20(sHsbMax), dB20(sHsbMax)], "b--", linewidth=1.0, label='Stopband Max')
    ax1.set_xlim(0, np.pi)
    ax1.set_ylim(-90, 3)
    
    ax2 = ax1.twiny()
    ax2.set_xlim(0, sFs / 2)
    
    update_ticks(ax1, ax2, sFs)
    
    ax1.set_xlabel('Normalized Frequency (π radians/sample)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.legend()
    
    ax2.set_xlabel('Frequency (Hz)')

    def on_xlims_change(event_ax):
        update_ticks(ax1, ax2, sFs)

    ax1.callbacks.connect('xlim_changed', on_xlims_change)

    plt.show()
    
def update_ticks(ax, ax2, sFs):
    def pi_formatter(x, pos):
        frac = x / np.pi
        if frac == 0:
            return '0'
        elif frac == 1:
            return r'$\pi$'
        elif frac == 0.5:
            return r'$\frac{\pi}{2}$'
        elif frac == 0.25:
            return r'$\frac{\pi}{4}$'
        elif frac % 1 == 0:
            return r'${}\pi$'.format(int(frac))
        else:
            return r'${:.2g}\pi$'.format(frac)
        
    def hz_formatter(x, pos, sFs):
            return f'{x:.2f} Hz'
        
    ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    xlim = ax.get_xlim()
    if (xlim[1] - xlim[0]) < (np.pi / 4):  # Adjust finer scale when zoomed in
        ax.xaxis.set_major_locator(MultipleLocator(np.pi / 16))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(np.pi / 8))
    
    hz_ticks = np.linspace(0, sFs / 2, 9)
    ax2.set_xlim(0, sFs / 2)
    ax2.set_xticks(hz_ticks)
    ax2.set_xticklabels([hz_formatter(tick, pos, sFs) for pos, tick in enumerate(hz_ticks)])
    ax.figure.canvas.draw()