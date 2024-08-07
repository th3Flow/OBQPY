# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 01:23:56 2024

@author: mayerflo
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sigP
from matplotlib.ticker import FuncFormatter, MultipleLocator

def dB20(array):
    with np.errstate(divide='ignore'):
        return 20 * np.log10(array)


def fir_calc(sFs, sFpb, sFsb, sApb, sAsb, sN):

    sBands = np.array([0., sFpb/sFs, sFsb/sFs, .5])

    # Remez weight calculation:
    # https://www.dsprelated.com/showcode/209.php

    sErr_pb = (1 - 10**(-sApb/20))/2      # /2 is not part of the article above, but makes it work much better.
    sErr_sb = 10**(-sAsb/20)

    sW_pb = 1/sErr_pb
    sW_sb = 1/sErr_sb
    
    vHFilt = sigP.remez(
            sN+1,                        # Desired number of taps
            sBands,                      # All the band inflection points
            [1,0],                       # Desired gain for each of the bands: 1 in the pass band, 0 in the stop band
            [sW_pb, sW_sb]
            )               
    
    (vw,vH) = sigP.freqz(vHFilt)
    
    sHpbMin = min(np.abs(vH[0:int(sFpb/sFs*2 * len(vH))]))
    sHpbMax = max(np.abs(vH[0:int(sFpb/sFs*2 * len(vH))]))
    sRpb = 1 - (sHpbMax - sHpbMin)
    
    sHsbMax = max(np.abs(vH[int(sFsb/sFs*2 * len(vH)+1):len(vH)]))
    sRsb = sHsbMax

    return (vHFilt, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax)

def firFindOptN(sFs, sFpb, sFsb, sApb, sAsb, sNmin = 1, sNmax = 1000):
    for sN in range(sNmin, sNmax):
        (vHFilt, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = fir_calc(sFs, sFpb, sFsb, sApb, sAsb, sN)
        if -dB20(sRpb) <= sApb and -dB20(sRsb) >= sAsb:
            print("Trying up to N=%d" % sN)
            print("Rpb: %fdB" % (-dB20(sRpb)))
            print("Rsb: %fdB" % -dB20(sRsb))
            return sN
    return None

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