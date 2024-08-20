# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 08:13:57 2024

@author: mayerflo
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sigP
from matplotlib.ticker import FuncFormatter, MultipleLocator

def anFiltEqu(vHFilt, sFs, sFpb, sFsb, strType):
    
    (vw, vH) = sigP.freqz(vHFilt)

    # Calculate passband ripple
    # Find pass band ripple
    sHpbMin = min(np.abs(vH[0:int(sFpb/sFs*2 * len(vH))]))
    sHpbMax = max(np.abs(vH[0:int(sFpb/sFs*2 * len(vH))]))
    sRpb = 1 - (sHpbMax - sHpbMin)#20 * np.log10(sHpbMax / sHpbMin)  # Ripple in dB

    # Calculate stopband attenuations
    # Find stop band attenuation
    sHsbMax = max(np.abs(vH[int(sFsb/sFs*2 * len(vH)+1):len(vH)]))
    sRsb = sHsbMax  # Attenuation in dB

    return vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax

def anFiltKaiser(vHFilt, sFs, sFpb, sFsb, strType):
    (vw, vH) = sigP.freqz(vHFilt)

    # Normalize frequencies
    freqs = vw * sFs / (2 * np.pi)

    # Calculate passband ripple
    # Find passband indices using sFpb (passband edge)
    vPassInd = np.where((freqs >= 0) & (freqs <= sFpb))[0]
    sPbResp = np.abs(vH[vPassInd])
    sHpbMin = np.min(sPbResp)
    sHpbMax = np.max(sPbResp)
    sRpb = 20 * np.log10(1 - (sHpbMax - sHpbMin))  # Ripple in dB

    # Calculate stopband attenuation
    # Find stopband indices using sFsb (stopband edge)
    vStopInd = np.where((freqs >= sFsb) & (freqs <= sFs / 2))[0]
    sStpResp = np.abs(vH[vStopInd])
    sHsbMax = np.max(sStpResp)
    sRsb = -20 * np.log10(sHsbMax)  # Attenuation in dB

    return vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax