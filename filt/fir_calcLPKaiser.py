# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:02:31 2024

@author: mayerflo
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sigP
from matplotlib.ticker import FuncFormatter, MultipleLocator

import filt

def fir_calcLPKaiser(sFs, sFpb, sFsb, sApb, sAsb):

    sWidthDig = (sFsb - sFpb) / (sFs /2)
    sCutDig   = sFpb / (sFs /2) + sWidthDig / 2
    sFiltordK, sBeta = sigP.kaiserord(sAsb-sApb, sWidthDig)
    
    vHFilt = sigP.firwin(
            sFiltordK, 
            sCutDig, 
            window=('kaiser',sBeta),
            pass_zero = 'lowpass'
            )           
    
    (vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax) = filt.anFiltKaiser(vHFilt, sFs, sFpb, sFsb, 'lowpass')

    return (vHFilt, vw, vH, sRpb, sRsb, sHpbMin, sHpbMax, sHsbMax)