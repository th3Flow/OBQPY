import numpy as np
import scipy.signal as sigP
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import sg, sa, sp, obq, misc

import os, sys

# Initial setup parameters
sNbins = 4096
sFs = 4096
sBSize = 8
sSigFmax = 63
bSNRideal = True

# Signal generation
v_n = np.arange(sNbins).reshape(-1, 1)
vxFrequ = (np.arange(0, sSigFmax, step=2)).reshape(-1, 1)
vxPhaseInit = np.random.rand(len(vxFrequ), 1) * 2 * np.pi
vx, vTime = sg.signalGen(v_n, vxFrequ, vxPhaseInit, sFs, 'real')
vx = sg.MFnormalize(vx, -1, 1)

# Initial quantization
vBSequSingle, ve, ve_hat = obq.iterSequQ(vx, np.tril(np.ones((sNbins, sNbins))), 0)
vW = np.ones(sNbins)  # Initial filter coefficients for baseline comparison

# Initialize the best improvement value and iteration counter
best_ser = -np.inf
best_params = None
iteration = 0

# Define the objective function to maximize sVBBlock_SERdb
def calculate_ser(params):
    global best_ser, best_params, iteration
    iteration += 1
    
    sPassHz, sTransWHz, sApass, sAstop = params
    sPassDig = sPassHz / (sFs / 2)
    sStopDig = (sPassHz + sTransWHz) / (sFs / 2)
    sCutOffDig = sPassDig + (sStopDig - sPassDig) / 2

    sTaps, sBeta = sigP.kaiserord(sAstop, sStopDig - sPassDig)

    vWLs = sigP.firwin(sTaps, sCutOffDig, window=('kaiser', sBeta))
    vNormWLs = vWLs / np.sum(vWLs)
    
    vBSequBlock, vEBlock = obq.iterBlockQnew(vx, vNormWLs, sBSize, sBSize, 'grb')
    
    if bSNRideal: 
        vxSigFilt = sp.idealBinFilt(sNbins, sg.freq2Bin(sSigFmax, sNbins, sFs), 'normal') @ vx
        vbSequErrFilt = sp.idealBinFilt(sNbins, sg.freq2Bin(sSigFmax, sNbins, sFs), 'normal') @ (vx - vBSequSingle)
        vbBlockSequErrFilt = sp.idealBinFilt(sNbins, sg.freq2Bin(sSigFmax, sNbins, sFs), 'normal') @ (vx - vBSequBlock)
    else:
        vxSigFilt = np.convolve(vNormWLs, vx, 'same')
        vbSequErrFilt = np.convolve(vNormWLs, vx - vBSequSingle, 'same')
        vbBlockSequErrFilt = np.convolve(vNormWLs, vx - vBSequBlock, 'same')
    
    _, sVB_SERdb, _ = sa.evalN(vbSequErrFilt, vxSigFilt)
    _, sVBBlock_SERdb, _ = sa.evalN(vbBlockSequErrFilt, vxSigFilt)
    
    current_ser = sVBBlock_SERdb
    
    if current_ser > best_ser:
        best_ser = current_ser
        best_params = params
    
    print(f"Run {iteration}")
    print(f"Current Parameters: sPassHz={sPassHz:.2f}, sTransWHz={sTransWHz:.2f}, sApass={sApass:.2f}, sAstop={sAstop:.2f}")
    print(f"Current SER: {current_ser:.2f} dB (Block) | {sVB_SERdb:.2f} dB (Single)")
    print(f"Best SER so far: {best_ser:.2f} dB for Parameters: sPassHz={best_params[0]:.2f}, sTransWHz={best_params[1]:.2f}, sApass={best_params[2]:.2f}, sAstop={best_params[3]:.2f}")
    
    return -current_ser  # Minimize the negative SER to maximize the SER

# Define bounds for the parameters: (sPassHz, sTransWHz, sApass, sAstop)
bounds = [
    (sSigFmax + 1, sSigFmax + 75),  # sPassHz
    (10, 100),       # sTransWHz
    (0, 1),          # sApass
    (20, 100)        # sAstop
]

# Run the differential evolution optimization
result = differential_evolution(
    calculate_ser,
    bounds,
    strategy='best2bin',
    maxiter=1000,
    popsize=15,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    disp=True
)

# Extract the optimal values
optimal_sPassHz, optimal_sTransWHz, optimal_sApass, optimal_sAstop = result.x
optimal_ser = -result.fun  # Convert back to positive

print(f"Optimal Passband Frequency: {optimal_sPassHz} Hz")
print(f"Optimal Transition Width: {optimal_sTransWHz} Hz")
print(f"Optimal Passband Ripple: {optimal_sApass} dB")
print(f"Optimal Stopband Attenuation: {optimal_sAstop} dB")
print(f"Optimal SER: {optimal_ser:.2f} dB")