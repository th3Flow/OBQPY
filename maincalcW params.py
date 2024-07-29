import numpy as np
import scipy.signal as sigP
from scipy.optimize import differential_evolution

import sg, sa, sp, obq

# Your existing signal and setup code
sNbins = 2**12
sFs = 4096
sBSize = 8
sSigFmax = 64

# Load or generate your signal vx
vx = np.load('signal.npy')

# Initialize the best improvement value and iteration counter
best_ser = -np.inf
best_params = None
iteration = 0
sTaps_limit = 129  # Define the limit for sTaps

# Generate the baseline Sigma Delta Quantizer
vBSequSingle, ve, ve_hat = obq.iterSequQ(vx, np.tril(np.ones((sNbins, sNbins))), 0)
vW = np.ones(sNbins)  # Initial filter coefficients for baseline comparison

# Define the objective function to maximize SER improvement
def calculate_ser(params):
    global best_ser, best_params, iteration
    iteration += 1
    
    sTransWidthHz, sAttdB, sCutOffHz = params
    sTransWidthDig = sTransWidthHz / (sFs / 2)
    sCutOffDig = sCutOffHz / (sFs / 2)
    
    # Compute the order and Kaiser parameter for the FIR filter
    sTaps, sBeta = sigP.kaiserord(sAttdB, sTransWidthDig)
    
    # Check if sTaps exceeds the limit
    if sTaps > sTaps_limit:
        penalty = np.inf  # High penalty value
    else:
        penalty = 0  # No penalty

    # Use firwin with a Kaiser window to create a lowpass FIR filter
    vWLs = sigP.firwin(sTaps, sCutOffDig, window=('kaiser', sBeta))
    vNormWLs = vWLs / np.sum(vWLs)
    
    # Run your optimization process (obq.iterBlockQnew)
    vBSequBlock, vEBlock = obq.iterBlockQnew(vx, vNormWLs, sBSize, 'grb')
    
    # Filtered Signals
    vxSigFilt = np.convolve(vNormWLs, vx, 'same')
    vbSequErrFilt = np.convolve(vNormWLs, vx - vBSequSingle, 'same')
    vbBlockSequErrFilt = np.convolve(vNormWLs, vx - vBSequBlock, 'same')
    
    # Calculate SER
    _, sVB_SERdb, _ = sa.evalN(vbSequErrFilt, vxSigFilt)
    _, sVBBlock_SERdb, _ = sa.evalN(vbBlockSequErrFilt, vxSigFilt)
    
    # Check if the improvement is at least 3 dB
    if sVBBlock_SERdb < sVB_SERdb + 3:
        improvement_penalty = np.inf  # High penalty value
    else:
        improvement_penalty = 0  # No penalty
    
    current_ser = sVBBlock_SERdb
    
    # Apply penalties if constraints are not met
    if penalty > 0 or improvement_penalty > 0:
        objective_value = penalty + improvement_penalty
    else:
        objective_value = -current_ser  # Negate because differential_evolution minimizes the objective
    
    if current_ser > best_ser:
        best_ser = current_ser
        best_params = params
    
    # Print the status update
    print(f"Run {iteration}")
    print(f"Current Parameters: sTransWidthHz={sTransWidthHz:.2f}, sAttdB={sAttdB:.2f}, sCutOffHz={sCutOffHz:.2f}")
    print(f"Current SER: {current_ser:.2f} dB")
    print(f"Best SER so far: {best_ser:.2f} dB for Parameters: sTransWidthHz={best_params[0]:.2f}, sAttdB={best_params[1]:.2f}, sCutOffHz={best_params[2]:.2f}")
    
    return objective_value

# Define bounds for the parameters: (sTransWidthHz, sAttdB, sCutOffHz)
bounds = [
    (50, 150),        # sTransWidthHz
    (20, 50),         # sAttdB
    (sSigFmax + 1, sSigFmax + 75)  # sCutOffHz
]

# Run the differential evolution optimization with enhanced exploration settings
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
optimal_sTransWidthHz, optimal_sAttdB, optimal_sCutOffHz = result.x
optimal_ser = -result.fun  # Convert back to positive

print(f"Optimal Transition Width: {optimal_sTransWidthHz} Hz")
print(f"Optimal Attenuation: {optimal_sAttdB} dB")
print(f"Optimal Cutoff Frequency: {optimal_sCutOffHz} Hz")
print(f"Optimal SER: {optimal_ser:.2f} dB")
