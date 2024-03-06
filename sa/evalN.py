import numpy as np
import math

def evalN(vNoise, vRef):
    """
    Compute the SNR of a signal and a reference signal.

    Inputs:
        vNoise: noise signal
        vRef: reference signal

    Outputs:
        sMSE: Mean Squared Error
        sSNRdB: resulting SNR in dB
        sPSNRdB: resulting peakSNR in dB

    Authors:
        Florian Mayer <florian.mayer@fh-joanneum.at>

    """
    vNoise += math.ulp(1.00)
    vRef += math.ulp(1.00)
    
    # Mean-Squared Error
    sMSE = np.sum((vNoise) ** 2) / np.size(vRef)
    varRef = np.sum(np.abs(vRef) ** 2) / np.size(vRef)
    varNoise = np.sum(np.abs(vNoise) ** 2) / np.size(vNoise)

    # Signal-to-Noise Ratio in dB
    sSNRdB = 10 * np.log10(varRef / varNoise)

    # Peak Signal-to-Noise Ratio in dB
    sPSNRdB = 10 * np.log10(np.max(vRef) ** 2 / sMSE)

    return sMSE, sSNRdB, sPSNRdB