# system packages
import numpy as np

def freq2Omega(sF, sFs):
    """Convert frequency from Hz to radians/sample."""
    sOmega = (2 * np.pi * sF) / sFs
    return sOmega

def omega2Freq(sOmega, sFs):
    """Convert frequency from radians/sample to Hz."""
    sF = (sOmega * sFs) / (2 * np.pi)
    return sF