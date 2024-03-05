import numpy as np

def idealBinFilt(sNbins, sMaxBin, sType):
    """
    Create ideal bandpass or bandstop filter coefficients.

    Parameters:
    sNbins : int
        Number of bins (samples) in the filter.
    sMaxBin : int
        Maximum bin (frequency) to pass or stop.
    sType : str
        Type of filter: 'inv' for bandstop, anything else for bandpass.

    Returns:
    vFiltCoeffs : numpy.ndarray
        The time-domain filter coefficients.
    """
    # Generate the bin indices
    v_n = np.arange(sNbins)
    
    # Create the ideal filter based on the type
    if sType == 'inv':
        # Bandstop filter
        vIdealFilter = np.ones(sNbins)
        vIdealFilter[:sMaxBin+1] = 0
        vIdealFilter[-sMaxBin:] = 0
    else:
        # Bandpass filter
        vIdealFilter = np.zeros(sNbins)
        vIdealFilter[:sMaxBin+1] = 1
        vIdealFilter[-sMaxBin:] = 1

    # Perform IFFT to get filter coefficients in time domain
    vFiltCoeffs = np.fft.ifft(vIdealFilter)
    
    # Normalize to get unity gain
    vFiltCoeffs = vFiltCoeffs / np.sum(vFiltCoeffs)
    
    return vFiltCoeffs.real  # Return only the real part in case of negligible imaginary parts
