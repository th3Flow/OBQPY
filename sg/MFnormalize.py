import numpy as np

def MFnormalize(vSignal, vRange):
    sSignalMean = np.mean(vSignal)
    # sSignalMean = (np.max(vSignal) + np.min(vSignal)) / 2
    vTempSignal = vSignal - sSignalMean
    sRangeSig = np.max(vTempSignal) - np.min(vTempSignal)

    sNormSigMean = np.mean(vRange)
    sRangeNormSig = np.max(vRange) - np.min(vRange)
    vNormSig = vTempSignal * (sRangeNormSig / sRangeSig) + sNormSigMean

    return vNormSig