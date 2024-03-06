import numpy as np

def iterSequQ(vx, mW, vC, sL):
    """
    Args:
        vx: Input vector.
        mW: Weight matrix.
        vC: Constant/Init vector.
        sL: Number of Decisions
        
    Returns:
        vb: Quantized one-bit vector
        ve: Error vector
    """
    vxLen = len(vx)
    ve = np.zeros(vxLen)         
    vb = np.zeros(vxLen) 

    for sn in range(vxLen):
        vb[sn] = 1                                   
        se_p = mW[sn, :] @ vx - mW[sn, :] @ vb + mW[sn, :] @ vC
        vb[sn] = -1
        se_n = mW[sn, :] @ vx - mW[sn, :] @ vb + mW[sn, :] @ vC
        
        if abs(se_p)**2 <= abs(se_n)**2:
            vb[sn] = 1
            ve[sn] = se_p
        else:
            vb[sn] = -1
            ve[sn] = se_n
                   
    return vb, ve