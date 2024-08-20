import numpy as np
#Linear Algebraic, signal processing
import scipy.linalg as scLinAlg
import scipy.signal as sigP
import obq, misc

def iterBlockQ(vx, vw, sM, sType):
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
    swLen = len(vw)
    #swLen_h = int(swLen/2)
    sxLen = len(vx)
    
    ve = np.zeros((sxLen,1)).flatten()         
    vb = np.zeros((sxLen,1)).flatten()   

    vwFull = np.zeros((sxLen,1)).flatten()
    vwFull[0:swLen] = vw

    #vxPadded = np.pad(vx,(sCutSIdx,(sCutEIdx + additionalPadding)), mode = "reflect")

    ve_init = np.zeros((sM,1)).flatten()
    vbBlock = np.zeros((sM,1)).flatten() 
    veBlock = np.zeros((sM,1)).flatten()

    if np.mod(sxLen,sM):
        print("vx should be a multiple of sM")
    else:
        mW_0 = np.tril(scLinAlg.toeplitz(vwFull[0:sM]))
        for m in range(int(sxLen/sM)):                                   
            ve_hat = ve_init.copy()     #Initialize ve_hat before we actually proceed with the error calculation
            for k in range(m):  #Generation of the vector e_hat
                sRowIdx = sM*(m-k)
                sColIdx = (m-k-1)*sM+1
                mW_m = scLinAlg.toeplitz(vwFull[sRowIdx:sRowIdx+sM],np.flip(vwFull[sColIdx:sColIdx+sM]))
                ve_hat += mW_m @ (vx[k*sM:k*sM+sM] - vb[k*sM:k*sM+sM])
            
            if (sType == 'grb'):
                vbBlock, veBlock = obq.OptBlock(vx[m*sM:m*sM+sM],mW_0,ve_hat)                
            else:
                vbBlock, veBlock = obq.combOptBlock(vx[m*sM:m*sM+sM], mW_0, ve_hat)    
                
            vb[m*sM:m*sM+sM] = vbBlock
            ve[m*sM:m*sM+sM] = veBlock

            sL2_sqBlock = np.sum(veBlock**2)
            print("BlockNumber: %d, ErrVal: %3.5f" % (m, sL2_sqBlock))
        
    return vb, ve