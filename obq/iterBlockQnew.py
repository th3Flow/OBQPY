import numpy as np
#Linear Algebraic, signal processing
import scipy.linalg as scLinAlg
import scipy.signal as sigP
import obq, misc, sp

def iterBlockQnew(vx, vw, sM, sH, sType):
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
    swLen       = len(vw)
    sxLen       = len(vx)
    sW_hatRLen    = swLen + sM - 1
    
    ve      = np.zeros((sxLen,1)).flatten()         
    vb      = np.zeros((sxLen,1)).flatten()   
    sEffZones = swLen//sM

    vC = np.zeros((sW_hatRLen,1)).flatten()
    mC = np.zeros((sW_hatRLen,sEffZones))
    sNumBlocks = (sxLen - sM) // sH + 1  # Calculate the number of iterations considering hop size
    
    
    vbBlock     = np.zeros((sM,1)).flatten() 
    veBlock     = np.zeros((sW_hatRLen,1)).flatten()
    veL2Block   = np.zeros((sNumBlocks,1)).flatten()
    
    #vErrWeigths = sp.linWeightVec(sEffZones,0.2)

    if np.mod(sxLen,sM):
        print("vx should be a multiple of sM")
    else:
        mW_hat = sp.convMtx(vw,sM,'colWise')
        
        for m in range(sNumBlocks): 
            sStIdx = m * sH
            sEndIdx = sStIdx + sM
            
            vCe = vC.copy()     #Initialize ve_hat before we actually proceed with the error calculation
            
            mC          = np.roll(mC, shift=1, axis=1)
            mC[:,1::]   = sp.circShiftZ(mC[:,1::], -sH, 0)
            if (m > 0):  #Generation of the vector e_hat
               mC[:,0] = sp.circShiftZ(veBlock, -sH, 0)
               
            vCe = mC[:,0]
            
            if (sType == 'grb'):
                vbBlock, veBlock = obq.OptBlock(vx[sStIdx:sEndIdx], mW_hat, vCe, np.sum(veL2Block))               
            else:
                vbBlock, veBlock = obq.combOptBlock(vx[sStIdx:sEndIdx], mW_hat)   
                   
            vb[sStIdx:sEndIdx] = vbBlock
            

            veL2Block[m] = np.sum(veBlock[0:sH]**2)
            print("BlockNumber: %d, ErrVal: %3.5f" % (m, veL2Block[m]))                       
        
    return vb, ve