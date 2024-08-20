import numpy as np
#Linear Algebraic, signal processing
import scipy.linalg as scLinAlg
import scipy.signal as sigP
import obq, misc, sp

def iterBlockQOS(vx, vw, sM, sType):
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
    sW_hatRLen  = swLen + sM - 1
    
    sOverlap    = sM // 2 - 1
    vxPad       = np.pad(vx, pad_width=(sOverlap, 0), mode='constant', constant_values=0)
    sEndIdx     = sOverlap
    
    ve          = np.zeros((sxLen,1)).flatten()         
    vb          = np.zeros((sxLen,1)).flatten()   
    sEffZones   = sW_hatRLen//sM

    vC = np.zeros((sW_hatRLen,1)).flatten()
    mC = np.zeros((sW_hatRLen,sEffZones))
    sNumBlocks = (sxLen - sM) // sM + 1  # Calculate the number of iterations considering hop size
    
    
    #vbBlock     = np.zeros((sM,1)).flatten() 
    #veBlock     = np.zeros((sW_hatRLen,1)).flatten()
    veL2Block   = np.zeros((sNumBlocks,1)).flatten()


    if np.mod(len(sxLen),sM):
        print("vx should be a multiple of sM")
    else:
        mW_hat = sp.convMtx(vw,sM,'colWise')
        
        for m in range(sNumBlocks): 
            sStIdx = m * sM + sEndIdx - sOverlap
            sEndIdx = sStIdx + sM
            
            if (sType == 'grb'):
                vbBlock, veBlock = obq.OptBlock(vx[sStIdx:sEndIdx], mW_hat)               
            else:
                vbBlock, veBlock = obq.combOptBlock(vx[sStIdx:sEndIdx], mW_hat)   
                   
            vb[sStIdx:sEndIdx] = vbBlock
            

            veL2Block[m] = np.sum(veBlock[0:sM]**2)
            print("BlockNumber: %d, ErrVal: %3.5f" % (m, np.sum(veL2Block)))                       
        
    return vb, ve