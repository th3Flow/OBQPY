import numpy as np
#Linear Algebraic, signal processing
import scipy.linalg as scLinAlg
import scipy.signal as sigP
import obq, misc, sp

def iterBlockQnew(vx, vw, sM, sType):
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

    vC = np.zeros((sW_hatRLen,1)).flatten()
    vbBlock = np.zeros((sM,1)).flatten() 
    veBlock = np.zeros((sM,1)).flatten()


    if np.mod(sxLen,sM):
        print("vx should be a multiple of sM")
    else:
        mW_hat = sp.convMtx(vw,sM,'colWise')
        for m in range(int(sxLen/sM)): 
            
            vCe_hat = vC.copy()     #Initialize ve_hat before we actually proceed with the error calculation
                
            if (m > 0):  #Generation of the vector e_hat
                vCe_hat[:(sW_hatRLen-sM)] = veBlock[-(sW_hatRLen-sM):]
            
            if (sType == 'grb'):
                vbBlock, veBlock = obq.OptBlock(vx[m*sM:m*sM+sM],mW_hat,vCe_hat)                
            else:
                vbBlock, veBlock = obq.combOptBlock(vx[m*sM:m*sM+sM], mW_hat)    
                   
            vb[m*sM:m*sM+sM] = vbBlock
            

            sL2_sqBlock = np.sum(veBlock**2)
            print("BlockNumber: %d, ErrVal: %3.5f" % (m, sL2_sqBlock))                       
        
    return vb, ve