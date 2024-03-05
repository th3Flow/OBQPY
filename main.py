# system packages
import numpy as np
import matplotlib.pyplot as mtplt
import matplotlib.gridspec as gridspec
import pandas as pd


from scipy.linalg import toeplitz
# individual packages
import sg, sa, sp, obc

mtplt.close('all')

sNbins = 2**10
sFs = sNbins
T = 1 / sFs

sRecFilterFrequ = 132

### Signal generation ###
v_n = np.arange(sNbins).reshape(-1, 1)
vxFrequ = (np.arange(1, 131, step=3) * sFs / sNbins).reshape(-1, 1)
vxPhaseInit = np.random.rand(len(vxFrequ), 1) * 2 * np.pi

vx, vTime = sg.signalGen(v_n, vxFrequ, vxPhaseInit, sFs, 'real')
vx = sg.MFnormalize(vx, np.array([-1, 1]))

### Generate ideal matrices ###
vRIdeal = sp.idealBinFilt(sNbins, sg.freq2Bin(sRecFilterFrequ, sNbins, sFs), 'normal')
mRIdeal = toeplitz(vRIdeal)
disp_mRIdeal = pd.DataFrame(mRIdeal)
disp_mRIdeal

### Frequency Analysis ###
vX = np.fft.fft(vx)
vXMag = 20*sa.safelog10(np.abs(vX) / np.max(abs(vX)))
# Frequency bins
vXFreq = np.fft.fftfreq(sNbins, T)





###### PLOTTING ######
figOne = mtplt.figure()
Pltgs = gridspec.GridSpec(3, 2)

pltDiscTime = figOne.add_subplot(Pltgs[0,:])
pltDiscTime.plot(vx)
pltDiscTime.set_title('Discrete-Time Signal')
pltDiscTime.set_xlabel('Samples $n$', fontsize = 11)
pltDiscTime.set_ylabel('Amplitude', fontsize = 11)
pltDiscTime.set_xlim([0,sNbins])
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltObsOne = figOne.add_subplot(Pltgs[1,0])
pltObsOne.set_title('Single-Sequential Structure')
pltObsOne.set_xlabel('Samples $n$', fontsize = 11)
pltObsOne.set_ylabel('Amplitude', fontsize = 11)
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltObsTwo = figOne.add_subplot(Pltgs[1,1])
pltObsTwo.set_title('Block-Sequential Structure')
pltObsTwo.set_xlabel('Samples $n$', fontsize = 11)
pltObsTwo.set_ylabel('Amplitude', fontsize = 11)
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltFreqOne = figOne.add_subplot(Pltgs[2,0])
pltFreqOne.plot(vXFreq[:sNbins // 2], vXMag[:sNbins // 2])
pltFreqOne.set_title('Frequency Spectrum One')
pltFreqOne.set_xlabel('Samples $n$', fontsize = 11)
pltFreqOne.set_ylabel('Amplitude', fontsize = 11)
pltFreqOne.set_xlim([0,sNbins/2])
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

pltFreqTwo = figOne.add_subplot(Pltgs[2,1])
pltFreqTwo.set_title('Frequency Spectrum Two')
pltFreqTwo.set_xlabel('Samples $n$', fontsize = 11)
pltFreqTwo.set_ylabel('Amplitude', fontsize = 11)
mtplt.minorticks_on()
mtplt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray')

mtplt.tight_layout()
mtplt.show()