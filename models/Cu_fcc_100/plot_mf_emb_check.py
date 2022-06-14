import numpy as np
import matplotlib.pyplot as plt
import h5py


nbe=50
nbath_per_ene=3
fh = h5py.File('imp_rks_ldos_nbe%i_nbpe%i.h5'%(nbe, nbath_per_ene), 'r')

freqs = np.asarray(fh['freqs'])
ldos = np.asarray(fh['ldos'])
A = np.asarray(fh['A'])


#for i in range(6):
for i in [0,1,3]:
    plt.plot(freqs,ldos[:,i], linestyle=':', color='C'+str(i))
    plt.plot(freqs,A[:,i], linestyle='-', color='C'+str(i))

plt.show()
