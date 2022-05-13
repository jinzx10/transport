import numpy as np
import matplotlib.pyplot as plt
import h5py


fh = h5py.File('imp_rks_ldos.h5', 'r')

freqs = np.asarray(fh['freqs'])
ldos = np.asarray(fh['ldos'])
A = np.asarray(fh['A'])


for i in range(6):
    plt.plot(freqs,ldos[:,i], linestyle=':', color='C'+str(i))
    plt.plot(freqs,A[:,i], linestyle='-', color='C'+str(i))

plt.show()
