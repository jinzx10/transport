import numpy as np
import matplotlib.pyplot as plt
import h5py

mu = 0.04
gate = 0.000
nbe=100
nbath_per_ene=9
eta=0.03

fname = 'imp_lead_rhf_ldos_nbe%i_nbpe%i_mu%5.3f_gate%5.3f_eta%5.3f.h5'%(nbe, nbath_per_ene, mu, gate, eta)
fh = h5py.File(fname, 'r')

freqs = np.asarray(fh['freqs'])
ldos = np.asarray(fh['ldos_ref'])
A = np.asarray(fh['A'])


for i in range(6):
#for i in [1]:
    plt.plot(freqs,ldos[:,i], linestyle=':', color='C'+str(i), label='ref')
    plt.plot(freqs,A[:,i], linestyle='-', color='C'+str(i), label='emb')

plt.legend()
plt.show()

exit()

fname = 'imp_rhf_ldos_nbe%i_nbpe%i_mu%5.3f_gate%5.3f.h5'%(nbe, nbath_per_ene, mu, gate)
fh = h5py.File(fname, 'r')

freqs = np.asarray(fh['freqs'])
ldos = np.asarray(fh['ldos_ref'])
A = np.asarray(fh['A'])


#for i in range(6):
for i in [1]:
    plt.plot(freqs,ldos[:,i], linestyle=':', color='C'+str(i), label='ref')
    plt.plot(freqs,A[:,i], linestyle='-', color='C'+str(i), label='emb')

#plt.legend()



plt.show()
