import numpy as np
import matplotlib.pyplot as plt
import h5py

imp_atom = 'Co'
gate = 0.000
eta = 0.02
gate_label = 'gate%5.3f'%(gate)
eta_label = 'eta%5.3f'%(eta)

fh = h5py.File('ldos_mf_' + imp_atom + '_' + gate_label + '_' + eta_label + '.h5', 'r')

freqs = fh['freqs']
ldos_ks = fh['ldos_ks']
ldos_hf = fh['ldos_hf']
ldos_emb_hf = fh['ldos_emb_hf']

#for i in range(6):
for i in [1]:
    plt.plot(freqs,ldos_ks[i,:], linestyle='-', color='C'+str(i), label='ks')
    plt.plot(freqs,ldos_hf[i,:], linestyle=':', color='C'+str(i), label='hf')
    plt.plot(freqs,ldos_emb_hf[i,:], linestyle='--', color='C'+str(i), label='emb hf')

plt.legend()
plt.show()
