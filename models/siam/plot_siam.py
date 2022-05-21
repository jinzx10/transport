import numpy as np
import matplotlib.pyplot as plt
import h5py


# read cc data
fh = h5py.File('ldos_siam_cc.h5', 'r')

freqs_cc = np.asarray(fh['freqs_cc'])
ldos_cc = np.asarray(fh['ldos_cc'])

freqs_mf = np.asarray(fh['freqs_mf'])
ldos_mf = np.asarray(fh['ldos_mf'])

mu_cc = np.asarray(fh['mu'])

# check integrated LDoS
dw_mf = freqs_mf[1:]-freqs_mf[:-1]
dw_cc = freqs_cc[1:]-freqs_cc[:-1]

mf_int = np.sum(ldos_mf[:-1]*dw_mf)
cc_int = np.sum(ldos_cc[:-1]*dw_cc)

print(mf_int)
print(cc_int)

plt.plot(freqs_cc, ldos_cc, linestyle='-', label='cc')
plt.plot(freqs_mf, ldos_mf, linestyle=':', label='mf')
plt.axvline(x=mu_cc, linewidth=0.5, linestyle=':', color='k')

# read dmrg data
fh = h5py.File('ldos_siam_dmrg.h5', 'r')
freqs_dmrg = np.asarray(fh['all_freqs'])
ldos_mf_dmrg = np.asarray(fh['ldos_mf'])
ldos_dmrg = np.asarray(fh['ldos_dmrg'])
mu_dmrg = np.asarray(fh['mu'] )

plt.plot(freqs_dmrg, ldos_dmrg, marker='o', label='dmrg')
plt.plot(freqs_dmrg, ldos_mf_dmrg, marker='+', label='mf')
plt.axvline(x=mu_dmrg, linewidth=0.5, linestyle=':', color='k')


plt.legend()
plt.show()

exit()
