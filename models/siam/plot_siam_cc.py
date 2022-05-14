import numpy as np
import matplotlib.pyplot as plt
import h5py


fh = h5py.File('ldos_siam_cc.h5', 'r')

freqs_cc = np.asarray(fh['freqs_cc'])
ldos_cc = np.asarray(fh['ldos_cc'])

freqs_mf = np.asarray(fh['freqs_mf'])
ldos_mf = np.asarray(fh['ldos_mf'])

dw_mf = freqs_mf[1:]-freqs_mf[:-1]
dw_cc = freqs_cc[1:]-freqs_cc[:-1]

mf_int = np.sum(ldos_mf[:-1]*dw_mf)
cc_int = np.sum(ldos_cc[:-1]*dw_cc)

print(mf_int)
print(cc_int)

plt.plot(freqs_cc, ldos_cc, linestyle='-')
plt.plot(freqs_mf, ldos_mf, linestyle=':')
plt.show()


exit()
