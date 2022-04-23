import numpy as np
import matplotlib.pyplot as plt
import h5py


l = -0.225
h = -0.175




suffix = 'gate' + str(gate).replace('.','') + '_nb' + str(nb) + '_mu' + str(mu).replace('.','') + '_log_cc_delta001_base15_nbpe1'

fname = 'ldos_' + suffix + '.dat'
fh = h5py.File(fname, 'r')

freqs = fh['freqs']
nw = len(freqs)

ldos = np.zeros((6, nw))

for i in range(6):
    ldos[i,:] = fh['ldos_' + method + '_'+str(i)]

ldos = np.sum(ldos, axis=0)

ldos_cumsum = np.cumsum(ldos)
plt.plot(freqs, ldos_cumsum)
