import numpy as np
import matplotlib.pyplot as plt
import h5py

'''
#==========================================================================
fh = h5py.File('ldos-17-log-cc.dat', 'r')

freqs = fh['freqs']
nw = len(freqs)

ldos_hf = np.zeros((6, nw))
ldos_cc = np.zeros((6, nw))

for i in range(6):
    ldos_hf[i,:] = fh['ldos_hf_'+str(i)]
    ldos_cc[i,:] = fh['ldos_cc_'+str(i)]

ldos_hf_sum = np.sum(ldos_hf, axis=0)
ldos_cc_sum = np.sum(ldos_cc, axis=0)

plt.plot(freqs, ldos_hf_sum, ls=':', color='b')
plt.plot(freqs, ldos_cc_sum, ls=':', color='r')
'''

#==========================================================================
fh = h5py.File('ldos_gate-0062_nb25_mu-0161393_log_cc_delta001_base15_nbpe1.dat', 'r')

freqs = fh['freqs']
nw = len(freqs)

ldos_hf = np.zeros((6, nw))
ldos_cc = np.zeros((6, nw))

for i in range(6):
    ldos_hf[i,:] = fh['ldos_hf_'+str(i)]
    ldos_cc[i,:] = fh['ldos_cc_'+str(i)]

ldos_hf_sum = np.sum(ldos_hf, axis=0)
ldos_cc_sum = np.sum(ldos_cc, axis=0)

plt.plot(freqs, ldos_hf_sum, ls=':', color='b')
plt.plot(freqs, ldos_cc_sum, ls=':', color='r')
#==========================================================================
fh = h5py.File('ldos_gate-0062_nb35_mu-0161393_log_cc_delta001_base15_nbpe1.dat', 'r')

freqs = fh['freqs']
nw = len(freqs)

ldos_hf = np.zeros((6, nw))
ldos_cc = np.zeros((6, nw))

for i in range(6):
    ldos_hf[i,:] = fh['ldos_hf_'+str(i)]
    ldos_cc[i,:] = fh['ldos_cc_'+str(i)]

ldos_hf_sum = np.sum(ldos_hf, axis=0)
ldos_cc_sum = np.sum(ldos_cc, axis=0)

plt.plot(freqs, ldos_hf_sum, ls='--', color='b')
plt.plot(freqs, ldos_cc_sum, ls='--', color='r')
'''

#==========================================================================
fh = h5py.File('ldos_nb25_mu-00775_log_cc_delta001_base15_nbpe1.dat', 'r')

freqs = fh['freqs']
nw = len(freqs)

ldos_hf = np.zeros((6, nw))
ldos_cc = np.zeros((6, nw))

for i in range(6):
    ldos_hf[i,:] = fh['ldos_hf_'+str(i)]
    ldos_cc[i,:] = fh['ldos_cc_'+str(i)]

ldos_hf_sum = np.sum(ldos_hf, axis=0)
ldos_cc_sum = np.sum(ldos_cc, axis=0)

plt.plot(freqs, ldos_hf_sum, ls=':', color='b')
plt.plot(freqs, ldos_cc_sum, ls=':', color='r')

#==========================================================================
fh = h5py.File('ldos_nb35_mu-00775_log_cc_delta001_base15_nbpe1.dat', 'r')

freqs = fh['freqs']
nw = len(freqs)

ldos_hf = np.zeros((6, nw))
ldos_cc = np.zeros((6, nw))

for i in range(6):
    ldos_hf[i,:] = fh['ldos_hf_'+str(i)]
    ldos_cc[i,:] = fh['ldos_cc_'+str(i)]

ldos_hf_sum = np.sum(ldos_hf, axis=0)
ldos_cc_sum = np.sum(ldos_cc, axis=0)

plt.plot(freqs, ldos_hf_sum, ls='--', color='b')
plt.plot(freqs, ldos_cc_sum, ls='--', color='r')

#==========================================================================
fh = h5py.File('ldos_nb50_mu-00775_log_cc_delta001_base15_nbpe1.dat', 'r')

freqs = fh['freqs']
nw = len(freqs)

ldos_hf = np.zeros((6, nw))
ldos_cc = np.zeros((6, nw))

for i in range(6):
    ldos_hf[i,:] = fh['ldos_hf_'+str(i)]
    ldos_cc[i,:] = fh['ldos_cc_'+str(i)]

ldos_hf_sum = np.sum(ldos_hf, axis=0)
ldos_cc_sum = np.sum(ldos_cc, axis=0)

plt.plot(freqs, ldos_hf_sum, ls='-', color='b')
plt.plot(freqs, ldos_cc_sum, ls='-', color='r')

#==========================================================================
fh = h5py.File('ldos_nb70_mu-00775_log_cc_delta001_base15_nbpe1.dat', 'r')

freqs = fh['freqs']
nw = len(freqs)

ldos_hf = np.zeros((6, nw))
ldos_cc = np.zeros((6, nw))

for i in range(6):
    ldos_hf[i,:] = fh['ldos_hf_'+str(i)]
    ldos_cc[i,:] = fh['ldos_cc_'+str(i)]

ldos_hf_sum = np.sum(ldos_hf, axis=0)
ldos_cc_sum = np.sum(ldos_cc, axis=0)

plt.plot(freqs, ldos_hf_sum, marker='o', color='b')
plt.plot(freqs, ldos_cc_sum, marker='o', color='r')
'''




plt.show()
