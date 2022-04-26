import numpy as np
import matplotlib.pyplot as plt
import h5py

mu = -0.161393

def plot_ldos(gate, nb, method, eri_scale, linestyle, color, cumsum = False):
    # get the occupancy
    suffix = 'gate' + str(gate) + '_eri' + str(eri_scale) + '_nb' + str(nb) + '_mu' + str(mu) + '_log_cc_delta0.01_base1.5_nbpe1'
    outfile='run_dmft_' + suffix + '.out'
    with open(outfile, 'r') as f:
        for line in f.readlines():
            if 'At mu' in line:
                occ = line.split()[7][:4]
                break

    fname = 'ldos_' + suffix + '.dat'
    fh = h5py.File(fname, 'r')
    
    freqs = fh['freqs']
    nw = len(freqs)
    
    ldos = np.zeros((5, nw))
    
    for i in range(5):
        ldos[i,:] = fh['ldos_' + method + '_'+str(i+1)]
    
    #ldos = np.zeros((6, nw))
    #
    #for i in range(6):
    #    ldos[i,:] = fh['ldos_' + method + '_'+str(i)]
    
    ldos = np.sum(ldos, axis=0)

    
    label = method + '   nbath=' + str(nb) + '   gate=' + str(gate)
    if method == 'cc':
        label = label + '   Nelec=' + occ

    if cumsum == False:
        plt.plot(freqs, ldos, ls=linestyle, color=color, label=label)
    else:
        plt.plot(freqs, np.cumsum(ldos)*(freqs[1]-freqs[0]), ls=linestyle, color=color, label=label)


#plot_ldos(-0.07, 30, 'cc', 0.0, '-', 'g')
#plot_ldos(-0.07, 30, 'hf', 0.0, ':', 'g')
#plot_ldos(-0.07, 30, 'cc', 0.5, '-', 'b')
#plot_ldos(-0.07, 30, 'hf', 0.5, ':', 'b')
plot_ldos(-0.07, 30, 'cc', 1.0, '-', 'r')
plot_ldos(-0.07, 30, 'hf', 1.0, ':', 'r')

plt.legend()
plt.xlabel('E')
#plt.ylabel('LDoS')
#plt.title('Total LDoS on Co')
plt.show()


