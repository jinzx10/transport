import numpy as np
import matplotlib.pyplot as plt
import h5py

mu = -0.161393

def plot_ldos(gate, nb, method, linestyle, color, cumsum = False):
    # get the occupancy
    suffix = 'gate' + str(gate).replace('.','') + '_nb' + str(nb) + '_mu' + str(mu).replace('.','') + '_log_cc_delta001_base15_nbpe1'
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
    
    ldos = np.zeros((6, nw))
    
    for i in range(6):
        ldos[i,:] = fh['ldos_' + method + '_'+str(i)]
    
    ldos = np.sum(ldos, axis=0)

    
    label = method + '   nbath=' + str(nb) + '   gate=' + str(gate)
    if method == 'cc':
        label = label + '   Nelec=' + occ

    if cumsum == False:
        plt.plot(freqs, ldos, ls=linestyle, color=color, label=label)
    else:
        plt.plot(freqs, np.cumsum(ldos)*(freqs[1]-freqs[0]), ls=linestyle, color=color, label=label)


#plot_ldos(-0.02, 25, 'cc', '-', 'g')
#plot_ldos(-0.02, 25, 'hf', ':', 'g')
#
#plot_ldos(-0.04, 25, 'cc', '-', 'b')
#plot_ldos(-0.04, 25, 'hf', ':', 'b')
#
#plot_ldos(-0.06, 25, 'cc', '-', 'c')
#plot_ldos(-0.06, 25, 'hf', ':', 'c')
#
#plot_ldos(-0.08, 25, 'cc', '-', 'm')
#plot_ldos(-0.08, 25, 'hf', ':', 'm')
#
#plot_ldos(-0.1, 25, 'cc', '-', 'y')
#plot_ldos(-0.1, 25, 'hf', ':', 'y')

#plot_ldos(0.09, 30, 'cc', '-', 'm')
#plot_ldos(0.09, 30, 'hf', ':', 'm')

plot_ldos(-0.07, 30, 'cc', '-', 'm')
plot_ldos(-0.07, 30, 'hf', ':', 'm')
#plot_ldos(-0.09, 30, 'cc', '-', 'm')
#plot_ldos(-0.09, 30, 'hf', ':', 'm')
#
#plot_ldos(-0.07, 30, 'cc', '-', 'm')
#plot_ldos(-0.07, 30, 'hf', ':', 'm')
plt.axvline(x=mu, linestyle=':', color='black', lw=1)

#plot_ldos(-0.04, 25, 'cc', '-', 'k', True)
#plot_ldos(-0.04, 25, 'hf', ':', 'k', True)
#
#plot_ldos(-0.08, 25, 'cc', '-', 'g', True)
#plot_ldos(-0.08, 25, 'hf', ':', 'g', True)
#
#plot_ldos(-0.12, 25, 'cc', '-', 'c', True)
#plot_ldos(-0.12, 25, 'hf', ':', 'c', True)
#
#plot_ldos(-0.16, 25, 'cc', '-', 'y', True)
#plot_ldos(-0.16, 25, 'hf', ':', 'y', True)
#
#plot_ldos(-0.3, 25, 'cc', '-', 'r', True)
#plot_ldos(-0.3, 25, 'hf', ':', 'r', True)

plt.legend()
plt.xlabel('E')
#plt.ylabel('LDoS')
#plt.title('Total LDoS on Co')
plt.show()


