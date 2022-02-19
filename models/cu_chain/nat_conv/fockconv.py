from pyscf.pbc.lib import chkfile
import numpy as np

nat_start = 2 
nat_end = 20
nat_step = 2

nat_range = range(nat_start, nat_end+1, nat_step)

rdF = []
udF = []

for nat in nat_range:
    cell = chkfile.load_cell(datadir + '/cu_' + str(nat).zfill(2) + '.chk')
    idx_start, idx_end = cell.aoslice_by_atom()[0][2:4]

    rF = np.load(datadir + '/rFock_' + str(nat).zfill(2) + '.npy')
    rF = rF[0, idx_start:idx_end, idx_start:idx_end]

    if nat > nat_start:
        rdF.append( np.linalg.norm(rF[0, idx_start:idx_end, idx_start:idx_end] 
                                    - rF_old[0, idx_start:idx_end, idx_start:idx_end]) )

    rF_old = rF

    uF = np.load(datadir + '/uFock_' + str(nat).zfill(2) + '.npy')
    uF = uF[0, 0, idx_start:idx_end, idx_start:idx_end]

    if nat > nat_start:
        udF.append( np.linalg.norm(uF[0, 0, idx_start:idx_end, idx_start:idx_end] 
                                    - uF_old[0, 0, idx_start:idx_end, idx_start:idx_end]) )

    uF_old = uF

print('rdF = ', rdF)
print('udF = ', udF)
