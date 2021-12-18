from pyscf.pbc.lib import chkfile
from pyscf.pbc import scf, gto, df
import matplotlib.pyplot as plt
import numpy as np

rdf = []
udf = []

datadir='data-211217-150954'

nat_start = 2
nat_end = 20

for nat in range(nat_start, nat_end+1, 2):

    cell = chkfile.load_cell(datadir + '/au_' + str(nat).zfill(2) + '.chk')

    # indices of the selected (first) atom's AOs in the Fock matrix
    idx_start, idx_end = cell.aoslice_by_atom()[0][2:4]

    #============ restricted ============
    rhf = scf.RHF(cell).density_fit()
    rhf.with_df._cderi = datadir + '/rhf_cderi_' + str(nat).zfill(2) + '.h5'

    rhf_data = chkfile.load(datadir + '/rhf_' + str(nat).zfill(2) + '.chk', 'scf')
    rhf.__dict__.update(rhf_data)
    
    rf = rhf.get_fock()

    if nat > nat_start:
        rdf.append(np.linalg.norm(rf[idx_start:idx_end, idx_start:idx_end]-rf_old[idx_start:idx_end, idx_start:idx_end]))

    rf_old = rf

    #============ unrestricted ============
    uhf = scf.UHF(cell).density_fit()    
    uhf.with_df._cderi = datadir + '/uhf_cderi_' + str(nat).zfill(2) + '.h5'

    uhf_data = chkfile.load(datadir + '/uhf_' + str(nat).zfill(2) + '.chk', 'scf')
    uhf.__dict__.update(uhf_data)

    uf = uhf.get_fock()

    if nat > nat_start:
        udf.append(np.linalg.norm(uf[0][idx_start:idx_end, idx_start:idx_end]-uf_old[0][idx_start:idx_end, idx_start:idx_end]))

    uf_old = uf

    print(nat, 'done')

print('nat = ', list(range(nat_start, nat_end, 2)))
print('rdf = ', rdf)
print('udf = ', udf)

