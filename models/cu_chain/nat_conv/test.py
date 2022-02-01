from pyscf.pbc.lib import chkfile
from pyscf.pbc import scf, gto, df
import matplotlib.pyplot as plt
import numpy as np
import sys

rdf = []
udf = []

datadir='data-' + sys.argv[1]

nat = 4

cell = chkfile.load_cell(datadir + '/au_' + str(nat).zfill(2) + '.chk')
print('cell', nat, 'loaded')

# indices of the selected (first) atom's AOs in the Fock matrix
idx_start, idx_end = cell.aoslice_by_atom()[0][2:4]
print('idx_start = ', idx_start)
print('idx_end = ', idx_end)
print('nao = ', cell.nao)

##============ restricted ============
#mf = scf.RHF(cell).density_fit()
#print('mf initialized')
#mf.with_df._cderi = datadir + '/cderi_' + str(nat).zfill(2) + '.h5'
#print('df cderi loaded')
#
#mf_data = chkfile.load(datadir + '/rhf_' + str(nat).zfill(2) + '.chk', 'scf')
#print('mf data readed')
#mf.__dict__.update(mf_data)
#print('mf data updated')
#
#rf = mf.get_fock()
#print('fock computed')
#
#if nat > nat_start:
#    rdf.append(np.linalg.norm(rf[idx_start:idx_end, idx_start:idx_end]-rf_old[idx_start:idx_end, idx_start:idx_end]))
#
#rf_old = rf
#
##============ unrestricted ============
#mf = scf.UHF(cell).density_fit()    
#mf.with_df._cderi = datadir + '/cderi_' + str(nat).zfill(2) + '.h5'
#
#mf_data = chkfile.load(datadir + '/uhf_' + str(nat).zfill(2) + '.chk', 'scf')
#mf.__dict__.update(mf_data)
#
#uf = mf.get_fock()
#
#if nat > nat_start:
#    udf.append(np.linalg.norm(uf[0][idx_start:idx_end, idx_start:idx_end]-uf_old[0][idx_start:idx_end, idx_start:idx_end]))
#
#uf_old = uf
#
#print(nat, 'done')
#
#
#print('nat = ', list(nat_range))
#print('rdf = ', rdf)
#print('udf = ', udf)

