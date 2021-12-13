from pyscf.pbc.lib import chkfile
from pyscf.pbc import scf, gto
import matplotlib.pyplot as plt
import numpy as np

a = 2.9
nat = 2

cell = gto.Cell()    

for iat in range(0, nat):
    cell.atom.append(['Au', (iat*a, 0, 0)])

cell.exp_to_discard = 0.1

cell.build(    
        unit = 'angstrom',    
        a = [[a*nat,0,0],[0,1,0],[0,0,1]],    
        dimension = 1,    
        basis = 'def2-svp',
        ecp = 'def2-svp',
        verbose = 0
        )    

rdf = []
udf = []

datadir='data-20211210181649'

for nks in range(2, 21, 2):
    kpts = cell.make_kpts([nks,1,1], scaled_center=[0,0,0])    
    #print('kpts = ', kpts)

    # restricted
    rhf_chk = datadir + '/rhf_02_' + str(nks).zfill(2) + '.chk'
    rhf_h5 = datadir + '/rhf_cderi_02_' + str(nks).zfill(2) + '.h5'

    rhf_data = chkfile.load(rhf_chk, 'scf')
    
    rhf = scf.KRHF(cell).density_fit()    
    rhf.kpts = kpts
    rhf.with_df._cderi = rhf_h5 
    
    rdm1 = rhf.make_rdm1(mo_coeff_kpts=rhf_data['mo_coeff'], mo_occ_kpts=rhf_data['mo_occ'])
    
    rf = rhf.get_fock(dm = rdm1)

    if nks > 2:
        rdf.append(np.linalg.norm(rf[0]-rf_old[0]))

    rf_old = rf

    # unrestricted
    uhf_chk = datadir + '/uhf_02_' + str(nks).zfill(2) + '.chk'
    uhf_h5 = datadir + '/uhf_cderi_02_' + str(nks).zfill(2) + '.h5'

    uhf_data = chkfile.load(uhf_chk, 'scf')
    
    uhf = scf.KUHF(cell).density_fit()    
    uhf.kpts = kpts
    uhf.with_df._cderi = uhf_h5 
    
    rdm1 = uhf.make_rdm1(mo_coeff_kpts=uhf_data['mo_coeff'], mo_occ_kpts=uhf_data['mo_occ'])
    
    uf = uhf.get_fock(dm = rdm1)

    if nks > 2:
        udf.append(np.linalg.norm(uf[0][0]-uf_old[0][0]))

    uf_old = uf


print('rdf = ', rdf)
print('udf = ', udf)
