from pyscf.pbc.lib import chkfile
from pyscf.pbc import scf, gto
import matplotlib.pyplot as plt
import numpy as np

rdf = []
udf = []

datadir='data-211217-182949'

nkmin = 2
nkmax = 20
nat = 2

cell = chkfile.load_cell(datadir + '/au_' + str(nat).zfill(2) + '.chk')

for nks in range(nkmin, nkmax+1, 2):

    #================ restricted ================
    rhf = scf.KRHF(cell).density_fit()
    rhf.with_df._cderi = datadir + '/' + 'rhf_cderi_'+str(nat).zfill(2)+'_'+str(nks).zfill(2)+'.h5'

    rhf_data = chkfile.load(datadir + '/rhf_'+str(nat).zfill(2)+'_'+str(nks).zfill(2)+'.chk', 'scf')
    rhf.__dict__.update(rhf_data)
    
    rf = rhf.get_fock()

    if nks > nkmin:
        rdf.append(np.linalg.norm(rf[0]-rf_old[0]))

    rf_old = rf

    #================ unrestricted ================
    uhf = scf.KUHF(cell).density_fit()    
    uhf.with_df._cderi = datadir + '/' + 'uhf_cderi_'+str(nat).zfill(2)+'_'+str(nks).zfill(2)+'.h5'

    uhf_data = chkfile.load(datadir + '/uhf_'+str(nat).zfill(2)+'_'+str(nks).zfill(2)+'.chk', 'scf')
    uhf.__dict__.update(uhf_data)
    
    uf = uhf.get_fock()

    if nks > nkmin:
        udf.append(np.linalg.norm(uf[0][0]-uf_old[0][0]))

    uf_old = uf

    print(nks, 'done')

print('nks = ', list(range(nkmin, nkmax+1, 2)))
print('rdf = ', rdf)
print('udf = ', udf)


