from pyscf.pbc.lib import chkfile
from pyscf.pbc import scf, gto, df
import matplotlib.pyplot as plt
import numpy as np

rdf = []
udf = []

datadir='data-211213-155910'

# spacing between gold atoms
a = 2.9

# spacing between Cu and left lead
l = 2.5

# spacing between Cu and right lead
r = 2.5

# restricted calculation
# nat all odd (there is an extra Cu atom so the total number of electrons is even)
for nat in range(5,20,2):

    nl = nat // 2 # number of atoms in the left lead
    nr = nat - nl

    # build chain
    cell = gto.Cell()    
    for iat in range(0, nl):
        cell.atom.append(['Au', (iat*a, 0, 0)])
    cell.atom.append(['Cu', ((nl-1)*a+l, 0, 0)])
    for iat in range(0, nr):
        cell.atom.append(['Au', ((nl-1)*a+l+r+iat*a, 0, 0)])
    
    cell.exp_to_discard = 0.1
    cell.spin = (nat+1) % 2
    cell.ke_cutoff = 200
    cell.max_memory = 160000

    cell.build(    
            unit = 'angstrom',    
            a = [[l+r+(nl+nr-1)*a,0,0],[0,1,0],[0,0,1]],    
            dimension = 1,    
            basis = {'Au':'def2-svp', 'Cu':'lanl2dz'},
            ecp = {'Au':'def2-svp', 'Cu':'lanl2dz'},
            verbose = 0,
            ) 

    # auxbasis for density fitting
    ab = df.aug_etb(cell, beta=2.0)

    # restricted
    rhf_chk = datadir + '/rhf_' + str(nat).zfill(2) + '.chk'
    rhf_h5 = datadir + '/rhf_cderi_' + str(nat).zfill(2) + '.h5'

    rhf_data = chkfile.load(rhf_chk, 'scf')
    
    rhf = scf.RHF(cell).density_fit(auxbasis=ab)    
    rhf.with_df._cderi = rhf_h5 
    rhf.__dict__.update(rhf_data)
    
    rf = rhf.get_fock()

    if nat > 5:
        rdf.append(np.linalg.norm(rf[nl*28:nl*28+15,nl*28:nl*28+15]-rf_old[(nl-1)*28:(nl-1)*28+15,(nl-1)*28:(nl-1)*28+15]))

    rf_old = rf

    print(nat, 'done')


print('restricted nat = ', list(range(5, 20, 2)))
print('rdf = ', rdf)



# unrestricted calculation
for nat in range(4, 21, 2):

    nl = nat // 2 # number of atoms in the left lead
    nr = nat - nl

    # build chain
    cell = gto.Cell()    
    for iat in range(0, nl):
        cell.atom.append(['Au', (iat*a, 0, 0)])
    cell.atom.append(['Cu', ((nl-1)*a+l, 0, 0)])
    for iat in range(0, nr):
        cell.atom.append(['Au', ((nl-1)*a+l+r+iat*a, 0, 0)])
    
    cell.exp_to_discard = 0.1
    cell.spin = (nat+1) % 2
    cell.ke_cutoff = 200
    cell.max_memory = 160000

    cell.build(    
            unit = 'angstrom',    
            a = [[l+r+(nl+nr-1)*a,0,0],[0,1,0],[0,0,1]],    
            dimension = 1,    
            basis = {'Au':'def2-svp', 'Cu':'lanl2dz'},
            ecp = {'Au':'def2-svp', 'Cu':'lanl2dz'},
            verbose = 0,
            ) 

    # auxbasis for density fitting
    ab = df.aug_etb(cell, beta=2.0)

    # unrestricted
    uhf_chk = datadir + '/uhf_' + str(nat).zfill(2) + '.chk'
    uhf_h5 = datadir + '/uhf_cderi_' + str(nat).zfill(2) + '.h5'

    uhf_data = chkfile.load(uhf_chk, 'scf')
    
    uhf = scf.UHF(cell).density_fit(auxbasis=ab)    
    uhf.with_df._cderi = uhf_h5 
    uhf.__dict__.update(uhf_data)
    
    
    uf = uhf.get_fock()

    if nat > 4:
        udf.append(np.linalg.norm(uf[0][0][nl*28:nl*28+15,nl*28:nl*28+15]-uf_old[0][0][(nl-1)*28:(nl-1)*28+15,(nl-1)*28:(nl-1)*28+15]))

    uf_old = uf

    print(nat, 'done')

print('unrestricted nat = ', list(range(4, 21, 2)))
print('udf = ', udf)


