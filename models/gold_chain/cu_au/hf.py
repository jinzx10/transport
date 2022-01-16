#######################################################################
# This script performs Gamma point HF calculations
# for 1-D periodic copper-embedded gold atomic chains:
# ...-Au-Au-Cu-Au-Au-...
#######################################################################

from pyscf.pbc import gto, scf, df    
from pyscf.pbc.lib import chkfile
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--savedir', default = 'data', type = str)
parser.add_argument('--spacing', default = 2.9, type = float)
parser.add_argument('--left', default = 2.5, type = float)
parser.add_argument('--right', default = 2.5, type = float)
parser.add_argument('--smearing', default = 0.01, type = float)
args = parser.parse_args()

savedir = args.savedir
a = args.spacing
l = args.left
r = args.right
smearing = args.smearing

print('data will be saved to ', savedir)
print('spacing between gold atoms = ', a)
print('spacing between copper atom and left lead = ', l)
print('spacing between copper atom and right lead = ', r)
print('smearing = ', smearing)

# total number of atoms in the lead (left+right) within a unit cell
for nat in range(4, 21):
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
    cell.max_memory = 32000

    cell.build(    
            unit = 'angstrom',    
            a = [[l+r+(nl+nr-1)*a,0,0],[0,30,0],[0,0,30]],    
            dimension = 3,    
            basis = {'Au':'def2-svp', 'Cu':'def2-svp'},
            ecp = {'Au':'def2-svp'}, # Cu does not have ECP
            verbose = 4,
    ) 

    # save cell
    chkfile.save_cell(cell, savedir + '/cu_au_' + str(nat).zfill(2) + '.chk')

    # auxbasis for density fitting
    ab = df.aug_etb(cell, beta=2.0)

    kpts = cell.make_kpts([1,1,1], scaled_center=[0,0,0])    

    #================ unrestricted ================
    mf = scf.KUHF(cell).density_fit(auxbasis=ab)    
    mf.chkfile = savedir + '/uhf_' + str(nat).zfill(2) + '.chk'
    mf.with_df._cderi_to_save = savedir + '/cderi_' + str(nat).zfill(2) + '.h5'

    mf.kpts = kpts
    
    # scf solver
    #mf = mf.newton()
    mf.max_cycle = 500

    mf = scf.addons.smearing_(mf, sigma=smearing, method="fermi") 

    # initial guess
    ig = mf.get_init_guess()
    ig[1,:,:,:] = 0

    e = mf.kernel(dm0=ig)

    # save Fock matrix
    fock = mf.get_fock()
    np.save(savedir + '/' + 'uFock_' + str(nat).zfill(2) + '.npy', fock)

    #================ restricted ================
    if nat % 2 == 1:
        mf = scf.KRHF(cell).density_fit(auxbasis=ab)    
        mf.chkfile = savedir + '/rhf_'+str(nat).zfill(2) + '.chk'
        mf.with_df._cderi = savedir + '/cderi_' + str(nat).zfill(2) + '.h5'
        mf.kpts = kpts
        
        #mf = mf.newton()
        mf.max_cycle = 500
        mf = scf.addons.smearing_(mf, sigma=smearing, method="fermi") 

        e = mf.kernel()
        
        # save Fock matrix
        fock = mf.get_fock()
        np.save(savedir + '/' + 'rFock_' + str(nat).zfill(2) + '.npy', fock)



