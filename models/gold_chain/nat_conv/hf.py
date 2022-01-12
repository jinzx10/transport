#######################################################################
# This script performs Gamma point HF calculations
# for 1-D periodic gold atomic chains of variable length
#######################################################################

from pyscf.pbc import gto, scf, df
from pyscf.pbc.lib import chkfile
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--savedir', default = 'data', type = str)
parser.add_argument('--spacing', default = 2.9, type = float)
parser.add_argument('--natmin', default = 2, type = int)
parser.add_argument('--natmax', default = 20, type = int)
parser.add_argument('--ke_cutoff', default = 200, type = int)
parser.add_argument('--exp_to_discard', default = 0.1, type = float)
parser.add_argument('--max_memory', default = 32000, type = int)
parser.add_argument('--df_beta', default = 2.0, type = float)
parser.add_argument('--smearing', default = 0.01, type = float)
args = parser.parse_args()

savedir = args.savedir
a = args.spacing
natmin = args.natmin
natmax = args.natmax
ke_cutoff = args.ke_cutoff
exp_to_discard = args.exp_to_discard
max_memory = args.max_memory
df_beta = args.df_beta
smearing = args.smearing

print('data will be saved to ', savedir)
print('spacing between gold atoms = ', a)
print('minimum number of atoms = ', natmin)
print('maximum number of atoms = ', natmax)
print('kinetic energy cutoff = ', ke_cutoff)
print('GTO exponent threshold = ', exp_to_discard)
print('cell max memory = ', max_memory, 'M')
print('density fitting beta = ', df_beta)
print('smearing = ', smearing)

# keep an even number of atoms in the unit cell
for nat in range(natmin, natmax+1, 2):

    cell = gto.Cell()

    for iat in range(0, nat):
        cell.atom.append(['Au', (iat*a, 0, 0)])

    cell.exp_to_discard = exp_to_discard
    cell.ke_cutoff = ke_cutoff
    cell.max_memory = max_memory

    cell.build(
            unit = 'angstrom',
            a = [[a*nat,0,0],[0,30,0],[0,0,30]],
            dimension = 3,
            basis = 'def2-svp',
            ecp = 'def2-svp',
            verbose = 4,
    )

    # save cell
    chkfile.save_cell(cell, savedir + '/au_' + str(nat).zfill(2) + '.chk')

    # auxbasis for density fitting
    ab = df.aug_etb(cell, beta=df_beta)

    #================ unrestricted ================
    mf = scf.UHF(cell).density_fit(auxbasis = ab)

    # data file
    mf.chkfile = savedir + '/uhf_' + str(nat).zfill(2) + '.chk'
    mf.with_df._cderi_to_save = savedir + '/cderi_' + str(nat).zfill(2) + '.h5'

    # scf solver
    #mf = mf.newton()
    mf.max_cycle = 500

    mf = scf.addons.smearing_(mf, sigma=smearing, method="fermi") 

    # initial guess
    ig = mf.get_init_guess()
    ig[1,:,:] = 0

    e = mf.kernel(dm0=ig)

    print('nat = ', nat, '   uhf energy per atom = ', e/nat)

    # save Fock matrix
    fock = mf.get_fock()
    np.save(savedir + '/' + 'uFock_' + str(nat).zfill(2) + '.npy', fock)

    #================ restricted ================
    mf = scf.RHF(cell).density_fit(auxbasis = ab)

    # data file
    mf.chkfile = savedir + '/rhf_' +str(nat).zfill(2) + '.chk'
    #mf.with_df._cderi_to_save = savedir + '/rhf_cderi_' + str(nat).zfill(2) + '.h5'

    # use previously calculated density fitting tensor
    mf.with_df._cderi = savedir + '/cderi_' + str(nat).zfill(2) + '.h5'

    # scf solver
    #mf = mf.newton()
    mf.max_cycle = 500

    mf = scf.addons.smearing_(mf, sigma=smearing, method="fermi") 

    e = mf.kernel()

    print('nat = ', nat, '   rhf energy per atom = ', e/nat)

    # save Fock matrix
    fock = mf.get_fock()
    np.save(savedir + '/' + 'rFock_' + str(nat).zfill(2) + '.npy', fock)


