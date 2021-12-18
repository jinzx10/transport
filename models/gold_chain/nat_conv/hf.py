#######################################################################
# This script performs Gamma point HF calculations
# for 1-D periodic gold atomic chains of variable length
#######################################################################

from pyscf.pbc import gto, scf, df
from pyscf.pbc.lib import chkfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--savedir', default = 'data', type = str)
parser.add_argument('--spacing', default = 2.9, type = float)
parser.add_argument('--natmin', default = 2, type = int)
parser.add_argument('--natmax', default = 20, type = int)
parser.add_argument('--ke_cutoff', default = 200, type = int)
parser.add_argument('--exp_to_discard', default = 0.1, type = float)
parser.add_argument('--max_memory', default = 10000, type = int)
args = parser.parse_args()

savedir = args.savedir
a = args.spacing
natmin = args.natmin
natmax = args.natmax
ke_cutoff = args.ke_cutoff
exp_to_discard = args.exp_to_discard
max_memory = args.max_memory

print('data will be saved to ', savedir)
print('spacing between gold atoms = ', a)
print('minimum number of atoms = ', natmin)
print('maximum number of atoms = ', natmax)
print('kinetic energy cutoff = ', ke_cutoff)
print('GTO exponent threshold = ', exp_to_discard)
print('cell max memory = ', max_memory, 'M')

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
            a = [[a*nat,0,0],[0,a,0],[0,0,a]],
            dimension = 1,
            basis = 'def2-svp',
            ecp = 'def2-svp',
            verbose = 4,
            )

    # save cell
    chkfile.save_cell(cell, savedir + '/au_' + str(nat).zfill(2) + '.chk')

    # auxbasis for density fitting
    ab = df.aug_etb(cell, beta=2.0)

    #================ unrestricted ================
    mf = scf.UHF(cell).density_fit(auxbasis = ab)

    # data file
    mf.chkfile = savedir + '/uhf_' + str(nat).zfill(2) + '.chk'
    mf.with_df._cderi_to_save = savedir + '/uhf_cderi_' + str(nat).zfill(2) + '.h5'

    # scf solver
    mf = mf.newton()
    mf.max_cycle = 500

    # initial guess
    ig = mf.get_init_guess()
    ig[1,:,:] = 0

    e = mf.kernel(dm0=ig)

    print('nat = ', nat, '   uhf energy per atom = ', e/nat)

    #================ restricted ================
    mf = scf.RHF(cell).density_fit(auxbasis = ab)

    # data file
    mf.chkfile = savedir + '/rhf_' +str(nat).zfill(2) + '.chk'
    mf.with_df._cderi_to_save = savedir + '/rhf_cderi_' + str(nat).zfill(2) + '.h5'

    # scf solver
    mf = mf.newton()
    mf.max_cycle = 500

    e = mf.kernel()

    print('nat = ', nat, '   rhf energy per atom = ', e/nat)


