import numpy as np
import os, h5py, argparse, copy

from pyscf.pbc import gto, df, scf
from pyscf.pbc.lib import chkfile

from libdmet_solid.system import lattice
from libdmet.utils import plot
from libdmet_solid.basis_transform import make_basis, eri_transform
from libdmet_solid.lo.iao import reference_mol

from pyscf.scf.hf import eig as eiggen

import matplotlib.pyplot as plt

# switch to 'production' for serious jobs
mode = 'TEST'

############################################################
#                       basis
############################################################
imp_atom = 'IMP_ATOM' if mode == 'production' else 'Co'

imp_basis = 'def2-svp'
Cu_basis = 'def2-svp-bracket'

############################################################
#           directory for saving/loading data
############################################################
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default = 'data', type = str)
args = parser.parse_args()

datadir = args.datadir + '/'
print('data will be saved to:', datadir)

if not os.path.exists(datadir):
    os.mkdir(datadir)

############################################################
#                       build cell
############################################################
# transport direction fcc 100 plane
# an example of nl=4, nr=3
# |   5        4        5      4     1      4      5       4      |   5
# |-l-1.5a   -l-a   -l-0.5a   -l   0(imp)   r    r+0.5a   r+a     | r+1.5a

# x-distance between impurity atom and nearest Cu plane
l = LEFT if mode == 'production' else 1.8
r = RIGHT if mode == 'production' else 1.8

# number of layers in the left/right lead included in contact SCF calculation
nl = NUM_LEFT if mode == 'production' else 4
nr = NUM_RIGHT if mode == 'production' else 3

# ensure that supercell periodic boundary is properly matched
assert((nl+nr)%2 == 1)

# Cu lattice constant (for the fcc cell)
a = LATCONST if mode == 'production' else 3.6

cell_label = imp_atom + '_' + imp_basis + '_Cu_' + Cu_basis \
        + '_nl' + str(nl) + '_nr' + str(nr) \
        + '_l' + str(l) + '_r' + str(r) + '_a' + str(a)

cell_fname = datadir + '/cell_' + cell_label + '.chk'

if os.path.isfile(cell_fname):

    cell = chkfile.load_cell(cell_fname)
    print('use saved cell file:', cell_fname)

else:
    
    cell = gto.Cell()
    
    cell.unit = 'angstrom'
    cell.verbose = 4
    cell.max_memory = 50000
    cell.dimension = 3

    cell.precision = 1e-10
    #cell.ke_cutoff = 200

    # should not discard! 4s and 5s have small exponent
    #cell.exp_to_discard = 0.1

    cell.a = [[l+r+(nl+nr-1)*0.5*a,0,0], [0,10,0], [0,0,10]]
    
    cell.atom.append([imp_atom, (0, 0, 0)])

    # left lead
    for i in range(0, nl):
        x = -l - (nl-1-i) * 0.5*a
        if (i%2 == 0 and nl%2 == 1) or (i%2 == 1 and nl%2 == 0): # 4-atom layer
            cell.atom.append(['Cu', (x,      0,  0.5*a)])
            cell.atom.append(['Cu', (x,      0, -0.5*a)])
            cell.atom.append(['Cu', (x,  0.5*a,      0)])
            cell.atom.append(['Cu', (x, -0.5*a,      0)])
        else: # 5-atom layer
            cell.atom.append(['Cu', (x,      0,      0)])
            cell.atom.append(['Cu', (x,  0.5*a,  0.5*a)])
            cell.atom.append(['Cu', (x, -0.5*a,  0.5*a)])
            cell.atom.append(['Cu', (x,  0.5*a, -0.5*a)])
            cell.atom.append(['Cu', (x, -0.5*a, -0.5*a)])

    # right lead
    for i in range(0, nr):
        x = i*0.5*a + r
        if i%2 == 0: # 4-atom layer
            cell.atom.append(['Cu', (x,      0,  0.5*a)])
            cell.atom.append(['Cu', (x,      0, -0.5*a)])
            cell.atom.append(['Cu', (x,  0.5*a,      0)])
            cell.atom.append(['Cu', (x, -0.5*a,      0)])
        else: # 5-atom layer
            cell.atom.append(['Cu', (x,      0,      0)])
            cell.atom.append(['Cu', (x,  0.5*a,  0.5*a)])
            cell.atom.append(['Cu', (x, -0.5*a,  0.5*a)])
            cell.atom.append(['Cu', (x,  0.5*a, -0.5*a)])
            cell.atom.append(['Cu', (x, -0.5*a, -0.5*a)])

    cell.spin = None # let build() decides
    cell.basis = {imp_atom : imp_basis, 'Cu' : Cu_basis}
    
    cell.build()
    
    # save cell
    chkfile.save_cell(cell, cell_fname)

nat_Cu = len(cell.atom) - 1

nelec = nat_Cu * 29

if imp_atom == 'Fe':
    nelec += 26
elif imp_atom == 'Co':
    nelec += 27
elif imp_atom == 'Ni':
    nelec += 28
else:
    print('do not know nelec of imp_atom')
    exit()

# currently only use restricted calculation
assert(nelec%2==0)
###############
'''
# plot cell
atoms = cell.atom
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for atm in atoms:
    coor = atm[1]
    cl = 'r' if atm[0] != 'Cu' else 'b'
    ax.scatter(coor[0], coor[1], coor[2], color=cl)

plt.show()

exit()
'''
###############

############################################################
#                   gate voltage
############################################################
gate = GATE if mode == 'production' else 0
gate_label = 'gate%5.3f'%(gate)

############################################################
#                   mean-field method
############################################################
# This is the method used to generate the density matrix;

# For HF+DMFT, the Fock matrix for building the impurity Hamiltonian
# will use HF veff from the previous DM anyway (even though it's a KS-converged DM)

# if False, use HF instead
use_dft = USE_DFT if mode == 'production' else True

# if use_dft
xcfun = 'XCFUN' if mode == 'production' else 'pbe0'

do_restricted = DO_RESTRICTED if mode == 'production' else True

# TODO may add support for ROHF/ROKS in the future
assert( (not do_restricted) or (cell.spin==0) )

if do_restricted:
    method_label = 'r'
else:
    method_label = 'u'

if use_dft:
    method_label += 'ks' + '_' + xcfun
else:
    method_label += 'hf'


############################################################
#               density fitting
############################################################

gdf_fname = datadir + '/cderi_' + cell_label + '.h5'

gdf = df.GDF(cell)

if os.path.isfile(gdf_fname):
    print('use saved gdf cderi:', gdf_fname)
    gdf._cderi = gdf_fname
else:
    gdf._cderi_to_save = gdf_fname
    gdf.auxbasis = {imp_atom : imp_basis + '-ri', 'Cu' : Cu_basis + '-ri'}
    gdf.build()

############################################################
#                   mean-field object
############################################################

if use_dft:
    if do_restricted:
        mf = scf.RKS(cell).density_fit()
    else:
        mf = scf.UKS(cell).density_fit()
    mf.xc = xcfun
else:
    if do_restricted:
        mf = scf.RHF(cell).density_fit()
    else:
        mf = scf.UHF(cell).density_fit()

mf.with_df = gdf

############################################################
#                   apply gate voltage
############################################################
nao_imp = cell.aoslice_by_atom()[0][3]
print('number of impurity AO orbitals = ', nao_imp)

ref_fname = datadir + '/ref_contact_' + cell_label + '.h5'
if os.path.isfile(ref_fname):
    fh = h5py.File(ref_fname, 'r')
    h1 = np.asarray(fh['h1'])
    ovlp = np.asarray(fh['ovlp'])
else:
    h1 = mf.get_hcore()
    ovlp = mf.get_ovlp()
    fh = h5py.File(ref_fname, 'w')
    fh['h1'] = h1
    fh['ovlp'] = ovlp

fh.close()

# h -> h + \sum_{\mu,\nu \in imp}|\mu> inv(S_imp)_{\mu\nu} <\nu|
h1 += gate * ovlp[:,0:nao_imp] @ np.linalg.solve(ovlp[0:nao_imp, 0:nao_imp], ovlp[0:nao_imp,:])

mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: ovlp

############################################################
#           low-level mean-field calculation
############################################################

mf_save_fname = datadir + '/' + cell_label + '_' + method_label + '_' + gate_label + '.chk'

mf_load_fname = 'MF_LOAD_FNAME' if mode == 'production' else None

mf_data = chkfile.load(mf_load_fname, 'scf')
print('load saved mf data', mf_load_fname)
mf.__dict__.update(mf_data)

nocc = nelec // 2

print('E(homo) = %6.3f, occ(homo) = %6.3f'%(mf.mo_energy[nocc-1], mf.mo_occ[nocc-1]))
print('E(lumo) = %6.3f, occ(lumo) = %6.3f'%(mf.mo_energy[nocc], mf.mo_occ[nocc]))



