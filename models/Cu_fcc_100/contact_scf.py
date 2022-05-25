#!/usr/bin/python
import numpy as np
import os, argparse
import matplotlib.pyplot as plt

from pyscf.pbc import gto, df, scf
from pyscf.pbc.lib import chkfile

from pyscf.scf.hf import eig as eiggen


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

if do_restricted:
    method_label = 'r'
else:
    method_label = 'u'

if use_dft:
    method_label += 'ks' + '_' + xcfun
else:
    method_label += 'hf'

# scf solver & addons
# if use_smearing, use Fermi smearing
# if False, scf will use newton solver to help convergence
use_smearing = USE_SMEARING if mode == 'production' else False 
smearing_sigma = SMEARING_SIGMA if mode == 'production' else 0.05

if use_smearing:
    solver_label = 'smearing' + str(smearing_sigma)
else:
    solver_label = 'newton'

############################################################
#                       build cell
############################################################
# transport direction fcc 100 plane
# |   5        4        5      4     1      4      5       4      |   5
# |-l-1.5a   -l-a   -l-0.5a   -l   0(imp)   r    r+0.5a   r+a     | r+1.5a

# x-distance between impurity atom and nearest Cu plane
l = LEFT if mode == 'production' else 1.8
r = RIGHT if mode == 'production' else 1.8

# Cu lattice constant (for the fcc cell)
a = LATCONST if mode == 'production' else 3.6

cell_label = imp_atom + '_' + imp_basis + '_Cu_' + Cu_basis \
        + '_' + str(l) + '_r' + str(r) + '_a' + str(a)

cell_fname = datadir + '/cell_' + cell_label + '.chk'

if os.path.isfile(cell_fname):
    cell = chkfile.load_cell(cell_fname)
    print('use saved cell file:', cell_fname)
else:
    
    cell = gto.Cell()
    
    cell.unit = 'angstrom'
    cell.verbose = 4
    cell.max_memory = 100000
    cell.dimension = 3

    cell.precision = 1e-10
    #cell.ke_cutoff = 200

    # should not discard! 4s and 5s have small exponent
    #cell.exp_to_discard = 0.1

    cell.a = [[l+r+3*a,0,0], [0,10,0], [0,0,10]]
    
    cell.atom.append([imp_atom, (0, 0, 0)])

    # 5-atom layer
    xl5 = [-l-1.5*a, -l-0.5*a, r+0.5*a]
    for x in xl5:
        cell.atom.append(['Cu', (x, 0, 0)])
        cell.atom.append(['Cu', (x,  0.5*a,  0.5*a)])
        cell.atom.append(['Cu', (x, -0.5*a,  0.5*a)])
        cell.atom.append(['Cu', (x,  0.5*a, -0.5*a)])
        cell.atom.append(['Cu', (x, -0.5*a, -0.5*a)])

    # 4-atom layer
    xl4 = [-l-a, -l, r, r+a]
    for x in xl4:
        cell.atom.append(['Cu', (x,      0,  0.5*a)])
        cell.atom.append(['Cu', (x,      0, -0.5*a)])
        cell.atom.append(['Cu', (x,  0.5*a,      0)])
        cell.atom.append(['Cu', (x, -0.5*a,      0)])

    cell.spin = 0
    cell.basis = {imp_atom : imp_basis, 'Cu' : Cu_basis}
    
    cell.build()
    
    # save cell
    chkfile.save_cell(cell, cell_fname)

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
#           low-level mean-field calculation
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

mf_fname = datadir + '/' + cell_label + '_' + method_label + '_' + solver_label + '.chk'

mf.with_df = gdf

if use_smearing:
    mf = scf.addons.smearing_(mf, sigma = smearing_sigma, method = 'fermi')
    mf.diis_space = 15
else:
    mf = mf.newton()

if os.path.isfile(mf_fname):
    print('load previous mf data', mf_fname)
    mf.__dict__.update( chkfile.load(mf_fname, 'scf') )
    mf.conv_tol = 1e-10
    mf.chkfile = mf_fname
    mf.max_cycle = 50
    mf.kernel()
else:
    mf.chkfile = mf_fname
    mf.conv_tol = 1e-10
    mf.max_cycle = 300

    mf.kernel()

###############################################
#       convergence sanity check begin
###############################################
if do_restricted:
    S_ao_ao = mf.get_ovlp()
    hcore_ao = mf.get_hcore()
    JK_ao = mf.get_veff()
    
    e, v = eiggen(hcore_ao+JK_ao, S_ao_ao)
    
    print('sanity check: mo_energy vs. fock eigenvalue = ', np.linalg.norm(mf.mo_energy-e))
    
    if nkpts == 1:
        DM_ao = mf.make_rdm1()
        dm_fock = (v * mo_occ[0]) @ v.T
        print('sanity check: dm diff between make_rdm1 and fock-solved = ', np.linalg.norm(dm_fock-DM_ao[0]))

###############################################
#       convergence sanity check end
###############################################

