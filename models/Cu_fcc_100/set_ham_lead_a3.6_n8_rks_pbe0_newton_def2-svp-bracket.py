import numpy as np
import os, h5py, argparse, copy

from pyscf.pbc import gto, df, scf
from pyscf.pbc.lib import chkfile

from libdmet_solid.system import lattice
from libdmet.utils import plot
from libdmet_solid.basis_transform import make_basis
from libdmet_solid.lo.iao import reference_mol

from pyscf.scf.hf import eig as eiggen

import matplotlib.pyplot as plt

# switch to 'production' for serious jobs
mode = 'production'
############################################################
#                       basis
############################################################
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
# |   5        4        5      4   |  5
# |   0       0.5a      a     1.5a | 2a

# Cu lattice constant (for the fcc cell)
a = 3.6 if mode == 'production' else 3.6

# number of layers
num_layer = 8 if mode == 'production' else 8

cell_label = 'Cu_' + Cu_basis + '_a' + str(a) + '_n' + str(num_layer)

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

    cell.a = [[num_layer/2*a,0,0], [0,10,0], [0,0,10]]
    
    for i in range(0,num_layer):
        x = i*0.5*a
        if i%2 == 0:
            # 5-atom layer
            cell.atom.append(['Cu', (x,      0,      0)])
            cell.atom.append(['Cu', (x,  0.5*a,  0.5*a)])
            cell.atom.append(['Cu', (x, -0.5*a,  0.5*a)])
            cell.atom.append(['Cu', (x,  0.5*a, -0.5*a)])
            cell.atom.append(['Cu', (x, -0.5*a, -0.5*a)])
        else:
            # 4-atom layer
            cell.atom.append(['Cu', (x,      0,  0.5*a)])
            cell.atom.append(['Cu', (x,      0, -0.5*a)])
            cell.atom.append(['Cu', (x,  0.5*a,      0)])
            cell.atom.append(['Cu', (x, -0.5*a,      0)])

    cell.spin = 0
    cell.basis = {'Cu' : Cu_basis}
    
    cell.build()
    
    # save cell
    chkfile.save_cell(cell, cell_fname)

nat_Cu = len(cell.atom)

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
#                   mean-field method
############################################################
# This is the method used to generate the density matrix;

# For HF+DMFT, the Fock matrix for building the impurity Hamiltonian
# will use HF veff from the previous DM anyway (even though it's a KS-converged DM)

# if False, use HF instead
use_dft = True if mode == 'production' else True

# if use_dft
xcfun = 'pbe0' if mode == 'production' else 'pbe0'

do_restricted = True if mode == 'production' else True

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

# scf solver & addons
# if use_smearing, use Fermi smearing
# if False, scf will use newton solver to help convergence
use_smearing = False if mode == 'production' else True
smearing_sigma = 0 if mode == 'production' else 0.05

if use_smearing:
    solver_label = 'smearing' + str(smearing_sigma)
else:
    solver_label = 'newton'


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
    gdf.auxbasis = {'Cu' : Cu_basis + '-ri'}
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
else:
    mf = mf.newton()
    mf.canonicalization = False

if os.path.isfile(mf_fname):
    print('load saved mf data', mf_fname)
    mf_data = chkfile.load(mf_fname, 'scf')
    mf.__dict__.update(mf_data)

    mf.chkfile = mf_fname
    mf.conv_tol = 1e-10
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
    
    DM_ao = mf.make_rdm1()
    dm_fock = (v * mf.mo_occ) @ v.T
    print('sanity check: dm diff between make_rdm1 and fock-solved = ', np.linalg.norm(dm_fock-DM_ao))

###############################################
#       convergence sanity check end
###############################################

###############################################
#       Number of AOs
###############################################
if Cu_basis == 'def2-svp-bracket':
    ncore_Cu = 9
    nval_Cu = 6
    nvirt_Cu = 9

nao_Cu = nval_Cu + nvirt_Cu # cores are discarded!
nao_Cu_tot = nao_Cu + ncore_Cu

############################################################
#               build Lattice object
############################################################
kmesh = [1,1,1]
kpts = cell.make_kpts(kmesh)
Lat = lattice.Lattice(cell, kmesh) 
nao = Lat.nao
nkpts = Lat.nkpts

############################################################
#               generate a kmf object
############################################################
if use_dft:
    if do_restricted:
        kmf = scf.KRKS(cell, kpts).density_fit()
    else:
        kmf = scf.KUKS(cell, kpts).density_fit()
    kmf.xc = xcfun
else:
    if do_restricted:
        kmf = scf.KRHF(cell, kpts).density_fit()
    else:
        kmf = scf.KUHF(cell, kpts).density_fit()

kmf.with_df = gdf

# add a k axis for subsequent IAO/PAO construction
# keep spin as the first dimension
if do_restricted:
    kmf.mo_coeff = mf.mo_coeff[np.newaxis,...]
    kmf.mo_energy = mf.mo_energy[np.newaxis,...]
    kmf.mo_occ = mf.mo_occ[np.newaxis,...]
else:
    kmf.mo_coeff = mf.mo_coeff[:,np.newaxis,...]
    kmf.mo_energy = mf.mo_energy[:,np.newaxis,...]
    kmf.mo_occ = mf.mo_occ[:,np.newaxis,...]

# spin: unrestricted -> 2; restricted -> 1
if len(mf.mo_energy.shape) == 1:
    spin = 1
else:
    spin = 2

############################################################
#           Orbital Localization
############################################################

MINAO = Cu_basis + '-minao'

# if True, use reference core/val
use_reference_mol = True

if use_reference_mol:

    # set IAO (core+val)
    pmol = reference_mol(cell, minao=MINAO)
    
    # set valence IAO
    pmol_val = pmol.copy()
    pmol_val.basis = Cu_basis + '-minao-val'
    pmol_val.build()
    basis_val = {}
    basis_val["Cu"] = copy.deepcopy(pmol_val._basis["Cu"])
    
    pmol_val = pmol.copy()
    pmol_val.basis = basis_val
    pmol_val.build()
    
    val_labels = pmol_val.ao_labels()
    for i in range(len(val_labels)):
        val_labels[i] = val_labels[i].replace("Cu 1s", "Cu 4s")
    pmol_val.ao_labels = lambda *args: val_labels
    
    # set core IAO
    pmol_core = pmol.copy()
    pmol_core.basis = Cu_basis + '-minao-core'
    pmol_core.build()
    basis_core = {}
    basis_core["Cu"] = copy.deepcopy(pmol_core._basis["Cu"])
    
    pmol_core = pmol.copy()
    pmol_core.basis = basis_core
    pmol_core.build()
    core_labels = pmol_core.ao_labels()
    
    ncore = len(pmol_core.ao_labels())
    nval = pmol_val.nao_nr()
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)

    print('ncore = ', ncore)
    print('nval = ', nval)
    print('nvirt = ', nvirt)
    
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt, C_ao_iao_core \
            = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=MINAO, full_return=True, \
            pmol_val=pmol_val, pmol_core=pmol_core, tol=1e-8)
else:
    # if core/val is not specified, the number of val will equal the size of MINAO
    # and the rest are virt
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=MINAO, full_return=True)

#TODO add unrestricted support in the future
assert(spin == 1)
############################################################
#               rearrange orbitals
############################################################
# rearrange IAO/PAO according to their positions

if use_reference_mol:
    # C_ao_lo: transformation matrix from AO to LO (IAO) basis, core orbitals excluded
    C_ao_lo   = np.zeros((nkpts,nao,nval+nvirt), dtype=C_ao_iao.dtype)

    for iat in range(nat_Cu):
        C_ao_lo[:,:,iat*nao_Cu:iat*nao_Cu+nval_Cu] = C_ao_iao_val[:,:,iat*nval_Cu:(iat+1)*nval_Cu]
        C_ao_lo[:,:,iat*nao_Cu+nval_Cu:(iat+1)*nao_Cu] = C_ao_iao_virt[:,:,iat*nvirt_Cu:(iat+1)*nvirt_Cu]

else:
    # TBD...
    print('rearranging orbitals not implemented!')

if np.max(np.abs(C_ao_lo.imag)) < 1e-8:
    C_ao_lo = C_ao_lo.real

############################################################
#               Plot MO and LO
############################################################
plot_orb = True

if plot_orb:
    plotdir = datadir + '/plot_' + cell_label + '_' + method_label + '/'
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)

    plot.plot_orb_k_all(cell, plotdir + '/lo', C_ao_lo, kpts, margin=0.0)
    plot.plot_orb_k_all(cell, plotdir + '/mo', kmf.mo_coeff, kpts, margin=0.0)

############################################################
#           Quantities in LO (IAO) basis
############################################################
hcore_ao = np.asarray(kmf.get_hcore())
JK_ao = np.asarray(kmf.get_veff())

hcore_lo = np.zeros((nkpts,nval+nvirt,nval+nvirt), dtype=hcore_ao.dtype)
JK_lo    = np.zeros((nkpts,nval+nvirt,nval+nvirt), dtype=JK_ao.dtype)

for ik in range(nkpts):
    hcore_lo[ik] = np.dot(np.dot(C_ao_lo[ik].T.conj(), hcore_ao[ik]), C_ao_lo[ik])
    JK_lo[ik] = np.dot(np.dot(C_ao_lo[ik].T.conj(), JK_ao[ik]), C_ao_lo[ik])

fn = datadir + '/data_lead_' + cell_label + '_' + method_label + '.h5'

fh = h5py.File(fn, 'w')
fh['hcore_lo'] = hcore_lo
fh['JK_lo'] = JK_lo
fh.close()

print('finished')







