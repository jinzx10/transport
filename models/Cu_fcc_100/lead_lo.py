import numpy as np
import os, h5py, argparse, copy

from pyscf.pbc import gto, scf, df
from pyscf.pbc.lib import chkfile

from libdmet_solid.system import lattice
from libdmet.utils import plot
from libdmet_solid.basis_transform import make_basis
from libdmet_solid.lo.iao import reference_mol

############################################################
#                       basic info
############################################################
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default = 'data', type = str)
args = parser.parse_args()

datadir = args.datadir + '/'
print('data will be read from', datadir)

Cu_basis = 'def2-svp-bracket'

############################################################
#               low-level mean-field method
############################################################
# This is the method used to generate the density matrix;
# The Fock matrix for building the impurity Hamiltonian
# will use HF anyway.

use_dft = USE_DFT

# if use_dft
xcfun = 'XCFUN'

do_restricted = DO_RESTRICTED

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
use_smearing = USE_SMEARING
smearing_sigma = SMEARING_SIGMA

if use_smearing:
    solver_label = 'smearing' + str(smearing_sigma)
else:
    solver_label = 'newton'

# Cu lattice constant (for the fcc cell)
a = LATCONST

if Cu_basis == 'def2-svp-bracket':
    ncore_Cu = 9
    nval_Cu = 6
    nvirt_Cu = 9

nao_Cu = nval_Cu + nvirt_Cu # cores are discarded!
nao_Cu_tot = nao_Cu + ncore_Cu

############################################################
#                   load cell
############################################################

cell_label = 'Cu_' + Cu_basis + '_a' + str(a)
cell_fname = datadir + '/cell_' + cell_label + '.chk'

cell = chkfile.load_cell(cell_fname)
print('use saved cell file:', cell_fname)

############################################################
#               density fitting
############################################################
gdf_fname = datadir + '/cderi_' + cell_label + '.h5'
gdf = df.GDF(cell)

print('use saved gdf cderi:', gdf_fname)
gdf._cderi = gdf_fname

############################################################
#           mean-field calculation
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

print('load previous mf data', mf_fname)
mf.__dict__.update( chkfile.load(mf_fname, 'scf') )
mf.conv_tol = 1e-10
mf.max_cycle = 50
mf.kernel()

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
kmf.mo_coeff = mf.mo_coeff[np.newaxis,...]
kmf.mo_energy = mf.mo_energy[np.newaxis,...]
kmf.mo_occ = mf.mo_occ[np.newaxis,...]

# make sure restricted/unrestricted quantities have the same dimension
# and set spin: unrestricted -> 2; restricted -> 1
if len(mo_energy.shape) == 2:
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

############################################################
#               rearrange orbitals
############################################################
# rearrange IAO/PAO according to their positions

if use_reference_mol:
    # C_ao_lo: transformation matrix from AO to LO (IAO) basis, core orbitals excluded
    C_ao_lo   = np.zeros((nkpts,nao,nval+nvirt), dtype=C_ao_iao.dtype)

    for iat in range(nat):
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

    plot.plot_orb_k_all(cell, plotdir + '/iao_', C_ao_iao, kpts, margin=0.0)
    plot.plot_orb_k_all(cell, plotdir + '/lo_', C_ao_lo, kpts, margin=0.0)

############################################################
#           Quantities in LO (IAO) basis
############################################################
hcore_ao = np.asarray(kmf.get_hcore())
JK_ao = np.asarray(kmf.get_veff())

hcore_lo = np.zeros((nkpts,nval+nvirt,nval+nvirt), dtype=hcore_ao.dtype)
JK_lo = np.zeros((nkpts,nval+nvirt,nval+nvirt), dtype=JK_ao.dtype)

for ik in range(nkpts):
    hcore_lo[ik] = np.dot(np.dot(C_ao_lo[ik].T.conj(), hcore_ao[ik]), C_ao_lo[ik])
    JK_lo[ik] = np.dot(np.dot(C_ao_lo[ik].T.conj(), JK_ao[ik]), C_ao_lo[ik])

fn = datadir + '/hcore_JK_lo_' + cell_label + '_' + method_label + '.h5'

fh = h5py.File(fn, 'w')
fh['hcore_lo'] = hcore_lo
fh['JK_lo'] = JK_lo
fh.close()

print('finished')


