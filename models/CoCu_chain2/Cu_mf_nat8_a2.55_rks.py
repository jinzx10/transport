############################################################
# This script performs a Gamma point mean-field calculation
# for a 1-D copper atomic chain
############################################################

from pyscf.pbc import gto, scf, df
from pyscf.pbc.lib import chkfile
import os, h5py, argparse, copy
import numpy as np

from libdmet_solid.system import lattice
from libdmet.utils import plot
from libdmet_solid.basis_transform import make_basis
from libdmet_solid.lo.iao import reference_mol

############################################################
#                       basis
############################################################
Cu_basis = 'def2-svp-bracket'
ncore_Cu = 9
nval_Cu = 6
nvirt_Cu = 9

nao_Cu = nval_Cu + nvirt_Cu # cores are discarded!

############################################################
#           directory for saving/loading data
############################################################
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default = 'data', type = str)

args = parser.parse_args()

if args.datadir is None:
    datadir = 'Cu_' + Cu_basis + '/'
else:
    datadir = args.datadir + '/'

print('data directory:', datadir)

if not os.path.exists(datadir):
    os.mkdir(datadir)

############################################################
#               low-level mean-field method
############################################################
# This is the method used to generate the density matrix;
# The Fock matrix for building the impurity Hamiltonian
# will use HF anyway.

# if False, use HF instead
use_pbe = True

if use_pbe:
    method_label = 'rks'
else:
    method_label = 'rhf'

############################################################
#                   build cell
############################################################
# total number of Cu atoms
nat = 8
assert(nat%2 == 0)


# atomic spacing
a = 2.55

cell_label = 'Cu_' + 'nat' + str(nat).zfill(2) + '_a' + str(a)

cell_fname = datadir + '/cell_' + cell_label + '.chk'

if os.path.isfile(cell_fname):
    cell = chkfile.load_cell(cell_fname)
else:
    
    cell = gto.Cell()
    
    cell.unit = 'angstrom'
    cell.verbose = 4
    cell.max_memory = 30000
    cell.dimension = 3
    cell.ke_cutoff = 500
    cell.a = [[nat*a,0,0], [0,20,0], [0,0,20]]
    
    for iat in range(0, nat):
        cell.atom.append(['Cu', (iat*a, 0, 0)])
    
    cell.spin = 0
    cell.basis = Cu_basis
    
    cell.build()
    
    # save cell
    chkfile.save_cell(cell, cell_fname)


############################################################
#               build Lattice object
############################################################
kmesh = [1,1,1]
kpts = cell.make_kpts(kmesh)
Lat = lattice.Lattice(cell, kmesh) 
nao = Lat.nao
nkpts = Lat.nkpts

############################################################
#               density fitting
############################################################
gdf_fname = datadir + '/cderi_' + cell_label + '.h5'
gdf = df.GDF(cell, kpts)

if os.path.isfile(gdf_fname):
    gdf._cderi = gdf_fname
else:
    gdf._cderi_to_save = gdf_fname
    gdf.auxbasis = Cu_basis + '-ri'
    gdf.build()

############################################################
#           mean-field calculation
############################################################

# if False, the scf will use the Newton solver for convergence
use_smearing = False
smearing_sigma = 0.05


if use_pbe:
    kmf = scf.KRKS(cell, kpts).density_fit()
    kmf.xc = 'pbe'
else:
    kmf = scf.KRHF(cell, kpts).density_fit()

mf_fname = datadir + '/' + cell_label + '_' + method_label + '_' + '.chk'

kmf.with_df = gdf

if use_smearing:
    kmf = scf.addons.smearing_(kmf, sigma = smearing_sigma, method = 'fermi')
else:
    kmf = kmf.newton()

if os.path.isfile(mf_fname):
    kmf.__dict__.update( chkfile.load(mf_fname, 'scf') )
else:
    kmf.conv_tol = 1e-12
    kmf.max_cycle = 200
    kmf.chkfile = mf_fname
    kmf.kernel()

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

    plot.plot_orb_k_all(cell, plotdir + '/iao_' + str(nat).zfill(2), C_ao_iao, kpts, margin=0.0)
    plot.plot_orb_k_all(cell, plotdir + '/lo_' + str(nat).zfill(2), C_ao_lo, kpts, margin=0.0)

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


