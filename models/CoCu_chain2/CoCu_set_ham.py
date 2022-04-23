#!/usr/bin/python
import numpy as np
import h5py, os, copy, argparse
from libdmet_solid.system import lattice
from libdmet_solid.basis_transform import make_basis, eri_transform
from libdmet.utils import plot

from pyscf.pbc import gto, df, scf
from pyscf.pbc.lib import chkfile

from libdmet_solid.lo.iao import reference_mol

############################################################
#           directory for saving/loading data
############################################################
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default = 'data', type = str)

args = parser.parse_args()

if args.datadir is None:
    datadir = 'CoCu_set_ham_data/'
else:
    datadir = args.datadir + '/'

print('data directory:', datadir)

############################################################
#                       basis
############################################################
Co_basis = 'def2-svp'

# 1s,2s,2p,3s,3p
ncore_Co = 9

# 3d,4s
nval_Co = 6

# 4p,4d,5s
nvirt_Co = 16

nao_Co = nval_Co + nvirt_Co # core orbitals are ignored!

Cu_basis = 'def2-svp-bracket'

ncore_Cu = 9
nval_Cu = 6
nvirt_Cu = 9

nao_Cu = nval_Cu + nvirt_Cu

############################################################
#               low-level mean-field method
############################################################
# This is the method used to generate the density matrix;
# The Fock matrix for building the impurity Hamiltonian
# will use HF anyway.

# TODO: Tianyu & Linqing mentioned that the hybridization might
# be better computed by the DFT's Fock matrix

# if False, use HF instead
use_pbe = False

do_restricted = False

if use_pbe:
    if do_restricted:
        method_label = 'rks'
    else:
        method_label = 'uks'
else:
    if do_restricted:
        method_label = 'rhf'
    else:
        method_label = 'uhf'

############################################################
#                       build cell
############################################################
# total number of Cu atoms (left + right)
nat_Cu = 9
assert(nat_Cu%2 == 1)

# number of Cu atoms in the left/right lead
nl = nat_Cu // 2
nr = nat_Cu - nl

# distance between Co and the nearest Cu atoms
l = 4.0
r = 4.0
    
label = 'CoCu_' + str(nat_Cu).zfill(2) + '_l' + str(l) + '_r' + str(r)

cell_fname = datadir + '/cell_' + label + '.chk'

if os.path.isfile(cell_fname):
    cell = chkfile.load_cell(cell_fname)
else:
    # atomic spacing in the lead
    a = 2.55
    
    cell = gto.Cell()
    
    cell.unit = 'angstrom'
    cell.verbose = 4
    cell.max_memory = 30000
    cell.dimension = 3
    cell.ke_cutoff = 500
    cell.a = [[l+r+(nat_Cu-1)*a,0,0], [0,20,0], [0,0,20]]
    
    cell.atom.append(['Co', ((nl-1)*a+l, 0, 0)])

    for iat in range(0, nl):
        cell.atom.append(['Cu', (iat*a, 0, 0)])

    for iat in range(0, nr):
        cell.atom.append(['Cu', ((nl-1)*a+l+r+iat*a, 0, 0)])
    
    cell.spin = 0
    cell.basis = {'Co' : Co_basis, 'Cu' : Cu_basis}
    
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
gdf_fname = datadir + '/cderi_' + label + '.h5'

gdf = df.GDF(cell, kpts)

if os.path.isfile(gdf_fname):
    gdf._cderi = gdf_fname
else:
    gdf._cderi_to_save = gdf_fname
    gdf.auxbasis = {'Co' : Co_basis + '-ri', 'Cu' : Cu_basis + '-ri'}
    gdf.build()

############################################################
#           low-level mean-field calculation
############################################################
# if False, the scf will use newton solver to help convergence
use_smearing = False
smearing_sigma = 0.05

if do_restricted:
    if use_pbe:
        kmf = scf.KRKS(cell, kpts).density_fit()
        kmf.xc = 'pbe'
        mf_fname = datadir + '/rks_' + label + '.chk'
    else:
        kmf = scf.KRHF(cell, kpts).density_fit()
        mf_fname = datadir + '/rhf_' + label + '.chk'
else:
    if use_pbe:
        kmf = scf.KUKS(cell, kpts).density_fit()
        kmf.xc = 'pbe'
        mf_fname = datadir + '/uks_' + label + '.chk'
    else:
        kmf = scf.KUHF(cell, kpts).density_fit()
        mf_fname = datadir + '/uhf_' + label + '.chk'

kmf.with_df = gdf

if use_smearing:
    kmf = scf.addons.smearing_(kmf, sigma = smearing_sigma, method = 'fermi')
else:
    kmf = kmf.newton()

if os.path.isfile(mf_fname):
    kmf.__dict__.update( chkfile.load(mf_fname, 'scf') )
else:
    kmf.chkfile = mf_fname
    kmf.conv_tol = 1e-12
    kmf.max_cycle = 200
    kmf.kernel()

# make sure restricted/unrestricted quantities have the same dimension
# and set spin: unrestricted -> 2; restricted -> 1
mo_energy = np.asarray(kmf.mo_energy)
mo_coeff = np.asarray(kmf.mo_coeff)
if len(mo_energy.shape) == 2:
    spin = 1
    mo_energy = mo_energy[np.newaxis, ...]
    mo_coeff = mo_coeff[np.newaxis, ...]
else:
    spin = 2

############################################################
#               Orbital Localization (IAO)
############################################################

MINAO = {'Co': Co_basis + '-minao', 'Cu': Cu_basis + '-minao'}

# if True, use reference core/val
use_reference_mol = True

if use_reference_mol:

    # set IAO (core+val)
    pmol = reference_mol(cell, minao=MINAO)
    
    # set valence IAO
    pmol_val = pmol.copy()
    pmol_val.basis = {'Co': Co_basis + '-minao-val', 'Cu': Cu_basis + '-minao-val'}
    pmol_val.build()
    basis_val = {}
    basis_val["Co"] = copy.deepcopy(pmol_val._basis["Co"])
    basis_val["Cu"] = copy.deepcopy(pmol_val._basis["Cu"])
    
    pmol_val = pmol.copy()
    pmol_val.basis = basis_val
    pmol_val.build()
    
    val_labels = pmol_val.ao_labels()
    for i in range(len(val_labels)):
        val_labels[i] = val_labels[i].replace("Co 1s", "Co 4s")
        val_labels[i] = val_labels[i].replace("Cu 1s", "Cu 4s")
    pmol_val.ao_labels = lambda *args: val_labels
    
    # set core IAO
    pmol_core = pmol.copy()
    pmol_core.basis = {'Co': Co_basis + '-minao-core', 'Cu': Cu_basis + '-minao-core'}
    pmol_core.build()
    basis_core = {}
    basis_core["Co"] = copy.deepcopy(pmol_core._basis["Co"])
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
# rearrange IAO/PAO according to their indices in cell.atom

if use_reference_mol:
    # C_ao_lo: transformation matrix from AO to LO (IAO) basis, all atoms, core orbitals are excluded
    # C_ao_lo_Co: transformation matrix from AO to LO (IAO) basis, only Co atom, core orbitals are excluded
    C_ao_lo    = np.zeros((spin,nkpts,nao,nval+nvirt), dtype=C_ao_iao.dtype)
    C_ao_lo_Co = np.zeros((spin,nkpts,nao,nao_Co), dtype=C_ao_lo.dtype)
    
    C_ao_lo[:,:,:,0:nval_Co] = C_ao_iao_val[:,:,:,0:nval_Co]
    C_ao_lo[:,:,:,nval_Co:nao_Co] = C_ao_iao_virt[:,:,:,0:nvirt_Co]

    for iat in range(nat_Cu):
        C_ao_lo[:,:,:,nao_Co+iat*nao_Cu:nao_Co+iat*nao_Cu+nval_Cu] 
        = C_ao_iao_val[:,:,:,nval_Co+iat*nval_Cu:nval_Co+(iat+1):nval_Cu]

        C_ao_lo[:,:,:,nao_Co+iat*nao_Cu+nval_Cu:nao_Co+(iat+1)*nao_Cu]
        = C_ao_iao_virt[:,:,:,nvirt_Co+iat*nvirt_Cu:nvirt_Co+(iat+1)*nvirt_Cu]

    C_ao_lo_Co = C_ao_lo[:,:,:,0:nao_Co]

else:
    # TODO...
    print('rearranging orbitals not implemented!')


############################################################
#           Plot MO and LO
############################################################
plot_orb = True

if plot_orb:
    plotdir = datadir + '/plot_' + label
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)

    plot.plot_orb_k_all(cell, plotdir + '/iao_' + str(nat_Cu).zfill(2), C_ao_iao, kpts, margin=0.0)
    plot.plot_orb_k_all(cell, plotdir + '/lo_' + str(nat_Cu).zfill(2), C_ao_lo, kpts, margin=0.0)

############################################################
#           Quantities in LO (IAO) basis
############################################################
S_ao_ao = kmf.get_ovlp()
print('S_ao_ao.shape = ', S_ao_ao.shape)

fname = datadir + '/C_ao_lo_' + label + '_' + method_label + '.h5'
f = h5py.File(fname, 'w')
f['C_ao_lo'] = C_ao_lo
f['C_ao_lo_Co'] = C_ao_lo_Co
f['S_ao_ao'] = S_ao_ao
f.close()

# get density matrix in IAO basis
DM_ao = kmf.make_rdm1() # DM_ao is real!

if len(DM_ao.shape) == 3:
    DM_ao = DM_ao[np.newaxis,...]

DM_lo    = np.zeros((spin,nkpts,nval+nvirt,nval+nvirt),dtype=DM_ao.dtype)
DM_lo_Co = np.zeros((spin,nkpts,nao_Co,nao_Co),dtype=DM_ao.dtype)

for s in range(spin):
    for ik in range(nkpts):
        # C^\dagger*S*C=I
        Cinv = np.dot(C_ao_lo[s,ik].T.conj(),S_ao_ao[ik])
        DM_lo[s,ik] = np.dot(np.dot(Cinv, DM_ao[s,ik]), Cinv.T.conj())
    
        Cinv_Co = np.dot(C_ao_lo_Co[s,ik].T.conj(),S_ao_ao[ik])
        DM_lo_Co[s,ik] = np.dot(np.dot(Cinv_Co, DM_ao[s,ik]), Cinv_Co.T.conj())

for s in range(spin):
    nelec_lo = np.trace(DM_lo[s].sum(axis=0)/nkpts)
    print ('Nelec (core excluded)', nelec_lo.real)
    
    nelec_lo_Co = np.trace(DM_lo_Co[s].sum(axis=0)/nkpts)
    print ('Nelec on Co', nelec_lo_Co.real)

fn = datadir + '/DM_lo_' + label + '_' + method_label + '.h5'
f = h5py.File(fn, 'w')
f['DM_lo'] = DM_lo
f['DM_lo_Co'] = DM_lo_Co
f.close()

# get 4-index ERI
#eri_lo = eri_transform.get_unit_eri_fast(cell, gdf, C_ao_lo=C_ao_lo, feri=gdf_fname)
eri_lo_Co = eri_transform.get_unit_eri_fast(cell, gdf, C_ao_lo=C_ao_lo_Co, feri=gdf_fname)

fn = datadir + '/eri_lo_' + label + '_' + method_label + '.h5'
f = h5py.File(fn, 'w')
#f['eri_lo'] = eri_lo.real
f['eri_lo_Co'] = eri_lo_Co.real
f.close()

assert(np.max(np.abs(eri_lo_Co.imag))<1e-8)

# get one-electron integrals
hcore_ao = kmf.get_hcore() # hcore_ao and JK_ao are all real!
print('hcore_ao.shape = ', hcore_ao.shape)
JK_ao = kmf.get_veff()
if len(JK_ao.shape) == 3:
    JK_ao = JK_ao[np.newaxis, ...]

hcore_lo    = np.zeros((spin,nkpts,nval+nvirt,nval+nvirt),dtype=hcore_ao.dtype)
hcore_lo_Co = np.zeros((spin,nkpts,nao_Co    ,nao_Co    ),dtype=hcore_ao.dtype)

JK_lo    = np.zeros((spin,nkpts,nval+nvirt,nval+nvirt),dtype=JK_ao.dtype)
JK_lo_Co = np.zeros((spin,nkpts,nao_Co    ,nao_Co    ),dtype=JK_ao.dtype)

for s in range(spin):
    for ik in range(nkpts):
        hcore_lo[s,ik] = np.dot(np.dot(C_ao_lo[s,ik].T.conj(), hcore_ao[ik]), C_ao_lo[s,ik])
        hcore_lo_Co[s,ik] = np.dot(np.dot(C_ao_lo_Co[s,ik].T.conj(), hcore_ao[ik]), C_ao_lo_Co[s,ik])
    
        JK_lo[s,ik] = np.dot(np.dot(C_ao_lo[s,ik].T.conj(), JK_ao[s,ik]), C_ao_lo[s,ik])
        JK_lo_Co[s,ik] = np.dot(np.dot(C_ao_lo_Co[s,ik].T.conj(), JK_ao[s,ik]), C_ao_lo_Co[s,ik])


fn = datadir + '/hcore_JK_lo_' + label + '_' + method_label + '.h5'

f = h5py.File(fn, 'w')
f['hcore_lo'] = hcore_lo
f['hcore_lo_Co'] = hcore_lo_Co
f['JK_lo'] = JK_lo
f['JK_lo_Co'] = JK_lo_Co
f.close()

# if using DFT, get HF JK term using DFT density
if use_pbe:
    if do_restricted:
        kmf_hf = scf.KRHF(cell, kpts, exxdiv='ewald')
    else:
        kmf_hf = scf.KUHF(cell, kpts, exxdiv='ewald')
    kmf_hf.with_df = gdf
    kmf_hf.with_df._cderi = gdf_fname
    kmf_hf.max_cycle = 0
    if do_restricted:
        JK_ao_hf = kmf_hf.get_veff(dm_kpts=DM_ao[0])
    else:
        JK_ao_hf = kmf_hf.get_veff(dm_kpts=DM_ao)
    
    JK_lo_hf    = np.zeros((spin,nkpts,nval+nvirt,nval+nvirt),dtype=JK_lo.dtype)
    JK_lo_hf_Co = np.zeros((spin,nkpts,nao_Co,nao_Co),dtype=JK_lo.dtype)
    for s in range(spin):
        for ik in range(nkpts):
            JK_lo_hf[s,ik] = np.dot(np.dot(C_ao_lo[s,ik].T.conj(), JK_ao_hf[s,ik]), C_ao_lo[s,ik])
            JK_lo_hf_Co[s,ik] = np.dot(np.dot(C_ao_lo_Co[s,ik].T.conj(), JK_ao_hf[s,ik]), C_ao_lo_Co[s,ik])
    
    fn = datadir + '/hcore_JK_lo_' + label + '_' + method_label + '.h5'
    f = h5py.File(fn, 'w')
    f['JK_lo_hf'] = JK_lo_hf
    f['JK_lo_hf_Co'] = JK_lo_hf_Co
    f.close()


