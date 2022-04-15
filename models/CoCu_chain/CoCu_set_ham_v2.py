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
    datadir = 'CoCu_set_ham_v2_data/'
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

# 4p,4d,4f,5s
nvirt_Co = 16

nao_Co = ncore_Co + nval_Co + nvirt_Co

Cu_basis = 'def2-svp-bracket'

ncore_Cu = 9
nval_Cu = 6
nvirt_Cu = 9 # same as Co, except without 4f

nao_Cu = ncore_Cu + nval_Cu + nvirt_Cu

############################################################
#               low-level mean-field method
############################################################
# This is the method used to generate the density matrix;
# The Fock matrix for building the impurity Hamiltonian
# will use HF anyway.

# TODO: Tianyu & Linqing mentioned that the hybridization might
# be better computed by the DFT's Fock matrix

# if False, use HF instead
use_pbe = True

############################################################
#                       build cell
############################################################
# total number of Cu atoms (left + right)
nat = 9
assert(nat%2 == 1)

# number of Cu atoms in the left/right lead
nl = nat // 2
nr = nat - nl

cell_fname = datadir + '/cell_CoCu_' + str(nat).zfill(2) + '.chk'

if os.path.isfile(cell_fname):
    cell = chkfile.load_cell(cell_fname)
else:
    # atomic spacing in the lead
    a = 2.55
    
    # distance between Co and the nearest Cu atoms
    l = 2.7
    r = 2.7
    
    cell = gto.Cell()
    
    cell.unit = 'angstrom'
    cell.verbose = 4
    cell.max_memory = 30000
    cell.dimension = 3
    cell.ke_cutoff = 500
    cell.a = [[l+r+(nat-1)*a,0,0], [0,20,0], [0,0,20]]
    
    for iat in range(0, nl):
        cell.atom.append(['Cu', (iat*a, 0, 0)])

    cell.atom.append(['Co', ((nl-1)*a+l, 0, 0)])

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

label = 'CoCu_' + str(nat).zfill(2) + '_' + str(kmesh[0]) + str(kmesh[1])+ str(kmesh[2])

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

if use_pbe:
    kmf = scf.KRKS(cell, kpts).density_fit()
    kmf.xc = 'pbe'
    mf_fname = datadir + '/ks_' + label + '.chk'
else:
    kmf = scf.KRHF(cell, kpts).density_fit()
    mf_fname = datadir + '/hf_' + label + '.chk'

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

############################################################
#               Orbital Localization (IAO)
############################################################

MINAO = {'Co': Co_basis + '-minao', 'Cu': Cu_basis + '-minao'}

# if True, use reference core/val
use_core_val = True

if use_core_val:

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
            pmol_val=pmol_val, pmol_core=pmol_core, tol=1e-9)

else:
    # if core/val is not specified, the number of val will equal the size of MINAO
    # and the rest are virt
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=MINAO, full_return=True)

############################################################
#               rearrange orbitals
############################################################
# rearrange IAO/PAO according to their positions

if use_core_val:
    # C_ao_lo: transformation matrix from AO to LO (IAO) basis, all atoms, core orbitals are excluded
    # C_ao_lo_Co: transformation matrix from AO to LO (IAO) basis, only Co atom, core orbitals are excluded
    C_ao_lo   = np.zeros((nkpts,nao,nval+nvirt), dtype=complex)
    C_ao_lo_Co = np.zeros((nkpts,nao,nval_Co+nvirt_Co), dtype=complex)
    
    # rearrange orbitals according to atomic index (Co is in the middle)
    # left lead
    for iat in range(nl):
    
        # val
        C_ao_lo[:,:,iat*(nval_Cu+nvirt_Cu):iat*(nval_Cu+nvirt_Cu)+nval_Cu] \
                = C_ao_iao[:,:,ncore+iat*nval_Cu:ncore+(iat+1)*nval_Cu]
    
        # virt
        C_ao_lo[:,:,iat*(nval_Cu+nvirt_Cu)+nval_Cu:(iat+1)*(nval_Cu+nvirt_Cu)] \
                = C_ao_iao[:,:,ncore+nval+iat*nvirt_Cu:ncore+nval+(iat+1)*nvirt_Cu]
    
    # Co
    # val
    C_ao_lo[:,:,nl*(nval_Cu+nvirt_Cu):nl*(nval_Cu+nvirt_Cu)+nval_Co] \
            = C_ao_iao[:,:,ncore+nl*nval_Cu:ncore+nl*nval_Cu+nval_Co]
    
    # virt
    C_ao_lo[:,:,nl*(nval_Cu+nvirt_Cu)+nval_Co:nl*(nval_Cu+nvirt_Cu)+nval_Co+nvirt_Co] \
            = C_ao_iao[:,:,ncore+nval+nl*nvirt_Cu:ncore+nval+nl*nvirt_Cu+nvirt_Co]
    
    
    # right lead
    for iat in range(nr):
    
        # val
        C_ao_lo[:,:,(nl+iat)*(nval_Cu+nvirt_Cu)+nval_Co+nvirt_Co:(nl+iat)*(nval_Cu+nvirt_Cu)+nval_Co+nvirt_Co+nval_Cu] \
                = C_ao_iao[:,:,ncore+nl*nval_Cu+nval_Co+iat*nval_Cu:ncore+nl*nval_Cu+nval_Co+(iat+1)*nval_Cu]
    
        # virt
        C_ao_lo[:,:,(nl+iat)*(nval_Cu+nvirt_Cu)+nval_Co+nvirt_Co+nval_Cu:(nl+iat+1)*(nval_Cu+nvirt_Cu)+nval_Co+nvirt_Co] \
                = C_ao_iao[:,:,ncore+nval+(nl+iat)*nvirt_Cu+nvirt_Co:ncore+nval+(nl+iat+1)*nvirt_Cu+nvirt_Co]
    
    C_ao_lo_Co = C_ao_lo[:,:,nl*(nval_Cu+nvirt_Cu):nl*(nval_Cu+nvirt_Cu)+nval_Co+nvirt_Co]

else:
    # TBD...
    print('rearranging orbitals not implemented!')


############################################################
#           Plot MO and LO
############################################################
plot_orb = False

if plot_orb:
    plotdir = datadir + '/plot_' + label
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)

    plot.plot_orb_k_all(cell, plotdir + '/iao_' + str(nat).zfill(2), C_ao_iao, kpts, margin=0.0)
    plot.plot_orb_k_all(cell, plotdir + '/lo_' + str(nat).zfill(2), C_ao_lo, kpts, margin=0.0)

############################################################
#           Quantities in LO (IAO) basis
############################################################
S_ao_ao = kmf.get_ovlp()

fname = datadir + '/C_ao_lo_' + label + '.h5'
f = h5py.File(fname, 'w')
f['C_ao_lo'] = C_ao_lo
f['C_ao_lo_Co'] = C_ao_lo_Co
f['S_ao_ao'] = S_ao_ao
f.close()

# get DFT density matrix in IAO basis
DM_ao = kmf.make_rdm1() # DM_ao is real!

DM_lo = np.zeros((nkpts,nval+nvirt,nval+nvirt),dtype=complex)
DM_lo_Co = np.zeros((nkpts,nval_Co+nvirt_Co,nval_Co+nvirt_Co),dtype=complex)

for ik in range(nkpts):
    # C^\dagger*S*C=I
    Cinv = np.dot(C_ao_lo[ik].T.conj(),S_ao_ao[ik])
    DM_lo[ik] = np.dot(np.dot(Cinv, DM_ao[ik]), Cinv.T.conj())

    Cinv_Co = np.dot(C_ao_lo_Co[ik].T.conj(),S_ao_ao[ik])
    DM_lo_Co[ik] = np.dot(np.dot(Cinv_Co, DM_ao[ik]), Cinv_Co.T.conj())

nelec_lo = np.trace(DM_lo.sum(axis=0)/nkpts)
print ('Nelec (core excluded)', nelec_lo.real)

nelec_lo_Co = np.trace(DM_lo_Co.sum(axis=0)/nkpts)
print ('Nelec on Co', nelec_lo_Co.real)

fn = datadir + '/DM_lo_' + label + '.h5'
f = h5py.File(fn, 'w')
f['DM_lo'] = DM_lo
f['DM_lo_Co'] = DM_lo_Co
f.close()

# get 4-index ERI
eri_lo = eri_transform.get_unit_eri_fast(cell, gdf, C_ao_lo=C_ao_lo, feri=gdf_fname)
eri_lo_Co = eri_transform.get_unit_eri_fast(cell, gdf, C_ao_lo=C_ao_lo_Co, feri=gdf_fname)

fn = datadir + '/eri_lo_' + label + '.h5'
f = h5py.File(fn, 'w')
f['eri_lo'] = eri_lo.real
f['eri_lo_Co'] = eri_lo_Co.real
f.close()

assert(np.max(np.abs(eri_lo.imag))<1e-8)

# get one-electron integrals
hcore_ao = kmf.get_hcore() # hcore_ao and JK_ao are all real!
JK_ao = kmf.get_veff()

hcore_lo = np.zeros((nkpts,nval+nvirt,nval+nvirt),dtype=complex)
hcore_lo_Co = np.zeros((nkpts,nval_Co+nvirt_Co,nval_Co+nvirt_Co),dtype=complex)

JK_lo = np.zeros((nkpts,nval+nvirt,nval+nvirt),dtype=complex)
JK_lo_Co = np.zeros((nkpts,nval_Co+nvirt_Co,nval_Co+nvirt_Co),dtype=complex)

for ik in range(nkpts):
    hcore_lo[ik] = np.dot(np.dot(C_ao_lo[ik].T.conj(), hcore_ao[ik]), C_ao_lo[ik])
    hcore_lo_Co[ik] = np.dot(np.dot(C_ao_lo_Co[ik].T.conj(), hcore_ao[ik]), C_ao_lo_Co[ik])

    JK_lo[ik] = np.dot(np.dot(C_ao_lo[ik].T.conj(), JK_ao[ik]), C_ao_lo[ik])
    JK_lo_Co[ik] = np.dot(np.dot(C_ao_lo_Co[ik].T.conj(), JK_ao[ik]), C_ao_lo_Co[ik])


fn = datadir + '/hcore_JK_lo_dft_' + label + '.h5'
f = h5py.File(fn, 'w')
f['hcore_lo'] = hcore_lo
f['hcore_lo_Co'] = hcore_lo_Co
f['JK_lo'] = JK_lo
f['JK_lo_Co'] = JK_lo_Co
f.close()

assert(np.max(np.abs(hcore_lo.sum(axis=0).imag/nkpts))<1e-8)
assert(np.max(np.abs(JK_lo.sum(axis=0).imag/nkpts))<1e-8)

# get HF JK term using DFT density
kmf_hf = scf.KRHF(cell, kpts, exxdiv='ewald')
kmf_hf.with_df = gdf
kmf_hf.with_df._cderi = gdf_fname
kmf_hf.max_cycle = 0
JK_ao = kmf_hf.get_veff(dm_kpts=DM_ao)

JK_lo = np.zeros((nkpts,nval+nvirt,nval+nvirt),dtype=complex)
JK_lo_Co = np.zeros((nkpts,nval_Co+nvirt_Co,nval_Co+nvirt_Co),dtype=complex)
for ik in range(nkpts):
    JK_lo[ik] = np.dot(np.dot(C_ao_lo[ik].T.conj(), JK_ao[ik]), C_ao_lo[ik])
    JK_lo_Co[ik] = np.dot(np.dot(C_ao_lo_Co[ik].T.conj(), JK_ao[ik]), C_ao_lo_Co[ik])

# HF only differs from DFT by the JK part
fn = datadir + '/JK_lo_hf_' + label + '.h5'
f = h5py.File(fn, 'w')
f['JK_lo'] = JK_lo
f['JK_lo_Co'] = JK_lo_Co
f.close()


