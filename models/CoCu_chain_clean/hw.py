#!/usr/bin/python
import numpy as np
import h5py, os, copy, argparse
from libdmet_solid.system import lattice
from libdmet_solid.basis_transform import make_basis, eri_transform
from libdmet.utils import plot

from pyscf.pbc import gto, df, scf
from pyscf.pbc.lib import chkfile

from libdmet_solid.lo.iao import reference_mol

import matplotlib.pyplot as plt


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

nao_Co = nval_Co + nvirt_Co # core orbitals are ignored!
nao_Co_tot = nao_Co + ncore_Co

Cu_basis = 'def2-svp-bracket'

ncore_Cu = 9
nval_Cu = 6
nvirt_Cu = 9

nao_Cu = nval_Cu + nvirt_Cu


############################################################
#           directory for saving/loading data
############################################################
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default = None, type = str)

args = parser.parse_args()

if args.datadir is None:
    datadir = 'Co_' + Co_basis + '_Cu_' + Cu_basis + '/'
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

# TODO: Tianyu & Linqing mentioned that the hybridization might
# be better computed by the DFT's Fock matrix

# if False, use HF instead
use_pbe = True

do_restricted = False

if do_restricted:
    method_label = 'r'
else:
    method_label = 'u'

if use_pbe:
    method_label += 'ks'
else:
    method_label += 'hf'

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
l = 2.7
r = 2.7

# Cu atomic spacing in the lead
a = 2.55

cell_label = 'CoCu_' + str(nat_Cu) + '_l' + str(l) + '_r' + str(r) + '_a' + str(a)
cell_fname = datadir + '/cell_' + cell_label + '.chk'

if os.path.isfile(cell_fname):
    print('load cell')
    cell = chkfile.load_cell(cell_fname)
else:
    
    cell = gto.Cell()
    
    cell.unit = 'angstrom'
    cell.verbose = 0
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
gdf_fname = datadir + '/cderi_' + cell_label + '.h5'

gdf = df.GDF(cell, kpts)

if os.path.isfile(gdf_fname):
    print('load cderi')
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
    if do_restricted:
        kmf = scf.KRKS(cell, kpts).density_fit()
    else:
        kmf = scf.KUKS(cell, kpts).density_fit()
    kmf.xc = 'pbe'
else:
    if do_restricted:
        kmf = scf.KRHF(cell, kpts).density_fit()
    else:
        kmf = scf.KUHF(cell, kpts).density_fit()

mf_fname = datadir + '/' + cell_label + '_' + method_label + '.chk'

kmf.with_df = gdf

if use_smearing:
    kmf = scf.addons.smearing_(kmf, sigma = smearing_sigma, method = 'fermi')
else:
    # newton or diis?
    kmf = kmf.newton()

if os.path.isfile(mf_fname):
    print('load mf data')
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

if spin == 1:
    C_ao_iao = C_ao_iao[np.newaxis,...]
    C_ao_iao_core = C_ao_iao_core[np.newaxis,...]
    C_ao_iao_val = C_ao_iao_val[np.newaxis,...]
    C_ao_iao_virt = C_ao_iao_virt[np.newaxis,...]

############################################################
#               rearrange orbitals
############################################################
# rearrange IAO/PAO according to their indices in cell.atom

if use_reference_mol:
    # C_ao_lo: transformation matrix from AO to LO (IAO) basis, all atoms, core orbitals are excluded
    # C_ao_lo_Co: transformation matrix from AO to LO (IAO) basis, only Co atom, core orbitals are excluded
    C_ao_lo_Co = np.zeros((spin,nkpts,nao,31), dtype=complex)
    C_ao_lo_Co[:,:,:,0:ncore_Co] = C_ao_iao_core[:,:,:,0:ncore_Co]
    C_ao_lo_Co[:,:,:,ncore_Co:ncore_Co+nval_Co] = C_ao_iao_val[:,:,:,0:nval_Co]
    C_ao_lo_Co[:,:,:,ncore_Co+nval_Co:31] = C_ao_iao_virt[:,:,:,0:nvirt_Co]

else:
    # TODO...
    print('rearranging orbitals not implemented!')

C_ao_lo_Co = C_ao_lo_Co.real



############################################################
#           Quantities in LO (IAO) basis
############################################################
S_ao_ao = kmf.get_ovlp()
print('S_ao_ao.shape = ', S_ao_ao.shape)


#plt.imshow(np.abs(C_ao_lo_Co[0,0]))
#plt.show()
#exit()

#C = C_ao_lo_Co[0,0,0:31,:]
#print('C.shape = ', C.shape)
#S_Co = S_ao_ao[0,:31,:31]
#identity = C[:,:].T.conj() @ S_Co @ C[:,:]
#print('diff from I = ', np.linalg.norm(identity-np.eye(31)))
#
#print('identity = ', identity[15,15])
#plt.imshow(np.abs(S_Co))
#plt.show()
#exit()
#
#identity2 = C_ao_lo_Co[0,0].T.conj() @ S_ao_ao @ C_ao_lo_Co[0,0]
#print('diff from I = ', np.linalg.norm(identity2-np.eye(31)))
#
#exit()




fname = 'Co_test.h5'
f = h5py.File(fname, 'w')
f['C_ao_lo_Co'] = C_ao_lo_Co
f['S_ao_ao'] = S_ao_ao
f.close()

# get density matrix in IAO basis
DM_ao = kmf.make_rdm1() # DM_ao is real!

if len(DM_ao.shape) == 3:
    DM_ao = DM_ao[np.newaxis,...]

DM_lo_Co = np.zeros((spin,nkpts,31,31),dtype=DM_ao.dtype)

for s in range(spin):
    for ik in range(nkpts):
        Cinv_Co = np.dot(C_ao_lo_Co[s,ik].T.conj(),S_ao_ao[ik])
        DM_lo_Co[s,ik] = np.dot(np.dot(Cinv_Co, DM_ao[s,ik]), Cinv_Co.T.conj())

nelec_Co_tmp = 0
for s in range(spin):
    nelec_lo_Co = np.trace(DM_lo_Co[s].sum(axis=0)/nkpts)
    print ('Nelec on Co', nelec_lo_Co.real)

    nelec_Co_tmp += nelec_lo_Co

print('total number of electrons on Co = ', nelec_Co_tmp)

fn = 'Co_test.h5'
f = h5py.File(fn, 'a')
f['DM_lo_Co'] = DM_lo_Co
f.close()

print(np.diag(DM_lo_Co[0,0]))

plt.imshow(np.abs(DM_lo_Co[0,0]), extent=[0,1,0,1])
plt.show()

