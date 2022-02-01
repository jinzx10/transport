#!/usr/bin/python
import numpy as np
import h5py, os
from libdmet_solid.system import lattice
from libdmet_solid.basis_transform import make_basis
from libdmet_solid.basis_transform import eri_transform
import libdmet_solid.utils.logger as log
from libdmet_solid.utils.misc import read_poscar

from pyscf.pbc import df, dft, scf, gto
from pyscf.pbc.lib import chkfile
from pyscf import lib
from pyscf import gto as gto_mol
from pyscf import scf as scf_mol

from fcdmft.gw.mol import gw_dc
from fcdmft.utils import cholesky

import copy
from libdmet_solid.utils.misc import kdot, mdot
from libdmet_solid.lo.iao import get_idx, get_idx_each_atom, \
        get_idx_each_orbital
from libdmet_solid.lo.iao import reference_mol, get_labels, get_idx_each

log.verbose = 'DEBUG1'

einsum = lib.einsum

# NOTE: lattice system setup by user
cell = gto.Cell()
cell.atom = "Cu 0.0 0.0 0.0"
latt = 2.544*np.sqrt(2.0)
cell.a = [[0.5*latt,0.0,0.5*latt], [0.5*latt,0.5*latt,0.0], [0.0,0.5*latt,0.5*latt]]
cell.unit = 'Angstrom'
cell.verbose = 7
cell.max_memory = 320000
cell.basis = 'def2-svp'
cell.precision = 1e-10
cell.dimension = 3
cell.spin = 0
cell.build()

kmesh = [3,3,3]
Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts
nao = Lat.nao
nkpts = Lat.nkpts

scell, phase = eri_transform.get_phase(cell, kpts)
old_atom = scell.atom
new_atom = [['Co',old_atom[0][1]]]
for i in range(1,len(old_atom)):
    new_atom.append(['Cu',old_atom[i][1]])
scell.atom = new_atom
scell.basis = {'Co':'def2-svp','Cu':'def2-svp-bracket'}
scell.spin = 0
scell.charge = 0
scell.verbose = 7
scell.build()

kmesh = [3,3,3]
Lat = lattice.Lattice(scell, kmesh)
kpts = Lat.kpts
nao = Lat.nao
nkpts = Lat.nkpts

gdf = df.GDF(scell, kpts)
gdf_fname = '../gdf_ints_333.h5'
gdf._cderi_to_save = gdf_fname
gdf.auxbasis = df.aug_etb(scell, beta=2.3, use_lval=True, l_val_set={'Co':2,'Cu':2})
gdf.auxbasis['Cu'] = 'def2-svp-bracket-ri'
gdf.auxbasis['Co'] = 'def2-svp-ri'
if not os.path.isfile(gdf_fname):
    gdf.build()

chkfname = 'cocu_333.chk'
if os.path.isfile(chkfname):
    kmf = dft.KRKS(scell, kpts).density_fit()
    kmf.xc = 'pbe'
    kmf = scf.addons.smearing_(kmf, sigma=5e-3, method="fermi")
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    data = chkfile.load(chkfname, 'scf')
    kmf.__dict__.update(data)
else:
    kmf = dft.KRKS(scell, kpts).density_fit()
    kmf.xc = 'pbe'
    kmf = scf.addons.smearing_(kmf, sigma=5e-3, method="fermi")
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    kmf.conv_tol = 1e-12
    kmf.max_cycle = 200
    kmf.diis_space = 15
    kmf.chkfile = chkfname
    kmf.kernel()

cell = scell

# set spin
mo_energy = np.asarray(kmf.mo_energy)
mo_coeff = np.asarray(kmf.mo_coeff)
if len(mo_energy.shape) == 2:
    spin = 1
    mo_energy = mo_energy[np.newaxis, ...]
    mo_coeff = mo_coeff[np.newaxis, ...]
else:
    spin = 2

# NOTE: choose IAO basis by user
# set IAO (core+val)
minao = {'Co':'def2-svp-minao', 'Cu':'def2-svp-minao'}
pmol = reference_mol(cell, minao=minao)
basis = pmol._basis

# set valence IAO
basis_val = {}
minao_val = {'Co':'def2-svp-minao-val', 'Cu':'def2-svp-minao-val'}
pmol_val = pmol.copy()
pmol_val.basis = minao_val
pmol_val.build()
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
basis_core = {}
minao_core = {'Co':'def2-svp-minao-core', 'Cu':'def2-svp-minao-core'}
pmol_core = pmol.copy()
pmol_core.basis = minao_core
pmol_core.build()
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

# First construct IAO and PAO.
C_ao_iao, C_ao_iao_val, C_ao_iao_virt, C_ao_iao_core \
        = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=minao, full_return=True, \
        pmol_val=pmol_val, pmol_core=pmol_core, tol=1e-9)

# C_ao_lo: transformation matrix from AO to LO (IAO) basis
nlo = nao - ncore
C_ao_lo_full = np.zeros((spin,nkpts,nao,nlo),dtype=complex)
for s in range(spin):
    C_ao_lo_full[s,:,:,:6] = C_ao_iao[:,:,ncore:(ncore+6)]
    C_ao_lo_full[s,:,:,6:22] = C_ao_iao[:,:,(ncore+nval):(ncore+nval+16)]
    C_ao_lo_full[s,:,:,22:(nval+16)] = C_ao_iao[:,:,(ncore+6):(ncore+nval)]
    C_ao_lo_full[s,:,:,(nval+16):] = C_ao_iao[:,:,(ncore+nval+16):]

nlo_co = 22
C_ao_lo = C_ao_lo_full[:,:,:,:nlo_co]

# C_mo_lo: transformation matrix from MO to LO (IAO) basis
S_ao_ao = kmf.get_ovlp()
'''
C_mo_lo = np.zeros((spin,nkpts,nao,nlo))
for s in range(spin):
    for ki in range(nkpts):
        C_mo_lo[s][ki] = np.dot(np.dot(mo_coeff[s][ki].T.conj(), S_ao_ao[ki]), C_ao_lo[s][ki])
fn = 'C_mo_lo.h5'
feri = h5py.File(fn, 'w')
feri['C_ao_lo'] = np.asarray(C_ao_lo)
feri['C_mo_lo'] = np.asarray(C_mo_lo)
feri.close()
'''

# get DFT density matrix in IAO basis
DM_ao = np.asarray(kmf.make_rdm1())
if len(DM_ao.shape) == 3:
    DM_ao = DM_ao[np.newaxis, ...]
DM_lo = np.zeros((spin,nkpts,nlo_co,nlo_co),dtype=DM_ao.dtype)
for s in range(spin):
    for ki in range(nkpts):
        Cinv = np.dot(C_ao_lo[s][ki].T.conj(),S_ao_ao[ki])
        DM_lo[s][ki] = np.dot(np.dot(Cinv, DM_ao[s][ki]), Cinv.T.conj())

for s in range(spin):
    nelec_lo = np.trace(DM_lo[s].sum(axis=0)/nkpts)
    print ('Nelec imp', nelec_lo.real)
fn = 'DM_iao_k.h5'
feri = h5py.File(fn, 'w')
feri['DM'] = np.asarray(DM_lo)
feri.close()
print (1./nkpts*DM_lo[0].sum(axis=0).diagonal())

# get 4-index ERI
eri = eri_transform.get_unit_eri_fast(cell, gdf, C_ao_lo=C_ao_lo, feri=gdf_fname)
fn = 'eri_imp111_iao.h5'
feri = h5py.File(fn, 'w')
feri['eri'] = np.asarray(eri.real)
feri.close()

# get one-electron integrals
hcore_ao = np.asarray(kmf.get_hcore())
JK_ao = np.asarray(kmf.get_veff())
if len(JK_ao.shape) == 3:
    JK_ao = JK_ao[np.newaxis, ...]

hcore_lo = np.zeros((spin,nkpts,nlo_co,nlo_co),dtype=hcore_ao.dtype)
JK_lo = np.zeros((spin,nkpts,nlo_co,nlo_co),dtype=JK_ao.dtype)
for s in range(spin):
    for ki in range(nkpts):
        hcore_lo[s,ki] = np.dot(np.dot(C_ao_lo[s,ki].T.conj(), hcore_ao[ki]), C_ao_lo[s,ki])
        JK_lo[s,ki] = np.dot(np.dot(C_ao_lo[s,ki].T.conj(), JK_ao[s,ki]), C_ao_lo[s,ki])

fn = 'hcore_JK_iao_k_dft.h5'
feri = h5py.File(fn, 'w')
feri['hcore'] = np.asarray(hcore_lo)
feri['JK'] = np.asarray(JK_lo)
feri.close()

hcore_lo_full = np.zeros((spin,nkpts,nlo,nlo),dtype=hcore_ao.dtype)
JK_lo_full = np.zeros((spin,nkpts,nlo,nlo),dtype=JK_ao.dtype)
for s in range(spin):
    for ki in range(nkpts):
        hcore_lo_full[s,ki] = np.dot(np.dot(C_ao_lo_full[s,ki].T.conj(), hcore_ao[ki]), C_ao_lo_full[s,ki])
        JK_lo_full[s,ki] = np.dot(np.dot(C_ao_lo_full[s,ki].T.conj(), JK_ao[s,ki]), C_ao_lo_full[s,ki])

fn = 'hcore_JK_iao_k_dft_full.h5'
feri = h5py.File(fn, 'w')
feri['hcore'] = np.asarray(hcore_lo_full)
feri['JK'] = np.asarray(JK_lo_full)
feri.close()

# get HF JK term using DFT density
kmf_hf = scf.KRHF(cell, kpts, exxdiv='ewald')
kmf_hf.with_df = gdf
kmf_hf.with_df._cderi = gdf_fname
kmf_hf.max_cycle = 0
JK_ao = np.asarray(kmf_hf.get_veff(dm_kpts=DM_ao[0]))
if len(JK_ao.shape) == 3:
    JK_ao = JK_ao[np.newaxis, ...]

JK_lo = np.zeros((spin,nkpts,nlo_co,nlo_co),dtype=JK_ao.dtype)
for s in range(spin):
    for ki in range(nkpts):
        JK_lo[s,ki] = np.dot(np.dot(C_ao_lo[s,ki].T.conj(), JK_ao[s,ki]), C_ao_lo[s,ki])

fn = 'hcore_JK_iao_k_hf.h5'
feri = h5py.File(fn, 'w')
feri['JK'] = np.asarray(JK_lo)
feri.close()

JK_lo_full = np.zeros((spin,nkpts,nlo,nlo),dtype=JK_ao.dtype)
for s in range(spin):
    for ki in range(nkpts):
        JK_lo_full[s,ki] = np.dot(np.dot(C_ao_lo_full[s,ki].T.conj(), JK_ao[s,ki]), C_ao_lo_full[s,ki])

fn = 'hcore_JK_iao_k_hf_full.h5'
feri = h5py.File(fn, 'w')
feri['JK'] = np.asarray(JK_lo_full)
feri.close()
