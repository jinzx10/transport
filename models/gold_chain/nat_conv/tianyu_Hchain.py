#!/usr/bin/python
import numpy as np
import h5py, os
from libdmet_solid.system import lattice
from libdmet_solid.basis_transform import make_basis
from libdmet_solid.basis_transform import eri_transform
import libdmet_solid.utils.logger as log

from pyscf.pbc import df, dft, scf, gto
from pyscf.pbc.lib import chkfile
from pyscf import lib
from pyscf import gto as gto_mol
from pyscf import scf as scf_mol

from fcdmft.gw.mol import gw_dc
from fcdmft.utils import cholesky

log.verbose = 'DEBUG1'

einsum = lib.einsum

# NOTE: lattice system setup by user
cell = gto.Cell()
cell.build(unit = 'angstrom',
           a = ''' 10.0    0.0     0.0
                   0.0     10.0    0.0
                   0.0     0.0     3.0 ''',
           atom = ''' H 5.0      5.0      1.0
                      H 5.0      5.0      2.0 ''',
           dimension = 3,
           max_memory = 32000,
           verbose = 5,
           basis='gth-dzv',
           pseudo='gth-pbe',
           precision=1e-12)

kmesh = [1,1,9]
Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts
nao = Lat.nao
nkpts = Lat.nkpts

gdf = df.GDF(cell, kpts)
gdf.auxbasis = df.aug_etb(cell, beta=2.0)
gdf_fname = 'gdf_ints_119.h5'
gdf._cderi_to_save = gdf_fname
if not os.path.isfile(gdf_fname):
    gdf.build()

chkfname = 'h_119.chk'
exxdiv = 'ewald'
if os.path.isfile(chkfname):
    kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv).density_fit()
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    kmf.conv_tol = 1e-12
    data = chkfile.load(chkfname, 'scf')
    kmf.__dict__.update(data)
else:
    kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv).density_fit()
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    kmf.conv_tol = 1e-12
    kmf.chkfile = chkfname
    kmf.kernel()

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
# C_ao_lo: transformation matrix from AO to LO (IAO) basis
MINAO = {'H':'gth-szv'}
C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=MINAO, full_return=True)
C_ao_lo = np.zeros((spin,nkpts,nao,nao),dtype=np.complex128)
for s in range(spin):
    C_ao_lo[s] = C_ao_iao

# C_mo_lo: transformation matrix from MO to LO (IAO) basis
S_ao_ao = kmf.get_ovlp()
C_mo_lo = np.zeros((spin,nkpts,nao,nao),dtype=np.complex128)
for s in range(spin):
    for ki in range(nkpts):
        C_mo_lo[s][ki] = np.dot(np.dot(mo_coeff[s][ki].T.conj(), S_ao_ao[ki]), C_ao_lo[s][ki])
fn = 'C_mo_lo.h5'
feri = h5py.File(fn, 'w')
feri['C_ao_lo'] = np.asarray(C_ao_lo)
feri['C_mo_lo'] = np.asarray(C_mo_lo)
feri.close()

# get DFT density matrix in IAO basis
DM_ao = np.asarray(kmf.make_rdm1())
if len(DM_ao.shape) == 3:
    DM_ao = DM_ao[np.newaxis, ...]
DM_lo = np.zeros((spin,nkpts,nao,nao),dtype=DM_ao.dtype)
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
hcore_lo = np.zeros((spin,nkpts,nao,nao),dtype=hcore_ao.dtype)
JK_lo = np.zeros((spin,nkpts,nao,nao),dtype=JK_ao.dtype)
for s in range(spin):
    for ki in range(nkpts):
        hcore_lo[s,ki] = np.dot(np.dot(C_ao_lo[s,ki].T.conj(), hcore_ao[ki]), C_ao_lo[s,ki])
        JK_lo[s,ki] = np.dot(np.dot(C_ao_lo[s,ki].T.conj(), JK_ao[s,ki]), C_ao_lo[s,ki])

fn = 'hcore_JK_iao_k_dft.h5'
feri = h5py.File(fn, 'w')
feri['hcore'] = np.asarray(hcore_lo)
feri['JK'] = np.asarray(JK_lo)
feri.close()
assert(np.max(np.abs(hcore_lo.sum(axis=1).imag/nkpts))<1e-6)
assert(np.max(np.abs(JK_lo.sum(axis=1).imag/nkpts))<1e-6)

# get HF JK term using DFT density
kmf_hf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
kmf_hf.with_df = gdf
kmf_hf.with_df._cderi = gdf_fname
kmf_hf.max_cycle = 0
JK_ao = np.asarray(kmf_hf.get_veff(dm_kpts=DM_ao[0]))
if len(JK_ao.shape) == 3:
    JK_ao = JK_ao[np.newaxis, ...]

JK_lo = np.zeros((spin,nkpts,nao,nao),dtype=JK_ao.dtype)
for s in range(spin):
    for ki in range(nkpts):
        JK_lo[s,ki] = np.dot(np.dot(C_ao_lo[s,ki].T.conj(), JK_ao[s,ki]), C_ao_lo[s,ki])

fn = 'hcore_JK_iao_k_hf.h5'
feri = h5py.File(fn, 'w')
feri['JK'] = np.asarray(JK_lo)
feri.close()
