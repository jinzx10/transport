#!/usr/bin/python
import numpy as np
import h5py, os
from libdmet_solid.system import lattice
from libdmet_solid.basis_transform import make_basis, eri_transform

from pyscf.pbc import gto, df, scf
from pyscf.pbc.lib import chkfile

############################################################
#                   build cell
############################################################
# total number of Cu atoms (left + right)
nat = 7

cell_fname = 'CoCu_' + str(nat).zfill(2) + '.chk'

if os.path.isfile(cell_fname):
    cell = chkfile.load_cell(cell_fname)
else:
    # atomic spacing in the lead
    a = 2.55
    
    # distance between Co and the nearest Cu atoms
    l = 2.7
    r = 2.7
    
    # number of Cu atoms in the left/right lead
    nl = nat // 2
    nr = nat - nl
    
    cell = gto.Cell()
    
    cell.unit = 'angstrom'
    cell.verbose = 4
    cell.max_memory = 180000
    cell.dimension = 3
    cell.ke_cutoff = 400
    cell.a = [[l+r+(nat-1)*a,0,0], [0,20,0], [0,0,20]]
    
    for iat in range(0, nl):
        cell.atom.append(['Cu', (iat*a, 0, 0)])
    cell.atom.append(['Co', ((nl-1)*a+l, 0, 0)])
    for iat in range(0, nr):
        cell.atom.append(['Cu', ((nl-1)*a+l+r+iat*a, 0, 0)])
    
    cell.spin = 0
    cell.basis = 'def2-svp-bracket'
    
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
gdf_fname = 'cderi_' + str(nat).zfill(2) + '_' + str(kmesh[0]) + str(kmesh[1])+ str(kmesh[2]) + '.h5'

gdf = df.GDF(cell, kpts)

if os.path.isfile(gdf_fname):
    gdf._cderi = gdf_fname
else:
    gdf._cderi_to_save = gdf_fname
    gdf.auxbasis = 'def2-svp-bracket-ri'
    gdf.build()

############################################################
#           low-level mean-field calculation
############################################################
smearing_sigma = 0.01
mf_fname = 'CoCu_' + str(nat).zfill(2) + '_' + str(kmesh[0]) + str(kmesh[1])+ str(kmesh[2]) + '.chk'

if os.path.isfile(mf_fname):
    kmf = scf.KRKS(cell, kpts).density_fit()
    kmf.xc = 'pbe'
    kmf = scf.addons.smearing_(kmf, sigma = smearing_sigma, method = 'fermi')
    kmf.with_df = gdf
    kmf.__dict__.update( chkfile.load(mf_fname, 'scf') )
else:
    kmf = scf.KRKS(cell, kpts).density_fit(auxbasis='def2-svp-bracket-ri')
    kmf.xc = 'pbe'
    kmf = scf.addons.smearing_(kmf, sigma = smearing_sigma, method = 'fermi')
    kmf.with_df = gdf
    kmf.chkfile = mf_fname
    kmf.conv_tol = 1e-12
    kmf.max_cycle = 200
    kmf.diis_space = 15
    kmf.kernel()


# set spin: unrestricted -> 2; restricted -> 1
# and make sure they have the same dimension
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

MINAO = {'Co': 'def2-svp-bracket-minao', 'Cu': 'def2-svp-bracket-minao'}
C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=MINAO, full_return=True)

C_ao_lo = np.zeros((spin,nkpts,nao,nao),dtype=complex)
for s in range(spin):
    C_ao_lo[s] = C_ao_iao


S_ao_ao = kmf.get_ovlp()

############################################################
#           Quantities in LO (IAO) basis
############################################################
C_mo_lo = np.zeros((spin,nkpts,nao,nao),dtype=complex)
for s in range(spin):
    for ik in range(nkpts):
        C_mo_lo[s][ik] = np.dot(np.dot(mo_coeff[s][ik].T.conj(), S_ao_ao[ik]), C_ao_lo[s][ik])
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
    for ik in range(nkpts):
        Cinv = np.dot(C_ao_lo[s][ik].T.conj(),S_ao_ao[ik])
        DM_lo[s][ik] = np.dot(np.dot(Cinv, DM_ao[s][ik]), Cinv.T.conj())

for s in range(spin):
    nelec_lo = np.trace(DM_lo[s].sum(axis=0)/nkpts)
    print ('Nelec imp', nelec_lo.real)
fn = 'DM_iao_k.h5'
feri = h5py.File(fn, 'w')
feri['DM'] = np.asarray(DM_lo)
feri.close()

# get 4-index ERI
eri = eri_transform.get_unit_eri_fast(cell, gdf, C_ao_lo=C_ao_lo, feri=gdf_fname)
fn = 'eri_iao.h5'
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
    for ik in range(nkpts):
        hcore_lo[s,ik] = np.dot(np.dot(C_ao_lo[s,ik].T.conj(), hcore_ao[ik]), C_ao_lo[s,ik])
        JK_lo[s,ik] = np.dot(np.dot(C_ao_lo[s,ik].T.conj(), JK_ao[s,ik]), C_ao_lo[s,ik])

fn = 'hcore_JK_iao_k_dft.h5'
feri = h5py.File(fn, 'w')
feri['hcore'] = np.asarray(hcore_lo)
feri['JK'] = np.asarray(JK_lo)
feri.close()
assert(np.max(np.abs(hcore_lo.sum(axis=1).imag/nkpts))<1e-6)
assert(np.max(np.abs(JK_lo.sum(axis=1).imag/nkpts))<1e-6)

# get HF JK term using DFT density
kmf_hf = scf.KRHF(cell, kpts, exxdiv='ewald')
kmf_hf.with_df = gdf
kmf_hf.with_df._cderi = gdf_fname
kmf_hf.max_cycle = 0
JK_ao = np.asarray(kmf_hf.get_veff(dm_kpts=DM_ao[0]))
if len(JK_ao.shape) == 3:
    JK_ao = JK_ao[np.newaxis, ...]

JK_lo = np.zeros((spin,nkpts,nao,nao),dtype=JK_ao.dtype)
for s in range(spin):
    for ik in range(nkpts):
        JK_lo[s,ik] = np.dot(np.dot(C_ao_lo[s,ik].T.conj(), JK_ao[s,ik]), C_ao_lo[s,ik])

fn = 'hcore_JK_iao_k_hf.h5'
feri = h5py.File(fn, 'w')
feri['JK'] = np.asarray(JK_lo)
feri.close()


