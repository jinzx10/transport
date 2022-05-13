#!/usr/bin/python
import numpy as np
import h5py, os, copy, argparse
from libdmet_solid.system import lattice
from libdmet_solid.basis_transform import make_basis, eri_transform
from libdmet.utils import plot

from pyscf.pbc import gto, df, scf
from pyscf.pbc.lib import chkfile

from libdmet_solid.lo.iao import reference_mol

from pyscf.scf.hf import eig as eiggen

############################################################
#                       basis
############################################################
Co_basis = 'def2-svp'

if Co_basis == 'def2-svp':
    # 1s,2s,2p,3s,3p
    ncore_Co = 9
    
    # 3d,4s
    nval_Co = 6
    
    # 4p,4d,4f,5s
    nvirt_Co = 16

if Co_basis == 'def2-svp-bracket':
    # 1s,2s,2p,3s,3p
    ncore_Co = 9
    
    # 3d,4s
    nval_Co = 6
    
    # 4p,4d,4f,5s
    nvirt_Co = 9

nao_Co = nval_Co + nvirt_Co # core orbitals are ignored!
nao_Co_tot = nao_Co + ncore_Co

Cu_basis = 'def2-svp-bracket'

ncore_Cu = 9
nval_Cu = 6
nvirt_Cu = 9

nao_Cu = nval_Cu + nvirt_Cu
nao_Cu_tot = nao_Cu + ncore_Cu


############################################################
#           directory for saving/loading data
############################################################
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default = 'data', type = str)

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

# if False, use HF instead
use_dft = True

# if use_dft
xcfun = 'pbe'

do_restricted = True

if do_restricted:
    method_label = 'r'
else:
    method_label = 'u'

if use_dft:
    method_label += 'ks' + '_' + xcfun
else:
    method_label += 'hf'

# scf solver & addons
# if use_smearing == True, use Fermi smearing
# if False, the scf will use newton solver to help convergence
use_smearing = False
smearing_sigma = 0.1

if use_smearing:
    method_label += '_smearing' + str(smearing_sigma)
#else:
#    method_label += '_newton'

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
    print('ready to load', cell_fname)
    cell = chkfile.load_cell(cell_fname)
    print('cell loaded!')
else:
    
    cell = gto.Cell()
    
    cell.unit = 'angstrom'
    cell.verbose = 4
    cell.max_memory = 30000
    cell.dimension = 3

    #cell.precision = 1e-10
    cell.ke_cutoff = 500

    # should not discard! 4s and 5s have small exponent
    #cell.exp_to_discard = 0.1

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
    print('ready to load', gdf_fname)
    gdf._cderi = gdf_fname
    print('gdf cderi loaded!')
else:
    gdf._cderi_to_save = gdf_fname
    gdf.auxbasis = {'Co' : Co_basis + '-ri', 'Cu' : Cu_basis + '-ri'}
    gdf.build()

############################################################
#           low-level mean-field calculation
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

mf_fname = datadir + '/' + cell_label + '_' + method_label + '.chk'

kmf.with_df = gdf

if use_smearing:
    kmf = scf.addons.smearing_(kmf, sigma = smearing_sigma, method = 'fermi')
    kmf.diis_space = 15
else:
    kmf = kmf.newton()

if os.path.isfile(mf_fname):
    print('ready to load', mf_fname)
    kmf.__dict__.update( chkfile.load(mf_fname, 'scf') )
    print('mean field data loaded!')
    kmf.conv_tol = 1e-10
    kmf.max_cycle = 300
    kmf.kernel()
else:
    kmf.chkfile = mf_fname
    kmf.conv_tol = 1e-10
    kmf.max_cycle = 300

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

#################################
#       sanity check begin
#################################
S_ao_ao = kmf.get_ovlp()

hcore_ao = kmf.get_hcore()
hcore_ao = hcore_ao[0]

JK_ao = kmf.get_veff()
JK_ao = JK_ao[0]

DM_ao = kmf.make_rdm1() # DM_ao is real!
DM_ao = DM_ao[0]

# check hcore_ao and JK_ao indeed generate DM_ao
fock = hcore_ao + JK_ao

e, v = eiggen(fock, S_ao_ao[0])


mo_energy = kmf.mo_energy
mo_energy = mo_energy[0]
print('sanity: mo_energy vs. fock eigenvalue = ', np.linalg.norm(mo_energy-e))

nocc = (27+nat_Cu*29) // 2

mo_occ = kmf.mo_occ
mo_occ = mo_occ[0]

dm_fock = (v * mo_occ) @ v.T

print('sanity nocc = ', nocc)

print('sanity sum(mo_occ) = ', np.sum(mo_occ))

print('sanity: dm diff between make_rdm1 and fock-solved = ', np.linalg.norm(dm_fock-DM_ao))

# verify DM_ao indeed generates JK_ao
DM_ao = DM_ao[np.newaxis,...]
JK_tmp = kmf.get_veff(dm=DM_ao)
print('sanity JK diff = ', np.linalg.norm(JK_tmp[0]-JK_ao))

#################################
#       sanity check end
#################################

if Co_basis == 'def2-svp':
    # 1s,2s,2p,3s,3p
    ncore_Co = 9
    
    # 3d,4s
    nval_Co = 6
    
    # 4p,4d,4f,5s
    nvirt_Co = 16

if Co_basis == 'def2-svp-bracket':
    # 1s,2s,2p,3s,3p
    ncore_Co = 9
    
    # 3d,4s
    nval_Co = 6
    
    # 4p,4d,4f,5s
    nvirt_Co = 9

nao_Co = nval_Co + nvirt_Co # core orbitals are ignored!
nao_Co_tot = nao_Co + ncore_Co

if Cu_basis == 'def2-svp-bracket':
    ncore_Cu = 9
    nval_Cu = 6
    nvirt_Cu = 9

nao_Cu = nval_Cu + nvirt_Cu
nao_Cu_tot = nao_Cu + ncore_Cu


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
    C_ao_iao_val = C_ao_iao_val[np.newaxis,...]
    C_ao_iao_virt = C_ao_iao_virt[np.newaxis,...]
    C_ao_iao_core = C_ao_iao_core[np.newaxis,...]

############################################################
#               rearrange orbitals
############################################################
# rearrange IAO/PAO according to their indices in cell.atom

if use_reference_mol:
    # C_ao_lo: transformation matrix from AO to LO (IAO) basis, all atoms, core orbitals are excluded
    # C_ao_lo_Co: transformation matrix from AO to LO (IAO) basis, only Co atom, core orbitals are excluded
    # C_ao_lo_tot: transformation matrix from AO to LO (IAO) basis, all atoms, all orbitals (core included!)
    C_ao_lo    = np.zeros((spin,nkpts,nao,nval+nvirt), dtype=C_ao_iao.dtype)
    C_ao_lo_Co = np.zeros((spin,nkpts,nao,nao_Co), dtype=C_ao_lo.dtype)
    C_ao_lo_tot = np.zeros((spin,nkpts,nao,nao), dtype=C_ao_lo.dtype)
    
    C_ao_lo[:,:,:,0:nval_Co] = C_ao_iao_val[:,:,:,0:nval_Co]
    C_ao_lo[:,:,:,nval_Co:nao_Co] = C_ao_iao_virt[:,:,:,0:nvirt_Co]

    for iat in range(nat_Cu):
        C_ao_lo[:,:,:,nao_Co+iat*nao_Cu:nao_Co+iat*nao_Cu+nval_Cu] \
                = C_ao_iao_val[:,:,:,nval_Co+iat*nval_Cu:nval_Co+(iat+1)*nval_Cu]

        C_ao_lo[:,:,:,nao_Co+iat*nao_Cu+nval_Cu:nao_Co+(iat+1)*nao_Cu] \
                = C_ao_iao_virt[:,:,:,nvirt_Co+iat*nvirt_Cu:nvirt_Co+(iat+1)*nvirt_Cu]

    C_ao_lo_Co = C_ao_lo[:,:,:,0:nao_Co]

    #----------------------------------------------------------
    C_ao_lo_tot[:,:,:,0:ncore_Co] = C_ao_iao_core[:,:,:,0:ncore_Co]
    C_ao_lo_tot[:,:,:,ncore_Co:ncore_Co+nval_Co] = C_ao_iao_val[:,:,:,0:nval_Co]
    C_ao_lo_tot[:,:,:,ncore_Co+nval_Co:nao_Co_tot] = C_ao_iao_virt[:,:,:,0:nvirt_Co]

    for iat in range(nat_Cu):
        C_ao_lo_tot[:,:,:,nao_Co_tot+iat*nao_Cu_tot:nao_Co_tot+iat*nao_Cu_tot+ncore_Cu] \
                = C_ao_iao_core[:,:,:,ncore_Co+iat*ncore_Cu:ncore_Co+(iat+1)*ncore_Cu]

        C_ao_lo_tot[:,:,:,nao_Co_tot+iat*nao_Cu_tot+ncore_Cu:nao_Co_tot+iat*nao_Cu_tot+ncore_Cu+nval_Cu] \
                = C_ao_iao_val[:,:,:,nval_Co+iat*nval_Cu:nval_Co+(iat+1)*nval_Cu]

        C_ao_lo_tot[:,:,:,nao_Co_tot+iat*nao_Cu_tot+ncore_Cu+nval_Cu:nao_Co_tot+(iat+1)*nao_Cu_tot] \
                = C_ao_iao_virt[:,:,:,nvirt_Co+iat*nvirt_Cu:nvirt_Co+(iat+1)*nvirt_Cu]
else:
    # TODO...
    print('rearranging orbitals not implemented!')

if np.max(np.abs(C_ao_lo_tot.imag)) < 1e-8:
    C_ao_lo = C_ao_lo.real
    C_ao_lo_Co = C_ao_lo_Co.real
    C_ao_lo_tot = C_ao_lo_tot.real

############################################################
#           Plot MO and LO
############################################################
plot_orb = False

if plot_orb:
    plotdir = datadir + '/plot_' + cell_label + '_' + method_label
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)

    plot.plot_orb_k_all(cell, plotdir + '/iao_', C_ao_iao, kpts, margin=0.0)
    plot.plot_orb_k_all(cell, plotdir + '/lo_', C_ao_lo, kpts, margin=0.0)

############################################################
#           Quantities in LO (IAO) basis
############################################################


S_ao_ao = kmf.get_ovlp()
print('S_ao_ao.shape = ', S_ao_ao.shape)


fname = datadir + '/C_ao_lo_' + cell_label + '_' + method_label + '.h5'
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

nelec_Co_tmp = 0
for s in range(spin):
    nelec_lo = np.trace(DM_lo[s].sum(axis=0)/nkpts)
    print ('Nelec (core excluded)', nelec_lo.real)
    
    nelec_lo_Co = np.trace(DM_lo_Co[s].sum(axis=0)/nkpts)
    print ('Nelec on Co', nelec_lo_Co.real)

    nelec_Co_tmp += nelec_lo_Co

print('total number of electrons on Co = ', nelec_Co_tmp)

fn = datadir + '/DM_lo_' + cell_label + '_' + method_label + '.h5'
f = h5py.File(fn, 'w')
f['DM_lo'] = DM_lo
f['DM_lo_Co'] = DM_lo_Co
f.close()

# get 4-index ERI
#eri_lo = eri_transform.get_unit_eri_fast(cell, gdf, C_ao_lo=C_ao_lo, feri=gdf_fname)
eri_lo_Co = eri_transform.get_unit_eri_fast(cell, gdf, C_ao_lo=C_ao_lo_Co, feri=gdf_fname)

fn = datadir + '/eri_lo_' + cell_label + '_' + method_label + '.h5'
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

# let C be AO-to-LO transformation matrix
# h^{LO} = \dg{C} h^{AO} C
for s in range(spin):
    for ik in range(nkpts):
        hcore_lo[s,ik] = np.dot(np.dot(C_ao_lo[s,ik].T.conj(), hcore_ao[ik]), C_ao_lo[s,ik])
        hcore_lo_Co[s,ik] = np.dot(np.dot(C_ao_lo_Co[s,ik].T.conj(), hcore_ao[ik]), C_ao_lo_Co[s,ik])
    
        JK_lo[s,ik] = np.dot(np.dot(C_ao_lo[s,ik].T.conj(), JK_ao[s,ik]), C_ao_lo[s,ik])
        JK_lo_Co[s,ik] = np.dot(np.dot(C_ao_lo_Co[s,ik].T.conj(), JK_ao[s,ik]), C_ao_lo_Co[s,ik])


fn = datadir + '/hcore_JK_lo_' + cell_label + '_' + method_label + '.h5'

f = h5py.File(fn, 'w')
f['hcore_lo'] = hcore_lo
f['hcore_lo_Co'] = hcore_lo_Co
f['JK_lo'] = JK_lo
f['JK_lo_Co'] = JK_lo_Co
f.close()

# if using DFT, get HF JK term using DFT density
if use_dft:
    if do_restricted:
        kmf_hf = scf.KRHF(cell, kpts, exxdiv='ewald')
    else:
        kmf_hf = scf.KUHF(cell, kpts, exxdiv='ewald')
    kmf_hf.with_df = gdf
    kmf_hf.with_df._cderi = gdf_fname
    kmf_hf.max_cycle = 0
    print('DM_ao.shape = ', DM_ao.shape)
    print('C_ao_lo.shape = ', C_ao_lo.shape)
    if do_restricted:
        JK_ao_hf = kmf_hf.get_veff(dm_kpts=DM_ao[0])
        JK_ao_hf = JK_ao_hf[np.newaxis,...]
    else:
        JK_ao_hf = kmf_hf.get_veff(dm_kpts=DM_ao)
    
    JK_lo_hf    = np.zeros((spin,nkpts,nval+nvirt,nval+nvirt),dtype=JK_lo.dtype)
    JK_lo_hf_Co = np.zeros((spin,nkpts,nao_Co,nao_Co),dtype=JK_lo.dtype)
    for s in range(spin):
        for ik in range(nkpts):
            JK_lo_hf[s,ik] = np.dot(np.dot(C_ao_lo[s,ik].T.conj(), JK_ao_hf[s,ik]), C_ao_lo[s,ik])
            JK_lo_hf_Co[s,ik] = np.dot(np.dot(C_ao_lo_Co[s,ik].T.conj(), JK_ao_hf[s,ik]), C_ao_lo_Co[s,ik])
    
    fn = datadir + '/hcore_JK_lo_' + cell_label + '_' + method_label + '.h5'
    f = h5py.File(fn, 'a')
    f['JK_lo_hf'] = JK_lo_hf
    f['JK_lo_hf_Co'] = JK_lo_hf_Co
    f.close()


#######################################

fname = datadir + '/CoCu_set_ham_test.h5'
fh = h5py.File(fname, 'w')
JK_ao = np.asarray(kmf.get_veff())[0]
DM_ao = np.asarray(kmf.make_rdm1())[0]
S_ao_ao = np.asarray(kmf.get_ovlp())[0]
hcore_ao = np.asarray(kmf.get_hcore())[0]
C_ao_lo_tot = C_ao_lo_tot[0,0]

DM_lo_tot = np.zeros_like(DM_ao)
Cinv = C_ao_lo_tot.T.conj() @ S_ao_ao
DM_lo_tot = Cinv @ DM_ao @ Cinv.T.conj()

fh['hcore_ao'] = hcore_ao
fh['JK_ao'] = JK_ao
fh['DM_ao'] = DM_ao
fh['S_ao_ao'] = S_ao_ao
fh['C_ao_lo_tot'] = C_ao_lo_tot
fh['DM_lo_tot'] = DM_lo_tot
fh.close()

print('hcore_ao.shape = ', hcore_ao.shape)
print('JK_ao.shape = ', JK_ao.shape)
print('S_ao_ao.shape = ', S_ao_ao.shape)
print('DM_ao.shape = ', DM_ao.shape)
print('C_ao_lo_tot.shape = ', C_ao_lo_tot.shape)
print('DM_lo_tot.shape = ', DM_lo_tot.shape)

#######################################


