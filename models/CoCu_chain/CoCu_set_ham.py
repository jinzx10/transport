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
ncore_Co = 9
nval_Co = 6
nvirt_Co = 16
nao_Co = ncore_Co + nval_Co + nvirt_Co

Cu_basis = 'def2-svp-bracket'
ncore_Cu = 9
nval_Cu = 6
nvirt_Cu = 9
nao_Cu = ncore_Cu + nval_Cu + nvirt_Cu

############################################################
#               low-level mean-field method
############################################################
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
    cell.max_memory = 100000
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
    # C_ao_lo: transformation matrix from AO to LO (IAO) basis
    # all : include core+val+virt for all atoms
    # nc  : include val+virt for all atoms
    # ncCo: include val+virt for Co atom only
    C_ao_lo_all  = np.zeros((spin,nkpts,nao,nao), dtype=complex)
    C_ao_lo_nc   = np.zeros((spin,nkpts,nao,nval+nvirt), dtype=complex)
    C_ao_lo_ncCo = np.zeros((spin,nkpts,nao,nval_Co+nvirt_Co), dtype=complex)
    
    '''
    rearrange orbitals strictly according to atomic index (Co is in the middle)
    for s in range(spin):
        # left lead
        for iat in range(nl):
            # core
            C_ao_lo_all[s,:,:,iat*nao_Cu:iat*nao_Cu+ncore_Cu] \
                    = C_ao_iao[:,:,iat*ncore_Cu:(iat+1)*ncore_Cu]
    
            # val
            C_ao_lo_all[s,:,:,iat*nao_Cu+ncore_Cu:iat*nao_Cu+ncore_Cu+nval_Cu] \
                    = C_ao_iao[:,:,ncore+iat*nval_Cu:ncore+(iat+1)*nval_Cu]
            C_ao_lo_nc[s,:,:,iat*(nval_Cu+nvirt_Cu):iat*(nval_Cu+nvirt_Cu)+nval_Cu] \
                    = C_ao_iao[:,:,ncore+iat*nval_Cu:ncore+(iat+1)*nval_Cu]
    
            # virt
            C_ao_lo_all[s,:,:,iat*nao_Cu+ncore_Cu+nval_Cu:(iat+1)*nao_Cu] \
                    = C_ao_iao[:,:,ncore+nval+iat*nvirt_Cu:ncore+nval+(iat+1)*nvirt_Cu]
            C_ao_lo_nc[s,:,:,iat*(nval_Cu+nvirt_Cu)+nval_Cu:(iat+1)*(nval_Cu+nvirt_Cu)] \
                    = C_ao_iao[:,:,ncore+nval+iat*nvirt_Cu:ncore+nval+(iat+1)*nvirt_Cu]
    
        # Co
        # core
        C_ao_lo_all[s,:,:,nl*nao_Cu:nl*nao_Cu+ncore_Co] \
                = C_ao_iao[:,:,nl*ncore_Cu:nl*ncore_Cu+ncore_Co]
    
        # val
        C_ao_lo_all[s,:,:,nl*nao_Cu+ncore_Co:nl*nao_Cu+ncore_Co+nval_Co] \
                = C_ao_iao[:,:,ncore+nl*nval_Cu:ncore+nl*nval_Cu+nval_Co]
        C_ao_lo_nc[s,:,:,nl*(nval_Cu+nvirt_Cu):nl*(nval_Cu+nvirt_Cu)+nval_Co] \
                = C_ao_iao[:,:,ncore+nl*nval_Cu:ncore+nl*nval_Cu+nval_Co]
    
        # virt
        C_ao_lo_all[s,:,:,nl*nao_Cu+ncore_Co+nval_Co:nl*nao_Cu+nao_Co] \
                = C_ao_iao[:,:,ncore+nval+nl*nvirt_Cu:ncore+nval+nl*nvirt_Cu+nvirt_Co]
        C_ao_lo_nc[s,:,:,nl*(nval_Cu+nvirt_Cu)+nval_Co:nl*(nval_Cu+nvirt_Cu)+nval_Co+nvirt_Co] \
                = C_ao_iao[:,:,ncore+nval+nl*nvirt_Cu:ncore+nval+nl*nvirt_Cu+nvirt_Co]
        
    
        # right lead
        for iat in range(nr):
            # core
            C_ao_lo_all[s,:,:,nl*nao_Cu+nao_Co+iat*nao_Cu:nl*nao_Cu+nao_Co+iat*nao_Cu+ncore_Cu] \
                    = C_ao_iao[:,:,nl*ncore_Cu+ncore_Co+iat*ncore_Cu:nl*ncore_Cu+ncore_Co+(iat+1)*ncore_Cu]
    
            # val
            C_ao_lo_all[s,:,:,nl*nao_Cu+nao_Co+iat*nao_Cu+ncore_Cu:nl*nao_Cu+nao_Co+iat*nao_Cu+ncore_Cu+nval_Cu] \
                    = C_ao_iao[:,:,ncore+nl*nval_Cu+nval_Co+iat*nval_Cu:ncore+nl*nval_Cu+nval_Co+(iat+1)*nval_Cu]
            C_ao_lo_nc[s,:,:,(nl+iat)*(nval_Cu+nvirt_Cu)+nval_Co+nvirt_Co:(nl+iat)*(nval_Cu+nvirt_Cu)+nval_Co+nvirt_Co+nval_Cu] \
                    = C_ao_iao[:,:,ncore+nl*nval_Cu+nval_Co+iat*nval_Cu:ncore+nl*nval_Cu+nval_Co+(iat+1)*nval_Cu]
    
            # virt
            C_ao_lo_all[s,:,:,nl*nao_Cu+nao_Co+iat*nao_Cu+ncore_Cu+nval_Cu:nl*nao_Cu+nao_Co+(iat+1)*nao_Cu] \
                    = C_ao_iao[:,:,ncore+nval+nl*nvirt_Cu+nvirt_Co+iat*nvirt_Cu:ncore+nval+nl*nvirt_Cu+nvirt_Co+(iat+1)*nvirt_Cu]
            C_ao_lo_nc[s,:,:,(nl+iat)*(nval_Cu+nvirt_Cu)+nval_Co+nvirt_Co+nval_Cu:(nl+iat+1)*(nval_Cu+nvirt_Cu)+nval_Co+nvirt_Co] \
                    = C_ao_iao[:,:,ncore+nval+nl*nvirt_Cu+nvirt_Co+iat*nvirt_Cu:ncore+nval+nl*nvirt_Cu+nvirt_Co+(iat+1)*nvirt_Cu]
    
    C_ao_lo_ncCo = C_ao_lo_nc[:,:,:,nl*(nval_Cu+nvirt_Cu):nl*(nval_Cu+nvirt_Cu)+nval_Co+nvirt_Co]
    '''

    # rearrange orbitals so that Co comes first
    for s in range(spin):
        # Co
        # core
        C_ao_lo_all[s,:,:,:ncore_Co] \
                = C_ao_iao[:,:,nl*ncore_Cu:nl*ncore_Cu+ncore_Co]
    
        # val
        C_ao_lo_all[s,:,:,ncore_Co:ncore_Co+nval_Co] \
                = C_ao_iao[:,:,ncore+nl*nval_Cu:ncore+nl*nval_Cu+nval_Co]
        C_ao_lo_nc[s,:,:,:nval_Co] \
                = C_ao_iao[:,:,ncore+nl*nval_Cu:ncore+nl*nval_Cu+nval_Co]
    
        # virt
        C_ao_lo_all[s,:,:,ncore_Co+nval_Co:nao_Co] \
                = C_ao_iao[:,:,ncore+nval+nl*nvirt_Cu:ncore+nval+nl*nvirt_Cu+nvirt_Co]
        C_ao_lo_nc[s,:,:,nval_Co:nval_Co+nvirt_Co] \
                = C_ao_iao[:,:,ncore+nval+nl*nvirt_Cu:ncore+nval+nl*nvirt_Cu+nvirt_Co]
        

        # left lead
        for iat in range(nl):
            # core
            C_ao_lo_all[s,:,:,nao_Co+iat*nao_Cu:nao_Co+iat*nao_Cu+ncore_Cu] \
                    = C_ao_iao[:,:,iat*ncore_Cu:(iat+1)*ncore_Cu]
    
            # val
            C_ao_lo_all[s,:,:,nao_Co+iat*nao_Cu+ncore_Cu:nao_Co+iat*nao_Cu+ncore_Cu+nval_Cu] \
                    = C_ao_iao[:,:,ncore+iat*nval_Cu:ncore+(iat+1)*nval_Cu]
            C_ao_lo_nc[s,:,:,nval_Co+nvirt_Co+iat*(nval_Cu+nvirt_Cu):nval_Co+nvirt_Co+iat*(nval_Cu+nvirt_Cu)+nval_Cu] \
                    = C_ao_iao[:,:,ncore+iat*nval_Cu:ncore+(iat+1)*nval_Cu]
    
            # virt
            C_ao_lo_all[s,:,:,nao_Co+iat*nao_Cu+ncore_Cu+nval_Cu:nao_Co+(iat+1)*nao_Cu] \
                    = C_ao_iao[:,:,ncore+nval+iat*nvirt_Cu:ncore+nval+(iat+1)*nvirt_Cu]
            C_ao_lo_nc[s,:,:,nval_Co+nvirt_Co+iat*(nval_Cu+nvirt_Cu)+nval_Cu:nval_Co+nvirt_Co+(iat+1)*(nval_Cu+nvirt_Cu)] \
                    = C_ao_iao[:,:,ncore+nval+iat*nvirt_Cu:ncore+nval+(iat+1)*nvirt_Cu]
    
    
        # right lead
        for iat in range(nr):
            # core
            C_ao_lo_all[s,:,:,nl*nao_Cu+nao_Co+iat*nao_Cu:nl*nao_Cu+nao_Co+iat*nao_Cu+ncore_Cu] \
                    = C_ao_iao[:,:,nl*ncore_Cu+ncore_Co+iat*ncore_Cu:nl*ncore_Cu+ncore_Co+(iat+1)*ncore_Cu]
    
            # val
            C_ao_lo_all[s,:,:,nl*nao_Cu+nao_Co+iat*nao_Cu+ncore_Cu:nl*nao_Cu+nao_Co+iat*nao_Cu+ncore_Cu+nval_Cu] \
                    = C_ao_iao[:,:,ncore+nl*nval_Cu+nval_Co+iat*nval_Cu:ncore+nl*nval_Cu+nval_Co+(iat+1)*nval_Cu]
            C_ao_lo_nc[s,:,:,(nl+iat)*(nval_Cu+nvirt_Cu)+nval_Co+nvirt_Co:(nl+iat)*(nval_Cu+nvirt_Cu)+nval_Co+nvirt_Co+nval_Cu] \
                    = C_ao_iao[:,:,ncore+nl*nval_Cu+nval_Co+iat*nval_Cu:ncore+nl*nval_Cu+nval_Co+(iat+1)*nval_Cu]
    
            # virt
            C_ao_lo_all[s,:,:,nl*nao_Cu+nao_Co+iat*nao_Cu+ncore_Cu+nval_Cu:nl*nao_Cu+nao_Co+(iat+1)*nao_Cu] \
                    = C_ao_iao[:,:,ncore+nval+nl*nvirt_Cu+nvirt_Co+iat*nvirt_Cu:ncore+nval+nl*nvirt_Cu+nvirt_Co+(iat+1)*nvirt_Cu]
            C_ao_lo_nc[s,:,:,(nl+iat)*(nval_Cu+nvirt_Cu)+nval_Co+nvirt_Co+nval_Cu:(nl+iat+1)*(nval_Cu+nvirt_Cu)+nval_Co+nvirt_Co] \
                    = C_ao_iao[:,:,ncore+nval+nl*nvirt_Cu+nvirt_Co+iat*nvirt_Cu:ncore+nval+nl*nvirt_Cu+nvirt_Co+(iat+1)*nvirt_Cu]
    
    C_ao_lo_ncCo = C_ao_lo_nc[:,:,:,:nval_Co+nvirt_Co]

else:
    # TBD...
    print('rearranging orbitals not implemented!')


############################################################
#           Plot MO and LO
############################################################
plot_orb = True

if plot_orb:
    plotdir = datadir + '/plot_' + label
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)

    plot.plot_orb_k_all(cell, plotdir + '/iao_' + str(nat).zfill(2), C_ao_iao, kpts, margin=0.0)
    plot.plot_orb_k_all(cell, plotdir + '/iao_all_' + str(nat).zfill(2), C_ao_lo_all[0], kpts, margin=0.0)
    plot.plot_orb_k_all(cell, plotdir + '/iao_nc_' + str(nat).zfill(2), C_ao_lo_nc[0], kpts, margin=0.0)

############################################################
#           Quantities in LO (IAO) basis
############################################################
S_ao_ao = kmf.get_ovlp()

C_mo_lo_all  = np.zeros((spin,nkpts,nao,nao), dtype=complex)
C_mo_lo_nc   = np.zeros((spin,nkpts,nao,nval+nvirt), dtype=complex)
C_mo_lo_ncCo = np.zeros((spin,nkpts,nao,nval_Co+nvirt_Co), dtype=complex)

for s in range(spin):
    for ik in range(nkpts):
        C_mo_lo_all[s][ik] = np.dot(np.dot(mo_coeff[s][ik].T.conj(), S_ao_ao[ik]), C_ao_lo_all[s][ik])
        C_mo_lo_nc[s][ik] = np.dot(np.dot(mo_coeff[s][ik].T.conj(), S_ao_ao[ik]), C_ao_lo_nc[s][ik])
        C_mo_lo_ncCo[s][ik] = np.dot(np.dot(mo_coeff[s][ik].T.conj(), S_ao_ao[ik]), C_ao_lo_ncCo[s][ik])

fname = datadir + '/C_lo_' + label + '.h5'
f = h5py.File(fname, 'w')
f['C_ao_lo_all'] = np.asarray(C_ao_lo_all)
f['C_ao_lo_nc'] = np.asarray(C_ao_lo_nc)
f['C_ao_lo_ncCo'] = np.asarray(C_ao_lo_ncCo)
f['C_mo_lo_all'] = np.asarray(C_mo_lo_all)
f['C_mo_lo_nc'] = np.asarray(C_mo_lo_nc)
f['C_mo_lo_ncCo'] = np.asarray(C_mo_lo_ncCo)
f['S_ao_ao'] = np.asarray(S_ao_ao)
f.close()

# get DFT density matrix in IAO basis
DM_ao = np.asarray(kmf.make_rdm1()) # DM_ao is real!
if len(DM_ao.shape) == 3:
    DM_ao = DM_ao[np.newaxis, ...]

DM_lo_all  = np.zeros((spin,nkpts,nao,nao),dtype=DM_ao.dtype)
DM_lo_nc   = np.zeros((spin,nkpts,nval+nvirt,nval+nvirt),dtype=DM_ao.dtype)
DM_lo_ncCo = np.zeros((spin,nkpts,nval_Co+nvirt_Co,nval_Co+nvirt_Co),dtype=DM_ao.dtype)

for s in range(spin):
    for ik in range(nkpts):
        Cinv_all = np.dot(C_ao_lo_all[s][ik].T.conj(),S_ao_ao[ik])
        DM_lo_all[s][ik] = np.dot(np.dot(Cinv_all, DM_ao[s][ik]), Cinv_all.T.conj())

        Cinv_nc = np.dot(C_ao_lo_nc[s][ik].T.conj(),S_ao_ao[ik])
        DM_lo_nc[s][ik] = np.dot(np.dot(Cinv_nc, DM_ao[s][ik]), Cinv_nc.T.conj())

        Cinv_ncCo = np.dot(C_ao_lo_ncCo[s][ik].T.conj(),S_ao_ao[ik])
        DM_lo_ncCo[s][ik] = np.dot(np.dot(Cinv_ncCo, DM_ao[s][ik]), Cinv_ncCo.T.conj())


for s in range(spin):
    nelec_lo_all = np.trace(DM_lo_all[s].sum(axis=0)/nkpts)
    print ('Nelec all', nelec_lo_all.real)

    nelec_lo_nc = np.trace(DM_lo_nc[s].sum(axis=0)/nkpts)
    print ('Nelec nc', nelec_lo_nc.real)

    nelec_lo_ncCo = np.trace(DM_lo_ncCo[s].sum(axis=0)/nkpts)
    print ('Nelec ncCo', nelec_lo_ncCo.real)

fn = datadir + '/DM_lo_' + label + '.h5'
f = h5py.File(fn, 'w')
f['DM_lo_all'] = np.asarray(DM_lo_all)
f['DM_lo_nc'] = np.asarray(DM_lo_nc)
f['DM_lo_ncCo'] = np.asarray(DM_lo_ncCo)
f.close()

# get 4-index ERI
eri_lo_all = eri_transform.get_unit_eri_fast(cell, gdf, C_ao_lo=C_ao_lo_all, feri=gdf_fname)
eri_lo_nc = eri_transform.get_unit_eri_fast(cell, gdf, C_ao_lo=C_ao_lo_nc, feri=gdf_fname)
eri_lo_ncCo = eri_transform.get_unit_eri_fast(cell, gdf, C_ao_lo=C_ao_lo_ncCo, feri=gdf_fname)

fn = datadir + '/eri_lo_' + label + '.h5'
f = h5py.File(fn, 'w')
f['eri_lo_all'] = np.asarray(eri_lo_all.real)
f['eri_lo_nc'] = np.asarray(eri_lo_nc.real)
f['eri_lo_ncCo'] = np.asarray(eri_lo_ncCo.real)
f.close()

# get one-electron integrals
hcore_ao = np.asarray(kmf.get_hcore()) # hcore_ao and JK_ao are all real!
JK_ao = np.asarray(kmf.get_veff())
if len(JK_ao.shape) == 3:
    JK_ao = JK_ao[np.newaxis, ...]

hcore_lo_all = np.zeros((spin,nkpts,nao,nao),dtype=complex)
hcore_lo_nc = np.zeros((spin,nkpts,nval+nvirt,nval+nvirt),dtype=complex)
hcore_lo_ncCo = np.zeros((spin,nkpts,nval_Co+nvirt_Co,nval_Co+nvirt_Co),dtype=complex)

JK_lo_all = np.zeros((spin,nkpts,nao,nao),dtype=complex)
JK_lo_nc = np.zeros((spin,nkpts,nval+nvirt,nval+nvirt),dtype=complex)
JK_lo_ncCo = np.zeros((spin,nkpts,nval_Co+nvirt_Co,nval_Co+nvirt_Co),dtype=complex)

for s in range(spin):
    for ik in range(nkpts):
        hcore_lo_all[s,ik] = np.dot(np.dot(C_ao_lo_all[s,ik].T.conj(), hcore_ao[ik]), C_ao_lo_all[s,ik])
        hcore_lo_nc[s,ik] = np.dot(np.dot(C_ao_lo_nc[s,ik].T.conj(), hcore_ao[ik]), C_ao_lo_nc[s,ik])
        hcore_lo_ncCo[s,ik] = np.dot(np.dot(C_ao_lo_ncCo[s,ik].T.conj(), hcore_ao[ik]), C_ao_lo_ncCo[s,ik])

        JK_lo_all[s,ik] = np.dot(np.dot(C_ao_lo_all[s,ik].T.conj(), JK_ao[s,ik]), C_ao_lo_all[s,ik])
        JK_lo_nc[s,ik] = np.dot(np.dot(C_ao_lo_nc[s,ik].T.conj(), JK_ao[s,ik]), C_ao_lo_nc[s,ik])
        JK_lo_ncCo[s,ik] = np.dot(np.dot(C_ao_lo_ncCo[s,ik].T.conj(), JK_ao[s,ik]), C_ao_lo_ncCo[s,ik])


fn = datadir + '/hcore_JK_lo_dft_' + label + '.h5'
f = h5py.File(fn, 'w')
f['hcore_lo_all'] = np.asarray(hcore_lo_all)
f['hcore_lo_nc'] = np.asarray(hcore_lo_nc)
f['hcore_lo_ncCo'] = np.asarray(hcore_lo_ncCo)
f['JK_lo_all'] = np.asarray(JK_lo_all)
f['JK_lo_nc'] = np.asarray(JK_lo_nc)
f['JK_lo_ncCo'] = np.asarray(JK_lo_ncCo)
f.close()

assert(np.max(np.abs(hcore_lo_all.sum(axis=1).imag/nkpts))<1e-6)
assert(np.max(np.abs(JK_lo_all.sum(axis=1).imag/nkpts))<1e-6)

# get HF JK term using DFT density
kmf_hf = scf.KRHF(cell, kpts, exxdiv='ewald')
kmf_hf.with_df = gdf
kmf_hf.with_df._cderi = gdf_fname
kmf_hf.max_cycle = 0
JK_ao = np.asarray(kmf_hf.get_veff(dm_kpts=DM_ao[0]))
if len(JK_ao.shape) == 3:
    JK_ao = JK_ao[np.newaxis, ...]

JK_lo_all = np.zeros((spin,nkpts,nao,nao),dtype=complex)
JK_lo_nc = np.zeros((spin,nkpts,nval+nvirt,nval+nvirt),dtype=complex)
JK_lo_ncCo = np.zeros((spin,nkpts,nval_Co+nvirt_Co,nval_Co+nvirt_Co),dtype=complex)
for s in range(spin):
    for ik in range(nkpts):
        JK_lo_all[s,ik] = np.dot(np.dot(C_ao_lo_all[s,ik].T.conj(), JK_ao[s,ik]), C_ao_lo_all[s,ik])
        JK_lo_nc[s,ik] = np.dot(np.dot(C_ao_lo_nc[s,ik].T.conj(), JK_ao[s,ik]), C_ao_lo_nc[s,ik])
        JK_lo_ncCo[s,ik] = np.dot(np.dot(C_ao_lo_ncCo[s,ik].T.conj(), JK_ao[s,ik]), C_ao_lo_ncCo[s,ik])

# HF only differs from DFT by the JK part
fn = datadir + '/JK_lo_hf_' + label + '.h5'
f = h5py.File(fn, 'w')
f['JK_lo_all'] = np.asarray(JK_lo_all)
f['JK_lo_nc'] = np.asarray(JK_lo_nc)
f['JK_lo_ncCo'] = np.asarray(JK_lo_ncCo)
f.close()


