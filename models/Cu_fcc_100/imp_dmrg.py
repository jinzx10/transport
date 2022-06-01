import numpy as np
import h5py, time, sys, os, scipy
from mpi4py import MPI

from fcdmft.solver import scf_mu

import matplotlib.pyplot as plt
from matplotlib import colors

from pyscf import gto, ao2mo, cc, scf, dft
from pyscf.lib import chkfile

from pyscf.pbc import scf as pbcscf
from pyscf.pbc import df as pbcdf
from pyscf.pbc.lib import chkfile as pbcchkfile

from utils.surface_green import *
from utils.bath_disc import *

from pyscf.scf.hf import eig as eiggen

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

mode = 'MODE'

imp_atom = 'IMP_ATOM' if mode == 'production' else 'Co'

# chemical potential
mu = CHEM_POT if mode == 'production' else 0.05

# DMRG scratch & save folder
scratch = 'SCRATCH'
save_dir = 'SAVEDIR'
os.environ['TMPDIR'] = scratch

# number of active space occ/vir orbitals
nocc_act = NOCC_ACT if mode == 'production' else 6
nvir_act = NVIR_ACT if mode == 'production' else 9
n_no = nocc_act + nvir_act


######################################
#       DMRG parameters
######################################
# NOTE scratch conflict for different job!
# NOTE verbose=3 good enough
# NOTE extra_freqs & extra_delta are set below
#extra_freqs = None
#extra_delta = None
n_threads = NUM_THREADS if mode == 'production' else 8
reorder_method='gaopt'

# bond dimension: make DW ~ 1e-5/e-6 or smaller
# 2000-5000 for gs
gs_n_steps = 20
gs_bond_dims = [400] * 5 + [800] * 5 + [1500] * 5 + [2000] * 5
gs_tol = 1E-8
gs_noises = [1E-3] * 5 + [1E-4] * 3 + [1e-7] * 2 + [0]

# gf_bond_dims 1k~1.5k
gf_n_steps = 8
gf_bond_dims = [500] * 2 + [1000] * 4
gf_tol = 1E-4
gf_noises = [1E-4] * 2 + [1E-5] * 2 + [1E-7] * 2 + [0]

gmres_tol = 1E-7

cps_bond_dims = [2000]
#cps_noises = [0]
#cps_tol = 1E-10
#cps_n_steps=20
n_off_diag_cg = -2

# correlation method within DMRG (like CAS, but not recommended)
dyn_corr_method = None
ncore_dmrg = 0
nvirt_dmrg = 0

diag_only= False


############################################################
#       frequencies to compute spectra
############################################################
# coarse grid
wl_freqs = mu - 0.01
wh_freqs = mu + 0.01
delta = 0.002
nw = 20
freqs = np.linspace(wl_freqs, wh_freqs, nw)
dw = freqs[1] - freqs[0]

extra_delta = 0.001
extra_nw = 5
extra_dw = dw / extra_nw

extra_freqs = []
for i in range(nw):
    freqs_tmp = []
    if extra_nw % 2 == 0:
        for w in range(-extra_nw // 2, extra_nw // 2):
            freqs_tmp.append(freqs[i] + extra_dw * w)
    else:
        for w in range(-(extra_nw-1) // 2, (extra_nw+1) // 2):
            freqs_tmp.append(freqs[i] + extra_dw * w)
    extra_freqs.append(np.array(freqs_tmp))

all_freqs = np.array(sorted(list(set(list(freqs) + \
        ([x for xx in extra_freqs for x in xx] if extra_freqs is not None else [])))))

nwa = len(all_freqs)

############################################################
#               some auxiliary functions
############################################################
def scdm(coeff, overlap):
    from pyscf import lo
    aux = lo.orth.lowdin(overlap)
    no = coeff.shape[1]
    ova = coeff.T @ overlap @ aux
    piv = scipy.linalg.qr(ova, pivoting=True)[2]
    bc = ova[:, piv[:no]]
    ova = np.dot(bc.T, bc)
    s12inv = lo.orth.lowdin(ova)
    return coeff @ bc @ s12inv

def _get_jk(dm, eri):
    """
    Get J and K potential from rdm and ERI.
    vj00 = np.tensordot(dm[0], eri[0], ((0,1), (0,1))) # J a from a
    vj11 = np.tensordot(dm[1], eri[1], ((0,1), (0,1))) # J b from b
    vj10 = np.tensordot(dm[0], eri[2], ((0,1), (0,1))) # J b from a
    vj01 = np.tensordot(dm[1], eri[2], ((1,0), (3,2))) # J a from b
    vk00 = np.tensordot(dm[0], eri[0], ((0,1), (0,3))) # K a from a
    vk11 = np.tensordot(dm[1], eri[1], ((0,1), (0,3))) # K b from b
    JK = np.asarray([vj00 + vj01 - vk00, vj11 + vj10 - vk11])
    """
    dm = np.asarray(dm, dtype=np.double)
    eri = np.asarray(eri, dtype=np.double)
    if len(dm.shape) == 2:
        dm = dm[np.newaxis, ...]
    if len(eri.shape) == 4:
        eri = eri[np.newaxis, ...]
    spin = dm.shape[0]
    norb = dm.shape[-1]
    if spin == 1:
        eri = ao2mo.restore(8, eri, norb)
        vj, vk = scf.hf.dot_eri_dm(eri, dm, hermi=1)
    else:
        eri_aa = ao2mo.restore(8, eri[0], norb)
        eri_bb = ao2mo.restore(8, eri[1], norb)
        eri_ab = ao2mo.restore(4, eri[2], norb)
        vj00, vk00 = scf.hf.dot_eri_dm(eri_aa, dm[0], hermi=1)
        vj11, vk11 = scf.hf.dot_eri_dm(eri_bb, dm[1], hermi=1)
        vj01, _ = scf.hf.dot_eri_dm(eri_ab, dm[1], hermi=1, with_j=True, with_k=False)
        # ZHC NOTE the transpose, since the dot_eri_dm uses the convention ijkl, kl -> ij
        vj10, _ = scf.hf.dot_eri_dm(eri_ab.T, dm[0], hermi=1, with_j=True, with_k=False)
        # ZHC NOTE explicit write down vj, without broadcast
        vj = np.asarray([[vj00, vj11], [vj01, vj10]])
        vk = np.asarray([vk00, vk11])
    return vj, vk

def _get_veff(dm, eri):
    """
    Get HF effective potential from rdm and ERI.
    """
    dm = np.asarray(dm, dtype=np.double)
    if len(dm.shape) == 2:
        dm = dm[np.newaxis, ...]
    spin = dm.shape[0]
    vj, vk = _get_jk(dm, eri)
    if spin == 1:
        JK = vj - vk*0.5 
    else:
        JK = vj[0] + vj[1] - vk
    return JK


############################################################
#           CAS from CISD natural orbital
############################################################
def gen_cas(emb_mf):

    from pyscf import ci
    
    emb_cisd = ci.CISD(emb_mf)
    emb_cisd.max_cycle = 1000
    emb_cisd.max_space = 20
    emb_cisd.max_memory = 100000
    emb_cisd.conv_tol = 1e-10
    emb_cisd.verbose = 4
    
    emb_cisd_fname = 'emb_cisd_' + imp_atom + '.chk'
    if rank == 0:
        if os.path.isfile(emb_cisd_fname):
            print('load CISD data from', emb_cisd_fname)
            emb_cisd.__dict__.update( chkfile.load(emb_cisd_fname, 'cisd') )
        else:
            emb_cisd.chkfile = emb_cisd_fname
            print('CISD starts')
            emb_cisd.kernel()
            emb_cisd.dump_chk()
    
        dm_ci_mo = emb_cisd.make_rdm1()
        
        nmo = emb_cisd.nmo
        nocc = emb_cisd.nocc # number of occupied MO
        print('nocc = ', nocc)
        print('nmo = ', nmo)
    
        print('dm_ci_mo.shape = ', dm_ci_mo.shape)
    
        # find natural orbital coefficients below
    
        # dm_ci_mo virtual block
        no_occ_v, no_coeff_v = np.linalg.eigh(dm_ci_mo[nocc:,nocc:])
        no_occ_v = np.flip(no_occ_v) # sort eigenvalues from large to small
        no_coeff_v = np.flip(no_coeff_v, axis=1)
        print('vir NO occupancy:', no_occ_v)
    
        # occupied block
        no_occ_o, no_coeff_o = np.linalg.eigh(dm_ci_mo[:nocc,:nocc])
        no_occ_o = np.flip(no_occ_o)
        no_coeff_o = np.flip(no_coeff_o, axis=1)
        print('occ NO occupancy:', no_occ_o)
    
        # use natural orbitals closest to the Fermi level
        # these indices are within their own block (say, no_idx_v starts from 0)
        no_idx_v = range(0, nvir_act)
        no_idx_o = range(nocc-nocc_act, nocc)
        
        print('no_idx_v = ', no_idx_v)
        print('no_idx_o = ', no_idx_o)
    
        # semi-canonicalization
        # rotate occ/vir NOs so that they diagonalize the Fock matrix within their subspace
        fvv = np.diag(emb_mf.mo_energy[nocc:])
        fvv_no = no_coeff_v.T @ fvv @ no_coeff_v
        _, v_canon_v = np.linalg.eigh(fvv_no[:nvir_act,:nvir_act])
        
        foo = np.diag(emb_mf.mo_energy[:nocc])
        foo_no = no_coeff_o.T @ foo @ no_coeff_o
        _, v_canon_o = np.linalg.eigh(foo_no[-nocc_act:,-nocc_act:])
        
        # at this stage, no_coeff is bare MO-to-NO coefficient (before semi-canonicalization)
        no_coeff_v = emb_mf.mo_coeff[:,nocc:] @ no_coeff_v[:,:nvir_act] @ v_canon_v
        no_coeff_o = emb_mf.mo_coeff[:,:nocc] @ no_coeff_o[:,-nocc_act:] @ v_canon_o
        # now no_coeff is AO-to-NO coefficient (with semi-canonicalization)
        
        ne_sum = np.sum(no_occ_o[no_idx_o]) + np.sum(no_occ_v[no_idx_v])
        nelectron = int(round(ne_sum))
        
        print('number of electrons in NO: ', ne_sum)
        print('number of electrons: ', nelectron)
        print('number of NOs: ', n_no)
    
    
        # NOTE still do not understand scdm, but 'local' might be important!
        no_coeff_o = scdm(no_coeff_o, np.eye(no_coeff_o.shape[0]))
        no_coeff_v = scdm(no_coeff_v, np.eye(no_coeff_v.shape[0]))
        no_coeff = np.concatenate((no_coeff_o, no_coeff_v), axis=1)
    
        print('natural orbital coefficients computed!')
    
    # final natural orbital coefficients for CAS
    no_coeff = comm.bcast(no_coeff, root=0)

    # new mf object for CAS
    # first build CAS mol object
    mol_cas = gto.M()
    mol_cas.nelectron = nelectron
    mol_cas.verbose = 4
    mol_cas.symmetry = 'c1'
    mol_cas.incore_anyway = True
    emb_mf_cas = scf.RHF(mol_cas)
    
    # hcore & eri in NO basis
    h1e = no_coeff.T @ (emb_mf.get_hcore() @ no_coeff)
    g2e = ao2mo.restore(8, ao2mo.kernel(emb_mf._eri, no_coeff), n_no)
    
    
    dm_hf = emb_mf.make_rdm1()
    dm_cas_no = no_coeff.T @ dm_hf @ no_coeff
    JK_cas_no = _get_veff(dm_cas_no, g2e)[0]
    JK_full_no = no_coeff.T @ emb_mf.get_veff() @ no_coeff
    h1e = h1e + JK_full_no - JK_cas_no
    h1e = 0.5 * (h1e + h1e.T)
    
    h1e = comm.bcast(h1e, root=0)
    g2e = comm.bcast(g2e, root=0)
    
    # set up integrals for emb_mf_cas
    emb_mf_cas.get_hcore = lambda *args: h1e
    emb_mf_cas.get_ovlp = lambda *args: np.eye(n_no)
    emb_mf_cas._eri = g2e
    
    if rank == 0:
        # NOTE why max_cycle is not 0 or do a full converged calculation?
        print('scf within CAS')
        emb_mf_cas.kernel(dm0=dm_cas_no)
        print('scf within CAS finished!')
    comm.Barrier()
    
    emb_mf_cas.mo_occ = comm.bcast(emb_mf_cas.mo_occ, root=0)
    emb_mf_cas.mo_energy = comm.bcast(emb_mf_cas.mo_energy, root=0)
    emb_mf_cas.mo_coeff = comm.bcast(emb_mf_cas.mo_coeff, root=0)

    # returns a mf object for the small CAS space, and natural orbital coefficients
    # that construct the CAS orbitals
    return emb_mf_cas, no_coeff


############################################################
#               embedding rdm
############################################################
def get_rdm_emb(emb_mf):
    from fcdmft.solver.gfdmrg import dmrg_mo_pdm

    emb_mf_cas, no_coeff = gen_cas(emb_mf)

    # low level dm in CAS space (natural orbital basis)
    dm_low_cas = emb_mf_cas.make_rdm1()

    # low level dm in AO space
    dm_low = emb_mf.make_rdm1()

    max_memory = int(emb_mf.max_memory * 1E6/nprocs) # in unit of bytes, per mpi proc

    if rank == 0:
        if not os.path.isdir(scratch):
            os.mkdir(scratch)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

    # DM within CAS
    dm_cas = dmrg_mo_pdm(emb_mf_cas, ao_orbs=range(0, n_no), mo_orbs=None, scratch=scratch, \
            reorder_method=reorder_method, n_threads=n_threads, memory=max_memory, \
            gs_bond_dims=gs_bond_dims, gs_n_steps=gs_n_steps, gs_tol=gs_tol, gs_noises=gs_noises, \
            load_dir=None, save_dir=save_dir, verbose=3, mo_basis=False, ignore_ecore=False, \
            mpi=True, dyn_corr_method=dyn_corr_method, ncore=ncore_dmrg, nvirt=nvirt_dmrg)

    print('dm_cas.shape:', dm_cas.shape)
    dm_cas = dm_cas[0] + dm_cas[1]

    # difference between high and low level DM for CAS space
    ddm = dm_cas - dm_low_cas
    
    # transform the difference to AO basis
    ddm = no_coeff @ ddm @ no_coeff.T

    # total embedding DM in AO space, equals to low-level dm plus high-level correction
    rdm = dm_low + ddm

    return rdm


############################################################
#           embedding model gf (AO basis)
############################################################
def get_gf_emb(emb_mf):
    from fcdmft.solver.gfdmrg import dmrg_mo_gf

    emb_mf_cas, no_coeff = gen_cas(emb_mf)

    if rank == 0:
        if not os.path.isdir(scratch):
            os.mkdir(scratch)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

    max_memory = int(emb_mf.max_memory * 1E6/nprocs) # in unit of bytes, per mpi proc

    # gf_dmrg_cas has dimension (n_no, n_no, nwa) (frequency is its last dimension)
    dm_dmrg_cas, gf_dmrg_cas = dmrg_mo_gf( \
            emb_mf_cas, freqs=freqs, delta=delta, ao_orbs=range(0,n_no), mo_orbs=None, \
            extra_freqs=extra_freqs, extra_delta=extra_delta, scratch=scratch, add_rem='+-', \
            n_threads=n_threads, reorder_method=reorder_method, memory=max_memory, \
            gs_bond_dims=gs_bond_dims, gf_bond_dims=gf_bond_dims, gf_n_steps=gf_n_steps, \
            gs_n_steps=gs_n_steps, gs_tol=gs_tol, gf_noises=gf_noises, gf_tol=gf_tol, \
            gs_noises=gs_noises, gmres_tol=gmres_tol, load_dir=None, save_dir=save_dir, \
            cps_bond_dims=cps_bond_dims, cps_noises=[0], cps_tol=gs_tol, cps_n_steps=gs_n_steps, \
            verbose=3, mo_basis=False, ignore_ecore=False, n_off_diag_cg=n_off_diag_cg, \
            mpi=True, dyn_corr_method=dyn_corr_method, ncore=ncore_dmrg, nvirt=nvirt_dmrg, diag_only=diag_only)

    print('gf_dmrg_cas.shape', gf_dmrg_cas.shape)
    
    fh = h5py.File('gf_dmrg_cas_' + imp_atom + '.h5', 'w')
    fh['gf_dmrg_cas'] = gf_dmrg_cas
    fh.close()
    
    # hf gf in CAS space (in NO basis)
    gf_hf_cas = np.zeros((nwa, n_no, n_no), dtype=complex)
    for iw in range(nwa):
        z = all_freqs[iw] + 1j*extra_delta
        gf_mo_cas = np.diag( 1. / (z - emb_mf_cas.mo_energy) )
        gf_hf_cas[iw,:,:] = emb_mf_cas.mo_coeff @ gf_mo_cas @ emb_mf_cas.mo_coeff.T

    # NO active space self energy
    sigma_cas = np.zeros((nwa, n_no, n_no), dtype=complex)
    for iw in range(nwa):
        sigma_cas[iw,:,:] = np.linalg.inv(gf_hf_cas[iw,:,:]) - np.linalg.inv(gf_dmrg_cas[:,:,iw])
    
    # convert sigma_cas to self energy in AO space
    sigma_ao = np.zeros((nwa, nemb, nemb), dtype=complex)
    for iw in range(nwa):
        sigma_ao[iw,:,:] = no_coeff @ (sigma_cas[iw,:,:] @ no_coeff.T)
    
    # hf gf in AO basis
    gf_hf = np.zeros((nwa, nemb, nemb), dtype=complex)
    for iw in range(nwa):
        z = all_freqs[iw] + 1j*extra_delta
        gf_mo = np.diag( 1. / (z - emb_mf.mo_energy) )
        gf_hf[iw,:,:] = emb_mf.mo_coeff @ gf_mo @ emb_mf.mo_coeff.T

    # final GF in AO space (contains active space DMRG self energy)
    gf_ao = np.zeros((nwa, nemb, nemb), dtype=complex)
    for iw in range(nwa):
        gf_ao[iw,:,:] = np.linalg.inv( np.linalg.inv(gf_hf[iw,:,:]) - sigma_ao[iw,:,:] )

    return gf_ao


############################################################
#           read contact's mean-field data
############################################################
contact_dir = '/home/zuxin/projects/transport/models/Cu_fcc_100/' + imp_atom + '/'
if rank == 0:
    print('read contact\'s mean field data from', contact_dir)

imp_basis = 'def2-svp'
Cu_basis = 'def2-svp-bracket'

if imp_atom == 'Co':
    nl = 4
else:
    nl = 2

nr = 3
l = 1.8
r = 1.8
a = 3.6

cell_label = imp_atom + '_' + imp_basis + '_Cu_' + Cu_basis \
        + '_nl' + str(nl) + '_nr' + str(nr) \
        + '_l' + str(l) + '_r' + str(r) + '_a' + str(a)

cell_fname = contact_dir + '/cell_' + cell_label + '.chk'

gdf_fname = contact_dir + '/cderi_' + cell_label + '.h5'

xcfun = 'pbe0'
method_label = 'rks_' + xcfun
solver_label = 'newton'

data_fname = contact_dir + '/data_contact_' + cell_label + '_' \
        + method_label + '_' + solver_label + '.h5'

#------------ read core Hamiltonian and HF veff (built with DFT DM)  ------------
fh = h5py.File(data_fname, 'r')

# imp atom block only
hcore_lo_imp = np.asarray(fh['hcore_lo_imp'])
JK_lo_ks_imp = np.asarray(fh['JK_lo_imp'])
JK_lo_hf_imp = np.asarray(fh['JK_lo_hf_imp'])

# entire center region, imp + some Cu atoms
hcore_lo_contact = np.asarray(fh['hcore_lo'])
JK_lo_ks_contact = np.asarray(fh['JK_lo'])
JK_lo_hf_contact = np.asarray(fh['JK_lo_hf'])

#------------ read density matrix ------------
DM_lo_imp = np.asarray(fh['DM_lo_imp'])

#------------ read ERI ------------
eri_lo_imp = np.asarray(fh['eri_lo_imp'])
fh.close()

#************ permute eri for unrestricted case ************
# see Tianyu's run_dmft.py 
if eri_lo_imp.shape[0] == 3:
    eri_permuted = np.zeros_like(eri_lo_imp)
    eri_permuted[0] = eri_lo_imp[0]
    eri_permuted[1] = eri_lo_imp[2]
    eri_permuted[2] = eri_lo_imp[1]
    eri_lo_imp = eri_permuted.copy()
    del eri_permuted

# size of the imp block (should be 6+16=22 for def2-svp)
nao_imp = hcore_lo_imp.shape[2]

# 3d+4s
nval_imp = 6

# size of the contact block (should be 22+nat_Cu_contact*(6+9) for {imp:'def2-svp', Cu:'def2-svp-bracket'})
nao_contact = hcore_lo_contact.shape[2]

# restricted -> spin=1
# unrestricted -> spin=2
spin, nkpts = hcore_lo_contact.shape[0:2]

# hcore is a 4-d array with shape (spin, nkpts, nao, nao)
if rank == 0:
    print('hcore_lo_contact.shape = ', hcore_lo_contact.shape)
    print('JK_lo_ks_contact.shape = ', JK_lo_ks_contact.shape)
    print('JK_lo_hf_contact.shape = ', JK_lo_hf_contact.shape)
    print('hcore_lo_imp.shape = ', hcore_lo_imp.shape)
    print('JK_lo_ks_imp.shape = ', JK_lo_ks_imp.shape)
    print('JK_lo_hf_imp.shape = ', JK_lo_hf_imp.shape)
    print('DM_lo_imp.shape = ', DM_lo_imp.shape)
    print('eri_lo_imp.shape = ', eri_lo_imp.shape)
    print('')

    print('hcore_lo_contact.dtype = ', hcore_lo_contact.dtype)
    print('JK_lo_ks_contact.dtype = ', JK_lo_ks_contact.dtype)
    print('JK_lo_hf_contact.dtype = ', JK_lo_hf_contact.dtype)
    print('hcore_lo_imp.dtype = ', hcore_lo_imp.dtype)
    print('JK_lo_ks_imp.dtype = ', JK_lo_ks_imp.dtype)
    print('JK_lo_hf_imp.dtype = ', JK_lo_hf_imp.dtype)
    print('DM_lo_imp.dtype = ', DM_lo_imp.dtype)
    print('eri_lo_imp.dtype = ', eri_lo_imp.dtype)
    print('')

    print('nao_imp = ', nao_imp)
    print('nval_imp = ', nval_imp)
    print('nao_contact = ', nao_contact)

    print('finish reading contact mean field data\n')

comm.Barrier()

############################################################
#               gate voltage (untested)
############################################################
# shift the imp one-body Hamiltonian
gate = 0.0

if rank == 0:
    print('gate voltage = ', gate)

for s in range(spin):
    for ik in range(nkpts):
        hcore_lo_contact[s,ik,0:nao_imp,0:nao_imp] += gate*np.eye(nao_imp) 
        hcore_lo_imp[s,ik] += gate*np.eye(nao_imp) 

############################################################
#               read lead's mean-field data
############################################################

bath_dir = '/home/zuxin/projects/transport/models/Cu_fcc_100/Cu/'

if rank == 0:
    print('start reading lead mean field data from', bath_dir)

#------------ check Cu HOMO/LUMO energy ------------

# should be the same as the one for computing contact
#a = 3.6
num_layer = 8
bath_cell_label = 'Cu_' + Cu_basis + '_a' + str(a) + '_n' + str(num_layer)

bath_cell_fname = bath_dir + 'cell_' + bath_cell_label + '.chk'
bath_cell = pbcchkfile.load_cell(bath_cell_fname)
nat_Cu_lead = len(bath_cell.atom)

bath_gdf_fname = bath_dir + 'cderi_' + bath_cell_label + '.h5'
bath_gdf = pbcdf.GDF(bath_cell)
bath_gdf._cderi = bath_gdf_fname

if 'ks' in method_label:
    bath_mf = pbcscf.RKS(bath_cell).density_fit()
    xcfun = 'pbe0'
    bath_mf.xc = xcfun
    bath_method_label = 'rks_' + xcfun
else:
    bath_mf = pbcscf.RHF(bath_cell).density_fit()
    bath_method_label = 'rhf'

solver_label = 'newton'

bath_mf_fname = bath_dir + bath_cell_label + '_' + bath_method_label + '_' + solver_label + '.chk'

bath_mf.with_df = bath_gdf
bath_mf.__dict__.update( pbcchkfile.load(bath_mf_fname, 'scf') )

ihomo = 29*nat_Cu_lead//2-1
ilumo = 29*nat_Cu_lead//2

E_Cu_homo = np.asarray(bath_mf.mo_energy)[ihomo]
E_Cu_lumo = np.asarray(bath_mf.mo_energy)[ilumo]

if rank == 0:
    print('ihomo = ', ihomo, '      occ = ', np.asarray(bath_mf.mo_occ)[ihomo], '      E = ', E_Cu_homo)
    print('ilumo = ', ilumo, '      occ = ', np.asarray(bath_mf.mo_occ)[ilumo], '      E = ', E_Cu_lumo)

comm.Barrier()

#------------ get H00 and H01 (for surface Green's function) ------------
bath_fname = bath_dir + '/data_lead_' + bath_cell_label + '_' + bath_method_label + '.h5'
fh = h5py.File(bath_fname, 'r')

hcore_lo_lead = np.asarray(fh['hcore_lo'])
JK_lo_lead = np.asarray(fh['JK_lo'])
F_lo_lead = hcore_lo_lead + JK_lo_lead

if rank == 0:
    print('F_lo_lead.shape = ', F_lo_lead.shape)

# number of orbitals per atom
nao_per_Cu = 15

# number of atoms per principal layer
nat_ppl = 9

# nao per principal layer
nao_ppl = nao_per_Cu * nat_ppl

H00 = F_lo_lead[0, :nao_ppl, :nao_ppl]
H01 = F_lo_lead[0, :nao_ppl, nao_ppl:2*nao_ppl]

if rank == 0:
    print('finish reading lead mean field data\n')

comm.Barrier()

##################
#plt.imshow(np.abs(F_lo_lead[0]))
#plt.show()
#exit()
##################

############################################################
#               impurity Hamiltonian
############################################################
hcore_lo_imp = 1./nkpts * np.sum(hcore_lo_imp, axis=1)
JK_lo_hf_imp = 1./nkpts * np.sum(JK_lo_hf_imp, axis=1)
DM_lo_imp = 1./nkpts * np.sum(DM_lo_imp, axis=1)

JK_00 = scf_mu._get_veff(DM_lo_imp, eri_lo_imp)

Hemb_imp = hcore_lo_imp + JK_lo_hf_imp - JK_00

if rank == 0:
    print('hcore_lo_imp.shape = ', hcore_lo_imp.shape)
    print('JK_lo_hf_imp.shape = ', JK_lo_hf_imp.shape)
    print('DM_lo_imp.shape = ', DM_lo_imp.shape)
    print('JK_00.shape = ', JK_00.shape)
    print('Hemb_imp.shape = ', Hemb_imp.shape)
    print('')

    if spin == 1:
        print('trace(DM_lo_imp) = ', np.trace(DM_lo_imp[0]))
    else:
        print('trace(DM_lo_imp[0]) = ', np.trace(DM_lo_imp[0]))
        print('trace(DM_lo_imp[1]) = ', np.trace(DM_lo_imp[1]))
    print('')

comm.Barrier()

############################################################
#               contact's Green's function
############################################################
hcore_lo_contact = 1./nkpts * np.sum(hcore_lo_contact, axis=1)
JK_lo_hf_contact = 1./nkpts * np.sum(JK_lo_hf_contact, axis=1)

if rank == 0:
    print('hcore_lo_contact.shape = ', hcore_lo_contact.shape)
    print('JK_lo_hf_contact.shape = ', JK_lo_hf_contact.shape)
    print('')

# return a 3-d array of size (spin, nao_contact, nao_contact)
def contact_Greens_function(z):
    g00 = Umerski1997(z, H00, H01)

    V_L = np.zeros((nao_contact, nao_ppl), dtype=complex)
    V_R = np.zeros((nao_contact, nao_ppl), dtype=complex)

    V_L[nao_imp:nao_imp+nao_ppl,:] = H01.T.conj()
    V_R[-nao_ppl:,:] = H01

    Sigma_L = V_L @ g00 @ V_L.T.conj()
    Sigma_R = V_R @ g00 @ V_R.T.conj()

    # contact block of the Green's function
    G_C = np.zeros((spin, nao_contact, nao_contact), dtype=complex)
    for s in range(spin):
        G_C[s,:,:] = np.linalg.inv( z*np.eye(nao_contact) - hcore_lo_contact[s] - JK_lo_hf_contact[s] \
                - Sigma_L - Sigma_R )
    return G_C

comm.Barrier()

############################################################
#               hybridization Gamma
############################################################
# number of orbitals that couple to the bath, usually nval_imp or nao_imp
n_hyb = nao_imp

# broadening for computing hybridization Gamma from self energy
hyb_broadening= 0.01
# -1/pi*imag(Sigma(e+i*delta))
# (spin, n_hyb, n_hyb)
def Gamma(e):
    z = e + 1j*hyb_broadening
    G_C = contact_Greens_function(z)
    
    Sigma_imp = np.zeros((spin, n_hyb, n_hyb),dtype=complex)
    for s in range(spin):
        Sigma_imp[s,:,:] = z*np.eye(n_hyb) - Hemb_imp[s,:n_hyb,:n_hyb] \
                - np.linalg.inv(G_C[s,:n_hyb,:n_hyb])

    return -1./np.pi*Sigma_imp.imag

############################################################
#       embedding Hamiltonian (one-body, spinless)
############################################################
#FIXME the second dim of v is actually number of orbitals that couple to the bath
# not nimp exactly if one selectively use only a few imp orb to calculation hyb
def emb_ham(h, e, v):
    nbe, nimp, nbath_per_ene = v.shape
    nbath = len(e) * nbath_per_ene
    nemb = nimp + nbath
    hemb = np.zeros((nemb, nemb))
    hemb[0:nimp, 0:nimp] = h[0:nimp, 0:nimp]
    
    # bath energy
    for ibe in range(nbe):
        for ib in range(nbath_per_ene):
            hemb[nimp+ib*nbe+ibe,nimp+ib*nbe+ibe] = e[ibe]
    
    for i in range(nimp):
        for ib in range(nbath_per_ene):
            hemb[nimp+ib*nbe:nimp+(ib+1)*nbe,i] = v[:,i,ib]
            hemb[i,nimp+ib*nbe:nimp+(ib+1)*nbe] = v[:,i,ib]

    return hemb

############################################################
#               one-side log grid
############################################################
# generate a log grid between w0 and w (converges to w0)
# return w0 + (w-w0)*l**(-i) where i ranges from 0 to num-1

def gen_log_grid(w0, w, l, num):
    grid = w0 + (w-w0) * l**(-np.arange(num,dtype=float))
    if w > w0:
        return grid[::-1]
    else:
        return grid

############################################################
#               TEST1: Co LDoS (mean-field)
############################################################
'''
# compute mean-field Co valence LDoS from the mean-field contact Green's function directly
wl = -1
wh = 6
nw = 1000
ldos_mf_broadening = 0.01
freqs = np.linspace(wl,wh,nw)
dw = freqs[1]-freqs[0]
ldos = np.zeros((spin,nw,nao_imp))

for iw in range(nw):
    z = freqs[iw] + 1j*ldos_mf_broadening
    GC = contact_Greens_function(z)
    GC_Co = GC[:,:nao_imp, :nao_imp]
    for s in range(spin):
        ldos[s,iw,:] = -1./np.pi*np.diag(GC_Co[s]).imag

if rank == 0:
    print('integrated LDoS')
    print(np.sum(ldos, axis=1)*dw)

ldos_tot_val = np.sum(ldos[:,:,0:6],axis=2)

if rank == 0:
    # ax1: individual valence
    # ax2: total valence
    fig, (ax1,ax2) = plt.subplots(1,2)
    for i in range(0,6):
        ax1.plot(freqs,ldos[0,:,i], color='C'+str(i))
        if spin == 2:
            ax1.plot(freqs,ldos[1,:,i], color='C'+str(i), linestyle='--')
    
    ax1.set_xlim((-0.8,0.8))
    
    ax2.plot(freqs,ldos_tot_val[0])
    if spin == 2:
        ax2.plot(freqs,ldos_tot_val[1])
    
    # plot Cu homo/lumo
    ax1.axvline(E_Cu_homo, color='b', linestyle=':', linewidth=0.5)
    ax1.axvline(E_Cu_lumo, color='r', linestyle=':', linewidth=0.5)
    
    ax1.set_xlim((-0.8,0.8))
    ax1.set_ylim((-0.1,30))
    
    ax2.set_xlim((-0.8,0.8))
    ax2.set_ylim((-0.1,60))
    
    fig.set_size_inches(12, 5)
    
    plt.show()

exit()
'''

#NOTE
# ideally the bath discretization should satisfy two criteria:
# 1. there should be a sufficient number of states near the Fermi level in order to see Kondo peak
#    the spacing should be prefarably smaller than the Kondo temperature
# 2. the mean-field LDoS should be able to recover the exact one (given the same mf solver)

# the second test are done in imp_rks_check.py

'''
############################################################
#       TEST2: rebuild Gamma by bath discretization
############################################################
#------------ bath discretization ------------
# evenly spaced grid
wlg = -0.6
whg = 0.6
nbe = 40
grid = np.linspace(wlg,whg,nbe)
nbath_per_ene = 6

# only test one spin component
Gamma_s = lambda e: Gamma(e)[spin-1]
e,v = direct_disc_hyb(Gamma_s, grid, nint=3, nbath_per_ene=nbath_per_ene)
comm.Barrier()
if rank == 0:
    print('bath discretization finished')
    print('e.shape = ', e.shape)
    print('v.shape = ', v.shape)
    print('bath energies = ', e)
    print('')

#------------ compare exact & rebuilt Gamma ------------
# exact hybridization Gamma
wl = -0.8
wh = 0.8
nw = 1000
freqs = np.linspace(wl, wh, nw)
hyb = np.zeros((nw, n_hyb, n_hyb))
for iw in range(nw):
    hyb[iw,:,:] = Gamma_s(freqs[iw])

# rebuild Gamma
gauss = lambda x,mu,sigma: 1.0/sigma/np.sqrt(2*np.pi)*np.exp(-0.5*((x-mu)/sigma)**2)
eta=0.01

Gamma_rebuilt = np.zeros((nw,n_hyb,n_hyb))
for iw in range(nw):
    for ib in range(len(e)):
        for ie in range(nbath_per_ene):
            Gamma_rebuilt[iw,:,:] += np.outer(v[ib,:,ie],v[ib,:,ie].conj()) * gauss(freqs[iw],e[ib],eta)

if rank == 0:
    for i in range(nval_Co):
        plt.plot(freqs, hyb[:,i,i], linestyle=':', label='exact'+str(i), color='C'+str(i))
        plt.plot(freqs, Gamma_rebuilt[:,i,i], linestyle='-', label='rebuilt'+str(i), color='C'+str(i))
    
    plt.xlim([wl,wh])
    plt.ylim([-0.01,0.3])
    
    fig = plt.gcf()
    fig.set_size_inches(6,4)
    
    plt.legend()
    plt.show()

exit()
'''

# NOTE 
# see imp_rks_check.py
############################################################
#   TEST3: embedding model Co LDoS solved by mean-field
############################################################
'''
#------------ bath discretization ------------
# evenly spaced grid
wlg = -0.8
whg = 1.8
nbe = 50
grid = np.linspace(wlg,whg,nbe+1)
nbath_per_ene = 1

nbath = nbe*nbath_per_ene
nemb = nbath + nao_imp

hemb = np.zeros((spin, nemb, nemb))

if rank == 0:
    print('hemb.shape = ', hemb.shape)
    print('bath discretization starts')

# one body part
for s in range(spin):
    Gamma_s = lambda e: Gamma(e)[s]
    e,v = direct_disc_hyb(Gamma_s, grid, nint=3, nbath_per_ene=nbath_per_ene)
    
    #------------ build & solve embedding model with mean-field ------------
    hemb[s,:,:] = emb_ham(Hemb_imp[s,:,:], e, v)

if rank == 0:
    print('bath discretization finished')

# electron repulsion integral
# only non-zero on the impurity
eri_imp = np.zeros([spin*(spin+1)//2, nemb, nemb, nemb, nemb])
eri_imp[:,:nao_imp,:nao_imp,:nao_imp,:nao_imp] = eri_lo_imp

dm0 = np.zeros((spin,nemb,nemb))
dm0[:,:nao_imp,:nao_imp] = DM_lo_Co.copy()

mol = gto.M()
mol.verbose = 4
mol.incore_anyway = True
mol.build()

mu = -0.16

if 'r' in method_label:
    mf = scf_mu.RHF(mol, mu)
    mf.get_hcore = lambda *args: hemb[0]
    mf._eri = ao2mo.restore(8, eri_imp[0], nemb)
    mf.mo_energy = np.zeros([nemb])
else:
    mf = scf_mu.UHF(mol, mu)
    mf.get_hcore = lambda *args: hemb
    mf._eri = eri_imp
    mf.mo_energy = np.zeros([2,nemb])

mf.get_ovlp = lambda *args: np.eye(nemb)
mf.max_cycle = 150
mf.conv_tol = 1e-12
mf.diis_space = 15


if rank == 0:
    print('scf starts')
    if spin == 1:
        mf.kernel(dm0[0])
    else:
        mf.kernel(dm0)
    print('scf finished')

mo_coeff = comm.bcast(mf.mo_coeff, root=0)
mo_energy = comm.bcast(mf.mo_energy, root=0)
mo_occ = comm.bcast(mf.mo_occ, root=0)
mf.mo_coeff = mo_coeff
mf.mo_energy = mo_energy
mf.mo_occ = mo_occ

rdm1 = mf.make_rdm1()

if rank == 0:
    if spin == 1:
        print('trace(rdm1[Co val+virt])', np.trace(rdm1[0:nao_imp,0:nao_imp]))
        print('trace(rdm1[Co val])', np.trace(rdm1[0:nval_Co,0:nval_Co]))
    else:
        print('trace(rdm1[Co alpha val+virt])', np.trace(rdm1[0, 0:nao_imp,0:nao_imp]))
        print('trace(rdm1[Co beta val+virt])', np.trace(rdm1[1, 0:nao_imp,0:nao_imp]))

        print('trace(rdm1[Co alpha val])', np.trace(rdm1[0, 0:nval_Co,0:nval_Co]))
        print('trace(rdm1[Co beta val])', np.trace(rdm1[1, 0:nval_Co,0:nval_Co]))

#------------ compute & plot Co LDoS ------------
fock = mf.get_fock()
if spin == 1:
    fock = fock[np.newaxis,...]

wld = -0.8
whd = 0.8
nwd = 200
delta = 0.01
freqs = np.linspace(wld,whd,nwd)

A = np.zeros((spin,nwd,nval_Co))
for s in range(spin):
    for iw in range(nwd):
        z = freqs[iw] + 1j*delta
        gf = np.linalg.inv(z*np.eye(nemb) - fock[s,:,:])
        A[s,iw,:] = -1./np.pi*np.diag(gf[0:nval_Co,0:nval_Co]).imag


# raw mean-field LDoS from contact Green's function
ldos = np.zeros((spin,nwd,nval_Co))
for iw in range(nwd):
    z = freqs[iw] + 1j*delta
    GC = contact_Greens_function(z)
    for s in range(spin):
        ldos[s,iw,:] = -1./np.pi*np.diag(GC[s,:nval_Co, :nval_Co]).imag

if rank == 0:
    for i in range(nval_Co):
        plt.plot(freqs,ldos[0,:,i], linestyle=':', color='C'+str(i))
        plt.plot(freqs,A[0,:,i], linestyle='-', color='C'+str(i))

    fig = plt.gcf()
    fig.set_size_inches(6,4)
    
    plt.xlim((wld,whd))
    plt.show()

exit()
'''

############################################################
#           TEST4: log discretization
############################################################
'''
#------------ bath discretization (log) ------------
mu = -0.16

# absolute band edge
wl = -0.6
wh = 0.6

# distance to mu
wl0 = mu - wl
wh0 = wh - mu

log_disc_base = 1.7

# total number of bath energies
nbe = 30
dif = round(np.log(abs(wh0/wl0))/np.log(log_disc_base)) // 2

# number of energies above/below the Fermi level
nl = nbe//2 - dif
nh = nbe - nl

print('nl, nh = ', nl, nh)

grid = np.concatenate((gen_log_grid(mu, wl, log_disc_base, nl), [mu], \
        gen_log_grid(mu, wh, log_disc_base, nh)))

nbath_per_ene = 1
nbath = nbe*nbath_per_ene
nemb = nbath + nao_imp

hemb = np.zeros((spin, nemb, nemb))

if rank == 0:
    print('grid = ', grid)
    print('hemb.shape = ', hemb.shape)
    print('bath discretization starts')

# one body part
for s in range(spin):
    Gamma_s = lambda e: Gamma(e)[s]
    e,v = direct_disc_hyb(Gamma_s, grid, nint=3, nbath_per_ene=nbath_per_ene)
    
    #------------ build & solve embedding model with mean-field ------------
    hemb[s,:,:] = emb_ham(Hemb_imp[s,:,:], e, v)

if rank == 0:
    print('bath discretization finished')

# electron repulsion integral
# only non-zero on the impurity
eri_imp = np.zeros([spin*(spin+1)//2, nemb, nemb, nemb, nemb])
eri_imp[:,:nao_imp,:nao_imp,:nao_imp,:nao_imp] = eri_lo_imp

dm0 = np.zeros((spin,nemb,nemb))
dm0[:,:nao_imp,:nao_imp] = DM_lo_Co.copy()

#------------ build & solve embedding model with mean-field ------------
mol = gto.M()
mol.verbose = 0
mol.incore_anyway = True
mol.build()

if 'r' in method_label:
    mf = scf_mu.RHF(mol, mu)
    mf.get_hcore = lambda *args: hemb[0]
    mf._eri = ao2mo.restore(8, eri_imp[0], nemb)
    mf.mo_energy = np.zeros([nemb])
else:
    mf = scf_mu.UHF(mol, mu)
    mf.get_hcore = lambda *args: hemb
    mf._eri = eri_imp
    mf.mo_energy = np.zeros([2,nemb])

mf.get_ovlp = lambda *args: np.eye(nemb)
mf.max_cycle = 150
mf.conv_tol = 1e-12
mf.diis_space = 15


if rank == 0:
    print('scf starts')
    if spin == 1:
        mf.kernel(dm0[0])
    else:
        mf.kernel(dm0)
    print('scf finished')

mo_coeff = comm.bcast(mf.mo_coeff, root=0)
mo_energy = comm.bcast(mf.mo_energy, root=0)
mo_occ = comm.bcast(mf.mo_occ, root=0)
mf.mo_coeff = mo_coeff
mf.mo_energy = mo_energy
mf.mo_occ = mo_occ

rdm1 = mf.make_rdm1()
fock = mf.get_fock()

if rank == 0:
    if spin == 1:
        print('trace(rdm1[Co val+virt])', np.trace(rdm1[0:nao_imp,0:nao_imp]))
        print('trace(rdm1[Co val])', np.trace(rdm1[0:nval_Co,0:nval_Co]))
    else:
        print('trace(rdm1[Co alpha val+virt])', np.trace(rdm1[0, 0:nao_imp,0:nao_imp]))
        print('trace(rdm1[Co beta val+virt])', np.trace(rdm1[1, 0:nao_imp,0:nao_imp]))

        print('trace(rdm1[Co alpha val])', np.trace(rdm1[0, 0:nval_Co,0:nval_Co]))
        print('trace(rdm1[Co beta val])', np.trace(rdm1[1, 0:nval_Co,0:nval_Co]))


#------------ compute & plot Co LDoS ------------
fock = mf.get_fock()
if spin == 1:
    fock = fock[np.newaxis,...]

wld = -0.8
whd = 0.8
nwd = 200
delta = 0.01
freqs = np.linspace(wld,whd,nwd)

A = np.zeros((spin,nwd,nval_Co))
for s in range(spin):
    for iw in range(nwd):
        z = freqs[iw] + 1j*delta
        gf = np.linalg.inv(z*np.eye(nemb) - fock[s,:,:])
        A[s,iw,:] = -1./np.pi*np.diag(gf[0:nval_Co,0:nval_Co]).imag

# raw mean-field LDoS from contact Green's function
ldos = np.zeros((spin,nwd,nval_Co))
for iw in range(nwd):
    z = freqs[iw] + 1j*delta
    GC = contact_Greens_function(z)
    for s in range(spin):
        ldos[s,iw,:] = -1./np.pi*np.diag(GC[s,:nval_Co, :nval_Co]).imag

if rank == 0:
    for i in range(nval_Co):
        plt.plot(freqs,ldos[0,:,i], linestyle=':', color='C'+str(i))
        plt.plot(freqs,A[0,:,i], linestyle='-', color='C'+str(i))

    fig = plt.gcf()
    fig.set_size_inches(6,4)
    
    plt.xlim((wld,whd))
    plt.show()

exit()
'''

comm.Barrier()

'''
############################################################
#       frequencies to compute spectra
############################################################

#NOTE for a range of 0.03 with nw=10, normal grid dw ~ 0.003
# extra_nw = 7 & extra_dw=0.002 implies a coverage of 0.002x3x2=0.012

# normal grid
wl_freqs = -0.16
wh_freqs = -0.13
delta = 0.01
nw = 10
freqs = np.linspace(wl_freqs, wh_freqs, nw)

extra_delta = 0.002
extra_freqs = []

extra_dw = 0.0006
extra_nw = 5

for i in range(nw):
    freqs_tmp = []
    if extra_nw % 2 == 0:
        for w in range(-extra_nw // 2, extra_nw // 2):
            freqs_tmp.append(freqs[i] + extra_dw * w)
    else:
        for w in range(-(extra_nw-1) // 2, (extra_nw+1) // 2):
            freqs_tmp.append(freqs[i] + extra_dw * w)
    extra_freqs.append(np.array(freqs_tmp))

#print('extra_freqs = ', extra_freqs)

all_freqs = np.array(sorted(list(set(list(freqs) + \
        ([x for xx in extra_freqs for x in xx] if extra_freqs is not None else [])))))
#print('all freqs = ', all_freqs)

nwa = len(all_freqs)
'''

############################################################
#       raw mean-field LDoS (from contact GF)
############################################################
ldos_mf = np.zeros((spin, nao_imp, nwa))
for iw in range(nwa):
    z = all_freqs[iw] + 1j*extra_delta
    GC = contact_Greens_function(z)

    for s in range(spin):
        ldos_mf[s,:,iw] = -1./np.pi*GC[s,:nao_imp, :nao_imp].diagonal().imag

'''
if rank == 0:
    dfreq = all_freqs[1:] - all_freqs[:-1]
    mf_int = np.sum(ldos_mf[:,:,:-1]*dfreq, axis=2)
    print('integrated mean-field LDoS within (%6.4f, %6.4f) = '%(all_freqs[0], all_freqs[-1]), mf_int)

    plt.plot(all_freqs, np.sum(ldos_mf[0], axis=0))
    plt.show()

exit()
'''

############################################################
#               bath discretization
############################################################
#------------ log discretization ------------
wlg = -0.6
whg = 0.6
nbe = 40 # total number of bath energies
nbath_per_ene = 3

log_disc_base = 1.5

# distance to mu
wl0 = mu - wlg
wh0 = whg - mu

dif = round(np.log(abs(wh0/wl0))/np.log(log_disc_base)) // 2

# number of energies above/below the Fermi level
nl = nbe//2 - dif
nh = nbe - nl

grid = np.concatenate((gen_log_grid(mu, wlg, log_disc_base, nl), [mu], \
        gen_log_grid(mu, whg, log_disc_base, nh)))

nbath = nbe * nbath_per_ene
nemb = nbath + nao_imp

hemb = np.zeros((spin, nemb, nemb))

if rank == 0:
    print('hemb.shape = ', hemb.shape)
    print('bath discretization starts')

# one body part
for s in range(spin):
    Gamma_s = lambda e: Gamma(e)[s]
    e,v = direct_disc_hyb(Gamma_s, grid, nint=3, nbath_per_ene=nbath_per_ene)
    
    hemb[s,:,:] = emb_ham(Hemb_imp[s,:,:], e, v)

if rank == 0:
    print('bath discretization finished')
    print('bath energies = ', e)

# electron repulsion integral
# only non-zero on the impurity
eri_imp = np.zeros([spin*(spin+1)//2, nemb, nemb, nemb, nemb])
eri_imp[:,:nao_imp,:nao_imp,:nao_imp,:nao_imp] = eri_lo_imp

dm0 = np.zeros((spin,nemb,nemb))
dm0[:,:nao_imp,:nao_imp] = DM_lo_imp.copy()

############################################################
#               build embedding model
############################################################
##########################
def get_emb_mf(mu, hemb, eri_imp, dm0):
    mol = gto.M()
    mol.verbose = 4
    mol.incore_anyway = True
    mol.build()

    emb_mf = scf_mu.RHF(mol, mu)
    emb_mf.get_hcore = lambda *args: hemb[0]
    emb_mf._eri = ao2mo.restore(8, eri_imp[0], nemb)
    emb_mf.mo_energy = np.zeros([nemb])
    
    emb_mf.get_ovlp = lambda *args: np.eye(nemb)
    emb_mf.max_cycle = 150
    emb_mf.conv_tol = 1e-10
    emb_mf.diis_space = 15

    if rank == 0:
        emb_mf.kernel(dm0)

    emb_mf.mo_coeff = comm.bcast(emb_mf.mo_coeff, root=0)
    emb_mf.mo_energy = comm.bcast(emb_mf.mo_energy, root=0)
    emb_mf.mo_occ = comm.bcast(emb_mf.mo_occ, root=0)

    return emb_mf

##########################
'''
mol = gto.M()
mol.verbose = 4
mol.incore_anyway = True
mol.build()

emb_mf = scf_mu.RHF(mol, mu)
emb_mf.get_hcore = lambda *args: hemb[0]
emb_mf._eri = ao2mo.restore(8, eri_imp[0], nemb)
emb_mf.mo_energy = np.zeros([nemb])

emb_mf.get_ovlp = lambda *args: np.eye(nemb)
emb_mf.max_cycle = 150
emb_mf.conv_tol = 1e-10
emb_mf.diis_space = 15

############################################################
#           embedding model mean-field SCF
############################################################
emb_mf_fname = 'emb_mf_' + imp_atom + '.chk'
if rank == 0: 
    if os.path.isfile(emb_mf_fname):
        print('ready to load emb_mf chkfile', emb_mf_fname)
        emb_mf.__dict__.update( chkfile.load(emb_mf_fname, 'scf') )
        print('mean field data loaded!')
        print('one more extra scf step...')
        emb_mf.kernel(emb_mf.make_rdm1())
    else:
        emb_mf.chkfile = emb_mf_fname
        print('scf starts')
        emb_mf.kernel(dm0[0])

    print('scf finished')

emb_mf.mo_coeff = comm.bcast(emb_mf.mo_coeff, root=0)
emb_mf.mo_energy = comm.bcast(emb_mf.mo_energy, root=0)
emb_mf.mo_occ = comm.bcast(emb_mf.mo_occ, root=0)
'''

emb_mf = get_emb_mf(mu, hemb, eri_imp, dm0[0])
dm_mf = emb_mf.make_rdm1()

if rank == 0:
    print('trace(dm_mf[imp val+virt])', np.trace(dm_mf[0:nao_imp,0:nao_imp]))
    print('trace(dm_mf[imp val])', np.trace(dm_mf[0:nval_imp,0:nval_imp]))

'''
############################################################
#           embedding model mean-field GF
############################################################
# HF GF in AO basis
gf_hf = np.zeros((nwa, nemb, nemb), dtype=complex)

# here the ldos_mf is obtained by running HF SCF on the embedding model
# which was constructed from DM originally converged with KS pbe0
# so it's not real mean-field ldos
ldos_mf_fake = np.zeros(nwa)
for iw in range(nwa):
    z = all_freqs[iw] + 1j*extra_delta
    gf_mo = np.diag( 1. / (z - emb_mf.mo_energy) )
    gf_hf[iw,:,:] = emb_mf.mo_coeff @ gf_mo @ emb_mf.mo_coeff.T
    ldos_mf_fake[iw] = -1./np.pi*gf_hf[iw,0,0].imag
'''

############################################################
#               embedding model rdm
############################################################

rdm = get_rdm_emb(emb_mf)
rdm_imp = rdm[0:nao_imp, 0:nao_imp]
if rank == 0:
    print('DMRG nelec on imp (val+virt) = ', np.trace(rdm_imp))
    print('DMRG nelec on imp (val)      = ', np.trace(rdm_imp[0:nval_imp, 0:nval_imp]))
    print('DMRG imp rdm diagonal = ', rdm_imp.diagonal())

#exit()

############################################################
#               embedding model gf
############################################################
gf = get_gf_emb(emb_mf)
ldos_dmrg = np.zeros((nwa, nao_imp))
for iw in range(nwa):
    gf_imp = gf[iw, :nao_imp, :nao_imp]
    ldos_dmrg[iw,:] = -1./np.pi*gf_imp.diagonal().imag

fh = h5py.File('ldos_dmrg_' + imp_atom + '.h5', 'w')
fh['all_freqs'] = all_freqs
fh['ldos_dmrg'] = ldos_dmrg
fh['ldos_mf'] = ldos_mf
fh['mu'] = mu
fh.close()

exit()

############################################################
#                   end of main
############################################################

