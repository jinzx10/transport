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
from utils.broydenroot import *

from pyscf.scf.hf import eig as eiggen

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

mode = 'MODE'

imp_atom = 'IMP_ATOM' if mode == 'production' else 'Co'

# whether perform an optimization or just a one-shot calculation
opt_mu = OPT_MU

# chemical potential (initial guess in optimization if opt_mu is True)
mu0 = CHEM_POT if mode == 'production' else 0.05

# DMRG scratch & save folder
scratch = 'SCRATCH'
save_dir = 'SAVEDIR'
os.environ['TMPDIR'] = scratch
os.environ['PYSCF_TMPDIR'] = scratch

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
#           an unknown auxiliary function
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

############################################################
#           CAS from CISD natural orbital
############################################################
def gen_cas(emb_mf):

    from pyscf import ci
    
    emb_cisd = ci.CISD(emb_mf)
    emb_cisd.max_cycle = 1000
    emb_cisd.max_space = 15
    #emb_cisd.max_memory = 100000
    emb_cisd.conv_tol = 1e-9
    emb_cisd.verbose = 4
    
    emb_cisd_fname = 'emb_cisd_' + imp_atom + '.chk'
    if rank == 0:
        '''
        if os.path.isfile(emb_cisd_fname):
            print('load CISD data from', emb_cisd_fname)
            emb_cisd.__dict__.update( chkfile.load(emb_cisd_fname, 'cisd') )
        else:
            emb_cisd.chkfile = emb_cisd_fname
            print('CISD starts')
            emb_cisd.kernel()
            emb_cisd.dump_chk()
        '''
        # always run CISD from scratch
        print('CISD starts')
        print('CISD max_memory = ', emb_cisd.max_memory)
        emb_cisd.kernel()
    
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
    JK_cas_no = scf_mu._get_veff(dm_cas_no, g2e)[0]
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
#                   gate voltage
############################################################
gate = GATE if mode == 'production' else 0
gate_label = 'gate%5.3f'%(gate)

############################################################
#           read contact's mean-field data
############################################################
#contact_dir = '/home/zuxin/projects/transport/models/Cu_fcc_100/' + imp_atom + '/'
contact_dir = 'PREFIX/projects/transport/models/Cu_fcc_100/' + imp_atom + '/'
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

data_fname = contact_dir + '/data_contact_' + cell_label + '_' \
        + method_label + '_' + gate_label + '.h5'

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
#           number of electrons on impurity
############################################################
# target number of electrons on the impurity for mu-optimization
nelec_lo_imp = np.trace(DM_lo_imp[0].sum(axis=0)/nkpts)
print('mf number of electrons on imp:', nelec_lo_imp)

############################################################
#               read lead's mean-field data
############################################################

#bath_dir = '/home/zuxin/projects/transport/models/Cu_fcc_100/Cu/'
bath_dir = 'PREFIX/projects/transport/models/Cu_fcc_100/Cu/'

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

# return a matrix of size (nemb,nemb) (does not have spin dimension!)
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
#   generate one-body part of embedding model Hamiltonian
############################################################
# given a chemical potential, this function choose a correponding
# log grid to discretize the bath and generate hemb of size (spin, nemb, nemb)

########################################
#   parameters for bath discretization
########################################
nbe = 50 # total number of bath energies
nbath_per_ene = 3
nbath = nbe * nbath_per_ene
nemb = nbath + nao_imp
wlg = -0.6
whg = 1.2
log_disc_base = 2.0
grid_type = 'custom1'
wlog=0.01
    
def gen_hemb(mu):
    grid = gen_grid(nbe, wlg, whg, mu, grid_type=grid_type, log_disc_base=log_disc_base, wlog=wlog)
    hemb = np.zeros((spin, nemb, nemb))
    
    # one body part
    for s in range(spin):
        Gamma_s = lambda e: Gamma(e)[s]
        e,v = direct_disc_hyb(Gamma_s, grid, nint=3, nbath_per_ene=nbath_per_ene)
        
        hemb[s,:,:] = emb_ham(Hemb_imp[s,:,:], e, v)

    if rank == 0:
        print('bath energies = ', e)

    return hemb


############################################################
#           electron repulsion integral
############################################################
# only non-zero on the impurity
eri_imp = np.zeros([spin*(spin+1)//2, nemb, nemb, nemb, nemb])
eri_imp[:,:nao_imp,:nao_imp,:nao_imp,:nao_imp] = eri_lo_imp

############################################################
#           embedding model DM initial guess
############################################################
dm0 = np.zeros((spin,nemb,nemb))
dm0[:,:nao_imp,:nao_imp] = DM_lo_imp.copy()

############################################################
#               build embedding model
############################################################
##########################
def get_emb_mf(mu):
    mol = gto.M()
    mol.verbose = 4
    mol.incore_anyway = True
    mol.build()

    hemb = gen_hemb(mu)

    emb_mf = scf_mu.RHF(mol, mu)
    emb_mf.get_hcore = lambda *args: hemb[0]
    emb_mf._eri = ao2mo.restore(8, eri_imp[0], nemb)
    emb_mf.mo_energy = np.zeros([nemb])
    
    emb_mf.get_ovlp = lambda *args: np.eye(nemb)
    emb_mf.max_cycle = 150
    emb_mf.conv_tol = 1e-10
    emb_mf.diis_space = 15

    if rank == 0:
        emb_mf.kernel(dm0[0])

    emb_mf.mo_coeff = comm.bcast(emb_mf.mo_coeff, root=0)
    emb_mf.mo_energy = comm.bcast(emb_mf.mo_energy, root=0)
    emb_mf.mo_occ = comm.bcast(emb_mf.mo_occ, root=0)

    return emb_mf

##########################

############################################################
#               optimize chemical potential
############################################################
if opt_mu:
    def dnelec(mu):
        emb_mf = get_emb_mf(mu)
        rdm = get_rdm_emb(emb_mf)
        rdm_imp = rdm[0:nao_imp, 0:nao_imp]
        nelec_imp_dmrg = np.trace(rdm_imp)
        return nelec_imp_dmrg - nelec_lo_imp
    
    mu, flag = broydenroot(dnelec, mu0, tol = 0.001, max_iter = 20)
    
    if flag == 0:
        print('optimized mu = ', mu)
        print('nelec diff = ', dnelec(mu))
    else:
        print('current mu = ', mu)
        print('nelec diff = ', dnelec(mu))
        print('mu optimization failed!')
else:
    mu = mu0
    print('no optimiziation is performed')
    print('mu = ', mu)

emb_mf = get_emb_mf(mu)
rdm = get_rdm_emb(emb_mf)
rdm_imp = rdm[0:nao_imp, 0:nao_imp]
nelec_imp_dmrg = np.trace(rdm_imp)
print('nelec diff = ', nelec_imp_dmrg - nelec_lo_imp)

exit()

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

