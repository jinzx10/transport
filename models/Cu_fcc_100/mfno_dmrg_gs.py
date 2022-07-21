import numpy as np
import h5py, time, sys, os, scipy
from mpi4py import MPI

from fcdmft.solver import scf_mu

import matplotlib.pyplot as plt
from matplotlib import colors

from pyscf import gto, ao2mo, cc, scf, dft

from pyscf.pbc import scf as pbcscf
from pyscf.pbc import df as pbcdf
from pyscf.pbc.lib import chkfile

from utils.surface_green import *
from utils.bath_disc import *
from utils.broydenroot import *
from utils.emb_helper import *

from pyscf.scf.hf import eig as eiggen

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

mode = 'MODE'

imp_atom = 'IMP_ATOM' if mode == 'production' else 'Co'

# chemical potential
mu = CHEM_POT if mode == 'production' else 0.06

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
#           CAS from HF natural orbital
############################################################
def gen_cas_hf(emb_mf):

    # emb_mf should already be converged

    if rank == 0:
        nocc = round(np.sum(emb_mf.mo_occ)) // 2
        print('nocc = ', nocc)
        no_idx = range(nocc-nocc_act, nocc+nvir_act)
        no_coeff = emb_mf.mo_coeff[:, no_idx]

    # final natural orbital coefficients for CAS
    no_coeff = comm.bcast(no_coeff, root=0)

    # new mf object for CAS
    # first build CAS mol object
    mol_cas = gto.M()
    mol_cas.nelectron = nocc_act*2
    mol_cas.verbose = 4
    mol_cas.symmetry = 'c1'
    mol_cas.incore_anyway = True
    emb_mf_cas = scf.RHF(mol_cas)
    
    # hcore & eri in NO basis
    h1e = no_coeff.T @ (emb_mf.get_hcore() @ no_coeff)

    #TODO
    #g2e = ao2mo.restore(8, ao2mo.kernel(emb_mf._eri, no_coeff), n_no)
    
    
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

    # low level CAS space DM (NO basis)
    dm_low_cas = emb_mf_cas.make_rdm1()

    # low level total embedding DM (AO basis)
    dm_low = emb_mf.make_rdm1()

    max_memory = int(emb_mf.max_memory * 1E6/nprocs) # in unit of bytes, per mpi proc

    if rank == 0:
        if not os.path.isdir(scratch):
            os.mkdir(scratch)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

    # high level CAS space DM (NO basis)
    dm_cas = dmrg_mo_pdm(emb_mf_cas, ao_orbs=range(0, n_no), mo_orbs=None, scratch=scratch, \
            reorder_method=reorder_method, n_threads=n_threads, memory=max_memory, \
            gs_bond_dims=gs_bond_dims, gs_n_steps=gs_n_steps, gs_tol=gs_tol, gs_noises=gs_noises, \
            load_dir=None, save_dir=save_dir, verbose=3, mo_basis=False, ignore_ecore=False, \
            mpi=True, dyn_corr_method=dyn_corr_method, ncore=ncore_dmrg, nvirt=nvirt_dmrg)

    print('dm_cas.shape:', dm_cas.shape)
    dm_cas = dm_cas[0] + dm_cas[1]

    # difference between high and low level CAS space DM (NO basis)
    ddm = dm_cas - dm_low_cas
    
    # transform the difference to AO basis
    ddm = no_coeff @ ddm @ no_coeff.T

    # total embedding DM (AO basis), low-level dm plus high-level correction
    rdm = dm_low + ddm

    return rdm

############################################################
#                   gate voltage
############################################################
gate = GATE if mode == 'production' else 0
gate_label = 'gate%5.3f'%(gate)

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

bath_dir = '/home/zuxin/projects/transport/models/Cu_fcc_100/Cu/'

if rank == 0:
    print('start reading lead mean field data from', bath_dir)

#------------ check Cu HOMO/LUMO energy ------------

# should be the same as the one for computing contact
#a = 3.6
num_layer = 8
bath_cell_label = 'Cu_' + Cu_basis + '_a' + str(a) + '_n' + str(num_layer)

bath_cell_fname = bath_dir + 'cell_' + bath_cell_label + '.chk'
bath_cell = chkfile.load_cell(bath_cell_fname)
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
bath_mf.__dict__.update( chkfile.load(bath_mf_fname, 'scf') )

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
hyb_broadening= 0.002
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


########################################
#       bath discretization
########################################
nbath_per_ene = 3
#grid = gen_mix_linear(wl=-0.6, wh=1.2, w0=mu, dw_coarse=0.1, dw_fine=0.001, w_fine=0.01)



nbe = 50 # total number of bath energies
wlg = -0.6
whg = 1.2
log_disc_base = 2.0
grid_type = 'custom1'
wlog=0.01
grid = gen_grid(nbe, wlg, whg, mu, grid_type=grid_type, log_disc_base=log_disc_base, wlog=wlog)



nbath = (len(grid)-1) * nbath_per_ene
nemb = nbath + nao_imp

print('nemb = ', nemb)


def gen_hemb(mu):
    hemb = np.zeros((spin, nemb, nemb))
    
    # one body part
    for s in range(spin):
        Gamma_s = lambda e: Gamma(e)[s]
        e,v = direct_disc_hyb(Gamma_s, grid, nint=5, nbath_per_ene=nbath_per_ene)
        
        hemb[s,:,:] = emb_ham(Hemb_imp[s,:,:], e, v)

    if rank == 0:
        print('bath energies = ', e)

    return hemb

############################################################
#               build embedding model
############################################################
print('mu = ', mu)

mol = gto.M()
mol.verbose = 4
mol.incore_anyway = True
mol.build()

emb_mf = RHF_imp(mol, mu, nao_imp, eri_lo_imp)


hemb = gen_hemb(mu)
emb_mf.get_hcore = lambda *args: hemb[0]
emb_mf.get_ovlp = lambda *args: np.eye(nemb)

emb_mf.mo_energy = np.zeros([nemb])
emb_mf.mo_occ = np.zeros([nemb])

# dm initial guess
iocc = np.array(hemb[0].diagonal() < mu, dtype=float)
dm0 = np.diag(iocc) * 2.0
dm0[:nao_imp,:nao_imp] = DM_lo_imp.copy()

if rank == 0:
    emb_mf.kernel(dm0)

emb_mf.mo_coeff = comm.bcast(emb_mf.mo_coeff, root=0)
emb_mf.mo_energy = comm.bcast(emb_mf.mo_energy, root=0)
emb_mf.mo_occ = comm.bcast(emb_mf.mo_occ, root=0)


nocc = round(np.sum(emb_mf.mo_occ)) // 2
print('nocc = ', nocc)
no_idx = range(nocc-nocc_act, nocc+nvir_act)
no_coeff = emb_mf.mo_coeff[:, no_idx]

# final natural orbital coefficients for CAS
no_coeff = comm.bcast(no_coeff, root=0)

print('ready to compute g2e')
g2e = ao2mo.kernel(emb_mf.eri_imp[0], no_coeff[:nao_imp,:])

print('ready to compute e2e')

e2e = 0
a=1
b=1
c=2
d=2
for i in range(nao_imp):
    for j in range(nao_imp):
        for k in range(nao_imp):
            for l in range(nao_imp):
                e2e += no_coeff[i,a] * no_coeff[j,b] * no_coeff[k,c] * no_coeff[l,d] * eri_lo_imp[0, i,j,k,l]
#e2e = np.zeros((nao_imp, nao_imp, nao_imp, nao_imp))
#for a in range(n_no):
#    print('a = ', a)
#    for b in range(n_no):
#        for c in range(n_no):
#            for d in range(n_no):
#                for i in range(nao_imp):
#                    for j in range(nao_imp):
#                        for k in range(nao_imp):
#                            for l in range(nao_imp):
#                                e2e[a,b,c,d] += no_coeff[i,a] * no_coeff[j,b] * no_coeff[k,c] * no_coeff[l,d] * eri_lo_imp[0, i,j,k,l]
#
#print('g2e diff = ', np.max(abs(e2e-g2e)))
print('e2e0000 = ', e2e)
print('max =', np.max(abs(g2e)))
print('g2e0000 = ', g2e[a,b,c,d])
exit()
############################################################
#           build small CAS from embedding mf
############################################################

rdm = get_rdm_emb(emb_mf)
rdm_imp = rdm[0:nao_imp, 0:nao_imp]
nelec_imp_dmrg = np.trace(rdm_imp)
print('nelec diff = ', nelec_imp_dmrg - nelec_lo_imp)

############################################################
#               embedding model rdm
############################################################

if rank == 0:
    print('DMRG nelec on imp (val+virt) = ', np.trace(rdm_imp))
    print('DMRG nelec on imp (val)      = ', np.trace(rdm_imp[0:nval_imp, 0:nval_imp]))
    print('DMRG imp rdm diagonal = ', rdm_imp.diagonal())

exit()

############################################################
#                   end of main
############################################################

