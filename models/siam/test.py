import numpy as np
import matplotlib.pyplot as plt
import h5py, time, sys, os, scipy

from mpi4py import MPI
from utils.bath_disc import *

from pyscf import gto, ao2mo, cc, scf, dft
from fcdmft.solver import scf_mu


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

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


##############################################
# test case: single-impurity Anderson model
##############################################


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

def gen_log_grid(w0, w, l, num):
    grid = w0 + (w-w0) * l**(-np.arange(num,dtype=float))
    if w > w0:
        return grid[::-1]
    else:
        return grid

def gen_siam_mf(on_site, Gamma, U, mu, grid):


    nbath = len(grid) - 1
    nemb = nbath + 1
    hemb = np.zeros((nemb, nemb))
    
    # hybridization Gamma
    hyb = lambda e: np.array([[Gamma]])
    e,v = direct_disc_hyb(hyb, grid, nint=3, nbath_per_ene=1)

    Himp = np.array([[on_site]])
    hemb[:,:] = emb_ham(Himp, e, v)
    
    eri_imp = np.zeros((nemb,nemb,nemb,nemb))
    eri_imp[0,0,0,0] = U
    
    mol = gto.M()
    mol.verbose = 4

    # set nelectron in mol, or use scf_mu (otherwise overload get_occ so that nocc>0)
    #mol.nelectron=nemb//2*2

    mol.build()
    
    emb_mf = scf_mu.RHF(mol, mu)
    #emb_mf = scf.RHF(mol)
    emb_mf.get_ovlp = lambda *args: np.eye(nemb)
    emb_mf.get_hcore = lambda *args: hemb
    emb_mf._eri = ao2mo.restore(8, eri_imp, nemb)
    emb_mf.mo_energy = np.zeros([nemb])
    emb_mf.max_cycle = 150
    emb_mf.conv_tol = 1e-12
    emb_mf.diis_space = 15

    return emb_mf


hyb_const = 0.2
U = 5.0
on_site = -2.501
mu = 0.0

#---------------- generate grid for bath --------------------

wl = -5.0
wh = 5.0
nbath = 100

# log grid
# distance to mu
wl0 = mu - wl
wh0 = wh - mu
log_disc_base = 1.3

# number of energies above/below the Fermi level
dif = round(np.log(abs(wh0/wl0))/np.log(log_disc_base)) // 2
nl = nbath//2 - dif
nh = nbath - nl

grid = np.concatenate((gen_log_grid(mu, wl, log_disc_base, nl), [mu], \
        gen_log_grid(mu, wh, log_disc_base, nh)))

'''
# linear grid
# 2/3 coarse bath states
# 1/3 fine bath states near mu
nbath_coarse = round(nbath/3) # one side
nbath_fine = nbath - nbath_coarse*2

# energy windows for fine bath states (2*w total)
w = 0.05

grid = np.concatenate( (\
        np.linspace(wl,mu-w-0.01,nbath_coarse), \
        np.linspace(-w,w,nbath_fine), \
        np.linspace(w+0.01,wh,nbath_coarse) ) )
'''
    
print('grid = ', grid)

#########################################################
# grid & broadening for spectrum

# no extra
wl_freqs = -0.1
wh_freqs = 0.1
nw = 31
freqs = np.linspace(wl_freqs, wh_freqs, nw)
all_freqs = freqs
nwa = len(all_freqs)
extra_freqs = None
extra_delta = None
delta_mf = 0.05
delta_dmrg = 0.05

'''
# normal grid dw = 1/40 ~ 0.025
# extra grid extra_dw*(2+2) = 0.006*4 = 0.024
# should be able to check the quality of extra grid by smoothness

wl_freqs = -0.2
wh_freqs = 0.2
#delta = 0.05
delta = 0.2
nw = 41
freqs = np.linspace(wl_freqs, wh_freqs, nw)

extra_delta = 0.01
extra_freqs = []

extra_dw = 0.005
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


all_freqs = np.array(sorted(list(set(list(freqs) + \
        ([x for xx in extra_freqs for x in xx] if extra_freqs is not None else [])))))

nwa = len(all_freqs)

print('all freqs = ', all_freqs)
print('extra_freqs = ', extra_freqs)
'''
#########################################################

siam_mf = gen_siam_mf(on_site, hyb_const, U, mu, grid)
siam_mf.kernel()

fock = siam_mf.get_fock()
nemb = fock.shape[0]

mo_energy = siam_mf.mo_energy
mo_coeff  = siam_mf.mo_coeff

print('mo_energy = ', mo_energy)
print('mo_energy diff = ', mo_energy[1:] - mo_energy[:-1])

# HF GF in AO basis
gf_hf = np.zeros((nwa, nemb, nemb), dtype=complex)
ldos_mf = np.zeros(nwa)
for iw in range(nwa):
    z = all_freqs[iw] + 1j*delta_mf
    gf_mo = np.diag( 1. / (z - mo_energy) )
    gf_hf[iw,:,:] = mo_coeff @ gf_mo @ mo_coeff.T
    ldos_mf[iw] = -1./np.pi*gf_hf[iw,0,0].imag

plt.plot(all_freqs, ldos_mf)
plt.xlim([all_freqs[0], all_freqs[-1]])
#plt.ylim([-0.01,0.5])
plt.show()
exit()

#########################################################
#               DMRG impurity solver
#########################################################
from fcdmft.solver.gfdmrg import dmrg_mo_pdm, dmrg_mo_gf

nimp = 1

# threshold for natural orbital occupation number
#thresh = 5e-3

# only construct virtual natural orbitals
#vno_only = False


# number of occ/virtual natural orbitals to keep
# override thresh & vno_only!
nocc_act = 5
nvir_act = 6


# do a CAS natural orbital construction before DMRG

# using CISD
from pyscf import ci
from pyscf.lib import chkfile

siam_ci = ci.CISD(siam_mf)
siam_ci.max_cycle = 1000
siam_ci.max_space = 40
siam_ci.conv_tol = 1e-10

chkfname = 'siam_ci.chk'

siam_ci.chkfile = chkfname
siam_ci.kernel()
siam_ci.dump_chk()

#if os.path.isfile(chkfname):
#    data = chkfile.load(chkfname, 'cisd')
#    siam_ci.__dict__.update(data)
#else:
#    siam_ci.chkfile = chkfname
#    if rank == 0:
#        siam_ci.kernel()
#        siam_ci.dump_chk()
#    comm.Barrier()

############################################
# make_casno_cisd return mf_cas, no_coeff, dm_ci_mo

# by default ci.make_rdm1 generate dm in MO basis!
dm_ci_mo = siam_ci.make_rdm1()

nmo = siam_ci.nmo
nocc = siam_ci.nocc # number of occupied MO
print('nocc = ', nocc)
print('nmo = ', nmo)

print('dm_ci_mo.shape = ', dm_ci_mo.shape)
#print('dm_ci_mo = ', dm_ci_mo)
#plt.imshow(np.abs(dm_ci_mo))
#plt.show()

# find natural orbital coeffcients below
# eventually we want AO-to-NO coefficients
# this is done in two steps
# 1. diagonalize the dm_ci_mo vir/occ blocks respective to find preliminary MO-to-NO coefficients
# 2. (semi-canonicalization) find fock matrix with those preliminary NO orbitals and diagonalize 
# the subspace fock. the resulting transformation are used to refine the preliminary MO-to-NO coefficients.


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
fvv = np.diag(siam_mf.mo_energy[nocc:])
fvv_no = no_coeff_v.T @ fvv @ no_coeff_v
_, v_canon_v = np.linalg.eigh(fvv_no[:nvir_act,:nvir_act])

foo = np.diag(siam_mf.mo_energy[:nocc])
foo_no = no_coeff_o.T @ foo @ no_coeff_o
_, v_canon_o = np.linalg.eigh(foo_no[-nocc_act:,-nocc_act:])

# at this stage, no_coeff is bare MO-to-NO coefficient (before semi-canonicalization)
no_coeff_v = siam_mf.mo_coeff[:,nocc:] @ no_coeff_v[:,:nvir_act] @ v_canon_v
no_coeff_o = siam_mf.mo_coeff[:,:nocc] @ no_coeff_o[:,-nocc_act:] @ v_canon_o
# now no_coeff is AO-to-NO coefficient (with semi-canonicalization)

ne_sum = np.sum(no_occ_o[no_idx_o]) + np.sum(no_occ_v[no_idx_v])
n_no = len(no_idx_o) + len(no_idx_v)
nelectron = int(round(ne_sum))

print('number of electrons in NO: ', ne_sum)
print('number of electrons: ', nelectron)
print('number of NOs: ', n_no)

# NOTE still do not understand scdm, but 'local' might be important!
# use local = True for now
no_coeff_o = scdm(no_coeff_o, np.eye(no_coeff_o.shape[0]))
no_coeff_v = scdm(no_coeff_v, np.eye(no_coeff_v.shape[0]))
no_coeff = np.concatenate((no_coeff_o, no_coeff_v), axis=1)
no_coeff = comm.bcast(no_coeff, root=0)

#---------------- new mf object for CAS ------------------
# first build CAS mol object
mol_cas = gto.M()
mol_cas.nelectron = nelectron
mol_cas.verbose = 4
mol_cas.symmetry = 'c1'
mol_cas.incore_anyway = True
siam_mf_cas = scf.RHF(mol_cas)

# hcore & eri in NO basis
h1e = no_coeff.T @ (siam_mf.get_hcore() @ no_coeff)
g2e = ao2mo.restore(8, ao2mo.kernel(siam_mf._eri, no_coeff), n_no)

dm_hf = siam_mf.make_rdm1()
# ovlp is identity in this case
#ovlp = mf.get_ovlp()
#CS = np.dot(no_coeff.T, ovlp)
#dm_cas_no = np.dot(CS, np.dot(dm_hf, CS.T))
dm_cas_no = no_coeff.T @ dm_hf @ no_coeff
JK_cas_no = _get_veff(dm_cas_no, g2e)[0]
JK_full_no = no_coeff.T @ siam_mf.get_veff() @ no_coeff
h1e = h1e + JK_full_no - JK_cas_no
h1e = 0.5 * (h1e + h1e.T)

h1e = comm.bcast(h1e, root=0)
g2e = comm.bcast(g2e, root=0)

# set up integrals for siam_mf_cas
siam_mf_cas.get_hcore = lambda *args: h1e
siam_mf_cas.get_ovlp = lambda *args: np.eye(n_no)
siam_mf_cas._eri = g2e

if rank == 0:
    # NOTE dm_cas_no should be already a converged dm, so max_cycle=1 is enough
    #siam_mf_cas.max_cycle = 1
    siam_mf_cas.kernel(dm_cas_no)
comm.Barrier()

siam_mf_cas.mo_occ = comm.bcast(siam_mf_cas.mo_occ, root=0)
siam_mf_cas.mo_energy = comm.bcast(siam_mf_cas.mo_energy, root=0)
siam_mf_cas.mo_coeff = comm.bcast(siam_mf_cas.mo_coeff, root=0)

# gf_hf (HF GF in AO basis) is computed above
# now we compute gf_hf_cas
mo_energy_cas = siam_mf_cas.mo_energy
mo_coeff_cas  = siam_mf_cas.mo_coeff

gf_hf_cas = np.zeros((nwa, n_no, n_no), dtype=complex)
for iw in range(nwa):
    z = all_freqs[iw] + 1j*delta_mf
    gf_mo_cas = np.diag( 1. / (z - mo_energy_cas) )
    gf_hf_cas[iw,:,:] = mo_coeff_cas @ gf_mo_cas @ mo_coeff_cas.T


# end of make_casno_cisd
# return siam_mf_cas, no_coeff, dm_ci_mo
############################################
# see cas_cisd
# cas_cisd pass return_dm=True to make_casno_cisd and leave get_cas_mo as default (True)

dm_ci_ao = siam_mf.mo_coeff @ dm_ci_mo @ siam_mf.mo_coeff.T
#print('diff = ', np.linalg.norm(siam_mf.mo_coeff@siam_mf.mo_coeff.T - np.eye(nmo)))
if rank == 0:
    print('CISD Nelec on impurity = ', np.trace(dm_ci_ao[:nimp,:nimp]))

#no_coeff = no_coeff[np.newaxis, ...]
dm_ao = siam_mf.make_rdm1()
dm_cas_ao = siam_mf_cas.make_rdm1()

# cas_cisd return mf_cas, no_coeff, gf_hf, gf_hf_cas, dm_ao, dm_cas_ao
# gf_hf is gf_low; gf_hf_cas is gf_low_cas

###############
mf_dmrg = siam_mf_cas
gf_orbs = range(len(siam_mf_cas.mo_energy))
diag_only= False

# exists in dmrg_gf (see dmft_solver.py)
# NOTE but not sure why save number of CAS orbitals in bare mf
#siam_mf.nocc_act = nocc_act
#siam_mf.nvir_act = nvir_act

# set scratch folder
scratch = 'SCRATCH'
if rank == 0:
    if not os.path.isdir(scratch):
        os.mkdir(scratch)
comm.Barrier()
os.environ['TMPDIR'] = scratch

save_dir = 'SAVEDIR'
if rank == 0:
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)


######################################
#       DMRG parameters
######################################
# extra_freqs & extra_delta are set above
#extra_freqs = None
#extra_delta = None
n_threads = NUM_THREADS
reorder_method='gaopt'
max_memory = siam_mf.max_memory * 1E6 # in unit of bytes, per mpi proc, see dmrg_mo_gf in gfdmrg.py

# bond dimension: make DW ~ 1e-5/e-6 or smaller
# 2000-5000 for gs
gs_n_steps = 20
gs_bond_dims = [400] * 5 + [800] * 5 + [1500] * 5 + [2000] * 5
gs_tol = 1E-8
gs_noises = [1E-3] * 5 + [1E-4] * 3 + [1e-7] * 2 + [0]

# gf_bond_dims 1k~1.5k
gf_n_steps = 6
gf_bond_dims = [200] * 2 + [500] * 4
gf_tol = 1E-3
gf_noises = [1E-4] * 2 + [1E-5] * 1 + [1E-7] * 1 + [0]

gmres_tol = 1E-7

# NOTE what do they mean?
cps_bond_dims = [2000]
#cps_noises = [0]
#cps_tol = 1E-10
#cps_n_steps=20
n_off_diag_cg = -2 #NOTE ???

# correlation method within DMRG (like CAS, but not recommended)
dyn_corr_method = None
ncore_dmrg = 0
nvirt_dmrg = 0

# scratch conflict for different job!
# memory: per mpi proc

# verbose=3 good enough


# gf_dmrg_cas has dimension (nao, nao, nwa) (len(freqs) is its last dimension)

###################
dm_dmrg_cas, gf_dmrg_cas = dmrg_mo_gf( \
        mf_dmrg, freqs=freqs, delta=delta_dmrg, ao_orbs=gf_orbs, mo_orbs=None, \
        extra_freqs=extra_freqs, extra_delta=extra_delta, scratch=scratch, add_rem='+-', \
        n_threads=n_threads, reorder_method=reorder_method, memory=max_memory, \
        gs_bond_dims=gs_bond_dims, gf_bond_dims=gf_bond_dims, gf_n_steps=gf_n_steps, \
        gs_n_steps=gs_n_steps, gs_tol=gs_tol, gf_noises=gf_noises, gf_tol=gf_tol, \
        gs_noises=gs_noises, gmres_tol=gmres_tol, load_dir=None, save_dir=save_dir, \
        cps_bond_dims=cps_bond_dims, cps_noises=[0], cps_tol=gs_tol, cps_n_steps=gs_n_steps, \
        verbose=3, mo_basis=False, ignore_ecore=False, n_off_diag_cg=n_off_diag_cg, \
        mpi=True, dyn_corr_method=dyn_corr_method, ncore=ncore_dmrg, nvirt=nvirt_dmrg, diag_only=diag_only)

print('gf_dmrg_cas.shape', gf_dmrg_cas.shape)

fh = h5py.File('gf_dmrg_cas.h5', 'w')
fh['gf_dmrg_cas'] = gf_dmrg_cas
fh.close()

#fh = h5py.File('gf_dmrg_cas.h5', 'r')
#gf_dmrg_cas = np.asarray(fh['gf_dmrg_cas'])
###################

# compute sigma_cas, NO active space self energy
sigma_cas = np.zeros((nwa, n_no, n_no), dtype=complex)
for iw in range(nwa):
    sigma_cas[iw,:,:] = np.linalg.inv(gf_hf_cas[iw,:,:]) - np.linalg.inv(gf_dmrg_cas[:,:,iw])

# convert sigma_cas to self energy in AO space
sigma_ao = np.zeros((nwa, nemb, nemb), dtype=complex)
for iw in range(nwa):
    sigma_ao[iw,:,:] = no_coeff @ (sigma_cas[iw,:,:] @ no_coeff.T)

# final GF in AO space (contains active space DMRG self energy)
gf_ao = np.zeros((nwa, nemb, nemb), dtype=complex)
ldos_dmrg = np.zeros(nwa)
for iw in range(nwa):
    gf_ao[iw,:,:] = np.linalg.inv( np.linalg.inv(gf_hf[iw,:,:]) - sigma_ao[iw,:,:] )
    ldos_dmrg[iw] = -1./np.pi * gf_ao[iw,0,0].imag

fh = h5py.File('ldos_siam_dmrg.h5', 'w')
fh['all_freqs'] = all_freqs
fh['ldos_mf'] = ldos_mf
fh['ldos_dmrg'] = ldos_dmrg
fh['mu'] = mu
fh.close()




