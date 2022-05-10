import numpy as np
import matplotlib.pyplot as plt
import h5py, time, sys, os, scipy

from mpi4py import MPI
from bath_disc import *

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

def gen_siam_mf(on_site, Gamma, U, mu, wl, wh, nbath):

    ## distance to mu
    #wl0 = mu - wl
    #wh0 = wh - mu
    #
    ## number of energies above/below the Fermi level
    #dif = round(np.log(abs(wh0/wl0))/np.log(log_disc_base)) // 2
    #nl = nbath//2 - dif
    #nh = nbath - nl
    #
    #grid = np.concatenate((gen_log_grid(mu, wl, log_disc_base, nl), [mu], \
    #        gen_log_grid(mu, wh, log_disc_base, nh)))
    
    # 2/3 coarse bath states
    # 1/3 fine bath states near mu
    nbath_coarse = round(nbath/3) # one side
    nbath_fine = nbath - nbath_coarse*2 + 1
    w = 0.05

    grid = np.concatenate( (\
            np.linspace(wl,mu-w-0.01,nbath_coarse), \
            np.linspace(-w,w,nbath_fine), \
            np.linspace(w+0.01,wh,nbath_coarse) ) )
    
    print('grid = ', grid)

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
    mol.nelectron=nemb//2*2

    mol.build()
    
    #emb_mf = scf_mu.RHF(mol, mu)
    emb_mf = scf.RHF(mol)
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
on_site = -U/2
siam_mf = gen_siam_mf(on_site, hyb_const, U, mu=0.0, wl=-1.0, wh=1.0, nbath=30)
siam_mf.kernel()

fock = siam_mf.get_fock()
print('fock.shape = ', fock.shape)

wl_mf = -6
wh_mf = 6
nw_mf = 200
delta = 0.2
freqs_mf = np.linspace(wl_mf, wh_mf, nw_mf)

# ldos (mean-field level)
ldos_mf = np.zeros(nw_mf)
for iw in range(nw_mf):
    z = freqs_mf[iw] + 1j*delta
    gf = np.linalg.inv(z*np.eye(fock.shape[0]) - fock[:,:])
    ldos_mf[iw] = -1./np.pi*gf[0,0].imag

'''
#------------ CC impurity solver ------------
siam_cc = cc.RCCSD(siam_mf)
siam_cc.conv_tol = 1e-7
siam_cc.conv_tol_normt = 1e-5
siam_cc.diis_space = 6
siam_cc.level_shift = 0.3
siam_cc.max_cycle = 300
siam_cc.verbose = 4
siam_cc.iterative_damping = 0.7
siam_cc.frozen = 0
siam_cc.kernel()
siam_cc.solve_lambda()

from fcdmft.solver import mpiccgf as ccgf

gmres_tol = 1e-3

wl_cc1 = -0.05
wh_cc1 = 0.05
nw_cc1 = 100
freqs_cc1 = np.linspace(wl_cc1, wh_cc1, nw_cc1)
eta1 = 0.005

ao_orbs = range(1)

gf = ccgf.CCGF(siam_cc, tol=gmres_tol)
g_ip = gf.ipccsd_ao(ao_orbs, freqs_cc1.conj(), siam_mf.mo_coeff, eta1).conj()
g_ea = gf.eaccsd_ao(ao_orbs, freqs_cc1, siam_mf.mo_coeff, eta1)
gf1 = g_ip + g_ea

ldos_cc1 = -1./np.pi*gf1[0,0,:].imag

wl_cc2 = -6
wh_cc2 = 6
nw_cc2 = 200
freqs_cc2 = np.linspace(wl_cc2, wh_cc2, nw_cc2)
eta2 = 0.2

ao_orbs = range(1)

gf = ccgf.CCGF(siam_cc, tol=gmres_tol)
g_ip = gf.ipccsd_ao(ao_orbs, freqs_cc2.conj(), siam_mf.mo_coeff, eta2).conj()
g_ea = gf.eaccsd_ao(ao_orbs, freqs_cc2, siam_mf.mo_coeff, eta2)
gf2 = g_ip + g_ea

ldos_cc2 = -1./np.pi*gf2[0,0,:].imag
ldos_cc = np.concatenate((ldos_cc1, ldos_cc2))
freqs_cc = np.concatenate((freqs_cc1, freqs_cc2))

idx = np.argsort(freqs_cc)
freqs_cc = freqs_cc[idx]
ldos_cc = ldos_cc[idx]

plt.plot(freqs_mf, ldos_mf)
plt.plot(freqs_cc1, ldos_cc1)
plt.plot(freqs_cc2, ldos_cc2)
#plt.plot(freqs_cc, ldos_cc)
plt.show()

exit()
'''

#------------ DMRG impurity solver ------------
from fcdmft.solver.gfdmrg import dmrg_mo_pdm

nimp = 1
nemb = fock.shape[0]

freqs = np.array([0.])
delta = 0.1

# threshold for natural orbital occupation number
#thresh = 5e-3

# only construct virtual natural orbitals
#vno_only = False


# number of occ/virtual natural orbitals to keep
# override thresh & vno_only!
nocc_act = 5
nvir_act = 6


# do a CAS natural orbital construction (using CISD) before DMRG
# cas_cisd

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
# make_casno_cisd return mf_cas, no_coeff, fock_cas, dm_cas, dm

# dm_ci is in MO basis!
dm_ci = siam_ci.make_rdm1()

nocc = siam_ci.nocc
nmo = siam_ci.nmo

print('dm_ci.shape = ', dm_ci.shape)
#print('dm_ci = ', dm_ci)
#plt.imshow(np.abs(dm_ci))
#plt.show()

print('nocc = ', nocc)
print('nmo = ', nmo)

# virtual block
no_occ_v, no_coeff_v = np.linalg.eigh(dm_ci[nocc:,nocc:])
no_occ_v = np.flip(no_occ_v) # sort eigenvalues from large to small
no_coeff_v = np.flip(no_coeff_v, axis=1)
print('vir NO occupancy:', no_occ_v)

# occupied block
no_occ_o, no_coeff_o = np.linalg.eigh(dm_ci[:nocc,:nocc])
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
# rotate occ/vir NOs so that they diagonal the Fock matrix within their subspace
fvv = np.diag(siam_mf.mo_energy[nocc:])
fvv_no = np.dot(no_coeff_v.T, np.dot(fvv, no_coeff_v))
_, v_canon_v = np.linalg.eigh(fvv_no[:nvir_act,:nvir_act])

foo = np.diag(siam_mf.mo_energy[:nocc])
foo_no = np.dot(no_coeff_o.T, np.dot(foo, no_coeff_o))
_, v_canon_o = np.linalg.eigh(foo_no[-nocc_act:,-nocc_act:])

# at this stage, no_coeff is MO-to-NO coefficient
no_coeff_v = np.dot(siam_mf.mo_coeff[:,nocc:], np.dot(no_coeff_v[:,:nvir_act], v_canon_v))
no_coeff_o = np.dot(siam_mf.mo_coeff[:,:nocc], np.dot(no_coeff_o[:,-nocc_act:], v_canon_o))
# now no_coeff is AO-to-NO coefficient

ne_sum = np.sum(no_occ_o[no_idx_o]) + np.sum(no_occ_v[no_idx_v])
n_no = len(no_idx_o) + len(no_idx_v)
nelectron = int(round(ne_sum))

print('number of electrons in NO: ', ne_sum)
print('number of electrons: ', nelectron)
print('number of NOs: ', n_no)

# FIXME still do not understand scdm, but 'local' might be important!
# use local = True for now
no_coeff_o = scdm(no_coeff_o, np.eye(no_coeff_o.shape[0]))
no_coeff_v = scdm(no_coeff_v, np.eye(no_coeff_v.shape[0]))
no_coeff = np.concatenate((no_coeff_o, no_coeff_v), axis=1)
no_coeff = comm.bcast(no_coeff, root=0)

# new mf object for CAS
mol_cas = gto.M()
mol_cas.nelectron = nelectron
mol_cas.verbose = 4
mol_cas.symmetry = 'c1'
mol_cas.incore_anyway = True
siam_mf_cas = scf.RHF(mol_cas)

# hcore & eri in NO basis
h1e = np.dot(no_coeff.T, np.dot(siam_mf.get_hcore(), no_coeff))
g2e = ao2mo.restore(8, ao2mo.kernel(siam_mf._eri, no_coeff), n_no)

dm_hf = siam_mf.make_rdm1()
# ovlp is identity in this case
#ovlp = mf.get_ovlp()
#CS = np.dot(no_coeff.T, ovlp)
#dm_cas_no = np.dot(CS, np.dot(dm_hf, CS.T))
dm_cas_no = np.dot(no_coeff.T, np.dot(dm_hf, no_coeff))
JK_cas_no = _get_veff(dm_cas_no, g2e)[0]
JK_full_no = np.dot(no_coeff.T, np.dot(siam_mf.get_veff(), no_coeff))
h1e = h1e + JK_full_no - JK_cas_no
h1e = 0.5 * (h1e + h1e.T)
h1e = comm.bcast(h1e, root=0)

# set up integrals for siam_mf_cas
siam_mf_cas.get_hcore = lambda *args: h1e
siam_mf_cas.get_ovlp = lambda *args: np.eye(n_no)
siam_mf_cas._eri = g2e

if rank == 0:
    siam_mf_cas.max_cycle = 1
    siam_mf_cas.kernel(dm_cas_no)
comm.Barrier()
siam_mf_cas.mo_occ = comm.bcast(siam_mf_cas.mo_occ, root=0)
siam_mf_cas.mo_energy = comm.bcast(siam_mf_cas.mo_energy, root=0)
siam_mf_cas.mo_coeff = comm.bcast(siam_mf_cas.mo_coeff, root=0)


# end of make_casno_cisd
############################################

# note that if return_dm=True and get_cas_mo=True, the dm returned by make_casno_cisd is CISD dm (in MO basis)!
dm_ci_ao = np.dot(siam_mf.mo_coeff, np.dot(dm_ci, siam_mf.mo_coeff.T))
#print('diff = ', np.linalg.norm(siam_mf.mo_coeff@siam_mf.mo_coeff.T - np.eye(nmo)))
if rank == 0:
    print('CISD Nelec = ', np.trace(dm_ci_ao[:nimp,:nimp]))

exit()

no_coeff = no_coeff[np.newaxis, ...]
#gf_hf = mf_gf(siam_mf, freqs, delta)
#gf_hf_cas = mf_gf(siam_mf_cas, freqs, delta)
dm_ao = siam_mf.make_rdm1()
dm_cas_ao = siam_mf_cas.make_rdm1()
#return mf_cas, no_coeff, gf_hf, gf_hf_cas, dm_ao, dm_cas_ao

exit()
###############
mf_dmrg = mf_cas
dm_orbs = range(len(mf_cas.mo_energy))

max_memory = mf.max_memory * 1E6

# set scratch folder
scratch = '/home/zuxin/projects/transport/models/CoCu_chain_clean/tmp_dmrg'
if rank == 0:
    if not os.path.isdir(scratch):
        print('scratch not found! mkdir now')
        os.mkdir(scratch)
comm.Barrier()
os.environ['TMPDIR'] = scratch

dm = dmrg_mo_pdm(mf_dmrg, ao_orbs=dm_orbs, mo_orbs=None, scratch=scratch, reorder_method=reorder_method, \
        n_threads=n_threads, memory=max_memory, gs_bond_dims=gs_bond_dims, \
        gs_n_steps=gs_n_steps, gs_tol=gs_tol, gs_noises=gs_noises, load_dir=load_dir, save_dir=save_dir, \
        verbose=3, mo_basis=False, ignore_ecore=False, mpi=True, dyn_corr_method=dyn_corr_method, \
        ncore=ncore, nvirt=nvirt)

# assemble CASCI density matrix
dm = dm[0] + dm[1]
ddm = dm - dm_low_cas
ddm = np.dot(no_coeff[0], np.dot(ddm, no_coeff[0].T))
dm_cas = dm_low + ddm

if rank == 0:
    logger.info(mf, 'DMRG Nelec = %s', np.trace(dm_cas[:nimp,:nimp]))

imp_occ = dm_cas[0,0]





