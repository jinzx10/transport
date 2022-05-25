import numpy as np
import h5py, time, sys, os, scipy
import matplotlib.pyplot as plt
from mpi4py import MPI

from fcdmft.solver import scf_mu

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
#           read contact's mean-field data
############################################################

contact_dir = '/home/zuxin/projects/transport/models/CoCu_chain_clean/data/'
if rank == 0:
    print('read contact\'s mean field data from', contact_dir)

Co_basis = 'def2-svp'
Cu_basis = 'def2-svp-bracket'

nat_Cu = 9
l = 2.7
r = 2.7
a = 2.55

cell_label = 'Co_' + Co_basis + '_Cu' + str(nat_Cu) + '_' + Cu_basis \
        + '_l' + str(l) + '_r' + str(r) + '_a' + str(a)

cell_fname = contact_dir + '/cell_' + cell_label + '.chk'

kmesh = [1,1,1]
kpts = np.array([[0,0,0]])
k_label = 'k' + str(kmesh[0]) + 'x' + str(kmesh[2]) + 'x' + str(kmesh[2])

gdf_fname = contact_dir + '/cderi_' + cell_label + '_' + k_label + '.h5'

xcfun = 'pbe0'
method_label = 'rks_' + xcfun
solver_label = 'newton'

data_fname = contact_dir + '/data_' \
        + cell_label + '_' + method_label + '_' + solver_label + '_' + k_label + '.h5'

#------------ read core Hamiltonian and HF veff (built with DFT DM)  ------------
fh = h5py.File(data_fname, 'r')

# Co atom block only
hcore_lo_Co = np.asarray(fh['hcore_lo_Co'])
JK_lo_ks_Co = np.asarray(fh['JK_lo_Co'])
JK_lo_hf_Co = np.asarray(fh['JK_lo_hf_Co'])

# entire center region, Co + 9 Cu atoms
hcore_lo_contact = np.asarray(fh['hcore_lo'])
JK_lo_ks_contact = np.asarray(fh['JK_lo'])
JK_lo_hf_contact = np.asarray(fh['JK_lo_hf'])

#------------ read density matrix ------------
DM_lo_Co = np.asarray(fh['DM_lo_Co'])

#------------ read ERI ------------
eri_lo_Co = np.asarray(fh['eri_lo_Co'])
fh.close()

#************ permute eri for unrestricted case ************
# see Tianyu's run_dmft.py 
if eri_lo_Co.shape[0] == 3:
    eri_permuted = np.zeros_like(eri_lo_Co)
    eri_permuted[0] = eri_lo_Co[0]
    eri_permuted[1] = eri_lo_Co[2]
    eri_permuted[2] = eri_lo_Co[1]
    eri_lo_Co = eri_permuted.copy()
    del eri_permuted

# size of the Co block (should be 6+16=22 for def2-svp)
nao_Co = hcore_lo_Co.shape[2]

# 3d+4s
nval_Co = 6

# size of the contact block (should be 22+nat_Cu_contact*(6+9) for {Co:'def2-svp', Cu:'def2-svp-bracket'})
nao_contact = hcore_lo_contact.shape[2]

# restricted -> spin=1
# unrestricted -> spin=2
spin, nkpts = hcore_lo_contact.shape[0:2]

# hcore is a 4-d array with shape (spin, nkpts, nao, nao)
if rank == 0:
    print('hcore_lo_contact.shape = ', hcore_lo_contact.shape)
    print('JK_lo_ks_contact.shape = ', JK_lo_ks_contact.shape)
    print('JK_lo_hf_contact.shape = ', JK_lo_hf_contact.shape)
    print('hcore_lo_Co.shape = ', hcore_lo_Co.shape)
    print('JK_lo_ks_Co.shape = ', JK_lo_ks_Co.shape)
    print('JK_lo_hf_Co.shape = ', JK_lo_hf_Co.shape)
    print('DM_lo_Co.shape = ', DM_lo_Co.shape)
    print('eri_lo_Co.shape = ', eri_lo_Co.shape)
    print('')

    print('hcore_lo_contact.dtype = ', hcore_lo_contact.dtype)
    print('JK_lo_ks_contact.dtype = ', JK_lo_ks_contact.dtype)
    print('JK_lo_hf_contact.dtype = ', JK_lo_hf_contact.dtype)
    print('hcore_lo_Co.dtype = ', hcore_lo_Co.dtype)
    print('JK_lo_ks_Co.dtype = ', JK_lo_ks_Co.dtype)
    print('JK_lo_hf_Co.dtype = ', JK_lo_hf_Co.dtype)
    print('DM_lo_Co.dtype = ', DM_lo_Co.dtype)
    print('eri_lo_Co.dtype = ', eri_lo_Co.dtype)
    print('')

    print('nao_Co = ', nao_Co)
    print('nval_Co = ', nval_Co)
    print('nao_contact = ', nao_contact)

    print('finish reading contact mean field data\n')

comm.Barrier()

############################################################
#               gate voltage (untested)
############################################################
# shift the Co one-body Hamiltonian
gate = -0.0

if rank == 0:
    print('gate voltage = ', gate)

for s in range(spin):
    for ik in range(nkpts):
        hcore_lo_contact[s,ik,0:nao_Co,0:nao_Co] += gate*np.eye(nao_Co) 
        hcore_lo_Co[s,ik] += gate*np.eye(nao_Co) 

############################################################
#               read lead's mean-field data
############################################################

bath_dir = '/home/zuxin/projects/transport/models/CoCu_chain_clean/Cu_def2-svp-bracket/'
nat_Cu_lead = 16

if rank == 0:
    print('start reading lead mean field data from', bath_dir)
    print('nat_Cu = ', nat_Cu_lead)

#------------ check Cu HOMO/LUMO energy ------------

# should be the same as the one for computing contact
#a = 2.55

bath_cell_label = 'Cu_' + 'nat' + str(nat_Cu_lead) + '_a' + str(a)

bath_cell_fname = bath_dir + 'cell_' + bath_cell_label + '.chk'
bath_cell = pbcchkfile.load_cell(bath_cell_fname)

bath_kpts = np.array([[0,0,0]])

bath_gdf_fname = bath_dir + 'cderi_' + bath_cell_label + '.h5'
bath_gdf = pbcdf.GDF(bath_cell, bath_kpts)
bath_gdf._cderi = bath_gdf_fname

if 'ks' in method_label:
    bath_kmf = pbcscf.KRKS(bath_cell, bath_kpts).density_fit()
    bath_kmf.xc = 'pbe'
    bath_method_label = 'rks'
else:
    bath_kmf = pbcscf.KRHF(bath_cell, bath_kpts).density_fit()
    bath_method_label = 'rhf'

bath_mf_fname = bath_dir + bath_cell_label + '_' + bath_method_label + '.chk'

bath_kmf.with_df = bath_gdf
bath_kmf.__dict__.update( pbcchkfile.load(bath_mf_fname, 'scf') )

ihomo = 29*nat_Cu_lead//2-1
ilumo = 29*nat_Cu_lead//2

E_Cu_homo = np.asarray(bath_kmf.mo_energy)[0,ihomo]
E_Cu_lumo = np.asarray(bath_kmf.mo_energy)[0,ilumo]

if rank == 0:
    print('ihomo = ', ihomo, '      occ = ', np.asarray(bath_kmf.mo_occ)[0,ihomo], '      E = ', E_Cu_homo)
    print('ilumo = ', ilumo, '      occ = ', np.asarray(bath_kmf.mo_occ)[0,ilumo], '      E = ', E_Cu_lumo)

comm.Barrier()

#------------ get H00 and H01 (for surface Green's function) ------------
bath_fname = bath_dir + '/hcore_JK_lo_' + bath_cell_label + '_' + bath_method_label + '.h5'
fh = h5py.File(bath_fname, 'r')

hcore_lo_lead = np.asarray(fh['hcore_lo'])
JK_lo_lead = np.asarray(fh['JK_lo'])
F_lo_lead = hcore_lo_lead + JK_lo_lead

if rank == 0:
    print('F_lo_lead.shape = ', F_lo_lead.shape)

# number of orbitals per atom
nao_per_Cu = 15

# number of atoms per principal layer
nat_ppl = 4

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
hcore_lo_Co = 1./nkpts * np.sum(hcore_lo_Co, axis=1)
JK_lo_hf_Co = 1./nkpts * np.sum(JK_lo_hf_Co, axis=1)
DM_lo_Co = 1./nkpts * np.sum(DM_lo_Co, axis=1)

JK_00 = scf_mu._get_veff(DM_lo_Co, eri_lo_Co)

Himp_Co = hcore_lo_Co + JK_lo_hf_Co - JK_00

if rank == 0:
    print('hcore_lo_Co.shape = ', hcore_lo_Co.shape)
    print('JK_lo_hf_Co.shape = ', JK_lo_hf_Co.shape)
    print('DM_lo_Co.shape = ', DM_lo_Co.shape)
    print('JK_00.shape = ', JK_00.shape)
    print('Himp_Co.shape = ', Himp_Co.shape)
    print('')

    if spin == 1:
        print('trace(DM_lo_Co) = ', np.trace(DM_lo_Co[0]))
    else:
        print('trace(DM_lo_Co[0]) = ', np.trace(DM_lo_Co[0]))
        print('trace(DM_lo_Co[1]) = ', np.trace(DM_lo_Co[1]))
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

    V_L[nao_Co:nao_Co+nao_ppl,:] = H01.T.conj()
    V_R[-nao_ppl:,:] = H01

    Sigma_L = V_L @ g00 @ V_L.T.conj()
    Sigma_R = V_R @ g00 @ V_R.T.conj()

    # contact block of the Green's function
    # TODO multiple spin
    G_C = np.zeros((spin, nao_contact, nao_contact), dtype=complex)
    for s in range(spin):
        G_C[s,:,:] = np.linalg.inv( z*np.eye(nao_contact) - hcore_lo_contact[s] - JK_lo_hf_contact[s] \
                - Sigma_L - Sigma_R )
    return G_C

sys.stdout.flush()
comm.Barrier()

############################################################
#               hybridization Gamma
############################################################
# number of orbitals that couple to the bath, usually nval_Co or nao_Co
n_hyb = nao_Co

# broadening for computing hybridization Gamma from self energy
hyb_broadening= 0.01
# -1/pi*imag(Sigma(e+i*delta))
# (spin, n_hyb, n_hyb)
def Gamma(e):
    z = e + 1j*hyb_broadening
    G_C = contact_Greens_function(z)
    
    Sigma_imp = np.zeros((spin, n_hyb, n_hyb),dtype=complex)
    for s in range(spin):
        Sigma_imp[s,:,:] = z*np.eye(n_hyb) - Himp_Co[s,:n_hyb,:n_hyb] \
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
ldos = np.zeros((spin,nw,nao_Co))

for iw in range(nw):
    z = freqs[iw] + 1j*ldos_mf_broadening
    GC = contact_Greens_function(z)
    GC_Co = GC[:,:nao_Co, :nao_Co]
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
nemb = nbath + nao_Co

hemb = np.zeros((spin, nemb, nemb))

if rank == 0:
    print('hemb.shape = ', hemb.shape)
    print('bath discretization starts')

# one body part
for s in range(spin):
    Gamma_s = lambda e: Gamma(e)[s]
    e,v = direct_disc_hyb(Gamma_s, grid, nint=3, nbath_per_ene=nbath_per_ene)
    
    #------------ build & solve embedding model with mean-field ------------
    hemb[s,:,:] = emb_ham(Himp_Co[s,:,:], e, v)

if rank == 0:
    print('bath discretization finished')

# electron repulsion integral
# only non-zero on the impurity
eri_imp = np.zeros([spin*(spin+1)//2, nemb, nemb, nemb, nemb])
eri_imp[:,:nao_Co,:nao_Co,:nao_Co,:nao_Co] = eri_lo_Co

dm0 = np.zeros((spin,nemb,nemb))
dm0[:,:nao_Co,:nao_Co] = DM_lo_Co.copy()

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
        print('trace(rdm1[Co val+virt])', np.trace(rdm1[0:nao_Co,0:nao_Co]))
        print('trace(rdm1[Co val])', np.trace(rdm1[0:nval_Co,0:nval_Co]))
    else:
        print('trace(rdm1[Co alpha val+virt])', np.trace(rdm1[0, 0:nao_Co,0:nao_Co]))
        print('trace(rdm1[Co beta val+virt])', np.trace(rdm1[1, 0:nao_Co,0:nao_Co]))

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
nemb = nbath + nao_Co

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
    hemb[s,:,:] = emb_ham(Himp_Co[s,:,:], e, v)

if rank == 0:
    print('bath discretization finished')

# electron repulsion integral
# only non-zero on the impurity
eri_imp = np.zeros([spin*(spin+1)//2, nemb, nemb, nemb, nemb])
eri_imp[:,:nao_Co,:nao_Co,:nao_Co,:nao_Co] = eri_lo_Co

dm0 = np.zeros((spin,nemb,nemb))
dm0[:,:nao_Co,:nao_Co] = DM_lo_Co.copy()

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
        print('trace(rdm1[Co val+virt])', np.trace(rdm1[0:nao_Co,0:nao_Co]))
        print('trace(rdm1[Co val])', np.trace(rdm1[0:nval_Co,0:nval_Co]))
    else:
        print('trace(rdm1[Co alpha val+virt])', np.trace(rdm1[0, 0:nao_Co,0:nao_Co]))
        print('trace(rdm1[Co beta val+virt])', np.trace(rdm1[1, 0:nao_Co,0:nao_Co]))

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


############################################################
#       raw mean-field LDoS (from contact GF)
############################################################
#ldos_mf = np.zeros((spin, nval_Co, nwa))
#for iw in range(nwa):
#    z = all_freqs[iw] + 1j*extra_delta
#    GC = contact_Greens_function(z)
#    for s in range(spin):
#        ldos_mf[s,:,iw] = -1./np.pi*np.diag(GC[s,:nval_Co, :nval_Co]).imag
#
#if rank == 0:
#    fname = 'ldos_build_impurity.h5'
#    fh = h5py.File(fname, 'w')
#    fh['ldos_mf'] = ldos_mf
#    fh['freqs'] = all_freqs
#    fh.close()
#    
#    dw = all_freqs[1:] - all_freqs[:-1]
#    mf_int = np.sum(ldos_mf[0,:,:-1]*dw, axis=1)
#    print('integrated mean-field LDoS within (%6.4f, %6.4f) = '%(all_freqs[0], all_freqs[-1]), mf_int)
#
#    #plt.plot(all_freqs, np.sum(ldos_mf, axis=1)[0])
#    #plt.show()

#exit()


############################################################
#               bath discretization
############################################################
mu = -0.145

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
nemb = nbath + nao_Co

hemb = np.zeros((spin, nemb, nemb))

if rank == 0:
    print('hemb.shape = ', hemb.shape)
    print('bath discretization starts')

# one body part
for s in range(spin):
    Gamma_s = lambda e: Gamma(e)[s]
    e,v = direct_disc_hyb(Gamma_s, grid, nint=3, nbath_per_ene=nbath_per_ene)
    
    hemb[s,:,:] = emb_ham(Himp_Co[s,:,:], e, v)

if rank == 0:
    print('bath discretization finished')
    print('bath energies = ', e)

# electron repulsion integral
# only non-zero on the impurity
eri_imp = np.zeros([spin*(spin+1)//2, nemb, nemb, nemb, nemb])
eri_imp[:,:nao_Co,:nao_Co,:nao_Co,:nao_Co] = eri_lo_Co

dm0 = np.zeros((spin,nemb,nemb))
dm0[:,:nao_Co,:nao_Co] = DM_lo_Co.copy()

############################################################
#               build embedding model
############################################################
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
emb_mf_fname = 'build_impurity_mf.chk'
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

dm_mf = emb_mf.make_rdm1()

if rank == 0:
    print('trace(dm_mf[Co val+virt])', np.trace(dm_mf[0:nao_Co,0:nao_Co]))
    print('trace(dm_mf[Co val])', np.trace(dm_mf[0:nval_Co,0:nval_Co]))

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


############################################################
#               CAS-CISD natural orbital
############################################################
nocc_act = NOCC_ACT
nvir_act = NVIR_ACT

from pyscf import ci

emb_cisd = ci.CISD(emb_mf)
emb_cisd.max_cycle = 1000
emb_cisd.max_space = 20
emb_cisd.max_memory = 100000
emb_cisd.conv_tol = 1e-10
emb_cisd.verbose = 4

emb_cisd_fname = 'build_impurity_cisd.chk'
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
    n_no = len(no_idx_o) + len(no_idx_v)
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
    emb_mf_cas.kernel(dm_cas_no)
    print('scf within CAS finished!')
comm.Barrier()

emb_mf_cas.mo_occ = comm.bcast(emb_mf_cas.mo_occ, root=0)
emb_mf_cas.mo_energy = comm.bcast(emb_mf_cas.mo_energy, root=0)
emb_mf_cas.mo_coeff = comm.bcast(emb_mf_cas.mo_coeff, root=0)

# gf_hf (HF GF in AO basis) is computed above
# now we compute gf_hf_cas
mo_energy_cas = emb_mf_cas.mo_energy
mo_coeff_cas  = emb_mf_cas.mo_coeff

gf_hf_cas = np.zeros((nwa, n_no, n_no), dtype=complex)
for iw in range(nwa):
    z = all_freqs[iw] + 1j*delta
    gf_mo_cas = np.diag( 1. / (z - mo_energy_cas) )
    gf_hf_cas[iw,:,:] = mo_coeff_cas @ gf_mo_cas @ mo_coeff_cas.T

# end of make_casno_cisd
# return emb_mf_cas, no_coeff, dm_ci_mo

############################################
# see cas_cisd
# cas_cisd pass return_dm=True to make_casno_cisd and leave get_cas_mo as default (True)

dm_ci_ao = emb_mf.mo_coeff @ dm_ci_mo @ emb_mf.mo_coeff.T
if rank == 0:
    print('CISD Nelec on impurity = ', np.trace(dm_ci_ao[:nao_Co,:nao_Co]))

#no_coeff = no_coeff[np.newaxis, ...]
dm_ao = emb_mf.make_rdm1()
dm_cas_ao = emb_mf_cas.make_rdm1()

# cas_cisd return mf_cas, no_coeff, gf_hf, gf_hf_cas, dm_ao, dm_cas_ao
# gf_hf is gf_low; gf_hf_cas is gf_low_cas
###############

mf_dmrg = emb_mf_cas
gf_orbs = range(n_no) # full GF in CAS space
diag_only= False

# exists in dmrg_gf (see dmft_solver.py)
# NOTE but not sure why save number of CAS orbitals in bare mf
#emb_mf.nocc_act = nocc_act
#emb_mf.nvir_act = nvir_act

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
# NOTE scratch conflict for different job!
# NOTE verbose=3 good enough
# NOTE extra_freqs & extra_delta are set above
#extra_freqs = None
#extra_delta = None
n_threads = NUM_THREADS
reorder_method='gaopt'
max_memory = emb_mf.max_memory * 1E6 # in unit of bytes, per mpi proc

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
gf_noises = [1E-4] * 2 + [1E-5] * 2 + [1E-7] * 1 + [0]

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


###################

from fcdmft.solver.gfdmrg import dmrg_mo_pdm, dmrg_mo_gf

# gf_dmrg_cas has dimension (nao, nao, nwa) (len(freqs) is its last dimension)
dm_dmrg_cas, gf_dmrg_cas = dmrg_mo_gf( \
        mf_dmrg, freqs=freqs, delta=delta, ao_orbs=gf_orbs, mo_orbs=None, \
        extra_freqs=extra_freqs, extra_delta=extra_delta, scratch=scratch, add_rem='+-', \
        n_threads=n_threads, reorder_method=reorder_method, memory=max_memory, \
        gs_bond_dims=gs_bond_dims, gf_bond_dims=gf_bond_dims, gf_n_steps=gf_n_steps, \
        gs_n_steps=gs_n_steps, gs_tol=gs_tol, gf_noises=gf_noises, gf_tol=gf_tol, \
        gs_noises=gs_noises, gmres_tol=gmres_tol, load_dir=None, save_dir=save_dir, \
        cps_bond_dims=cps_bond_dims, cps_noises=[0], cps_tol=gs_tol, cps_n_steps=gs_n_steps, \
        verbose=1, mo_basis=False, ignore_ecore=False, n_off_diag_cg=n_off_diag_cg, \
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




