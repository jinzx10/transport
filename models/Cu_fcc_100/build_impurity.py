import numpy as np
import h5py, time, sys
import matplotlib.pyplot as plt
from mpi4py import MPI

from fcdmft.solver import scf_mu

from matplotlib import colors


from pyscf import gto, ao2mo, cc, scf, dft

from pyscf.pbc import scf as pbcscf
from pyscf.pbc import df as pbcdf
from pyscf.pbc.lib import chkfile


from utils.surface_green import *
from utils.bath_disc import *

from pyscf.scf.hf import eig as eiggen

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()


############################################################
#           read contact's mean-field data
############################################################

contact_dir = 'data/'
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

bath_dir = 'Cu_def2-svp-bracket/'
nat_Cu_lead = 16

if rank == 0:
    print('start reading lead mean field data from', bath_dir)
    print('nat_Cu = ', nat_Cu_lead)

#------------ check Cu HOMO/LUMO energy ------------

# should be the same as the one for computing contact
#a = 2.55

bath_cell_label = 'Cu_' + 'nat' + str(nat_Cu_lead) + '_a' + str(a)

bath_cell_fname = bath_dir + 'cell_' + bath_cell_label + '.chk'
bath_cell = chkfile.load_cell(bath_cell_fname)

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
bath_kmf.__dict__.update( chkfile.load(bath_mf_fname, 'scf') )

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
#n_hyb = nval_Co 

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
#               bath discretization
############################################################
mu = -0.145

#------------ bath discretization (log) ------------
wlg = -0.6
whg = 0.6
nbe = 50 # total number of bath energies
nbath_per_ene = 6

log_disc_base = 1.4

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
    
    #------------ build & solve embedding model with mean-field ------------
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

#------------ build & solve embedding model with mean-field ------------
mol = gto.M()
mol.verbose = 0
#mol.incore_anyway = True
mol.build()

if 'r' in method_label:
    emb_mf = scf_mu.RHF(mol, mu)
    emb_mf.get_hcore = lambda *args: hemb[0]
    emb_mf._eri = ao2mo.restore(8, eri_imp[0], nemb)
    emb_mf.mo_energy = np.zeros([nemb])
else:
    emb_mf = scf_mu.UHF(mol, mu)
    emb_mf.get_hcore = lambda *args: hemb
    emb_mf._eri = eri_imp
    emb_mf.mo_energy = np.zeros([2,nemb])

emb_mf.get_ovlp = lambda *args: np.eye(nemb)
emb_mf.max_cycle = 150
emb_mf.conv_tol = 1e-12
emb_mf.diis_space = 15


if rank == 0:
    print('scf starts')
    if spin == 1:
        emb_mf.kernel(dm0[0])
    else:
        emb_mf.kernel(dm0)
    print('scf finished')

emb_mf.mo_coeff = comm.bcast(emb_mf.mo_coeff, root=0)
emb_mf.mo_energy = comm.bcast(emb_mf.mo_energy, root=0)
emb_mf.mo_occ = comm.bcast(emb_mf.mo_occ, root=0)
#emb_mf.mo_coeff = mo_coeff
#emb_mf.mo_energy = mo_energy
#emb_mf.mo_occ = mo_occ

rdm1 = emb_mf.make_rdm1()
fock = emb_mf.get_fock()
if spin == 1:
    fock = fock[np.newaxis,...]

if rank == 1:
    if spin == 1:
        print('trace(rdm1[Co val+virt])', np.trace(rdm1[0:nao_Co,0:nao_Co]))
        print('trace(rdm1[Co val])', np.trace(rdm1[0:nval_Co,0:nval_Co]))
    else:
        print('trace(rdm1[Co alpha val+virt])', np.trace(rdm1[0, 0:nao_Co,0:nao_Co]))
        print('trace(rdm1[Co beta val+virt])', np.trace(rdm1[1, 0:nao_Co,0:nao_Co]))

        print('trace(rdm1[Co alpha val])', np.trace(rdm1[0, 0:nval_Co,0:nval_Co]))
        print('trace(rdm1[Co beta val])', np.trace(rdm1[1, 0:nval_Co,0:nval_Co]))

#------------ compute Co mean-field LDoS ------------
wl_mf = -0.8
wh_mf = 0.8
nw_mf = 200
delta = 0.01
freqs_mf = np.linspace(wl_mf, wh_mf, nw_mf)

# from embedding problem (mean-field level)
A = np.zeros((spin, nval_Co, nw_mf))
for s in range(spin):
    for iw in range(nw_mf):
        z = freqs_mf[iw] + 1j*delta
        gf = np.linalg.inv(z*np.eye(nemb) - fock[s,:,:])
        A[s,:,iw] = -1./np.pi*np.diag(gf[0:nval_Co,0:nval_Co]).imag

# raw mean-field LDoS from contact Green's function
ldos_mf = np.zeros((spin, nval_Co, nw_mf))
for iw in range(nw_mf):
    z = freqs_mf[iw] + 1j*delta
    GC = contact_Greens_function(z)
    for s in range(spin):
        ldos_mf[s,:,iw] = -1./np.pi*np.diag(GC[s,:nval_Co, :nval_Co]).imag

if rank == 0:
    fname = 'ldos_' + method_label + '.h5'
    fh = h5py.File(fname, 'w')
    fh['ldos_mf'] = ldos_mf
    fh['freqs_mf'] = freqs_mf
    fh.close()

#------------ ground state CCSD ------------
if spin == 1:
    emb_cc = cc.RCCSD(emb_mf)
else:
    from fcdmft.solver import ucc_eri
    emb_cc = cc.UCCSD(emb_mf)

emb_cc.conv_tol = 1e-7
emb_cc.conv_tol_normt = 1e-5
emb_cc.diis_space = 6
#emb_cc.level_shift = 0.3
emb_cc.max_cycle = 300
emb_cc.verbose = 4
#emb_cc.iterative_damping = 0.7
emb_cc.frozen = 0

emb_cc_rdm = None

if rank == 0:
    emb_cc.kernel()
    emb_cc.solve_lambda()

    # by default make_rdm1 generates rdm1 in MO basis!
    rdm_mo = emb_cc.make_rdm1()

    if spin == 1:
        emb_cc_rdm = np.dot(emb_mf.mo_coeff, np.dot(rdm_mo, emb_mf.mo_coeff.T))
        print('CC Co Nelec = ', np.trace(emb_cc_rdm[:nao_Co, :nao_Co]))
    else:
        emb_cc_rdm = np.zeros((2,nemb,nemb))
        emb_cc_rdm[0] = np.dot(emb_mf.mo_coeff[0], np.dot(rdm_mo[0], emb_mf.mo_coeff[0].T))
        emb_cc_rdm[1] = np.dot(emb_mf.mo_coeff[1], np.dot(rdm_mo[1], emb_mf.mo_coeff[1].T))
        print('CC Co Nelec alpha = ', np.trace(emb_cc_rdm[0, :nao_Co, :nao_Co]))
        print('CC Co Nelec beta  = ', np.trace(emb_cc_rdm[1, :nao_Co, :nao_Co]))

emb_cc.t1 = comm.bcast(emb_cc.t1, root=0)
emb_cc.t2 = comm.bcast(emb_cc.t2, root=0)
emb_cc.l1 = comm.bcast(emb_cc.l1, root=0)
emb_cc.l2 = comm.bcast(emb_cc.l2, root=0)
emb_cc_rdm = comm.bcast(emb_cc_rdm, root=0)

sys.stdout.flush()
comm.Barrier()

#------------ CCGF ------------

from fcdmft.solver import mpiuccgf as uccgf
from fcdmft.solver import mpiccgf as ccgf

gmres_tol = 1e-3
ao_orbs = range(nval_Co)

# use a coarse grid with a large broadening acrossing the whole range
# plus a fine grid with a small broadening near the Fermi level
wl_cc1 = -0.8
wh_cc1 = 0.8
nw_cc1 = 500
freqs_cc1 = np.linspace(wl_cc1, wh_cc1, nw_cc1)
eta1 = 0.2

if spin == 1:
    gf = ccgf.CCGF(emb_cc, tol=gmres_tol)
    g_ip = gf.ipccsd_ao(ao_orbs, freqs_cc1.conj(), emb_mf.mo_coeff, eta1).conj()
    sys.stdout.flush()
    g_ea = gf.eaccsd_ao(ao_orbs, freqs_cc1, emb_mf.mo_coeff, eta1)
    sys.stdout.flush()

    g_ip = g_ip[np.newaxis,...]
    g_ea = g_ea[np.newaxis,...]

else:
    gf = uccgf.UCCGF(emb_cc, tol=gmres_tol)
    g_ip = gf.ipccsd_ao(ao_orbs, freqs_cc1.conj(), emb_mf.mo_coeff, eta1).conj()
    sys.stdout.flush()
    g_ea = gf.eaccsd_ao(ao_orbs, freqs_cc1, emb_mf.mo_coeff, eta1)
    sys.stdout.flush()

gf1 = g_ip + g_ea

ldos_cc1 = np.zeros((spin, nval_Co, nw_cc1))
for i in range(nval_Co):
    for s in range(spin):
        ldos_cc1[s,i,:] = -1./np.pi*gf1[s,i,i,:].imag

wl_cc2 = -0.17
wh_cc2 = -0.15
nw_cc2 = 100
freqs_cc2 = np.linspace(wl_cc2, wh_cc2, nw_cc2)
eta2 = 0.005

if spin == 1:
    gf = ccgf.CCGF(emb_cc, tol=gmres_tol)
    g_ip = gf.ipccsd_ao(ao_orbs, freqs_cc2.conj(), emb_mf.mo_coeff, eta2).conj()
    sys.stdout.flush()
    g_ea = gf.eaccsd_ao(ao_orbs, freqs_cc2, emb_mf.mo_coeff, eta2)
    sys.stdout.flush()

    g_ip = g_ip[np.newaxis,...]
    g_ea = g_ea[np.newaxis,...]

else:
    gf = uccgf.UCCGF(emb_cc, tol=gmres_tol)
    g_ip = gf.ipccsd_ao(ao_orbs, freqs_cc2.conj(), emb_mf.mo_coeff, eta2).conj()
    sys.stdout.flush()
    g_ea = gf.eaccsd_ao(ao_orbs, freqs_cc2, emb_mf.mo_coeff, eta2)
    sys.stdout.flush()

gf2 = g_ip + g_ea

ldos_cc2 = np.zeros((spin, nval_Co, nw_cc2))
for i in range(nval_Co):
    for s in range(spin):
        ldos_cc2[s,i,:] = -1./np.pi*gf2[s,i,i,:].imag

freqs_cc = np.concatenate((freqs_cc1, freqs_cc2))
ldos_cc = np.concatenate((ldos_cc1, ldos_cc2), axis=2)

idx = np.argsort(freqs_cc)
freqs_cc = freqs_cc[idx]
ldos_cc = ldos_cc[:,:,idx]

if rank == 0:
    fname = 'ldos_' + method_label + '.h5'
    fh = h5py.File(fname, 'a')
    fh['ldos_cc'] = ldos_cc
    fh['freqs_cc'] = freqs_cc 
    fh.close()

exit()



