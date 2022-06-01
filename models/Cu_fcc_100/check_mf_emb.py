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

from utils.diis import diis
from utils.surface_green import *
from utils.bath_disc import *

from pyscf.scf.hf import eig as eiggen

############################################################
# this script performs a sanity check to make sure that
# the embedding model solved by the same mf method (ks)
# recovers the imp block as in the original contact calculation
############################################################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

imp_atom = 'Co'

# chemical potential
mu = 0.06

############################################################
#       frequencies to compute spectra
############################################################
# coarse grid
wl_freqs = mu - 0.4
wh_freqs = mu + 0.3
delta = 0.01
nw = 100
freqs = np.linspace(wl_freqs, wh_freqs, nw)
dw = freqs[1] - freqs[0]

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

# entire center region, imp + some Cu atoms
hcore_lo_contact = np.asarray(fh['hcore_lo'])
JK_lo_ks_contact = np.asarray(fh['JK_lo'])

#------------ read density matrix ------------
DM_lo_imp = np.asarray(fh['DM_lo_imp'])

#------------ read ERI ------------
eri_lo_imp = np.asarray(fh['eri_lo_imp'])

# for mf embedding sanity check
C_ao_lo_tot = np.asarray(fh['C_ao_lo_tot'])[0,0]
DM_lo_tot = np.asarray(fh['DM_lo_tot'])
JK_ao = np.asarray(fh['JK_ao'])

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
    print('hcore_lo_imp.shape = ', hcore_lo_imp.shape)
    print('JK_lo_ks_imp.shape = ', JK_lo_ks_imp.shape)
    print('DM_lo_imp.shape = ', DM_lo_imp.shape)
    print('eri_lo_imp.shape = ', eri_lo_imp.shape)
    print('')

    print('hcore_lo_contact.dtype = ', hcore_lo_contact.dtype)
    print('JK_lo_ks_contact.dtype = ', JK_lo_ks_contact.dtype)
    print('hcore_lo_imp.dtype = ', hcore_lo_imp.dtype)
    print('JK_lo_ks_imp.dtype = ', JK_lo_ks_imp.dtype)
    print('DM_lo_imp.dtype = ', DM_lo_imp.dtype)
    print('eri_lo_imp.dtype = ', eri_lo_imp.dtype)
    print('')

    print('nao_imp = ', nao_imp)
    print('nval_imp = ', nval_imp)
    print('nao_contact = ', nao_contact)

    print('finish reading contact mean field data\n')

comm.Barrier()

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

##################
#plt.imshow(np.abs(F_lo_lead[0]))
#plt.show()
#exit()
##################

############################################################
#               impurity Hamiltonian
############################################################
hcore_lo_imp = 1./nkpts * np.sum(hcore_lo_imp, axis=1)
JK_lo_ks_imp = 1./nkpts * np.sum(JK_lo_ks_imp, axis=1)
DM_lo_imp = 1./nkpts * np.sum(DM_lo_imp, axis=1)

#------------ compute JK_00 ------------
# JK_00 stands for intra-(imp val+virt) two-body mean field potential 
# for DFT it is computed by taking the difference between veff with/without imp val+virt DM

# build contact object

cell = chkfile.load_cell(cell_fname)

gdf = pbcdf.GDF(cell)
gdf._cderi = gdf_fname

mf = pbcscf.RKS(cell).density_fit()
mf.xc = 'pbe0'
mf.with_df = gdf

## test: make sure veff generated by DM_ao is JK_ao
#veff_test = np.asarray(kmf.get_veff(dm=DM_ao))[0]
#print('diff = ', np.linalg.norm(veff_test-JK_ao))
#exit()

# total DM in LO basis, imp val+virt block removed
DM_lo_tmp = np.copy(DM_lo_tot)
DM_lo_tmp[9:31,9:31] = 0

## test: make sure DM_lo_imp is a block within DM_lo_tot 
#print('diff = ', np.linalg.norm(DM_lo_imp[0]-DM_lo_tot[9:31,9:31]))
#exit()

# test: make sure hcore+JK_ao generates DM_ao


# transform to AO basis
DM_ao_tmp = C_ao_lo_tot @ DM_lo_tmp @ C_ao_lo_tot.T.conj()
JK_ao_tmp = np.asarray(mf.get_veff(dm=DM_ao_tmp))

# transform the potential to LO basis
JK_lo_tmp = C_ao_lo_tot.T.conj() @ JK_ao_tmp @ C_ao_lo_tot
#JK_lo_tmp = np.dot(np.dot(C_ao_lo_tot.T.conj(), JK_ao_tmp), C_ao_lo_tot)
JK_lo_tmp = JK_lo_tmp[np.newaxis,...]

# take the different for the imp val+virt block
JK_00 = JK_lo_ks_imp - JK_lo_tmp[:,9:31,9:31]

Hemb_imp = hcore_lo_imp + JK_lo_ks_imp - JK_00

if rank == 0:
    print('hcore_lo_imp.shape = ', hcore_lo_imp.shape)
    print('JK_lo_ks_imp.shape = ', JK_lo_ks_imp.shape)
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
#           contact's Green's function 
############################################################
hcore_lo_contact = 1./nkpts * np.sum(hcore_lo_contact, axis=1)
JK_lo_ks_contact = 1./nkpts * np.sum(JK_lo_ks_contact, axis=1)

if rank == 0:
    print('hcore_lo_contact.shape = ', hcore_lo_contact.shape)
    print('JK_lo_ks_contact.shape = ', JK_lo_ks_contact.shape)
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

    # lead's surface Green's function is turned off!
    Sigma_L.fill(0.0)
    Sigma_R.fill(0.0)

    # contact block of the Green's function
    G_C = np.zeros((spin, nao_contact, nao_contact), dtype=complex)
    for s in range(spin):
        G_C[s,:,:] = np.linalg.inv( z*np.eye(nao_contact) - hcore_lo_contact[s] - JK_lo_ks_contact[s] \
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
#               bath discretization
############################################################
#------------ log discretization ------------
wlg = -0.6
whg = 1.8
nbe = 200 # total number of bath energies
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

grid = np.linspace(wlg,whg,nbe+1)
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
#eri_imp = np.zeros([spin*(spin+1)//2, nemb, nemb, nemb, nemb])
#eri_imp[:,:nao_imp,:nao_imp,:nao_imp,:nao_imp] = eri_lo_imp

dm0 = np.zeros((spin,nemb,nemb))
dm0[:,:nao_imp,:nao_imp] = DM_lo_imp.copy()

############################################################
#               user-defined mf object
############################################################
class RHF2(scf.hf.RHF):

    __doc__ = scf.hf.RHF.__doc__

    def __init__(self, mol, mu):

        self._cell = chkfile.load_cell(cell_fname)

        self.mu = mu
        scf.hf.RHF.__init__(self,mol)

        self._gdf = pbcdf.GDF(self._cell)
        self._gdf._cderi = gdf_fname

        self._mf = pbcscf.RKS(self._cell).density_fit()
        self._mf.xc = xcfun
        self._mf.with_df = self._gdf


        # file that stores extra data for sanity check
        fh = h5py.File(data_fname, 'r')
        self._C_ao_lo_tot = np.asarray(fh['C_ao_lo_tot'])[0,0]
        self._DM_lo_tot = np.asarray(fh['DM_lo_tot'])
        #self._S_ao_ao = np.asarray(fh['S_ao_ao'])
        fh.close()

        C = self._C_ao_lo_tot

        dm_tot_lo_tmp = np.copy(self._DM_lo_tot)
        dm_tot_lo_tmp[9:31,9:31] = 0
        dm_tot_ao_tmp = C @ dm_tot_lo_tmp @ C.T.conj()
        #dm_tot_ao_tmp = dm_tot_ao_tmp[np.newaxis,...]
        veff_ao_tmp = self._mf.get_veff(dm=dm_tot_ao_tmp)
        veff_ao_tmp = np.asarray(veff_ao_tmp)
        veff_lo_tmp = np.dot(np.dot(C.T.conj(), veff_ao_tmp), C)
        
        self._veff_ref = veff_lo_tmp[9:31,9:31]

        print('RHF2 object initialized!')

    def get_occ(self, mo_energy=None, mo_coeff=None):
        mo_occ = np.zeros_like(mo_energy)
        mo_occ[mo_energy<=self.mu] = 2.0
        return mo_occ


    def get_veff(self, mol, dm, dm_last=0, vhf_last=0):
        # imp def2-svp: core-9 val-6 virt-16

        # given an embedding dm, extract the imp block
        # imp val+virt has 22 orbitals
        dm_imp = dm[:22, :22]

        # DM in LO basis, all orbitals (core+val+virt)
        dm_tot_lo = np.copy(self._DM_lo_tot)

        # C is the AO-to-LO transformation matrix (all LO, including core)
        C = np.copy(self._C_ao_lo_tot)

        # replace the imp val+virt block
        dm_tot_lo[9:31,9:31] = dm_imp

        # P^{AO} = C P^{LO} \dg{C}

        # transform to AO basis
        dm_tot_ao = C @ dm_tot_lo @ C.T.conj()

        ## self._kmf is a pbc scf, need an extra k axis
        #dm_tot_ao = dm_tot_ao[np.newaxis,...]

        # compute veff
        veff_ao = self._mf.get_veff(dm=dm_tot_ao)

        # veff_ao is a tagged array, extract its content
        veff_ao = np.asarray(veff_ao)

        # transform to LO
        veff_lo = C.T.conj() @ veff_ao @ C
        #veff_lo = np.dot(np.dot(C.T.conj(), veff_ao), C)

        veff_diff = veff_lo[9:31,9:31] - self._veff_ref

        # veff in the embedding model
        # only non-zero in the imp block
        veff = np.zeros_like(dm)

        # extract imp val+virt
        veff[:22,:22] = veff_diff

        #return veff, veff_lo
        return veff



############################################################
#               build embedding model
############################################################
mol = gto.M()
mol.verbose = 4
mol.build()

mf = RHF2(mol, mu)
mf.get_hcore = lambda *args: hemb[0]
mf.get_ovlp = lambda *args: np.eye(nemb)
mf.mo_energy = np.zeros([nemb])


# sanity check
# mf.get_veff returns intra-(imp val+virt) veff
# Cu-to-imp veff is incorporated in mf.hcore (which is Himp_imp) and is stored as mf._veff_ref
veff_test = mf.get_veff(mol=mol,dm=dm0[0])
fock_imp_test = veff_test[:22,:22] + mf.get_hcore()[:22,:22]
fock_imp_ref = hcore_lo_imp + JK_lo_ks_imp
fock_imp_ref = fock_imp_ref[0]
print('sanity check: fock_imp diff = ', np.linalg.norm(fock_imp_ref-fock_imp_test))


veff_test = mf.get_veff(mol=mol,dm=dm0[0])
JK_lo_tot = np.dot(np.dot(C_ao_lo_tot.T.conj(), JK_ao[0]), C_ao_lo_tot)
print('diff = ', np.linalg.norm(veff_test[0:22,0:22]+mf._veff_ref-JK_lo_tot[9:31,9:31]))

veff_0 = mf.get_veff(mol=mol,dm=dm0[0])
fock_0 = veff_0 + mf.get_hcore()
print('fock_0.shape = ', fock_0.shape)

e,v = eiggen(fock_0, np.eye(nemb))
#print('e = ', e)
v_occ = v[:, e<mu]
dm_new = 2. * v_occ @ v_occ.T
print('imp nelec = ', np.trace(dm_new[:22,:22]))
print('dm diff = ', np.linalg.norm(dm_new[:22,:22]-dm0[0,:22,:22]) )

#exit()

# Fock iteration for embedding Hamiltonian
# for commutator-DIIS
smearing_sigma = 0
def fock2fockcomm(fock_in):
    e,v = eiggen(fock_in, np.eye(nemb))
    
    # use fermi broadening to assist convergence
    #v_occ = v[:, e<mu]
    #dm = 2. * v_occ @ v_occ.T


    if abs(smearing_sigma) > 1e-10:
        occ = 2./( 1. + np.exp((e-mu)/smearing_sigma) )
    else:
        occ = np.zeros_like(e)
        occ[e<=mu] = 2.0

    dm = (v*occ) @ v.T

    fock_out = mf.get_hcore() + mf.get_veff(mol=mol, dm=dm)
    comm = fock_out @ dm - dm @ fock_out
    return fock_out, comm


fock, flag = diis(fock2fockcomm, fock_0, max_iter=300)

if flag == 0:
    print('fock.shape = ', fock.shape)
else:
    exit()

e,v = eiggen(fock, np.eye(nemb))

if abs(smearing_sigma) > 1e-10:
    occ = 2./( 1. + np.exp((e-mu)/smearing_sigma) )
else:
    occ = np.zeros_like(e)
    occ[e<=mu] = 2.0
rdm1 = (v*occ) @ v.T

#exit()


#orb_imp_occ = np.sum((v*v)[:22,:], axis=0) * 2
#orb_imp_occ_cumsum = np.cumsum(orb_imp_occ)
#print('orb imp occ = ', orb_imp_occ)
#print('cumsum = ', orb_imp_occ_cumsum)
#exit()




#mf.kernel(dm0=dm0[0])

#############################################################

#rdm1 = mf.make_rdm1()

print('spin = ', spin)
print('trace(rdm1[imp val+virt])', np.trace(rdm1[0:nao_imp,0:nao_imp]))
print('trace(rdm1[imp val])', np.trace(rdm1[0:nval_imp,0:nval_imp]))

#------------ compute & plot imp LDoS ------------
#fock = mf.get_fock(dm=rdm1)

if spin == 1:
    fock = fock[np.newaxis,...]

wld = -0.8
whd = 0.8
nwd = 200
delta = 0.01
freqs = np.linspace(wld,whd,nwd)

# new imp ldos from embedding model
A = np.zeros((spin,nwd,nval_imp))
for s in range(spin):
    for iw in range(nwd):
        z = freqs[iw] + 1j*delta
        gf = np.linalg.inv(z*np.eye(nemb) - fock[s,:,:])
        A[s,iw,:] = -1./np.pi*np.diag(gf[0:nval_imp,0:nval_imp]).imag


# raw mean-field LDoS from contact Green's function
ldos = np.zeros((spin,nwd,nval_imp))
for iw in range(nwd):
    z = freqs[iw] + 1j*delta
    GC = contact_Greens_function(z)
    for s in range(spin):
        ldos[s,iw,:] = -1./np.pi*np.diag(GC[s,:nval_imp, :nval_imp]).imag

fh = h5py.File('imp_rks_ldos.h5', 'w')
fh['freqs'] = freqs
fh['A'] = A[0,:,:]
fh['ldos'] = ldos[0,:,:]
fh.close()





