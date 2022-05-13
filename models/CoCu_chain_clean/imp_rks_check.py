import numpy as np
import h5py, time, sys
import matplotlib.pyplot as plt
from mpi4py import MPI

from fcdmft.solver import scf_mu

from surface_green import *
from bath_disc import *

from matplotlib import colors

from pyscf import gto, ao2mo, cc, scf, dft, lib

from pyscf.pbc import scf as pbcscf
from pyscf.pbc import df as pbcdf
from pyscf.pbc.lib import chkfile

from pyscf.pbc import dft as pbcdft
from diis import diis

from pyscf.scf.hf import eig as eiggen

class RHF2(scf.hf.RHF):

    __doc__ = scf.hf.RHF.__doc__

    def __init__(self, mol, mu):

        nat_Cu = 9
        l = 2.7
        r = 2.7
        a = 2.55
        datadir = 'Co_def2-svp_Cu_def2-svp-bracket/backup'

        cell_label = 'CoCu_' + str(nat_Cu) + '_l' + str(l) + '_r' + str(r) + '_a' + str(a)
        cell_fname = datadir + '/cell_' + cell_label + '.chk'
        self._cell = chkfile.load_cell(cell_fname)

        self.mu = mu
        scf.hf.RHF.__init__(self,mol)

        gdf_fname = datadir + '/cderi_' + cell_label + '.h5'

        kmesh = [1,1,1]
        kpts = self._cell.make_kpts(kmesh)
        self._gdf = pbcdf.GDF(self._cell, kpts)
        self._gdf._cderi = gdf_fname

        self._kmf = pbcdft.KRKS(self._cell, kpts).density_fit()
        self._kmf.xc = 'pbe'
        self._kmf.with_df = self._gdf

        fname = datadir + '/CoCu_set_ham_test.h5'
        fh = h5py.File(fname, 'r')
        self._C_ao_lo_tot = np.asarray(fh['C_ao_lo_tot'])
        self._DM_lo_tot = np.asarray(fh['DM_lo_tot'])
        self._S_ao_ao = np.asarray(fh['S_ao_ao'])
        fh.close()

        C = self._C_ao_lo_tot

        dm_tot_lo_tmp = np.copy(self._DM_lo_tot)
        dm_tot_lo_tmp[9:31,9:31] = 0
        dm_tot_ao_tmp = C @ dm_tot_lo_tmp @ C.T.conj()
        dm_tot_ao_tmp = dm_tot_ao_tmp[np.newaxis,...]
        veff_ao_tmp = self._kmf.get_veff(dm=dm_tot_ao_tmp)
        veff_ao_tmp = np.asarray(veff_ao_tmp)[0]
        veff_lo_tmp = np.dot(np.dot(C.T.conj(), veff_ao_tmp), C)
        
        self._veff_ref = veff_lo_tmp[9:31,9:31]

    def get_occ(self, mo_energy=None, mo_coeff=None):
        mo_occ = np.zeros_like(mo_energy)
        mo_occ[mo_energy<=self.mu] = 2.0
        return mo_occ


    def get_veff(self, mol, dm, dm_last=0, vhf_last=0):
        # Co def2-svp: core-9 val-6 virt-16

        # given an embedding dm, extract the Co block
        # Co val+virt has 22 orbitals
        dm_imp = dm[:22, :22]

        # DM in LO basis, all orbitals (core+val+virt)
        dm_tot_lo = np.copy(self._DM_lo_tot)

        # C is the AO-to-LO transformation matrix (all LO, including core)
        C = np.copy(self._C_ao_lo_tot)

        # replace the Co val+virt block
        dm_tot_lo[9:31,9:31] = dm_imp

        # P^{AO} = C P^{LO} \dg{C}

        # transform to AO basis
        dm_tot_ao = C @ dm_tot_lo @ C.T.conj()

        # self._kmf is a pbc scf, need an extra k axis
        dm_tot_ao = dm_tot_ao[np.newaxis,...]

        # compute veff
        veff_ao = self._kmf.get_veff(dm=dm_tot_ao)

        # veff_ao is a tagged array, extract its content
        veff_ao = np.asarray(veff_ao)[0]

        # transform to LO
        veff_lo = C.T.conj() @ veff_ao @ C
        #veff_lo = np.dot(np.dot(C.T.conj(), veff_ao), C)

        veff_diff = veff_lo[9:31,9:31] - self._veff_ref

        # veff in the embedding model
        # only non-zero in the imp block
        veff = np.zeros_like(dm)

        # extract Co val+virt
        veff[:22,:22] = veff_diff

        #return veff, veff_lo
        return veff

'''

class RKS2(dft.rks.RKS):

    __doc__ = dft.rks.RKS.__doc__

    def __init__(self, mol, mu):

        nat_Cu = 9
        l = 2.7
        r = 2.7
        a = 2.55
        datadir = 'Co_def2-svp_Cu_def2-svp-bracket/'

        cell_label = 'CoCu_' + str(nat_Cu) + '_l' + str(l) + '_r' + str(r) + '_a' + str(a)
        cell_fname = datadir + '/cell_' + cell_label + '.chk'
        self._cell = chkfile.load_cell(cell_fname)

        self.mu = mu
        dft.rks.RKS.__init__(self,mol)

        gdf_fname = datadir + '/cderi_' + cell_label + '.h5'

        kmesh = [1,1,1]
        kpts = self._cell.make_kpts(kmesh)
        self._gdf = pbcdf.GDF(self._cell, kpts)
        self._gdf._cderi = gdf_fname

        self._kmf = pbcdft.KRKS(self._cell, kpts).density_fit()
        self._kmf.xc = 'pbe'
        self._kmf.with_df = self._gdf

        fname = datadir + '/imp_rks_check.h5'
        fh = h5py.File(fname, 'r')
        self._C_ao_lo_tot = np.asarray(fh['C_ao_lo_tot'])
        self._DM_lo_tot = np.asarray(fh['DM_lo_tot'])
        self._S_ao_ao = np.asarray(fh['S_ao_ao'])
        fh.close()

    def get_occ(self, mo_energy=None, mo_coeff=None):
        mo_occ = np.zeros_like(mo_energy)
        mo_occ[mo_energy<=self.mu] = 2.0
        return mo_occ


    def get_veff(self, mol, dm, dm_last=0, vhf_last=0):
        # Co def2-svp: core-9 val-6 virt-16

        # given an embedding dm, extract the Co block
        # Co val+virt has 22 orbitals
        dm_imp = dm[:22, :22]

        # DM in LO basis, all orbitals (core+val+virt)
        dm_tot_lo = np.copy(self._DM_lo_tot)

        # AO-to-LO transformation matrix (include all LO)
        C = np.copy(self._C_ao_lo_tot)

        # replace the Co val+virt block
        dm_tot_lo[9:31,9:31] = dm_imp

        # P^{AO} = C P^{LO} \dg{C}

        # transform to AO basis
        dm_tot_ao = C @ dm_tot_lo @ C.T.conj()

        # self._kmf is a pbc scf, need an extra k axis
        dm_tot_ao = dm_tot_ao[np.newaxis,...]

        # compute veff
        veff_ao = self._kmf.get_veff(dm=dm_tot_ao)

        # now veff_ao is a tagged array
        #vj = (veff_ao.vj)[0]
        #vk = (veff_ao.vk)[0]
        ecoul = veff_ao.ecoul
        exc = veff_ao.exc
        veff_ao = np.asarray(veff_ao)[0]

        # transform to LO
        veff_lo = np.dot(np.dot(C.T.conj(), veff_ao), C)
        #vj_lo = np.dot(np.dot(C.T.conj(), vj), C)
        #vk_lo = np.dot(np.dot(C.T.conj(), vk), C)

        veff = np.zeros_like(dm)
        #vj = np.zeros_like(dm)
        #vk = np.zeros_like(dm)

        # extract Co val+virt
        veff[:22,:22] = veff_lo[9:31,9:31]
        #vj[:22,:22] = vj[9:31,9:31]
        #vk[:22,:22] = vk[9:31,9:31]

        vxc = lib.tag_array(veff, ecoul=ecoul, exc=exc, vj=None, vk=None)
        return vxc

'''

        



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

############################################################
#           read contact's mean-field data
############################################################

contact_dir = 'Co_def2-svp_Cu_def2-svp-bracket/backup'

if rank == 0:
    print('reading contact\'s mean field data from', contact_dir)

nat_Cu_contact = 9
a = 2.55
l = 2.7
r = 2.7

cell_label = 'CoCu_' + str(nat_Cu_contact) + '_l' + str(l) + '_r' + str(r) + '_a' + str(a)
method_label = 'rks_pbe'

#------------ read core Hamiltonian and DFT veff (built with DFT DM)  ------------
fname = contact_dir + '/hcore_JK_lo_' + cell_label + '_' + method_label + '.h5'
fh = h5py.File(fname, 'r')

# Co atom block only
hcore_lo_Co = np.asarray(fh['hcore_lo_Co'])
JK_lo_ks_Co = np.asarray(fh['JK_lo_Co'])

# entire center region, Co + 9 Cu atoms
hcore_lo_contact = np.asarray(fh['hcore_lo'])
JK_lo_ks_contact = np.asarray(fh['JK_lo'])
fh.close()

#------------ read density matrix ------------
fname = contact_dir + '/DM_lo_' + cell_label + '_' + method_label + '.h5'
fh = h5py.File(fname, 'r')
DM_lo_Co = np.asarray(fh['DM_lo_Co'])
fh.close()

#------------ read ERI ------------
fname = contact_dir + '/eri_lo_' + cell_label + '_' + method_label + '.h5'
fh = h5py.File(fname, 'r')
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
    print('hcore_lo_Co.shape = ', hcore_lo_Co.shape)
    print('JK_lo_ks_Co.shape = ', JK_lo_ks_Co.shape)
    print('DM_lo_Co.shape = ', DM_lo_Co.shape)
    print('eri_lo_Co.shape = ', eri_lo_Co.shape)
    print('')

    print('hcore_lo_contact.dtype = ', hcore_lo_contact.dtype)
    print('JK_lo_ks_contact.dtype = ', JK_lo_ks_contact.dtype)
    print('hcore_lo_Co.dtype = ', hcore_lo_Co.dtype)
    print('JK_lo_ks_Co.dtype = ', JK_lo_ks_Co.dtype)
    print('DM_lo_Co.dtype = ', DM_lo_Co.dtype)
    print('eri_lo_Co.dtype = ', eri_lo_Co.dtype)
    print('')

    print('nao_Co = ', nao_Co)
    print('nval_Co = ', nval_Co)
    print('nao_contact = ', nao_contact)

    print('finish reading contact mean field data\n')

comm.Barrier()

############################################################
#               gate voltage
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

cell_fname = bath_dir + 'cell_' + bath_cell_label + '.chk'
cell = chkfile.load_cell(cell_fname)

kpts = [[0,0,0]]

gdf_fname = bath_dir + 'cderi_' + bath_cell_label + '.h5'
gdf = pbcdf.GDF(cell, kpts)
gdf._cderi = gdf_fname

if 'ks' in method_label:
    kmf = pbcscf.KRKS(cell, kpts).density_fit()
    kmf.xc = 'pbe'
    bath_method_label = 'rks'
else:
    kmf = pbcscf.KRHF(cell, kpts).density_fit()
    bath_method_label = 'rhf'

mf_fname = bath_dir + bath_cell_label + '_' + bath_method_label + '.chk'

kmf.with_df = gdf
kmf.__dict__.update( chkfile.load(mf_fname, 'scf') )

ihomo = 29*nat_Cu_lead//2-1
ilumo = 29*nat_Cu_lead//2

E_Cu_homo = np.asarray(kmf.mo_energy)[0,ihomo]
E_Cu_lumo = np.asarray(kmf.mo_energy)[0,ilumo]

if rank == 0:
    print('ihomo = ', ihomo, '      occ = ', np.asarray(kmf.mo_occ)[0,ihomo], '      E = ', E_Cu_homo)
    print('ilumo = ', ilumo, '      occ = ', np.asarray(kmf.mo_occ)[0,ilumo], '      E = ', E_Cu_lumo)

comm.Barrier()

#------------ get H00 and H01 (for surface Green's function) ------------
fname = bath_dir + '/hcore_JK_lo_' + bath_cell_label + '_' + bath_method_label + '.h5'
fh = h5py.File(fname, 'r')

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
JK_lo_ks_Co = 1./nkpts * np.sum(JK_lo_ks_Co, axis=1)
DM_lo_Co = 1./nkpts * np.sum(DM_lo_Co, axis=1)

#JK_00 = scf_mu._get_veff(DM_lo_Co, eri_lo_Co)
#print('JK_00 = ', JK_00[0])
#plt.imshow(np.abs(JK_00[0]))
#plt.show()
#
#exit()

#------------ compute JK_00 ------------
# JK_00 stands for intra-(Co val+virt) two-body mean field potential 
# for DFT it is computed by taking the difference between veff with/without Co val+virt DM

# build Co+9Cu cell & rks object
nat_Cu = 9
l = 2.7
r = 2.7
a = 2.55
datadir = 'Co_def2-svp_Cu_def2-svp-bracket/backup'

cell_fname = datadir + '/cell_' + cell_label + '.chk'
cell = chkfile.load_cell(cell_fname)

gdf_fname = datadir + '/cderi_' + cell_label + '.h5'

kmesh = [1,1,1]
kpts = cell.make_kpts(kmesh)
gdf = pbcdf.GDF(cell, kpts)
gdf._cderi = gdf_fname

kmf = pbcdft.KRKS(cell, kpts).density_fit()
kmf.xc = 'pbe'
kmf.with_df = gdf


fname = datadir + '/CoCu_set_ham_test.h5'
fh = h5py.File(fname, 'r')
C_ao_lo_tot = np.asarray(fh['C_ao_lo_tot'])
DM_lo_tot = np.asarray(fh['DM_lo_tot'])
S_ao_ao = np.asarray(fh['S_ao_ao'])
DM_ao = np.asarray(fh['DM_ao'])
JK_ao = np.asarray(fh['JK_ao'])
DM_ao = DM_ao[np.newaxis,...]
fh.close()

## test: make sure veff generated by DM_ao is JK_ao
#veff_test = np.asarray(kmf.get_veff(dm=DM_ao))[0]
#print('diff = ', np.linalg.norm(veff_test-JK_ao))
#exit()

# total DM in LO basis, Co val+virt block removed
DM_lo_tmp = np.copy(DM_lo_tot)
DM_lo_tmp[9:31,9:31] = 0

## test: make sure DM_lo_Co is a block within DM_lo_tot 
#print('diff = ', np.linalg.norm(DM_lo_Co[0]-DM_lo_tot[9:31,9:31]))
#exit()

# test: make sure hcore+JK_ao generates DM_ao


# transform to AO basis
DM_ao_tmp = C_ao_lo_tot @ DM_lo_tmp @ C_ao_lo_tot.T.conj()
DM_ao_tmp = DM_ao_tmp[np.newaxis,...]
JK_ao_tmp = np.asarray(kmf.get_veff(dm=DM_ao_tmp))[0]

# transform the potential to LO basis
JK_lo_tmp = C_ao_lo_tot.T.conj() @ JK_ao_tmp @ C_ao_lo_tot
#JK_lo_tmp = np.dot(np.dot(C_ao_lo_tot.T.conj(), JK_ao_tmp), C_ao_lo_tot)
JK_lo_tmp = JK_lo_tmp[np.newaxis,...]

# take the different for the Co val+virt block
JK_00 = JK_lo_ks_Co - JK_lo_tmp[0,9:31,9:31]

Himp_Co = hcore_lo_Co + JK_lo_ks_Co - JK_00


if rank == 0:
    print('hcore_lo_Co.shape = ', hcore_lo_Co.shape)
    print('JK_lo_ks_Co.shape = ', JK_lo_ks_Co.shape)
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

#print('JK_00 = ', JK_00[0])
#plt.imshow(np.abs(JK_00[0]))
#plt.show()
#exit()

############################################################
#               contact's Green's function
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

    V_L[nao_Co:nao_Co+nao_ppl,:] = H01.T.conj()
    V_R[-nao_ppl:,:] = H01

    Sigma_L = V_L @ g00 @ V_L.T.conj()
    Sigma_R = V_R @ g00 @ V_R.T.conj()

    # contact block of the Green's function
    # TODO multiple spin
    G_C = np.zeros((spin, nao_contact, nao_contact), dtype=complex)
    for s in range(spin):
        G_C[s,:,:] = np.linalg.inv( z*np.eye(nao_contact) - hcore_lo_contact[s] - JK_lo_ks_contact[s] \
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

# -1/pi*imag(Sigma(e+i*delta))
# (spin, n_hyb, n_hyb)
def Gamma(e):
    # broadening
    delta = 0.01

    z = e + 1j*delta
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
#   embedding model Co LDoS solved by mean-field
############################################################

mu = -0.145

#------------ bath discretization ------------
# evenly spaced grid
wlg = -0.6
whg = 0.6
nbe = 80
nbath_per_ene = 6

#grid = np.linspace(wlg,whg,nbe+1)
log_disc_base = 1.3

# distance to mu
wl0 = mu - wlg
wh0 = whg - mu

dif = round(np.log(abs(wh0/wl0))/np.log(log_disc_base)) // 2

# number of energies above/below the Fermi level
nl = nbe//2 - dif
nh = nbe - nl
grid = np.concatenate((gen_log_grid(mu, wlg, log_disc_base, nl), [mu], \
        gen_log_grid(mu, whg, log_disc_base, nh)))

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
#eri_imp = np.zeros([spin*(spin+1)//2, nemb, nemb, nemb, nemb])
#eri_imp[:,:nao_Co,:nao_Co,:nao_Co,:nao_Co] = eri_lo_Co

dm0 = np.zeros((spin,nemb,nemb))
dm0[:,:nao_Co,:nao_Co] = DM_lo_Co.copy()

mol = gto.M()
mol.verbose = 4
#mol.incore_anyway = True
mol.build()


#mf = RKS2(mol, mu)
mf = RHF2(mol, mu)
mf.get_hcore = lambda *args: hemb[0]
mf.get_ovlp = lambda *args: np.eye(nemb)
mf.mo_energy = np.zeros([nemb])
#mf._eri = ao2mo.restore(8, eri_imp[0], nemb)
mf.max_cycle = 150
mf.conv_tol = 1e-12
mf.diis_space = 15


# sanity check
# mf.get_veff returns intra-(Co val+virt) veff
# Cu-to-Co veff is incorporated in mf.hcore (which is Himp_Co) and is stored as mf._veff_ref
veff_test = mf.get_veff(mol=mol,dm=dm0[0])
fock_Co_test = veff_test[:22,:22] + mf.get_hcore()[:22,:22]
fock_Co_ref = hcore_lo_Co + JK_lo_ks_Co
fock_Co_ref = fock_Co_ref[0]
print('sanity check: fock_Co diff = ', np.linalg.norm(fock_Co_ref-fock_Co_test))


veff_test = mf.get_veff(mol=mol,dm=dm0[0])
JK_lo_tot = np.dot(np.dot(C_ao_lo_tot.T.conj(), JK_ao), C_ao_lo_tot)
print('diff = ', np.linalg.norm(veff_test[0:22,0:22]+mf._veff_ref-JK_lo_tot[9:31,9:31]))

veff_0 = mf.get_veff(mol=mol,dm=dm0[0])
fock_0 = veff_0 + mf.get_hcore()
print('fock_0.shape = ', fock_0.shape)

e,v = eiggen(fock_0, np.eye(nemb))
#print('e = ', e)
v_occ = v[:, e<mu]
dm_new = 2. * v_occ @ v_occ.T
print('Co nelec = ', np.trace(dm_new[:22,:22]))
print('dm diff = ', np.linalg.norm(dm_new[:22,:22]-dm0[0,:22,:22]) )

# one Fock iteration for embedding Hamiltonian
def fock2fock(fock_in):
    e,v = eiggen(fock_in, np.eye(nemb))
    v_occ = v[:, e<mu]
    dm = 2. * v_occ @ v_occ.T
    fock_out = mf.get_hcore() + mf.get_veff(mol=mol, dm=dm)
    return fock_out

# for Commutator-DIIS
smearing_sigma = 0.1
chem_pot = -0.16
def fock2fockcomm(fock_in):
    e,v = eiggen(fock_in, np.eye(nemb))
    
    # use fermi broadening to assist convergence
    #v_occ = v[:, e<mu]
    #dm = 2. * v_occ @ v_occ.T

    chem_pot = -0.16

    occ = 2./( 1. + np.exp((e-chem_pot)/smearing_sigma) )
    dm = (v*occ) @ v.T

    fock_out = mf.get_hcore() + mf.get_veff(mol=mol, dm=dm)
    comm = fock_out @ dm - dm @ fock_out
    return fock_out, comm


flag, fock = diis(fock2fockcomm, fock_0, max_iter=300)

if flag == 0:
    print('fock.shape = ', fock.shape)
else:
    exit()

e,v = eiggen(fock, np.eye(nemb))

occ = 2./( 1. + np.exp((e-chem_pot)/smearing_sigma) )
rdm1 = (v*occ) @ v.T

#exit()


#orb_Co_occ = np.sum((v*v)[:22,:], axis=0) * 2
#orb_Co_occ_cumsum = np.cumsum(orb_Co_occ)
#print('orb Co occ = ', orb_Co_occ)
#print('cumsum = ', orb_Co_occ_cumsum)
#exit()




#mf.kernel(dm0=dm0[0])

#############################################################

#rdm1 = mf.make_rdm1()

print('spin = ', spin)
print('trace(rdm1[Co val+virt])', np.trace(rdm1[0:nao_Co,0:nao_Co]))
print('trace(rdm1[Co val])', np.trace(rdm1[0:nval_Co,0:nval_Co]))

#------------ compute & plot Co LDoS ------------
#fock = mf.get_fock(dm=rdm1)

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

fh = h5py.File('imp_rks_ldos.h5', 'w')
fh['freqs'] = freqs
fh['A'] = A[0,:,:]
fh['ldos'] = ldos[0,:,:]
fh.close()
#for i in range(nval_Co):
#    plt.plot(freqs,ldos[0,:,i], linestyle=':', color='C'+str(i))
#    plt.plot(freqs,A[0,:,i], linestyle='-', color='C'+str(i))
#
#fig = plt.gcf()
#fig.set_size_inches(6,4)
#
#plt.xlim((wld,whd))
#plt.show()

exit()
