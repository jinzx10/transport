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
from utils.emb_helper import *

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

# with lead or not
with_lead = False

# center of bath discretization
disc_center = 0.1829

############################################################
#                   gate voltage
############################################################
gate = 0.000
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


############################################################
#               impurity Hamiltonian
############################################################
hcore_lo_imp = 1./nkpts * np.sum(hcore_lo_imp, axis=1)
JK_lo_ks_imp = 1./nkpts * np.sum(JK_lo_ks_imp, axis=1)
DM_lo_imp = 1./nkpts * np.sum(DM_lo_imp, axis=1)

#------------ build contact object ------------
print('ready to build mf contact')
cell = chkfile.load_cell(cell_fname)
 
gdf = pbcdf.GDF(cell)
gdf._cderi = gdf_fname

mf_contact = pbcscf.RKS(cell).density_fit()
mf_contact.xc = 'pbe0'
mf_contact.with_df = gdf
print('mf contact built!')

#------------ compute/load JK_00 ------------
# JK_00 stands for intra-(imp val+virt) two-body mean field potential 
# for DFT it is computed by taking the difference between veff with/without imp val+virt DM
JK_00_fname = 'JK_00_' + imp_atom + '.h5'
if os.path.isfile(JK_00_fname):
    fh = h5py.File(JK_00_fname, 'r')
    JK_00 = np.asarray(fh['JK_00'])
    veff_ref = np.asarray(fh['veff_ref'])
    fh.close()
else:
    # total DM in LO basis, imp val+virt block removed
    DM_lo_tmp = np.copy(DM_lo_tot)
    DM_lo_tmp[9:31,9:31] = 0
    
    # transform to AO basis
    DM_ao_tmp = C_ao_lo_tot @ DM_lo_tmp @ C_ao_lo_tot.T.conj()
    JK_ao_tmp = np.asarray(mf_contact.get_veff(dm=DM_ao_tmp))
    
    # transform the potential to LO basis
    JK_lo_tmp = C_ao_lo_tot.T.conj() @ JK_ao_tmp @ C_ao_lo_tot

    # take the different for the imp val+virt block
    JK_lo_tmp = JK_lo_tmp[np.newaxis,...]
    JK_00 = JK_lo_ks_imp - JK_lo_tmp[:,9:31,9:31]

    veff_ref = np.copy(JK_lo_tmp[0,9:31,9:31])

    fh = h5py.File(JK_00_fname, 'w')
    fh['veff_ref'] = veff_ref
    fh['JK_00'] = JK_00
    fh.close()

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

    # surface Green's function
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
        if with_lead:
            G_C[s,:,:] = np.linalg.inv( z*np.eye(nao_contact) - hcore_lo_contact[s] - JK_lo_ks_contact[s] \
                    - Sigma_L - Sigma_R )
        else:
            G_C[s,:,:] = np.linalg.inv( z*np.eye(nao_contact) - hcore_lo_contact[s] - JK_lo_ks_contact[s] )
    return G_C

comm.Barrier()

############################################################
#               hybridization Gamma
############################################################
# number of orbitals that couple to the bath, usually nval_imp or nao_imp
n_hyb = nao_imp

# broadening for computing hybridization Gamma from self energy
delta= 0.002
# -1/pi*imag(Sigma(e+i*delta))
# (spin, n_hyb, n_hyb)
def Gamma(e):
    z = e + 1j*delta
    G_C = contact_Greens_function(z)
    
    Sigma_imp = np.zeros((spin, n_hyb, n_hyb),dtype=complex)
    for s in range(spin):
        Sigma_imp[s,:,:] = z*np.eye(n_hyb) - Hemb_imp[s,:n_hyb,:n_hyb] \
                - np.linalg.inv(G_C[s,:n_hyb,:n_hyb])

    return -1./np.pi*Sigma_imp.imag


############################################################
#               bath discretization
############################################################
#------------ log discretization ------------
wlg = -0.4
whg = 0.6
nbe = 50 # total number of bath energies
nbath_per_ene = 3

grid_type = 'custom1'
log_disc_base = 2.0
wlog = 0.01

grid = gen_grid(nbe, wlg, whg, disc_center, grid_type=grid_type, log_disc_base=log_disc_base, wlog=wlog)

print('grid points:', grid)
print('number of grids:', len(grid))

nbath = nbe * nbath_per_ene
nemb = nbath + nao_imp

hemb = np.zeros((spin, nemb, nemb))

if rank == 0:
    print('hemb.shape = ', hemb.shape)
    print('bath discretization starts')

# one body part
for s in range(spin):
    Gamma_s = lambda e: Gamma(e)[s]
    e,v = direct_disc_hyb(Gamma_s, grid, nint=30, nbath_per_ene=nbath_per_ene)
    
    hemb[s,:,:] = emb_ham(Hemb_imp[s,:,:], e, v)

if rank == 0:
    print('bath discretization finished')
    print('bath energies = ', e)

############################################################
#               user-defined mf object
############################################################
class RHF2(scf.hf.RHF):

    __doc__ = scf.hf.RHF.__doc__

    def __init__(self, mol, mu):

        scf.hf.SCF.__init__(self, mol)

        self._mf = mf_contact
        self._veff_ref = veff_ref
        self._C_ao_lo_tot = C_ao_lo_tot
        self._DM_lo_tot = DM_lo_tot
        self.mu = mu


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
mol.incore_anyway = True
mol.build()

mf = RHF2(mol, mu)
mf.get_hcore = lambda *args: hemb[0]
mf.get_ovlp = lambda *args: np.eye(nemb)
mf.mo_energy = np.zeros([nemb])
mf.mo_occ = np.zeros([nemb])

# dm initial guess
iocc = np.array(hemb[0].diagonal() < mu, dtype=float)
dm0 = np.diag(iocc) * 2.0
dm0[:nao_imp,:nao_imp] = DM_lo_imp.copy()

#------------ sanity check ------------
# mf.get_veff returns intra-(imp val+virt) veff
# Cu-to-imp veff is incorporated in mf.hcore (which is Himp_imp) and is stored as mf._veff_ref
veff_test = mf.get_veff(mol=mol,dm=dm0)
fock_imp_test = veff_test[:22,:22] + mf.get_hcore()[:22,:22]
fock_imp_ref = hcore_lo_imp + JK_lo_ks_imp
fock_imp_ref = fock_imp_ref[0]
print('sanity check: fock_imp diff = ', np.linalg.norm(fock_imp_ref-fock_imp_test))

JK_lo_tot = np.dot(np.dot(C_ao_lo_tot.T.conj(), JK_ao[0]), C_ao_lo_tot)
print('sanity check: JK_lo diff = ', np.linalg.norm(veff_test[0:22,0:22]+mf._veff_ref-JK_lo_tot[9:31,9:31]))
#------------ sanity check end ------------

mf.kernel(dm0)
rdm1 = mf.make_rdm1()
fock = hemb[0] + mf.get_veff(mol=mol, dm=rdm1)

print('trace(rdm1[imp val+virt])', np.trace(rdm1[0:nao_imp,0:nao_imp]))
print('trace(rdm1[imp val])', np.trace(rdm1[0:nval_imp,0:nval_imp]))
print('rdm1 diff = ', np.linalg.norm(rdm1[0:nao_imp, 0:nao_imp]-DM_lo_imp[0]))




## Fock iteration for embedding Hamiltonian
## for commutator-DIIS
#smearing_sigma = 0
#def fock2fockcomm(fock_in):
#    e,v = eiggen(fock_in, np.eye(nemb))
#    
#    # use fermi broadening to assist convergence
#    #v_occ = v[:, e<mu]
#    #dm = 2. * v_occ @ v_occ.T
#
#
#    if abs(smearing_sigma) > 1e-10:
#        occ = 2./( 1. + np.exp((e-mu)/smearing_sigma) )
#    else:
#        occ = np.zeros_like(e)
#        occ[e<=mu] = 2.0
#
#    dm = (v*occ) @ v.T
#
#    fock_out = mf.get_hcore() + mf.get_veff(mol=mol, dm=dm)
#    comm = fock_out @ dm - dm @ fock_out
#    return fock_out, comm
#
#
#fock_0 = veff_test + mf.get_hcore()
#fock, flag = diis(fock2fockcomm, fock_0, max_iter=300)
#
#if flag == 0:
#    print('fock.shape = ', fock.shape)
#else:
#    exit()
#
#e,v = eiggen(fock, np.eye(nemb))
#
#if abs(smearing_sigma) > 1e-10:
#    occ = 2./( 1. + np.exp((e-mu)/smearing_sigma) )
#else:
#    occ = np.zeros_like(e)
#    occ[e<=mu] = 2.0
#rdm1 = (v*occ) @ v.T


#############################################################


#------------ compute & plot imp LDoS ------------

if spin == 1:
    fock = fock[np.newaxis,...]

wld = -0.4
whd = 0.8
nwd = 400
eta = 0.03
freqs = np.linspace(wld,whd,nwd)

# imp ldos from embedding model
A = np.zeros((spin,nwd,nval_imp))
for s in range(spin):
    for iw in range(nwd):
        z = freqs[iw] + 1j*eta

        gf_mo = np.diag(1./(z-mf.mo_energy))
        gf_ao = mf.mo_coeff @ gf_mo @ mf.mo_coeff.T

        A[s,iw,:] = -1./np.pi*np.diag(gf_ao[0:nval_imp,0:nval_imp]).imag

# mean-field LDoS from contact Green's function
ldos_ref = np.zeros((spin,nwd,nval_imp))
for iw in range(nwd):
    z = freqs[iw] + 1j*eta
    GC = contact_Greens_function(z)
    for s in range(spin):
        ldos_ref[s,iw,:] = -1./np.pi*np.diag(GC[s,:nval_imp, :nval_imp]).imag

fh = h5py.File('imp_rks_ldos_nbe%i_nbpe%i_mu%5.3f_gate%5.3f.h5'%(nbe, nbath_per_ene, mu, gate), 'w')
fh['freqs'] = freqs
fh['A'] = A[0,:,:]
fh['ldos_ref'] = ldos_ref[0,:,:]
fh['nbe'] = nbe
fh['nbath_per_ene'] = nbath_per_ene
fh.close()





