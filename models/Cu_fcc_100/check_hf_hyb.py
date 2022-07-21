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

imp_atom = 'Co'

gate = 0.000
gate_label = 'gate%5.3f'%(gate)

mu = 0.04
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

method_label = 'rhf'

data_fname = contact_dir + '/data_contact_' + cell_label + '_' \
        + method_label + '_' + gate_label + '.h5'

#------------ read core Hamiltonian and HF veff (built with DFT DM)  ------------
fh = h5py.File(data_fname, 'r')

# imp atom block only
hcore_lo_imp = np.asarray(fh['hcore_lo_imp'])
JK_lo_hf_imp = np.asarray(fh['JK_lo_imp'])

# entire center region, imp + some Cu atoms
hcore_lo_contact = np.asarray(fh['hcore_lo'])
JK_lo_hf_contact = np.asarray(fh['JK_lo'])

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
    print('hcore_lo_imp.shape = ', hcore_lo_imp.shape)
    print('DM_lo_imp.shape = ', DM_lo_imp.shape)
    print('eri_lo_imp.shape = ', eri_lo_imp.shape)
    print('')

    print('hcore_lo_contact.dtype = ', hcore_lo_contact.dtype)
    print('hcore_lo_imp.dtype = ', hcore_lo_imp.dtype)
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

    Sigma_L.fill(0.0)
    Sigma_R.fill(0.0)

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
delta= 0.005
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

Gamma_s = lambda e: Gamma(e)[0]

############################################################
#               exact Gamma
############################################################
wl = -1.6
wh = 3.0
nw = 500
plot_grid = np.linspace(wl, wh, nw)
hyb = np.zeros((nw, n_hyb, n_hyb))
for iw in range(nw):
    hyb[iw,:,:] = Gamma_s(plot_grid[iw])

#for i in [1,4]:
##for i in range(6):
#    plt.plot(plot_grid, hyb[:,i,i], linestyle='-', label='exact'+str(i), color='C'+str(i))
#
#plt.xlim([wl,wh])
##plt.ylim([-0.01,0.3])
#
#fig = plt.gcf()
#fig.set_size_inches(6,4)
#
#plt.legend()
#plt.show()
#
#exit()

############################################################
#       rebuild Gamma by bath discretization
############################################################
#------------ bath discretization ------------
# evenly spaced grid
wlg = -1.0
whg = 2.0
nbe = 100 # total number of bath energies
nbath_per_ene = 9

grid_type = 'linear'
log_disc_base = 2.0
wlog = 0.01

grid = gen_grid(nbe, wlg, whg, mu, grid_type, log_disc_base, wlog=wlog)

# only test one spin component
e,v = direct_disc_hyb(Gamma_s, grid, nint=5, nbath_per_ene=nbath_per_ene)
comm.Barrier()
if rank == 0:
    print('bath discretization finished')
    print('e.shape = ', e.shape)
    print('v.shape = ', v.shape)
    print('bath energies = ', e)
    print('')

# rebuild Gamma
#bdfun = lambda x,x0,b: 1.0/np.pi*b/((x-x0)**2+b**2)
bdfun = lambda x,mu,sigma: 1.0/sigma/np.sqrt(2*np.pi)*np.exp(-0.5*((x-mu)/sigma)**2)
eta=0.005

Gamma_rebuilt = np.zeros((nw,n_hyb,n_hyb))
for iw in range(nw):
    for ib in range(len(e)):
        for ie in range(nbath_per_ene):
            Gamma_rebuilt[iw,:,:] += np.outer(v[ib,:,ie],v[ib,:,ie].conj()) * bdfun(plot_grid[iw],e[ib],eta)

if rank == 0:
    for i in [1,2,3,4,5]:
    #for i in range(nval_imp):
        plt.plot(plot_grid, hyb[:,i,i], linestyle=':', label='exact'+str(i), color='C'+str(i))
        plt.plot(plot_grid, Gamma_rebuilt[:,i,i], linestyle='-', label='rebuilt'+str(i), color='C'+str(i))
    
    plt.xlim([wl,wh])
    #plt.ylim([-0.01,0.2])
    
    fig = plt.gcf()
    fig.set_size_inches(6,4)
    
    plt.legend()
    plt.show()


