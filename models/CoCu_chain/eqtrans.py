import numpy as np
import h5py, time
import matplotlib.pyplot as plt
from mpi4py import MPI

from fcdmft.solver import scf_mu as scf

from surface_green import *


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

############################################################
#           read contact's mean-field data
############################################################
contact_dir = 'Co_svp_Cu_svp_bracket_pbe_v2/'

label = 'CoCu_09_111'

#------------ read core Hamiltonian and DFT veff ------------
fname = contact_dir + '/hcore_JK_lo_dft_' + label + '.h5'
fh = h5py.File(fname, 'r')

# Co atom block only
hcore_Co = np.asarray(fh['hcore_lo_Co'])
JK_dft_Co = np.asarray(fh['JK_lo_Co'])

# entire center region, Co + 9 Cu atoms
hcore = np.asarray(fh['hcore_lo'])
JK_dft = np.asarray(fh['JK_lo'])
fh.close()

#------------ read HF veff ------------
fname = contact_dir + '/JK_lo_hf_' + label + '.h5'
fh = h5py.File(fname, 'r')

# Co atom block only
JK_hf_Co = np.asarray(fh['JK_lo_Co'])

# entire center region, Co + 9 Cu atoms
JK_hf = np.asarray(fh['JK_lo'])
fh.close()


#------------ read density matrix ------------
fname = contact_dir + '/DM_lo_' + label + '.h5'
fh = h5py.File(fname, 'r')
DM_Co = np.asarray(fh['DM_lo_Co'])
fh.close()

#------------ read ERI ------------
fname = contact_dir + '/eri_lo_' + label + '.h5'
fh = h5py.File(fname, 'r')
eri_Co = np.asarray(fh['eri_lo_Co'])
fh.close()

nao = hcore.shape[1]
nao_Co = hcore_Co.shape[1]

nval_Co = 6

############################################################
#               impurity block
############################################################
imag_tol = 1e-8

nkpts = hcore_Co.shape[0]
hcore_Co = 1./nkpts * np.sum(hcore_Co, axis=0)
JK_hf_Co = 1./nkpts * np.sum(JK_hf_Co, axis=0)

hcore = 1./nkpts * np.sum(hcore, axis=0)
JK_hf = 1./nkpts * np.sum(JK_hf, axis=0)
JK_dft = 1./nkpts * np.sum(JK_dft, axis=0)

assert( np.max(np.abs(DM_Co.imag)) < imag_tol )
assert( np.max(np.abs(hcore_Co.imag)) < imag_tol )
assert( np.max(np.abs(JK_hf_Co.imag)) < imag_tol )
assert( np.max(np.abs(hcore.imag)) < imag_tol )
assert( np.max(np.abs(JK_hf.imag)) < imag_tol )
assert( np.max(np.abs(JK_dft.imag)) < imag_tol )


DM_Co = DM_Co.real
hcore_Co = hcore_Co.real
JK_hf_Co = JK_hf_Co.real
hcore = hcore.real
JK_hf = JK_hf.real
JK_dft = JK_dft.real

JK_00 = scf._get_veff(DM_Co, eri_Co)
Himp_Co = hcore_Co + JK_hf_Co - JK_00[0]

#plt.imshow(np.abs(Himp_Co), extent=[0,1,0,1])
#plt.show()
#exit()

############################################################
#               read lead's mean-field data
############################################################

#------------ get H00 and H01 for surface Green's function ------------
bath_dir = 'Cu_svp_bracket_pbe_v2/'

label = 'Cu_16_111'
fname = bath_dir + '/ks_lo_' + label + '.h5'
fh = h5py.File(fname, 'r')
F_bath = np.asarray(fh['F_lo'])

# number of orbitals per atom
nao_per_Cu = 15

# number of atoms per block
nat_per_blk = 4

nao_per_blk = nao_per_Cu * nat_per_blk

H00 = F_bath[0, :nao_per_blk, :nao_per_blk]
H01 = F_bath[0, :nao_per_blk, nao_per_blk:2*nao_per_blk]

############################################################
#           display matrix elements as image 
############################################################
#plt.imshow(np.log(np.abs(F_bath[0])), extent=[0,1,0,1])
#plt.show()

############################################################
#           hybridization function
############################################################
def Gamma(e):
    # -1/pi*imag(Sigma(e+i*delta))
    delta = 1e-2
    z = e + 1j*delta
    g00 = Umerski1997(z, H00, H01)

    V_L = np.zeros((nao, nao_per_blk), dtype=complex)
    V_R = np.zeros((nao, nao_per_blk), dtype=complex)

    V_L[:nao_per_blk,:] = H01.T.conj()
    V_R[-nao_per_blk:,:] = H01

    Sigma_L = V_L @ g00 @ V_L.T.conj()
    Sigma_R = V_R @ g00 @ V_R.T.conj()


    # contact block of the Green's function
    G_C = np.linalg.inv( z*np.eye(nao) - hcore - JK_dft - Sigma_L - Sigma_R )
    #G_C = np.linalg.inv(z*np.eye(nao)-hcore-JK_hf)

    # Co block of the Green's function
    G_imp = G_C[nao_per_Cu*4:nao_per_Cu*4+nao_Co, nao_per_Cu*4:nao_per_Cu*4+nao_Co]

    # here the impurity merely contains the 6 Co valence orbitals
    # G_imp = inv(z-H_imp-Sigma_imp)
    Sigma_imp = z*np.eye(nval_Co) - Himp_Co[:nval_Co,:nval_Co] - np.linalg.inv(G_imp[:nval_Co,:nval_Co])

    return -1./np.pi*Sigma_imp.imag


############################################################
#           test the range of hybridization
############################################################
ec = 0
w = 0.5
ne = 5000
e = np.linspace(ec-w, ec+w, ne)
hyb = np.zeros((ne, nval_Co, nval_Co))
for ie in range(ne):
    hyb[ie,:,:] = Gamma(e[ie])

plt.plot(e,hyb[:,0,0])
plt.show()


############################################################
#               bath discretization
############################################################
#------------ generate grid ------------







#------------ ------------
############################################################
#
############################################################
#------------ ------------
