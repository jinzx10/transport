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

############################################################
#               impurity block
############################################################
imag_tol = 1e-8

nkpts = hcore_Co.shape[0]
hcore_Co = 1./nkpts * np.sum(hcore_Co, axis=0)
JK_Co = 1./nkpts * np.sum(JK_hf_Co, axis=0)

assert( np.max(np.abs(DM_Co.imag)) < imag_tol )
assert( np.max(np.abs(hcore_Co.imag)) < imag_tol )
assert( np.max(np.abs(JK_Co.imag)) < imag_tol )
DM_Co = DM_Co.real
hcore_Co = hcore_Co.real
JK_Co = JK_Co.real

JK_00 = scf._get_veff(DM_Co, eri_Co)
Himp_Co = hcore_Co + JK_Co - JK_00[0]

#plt.imshow(np.abs(hcore[0]), extent=[0,1,0,1])
#plt.show()

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
nlo_per_atm = 15

# number of atoms per block
nat_per_blk = 4

nlo_per_blk = nlo_per_atm * nat_per_blk

H00 = F_bath[0, :nlo_per_blk, :nlo_per_blk]
H01 = F_bath[0, :nlo_per_blk, nlo_per_blk:2*nlo_per_blk]


############################################################
#               show some matrix
############################################################
#plt.imshow(np.log(np.abs(F_bath[0])), extent=[0,1,0,1])
#plt.show()

############################################################
#           test the range of hybridization
############################################################
############################################################
#           test the range of hybridization
############################################################
delta = 1e-3
ec = 0
w = 1
ne = 1000
e = np.linspace(ec-w, ec+w, ne)

############################################################
#               bath discretization
############################################################
#------------ generate grid ------------







#------------ ------------
############################################################
#
############################################################
#------------ ------------
