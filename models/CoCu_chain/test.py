import numpy as np
import h5py, time
import matplotlib.pyplot as plt
from mpi4py import MPI

from fcdmft.solver import scf_mu as scf

from surface_green import *
from bath_disc import *

from matplotlib import colors


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

############################################################
#           read contact's mean-field data
############################################################

#****** the v2 version has Co IAO/PAO in the middle ******

# others put Co at the beginning
contact_dir = 'Co_svp_Cu_svp_bracket_pbe2/'
#contact_dir = 'Co_svp_Cu_svp_bracket_pbe_v2/'

label = 'CoCu_09_111'

#------------ read core Hamiltonian and DFT veff ------------
fname = contact_dir + '/hcore_JK_lo_dft_' + label + '.h5'
fh = h5py.File(fname, 'r')

# Co atom block only
hcore_Co = np.asarray(fh['hcore_lo_ncCo'])
JK_dft_Co = np.asarray(fh['JK_lo_ncCo'])

# entire center region, Co + 9 Cu atoms
hcore = np.asarray(fh['hcore_lo_nc'])
JK_dft = np.asarray(fh['JK_lo_nc'])
fh.close()


#------------ read HF veff ------------
fname = contact_dir + '/JK_lo_hf_' + label + '.h5'
fh = h5py.File(fname, 'r')

# Co atom block only
JK_hf_Co = np.asarray(fh['JK_lo_ncCo'])

# entire center region, Co + 9 Cu atoms
JK_hf = np.asarray(fh['JK_lo_nc'])
fh.close()


#------------ read density matrix ------------
# the Co's density matrix is used to 
fname = contact_dir + '/DM_lo_' + label + '.h5'
fh = h5py.File(fname, 'r')
DM_Co = np.asarray(fh['DM_lo_ncCo'])
fh.close()

#------------ read ERI ------------
fname = contact_dir + '/eri_lo_' + label + '.h5'
fh = h5py.File(fname, 'r')
eri_Co = np.asarray(fh['eri_lo_ncCo'])
fh.close()

nao = hcore.shape[1]
nao_Co = hcore_Co.shape[1]

#<===============================
# apply gate voltage
gate = -0.062
hcore_Co[0] = hcore_Co[0] + gate*np.eye(nao_Co)
hcore[0,:nao_Co,:nao_Co] = hcore[0,:nao_Co,:nao_Co] + gate*np.eye(hcore_Co.shape[1])
#===============================>
print(hcore_Co.shape)
print(hcore.shape)

exit()

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

#H_lo = hcore + JK_hf
#plt.imshow(np.log(np.abs(H_lo)**(-1)), extent=[0,1,0,1])
#plt.imshow(np.log(np.abs(H_lo)**(-1)), extent=[0,1,0,1], norm=colors.LogNorm())
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

offset = nao_per_blk
############################################################
#           display matrix elements as image 
############################################################
#plt.imshow(np.log(np.abs(F_bath[0])**(-1)), extent=[0,1,0,1], norm=colors.LogNorm())
#plt.imshow(np.log(np.abs(F_bath[0])), extent=[0,1,0,1], norm=colors.LogNorm())
#plt.show()

############################################################
#           hybridization function
############################################################

# the impurity is chosen to be the 6 valence orbitals of Co
nimp = 6

# contact's GF
def contact_Green_function(z):
    g00 = Umerski1997(z, H00, H01)

    V_L = np.zeros((nao, nao_per_blk), dtype=complex)
    V_R = np.zeros((nao, nao_per_blk), dtype=complex)

    V_L[:nao_per_blk,:] = H01.T.conj()
    V_R[-nao_per_blk:,:] = H01

    Sigma_L = V_L @ g00 @ V_L.T.conj()
    Sigma_R = V_R @ g00 @ V_R.T.conj()

    # contact block of the Green's function
    return np.linalg.inv( z*np.eye(nao) - hcore - JK_dft - Sigma_L - Sigma_R )


# -1/pi*imag(Sigma(e+i*delta))
delta = 0.01
def Gamma(e):

    # contact block of the Green's function
    G_C = contact_Green_function(e+1j*delta)
    #G_C = np.linalg.inv(z*np.eye(nao)-hcore-JK_hf)

    # Co block of the Green's function
    G_imp = G_C[nao_per_Cu*4:nao_per_Cu*4+nao_Co, nao_per_Cu*4:nao_per_Cu*4+nao_Co]

    # here the impurity merely contains the 6 Co valence orbitals
    # G_imp = inv(z-H_imp-Sigma_imp)
    Sigma_imp = z*np.eye(nimp) - Himp_Co[:nimp,:nimp] - np.linalg.inv(G_imp[:nimp,:nimp])

    return -1./np.pi*Sigma_imp.imag

'''
############################################################
#           test the range of hybridization
############################################################
ec = 0
w = 0.35
nz = 2000
z = np.linspace(ec-w, ec+w, nz)
hyb = np.zeros((nz, nimp, nimp))
for iz in range(nz):
    hyb[iz,:,:] = Gamma(z[iz])

#for i in range(6):
#    plt.plot(z,hyb[:,i,i])
plt.plot(z,hyb[:,0,0])
#plt.show()
#exit()

############################################################
#               bath discretization
############################################################
#------------ generate grid ------------
# one-side log grid
# generate a log grid between w0 and w (converges to w0)
# return w0 + (w-w0)*l**(-i) where i ranges from 0 to num-1

def gen_log_grid(w0, w, l, num):
    grid = w0 + (w-w0) * l**(-np.arange(num,dtype=float))
    if w > w0:
        return grid[::-1]
    else:
        return grid

wl0 = -0.25
wh0 = 0.4

nbe = 50

mu = -0.075

# absolute band range
wl, wh = wl0+mu, wh0+mu

base  = 1.3

dif = round(np.log(abs(wh0/wl0))/np.log(base)) // 2

# number of energies above/below the Fermi level
nl = nbe//2 - dif
nh = nbe - nl

grid = np.concatenate((gen_log_grid(mu, wl, base, nl), [mu], gen_log_grid(mu, wh, base, nh)))
#grid = np.linspace(wl,wh,nbe+1)

print('grid = ', grid)

nbath_per_ene = 1
e,v = direct_disc_hyb(Gamma, grid, nint=3, nbath_per_ene=nbath_per_ene)

print('e = ', e)
print('e.shape = ', e.shape)
print('v.shape = ', v.shape)



############################################################
#           check the rebuilt Gamma
############################################################
gauss = lambda x,mu,sigma: 1.0/sigma/np.sqrt(2*np.pi)*np.exp(-0.5*((x-mu)/sigma)**2)
eta=0.005

Gamma_rebuilt = np.zeros((nz,nimp,nimp))
for iz in range(nz):
    for ib in range(len(e)):
        for ie in range(nbath_per_ene):
            Gamma_rebuilt[iz,:,:] += np.outer(v[ib,:,ie],v[ib,:,ie].conj()) * gauss(z[iz],e[ib],eta)

plt.plot(z, Gamma_rebuilt[:,0,0])

for ie in range(nbe):
    plt.axvline(x=e[ie], linestyle=':', color='blue', lw=0.2)

plt.axvline(x=mu, linestyle=':', color='red', lw=2)
'''

#------------ ------------
############################################################
#           check the LDoS
############################################################
#------------ ------------
ne = 100
erange = np.linspace(-0.25,-0.075,ne)

ldos_mf = np.zeros((ne,6))

for ie in range(ne):
    z = erange[ie] + 1j*0.001
    A = -1./np.pi*contact_Green_function(z).imag
    ldos_mf[ie,:] = np.diag(A)[offset:offset+6]
    
ldos_mf = np.sum(ldos_mf, axis=1)

plt.plot(erange, ldos_mf)

plt.show()


############################################################
# 
############################################################
#------------ ------------
