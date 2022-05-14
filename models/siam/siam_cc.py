import numpy as np
import matplotlib.pyplot as plt
import h5py, time, sys

from pyscf import gto, ao2mo, cc, scf, dft
from fcdmft.solver import scf_mu

from utils.bath_disc import *

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

def gen_siam_mf(on_site, Gamma, U, mu, grid):


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
    #mol.nelectron=nemb//2*2

    mol.build()
    
    emb_mf = scf_mu.RHF(mol, mu)
    #emb_mf = scf.RHF(mol)
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
on_site = -2.0
mu = 0.5

#---------------- generate grid for bath --------------------

# log grid
# distance to mu
#wl0 = mu - wl
#wh0 = wh - mu
#
## number of energies above/below the Fermi level
#dif = round(np.log(abs(wh0/wl0))/np.log(log_disc_base)) // 2
#nl = nbath//2 - dif
#nh = nbath - nl

#grid = np.concatenate((gen_log_grid(mu, wl, log_disc_base, nl), [mu], \
#        gen_log_grid(mu, wh, log_disc_base, nh)))

# linear grid
wl = -5.0
wh = 5.0
nbath = 20
# 2/3 coarse bath states
# 1/3 fine bath states near mu
nbath_coarse = round(nbath/3) # one side
nbath_fine = nbath - nbath_coarse*2

# energy windows for fine bath states (2*w total)
w = 0.05

grid = np.concatenate( (\
        np.linspace(wl,mu-w-0.01,nbath_coarse), \
        np.linspace(-w,w,nbath_fine), \
        np.linspace(w+0.01,wh,nbath_coarse) ) )
    
print('grid = ', grid)

#########################################################

siam_mf = gen_siam_mf(on_site, hyb_const, U, mu, grid)

siam_mf.kernel()

fock = siam_mf.get_fock()

wl_mf = -6
wh_mf = 6
nw_mf = 100
delta = 0.2
freqs_mf = np.linspace(wl_mf, wh_mf, nw_mf)

# ldos (mean-field level)
ldos_mf = np.zeros(nw_mf)
for iw in range(nw_mf):
    z = freqs_mf[iw] + 1j*delta
    gf = np.linalg.inv(z*np.eye(fock.shape[0]) - fock[:,:])
    ldos_mf[iw] = -1./np.pi*gf[0,0].imag

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

#---------------- grid to compute LDoS ------------------
wl = -6
wh = 6

delta_cc1 = 0.1
nw1 = 2*round((mu-0.5-wl)/delta_cc1)
freqs_cc1 = np.linspace(-6,mu-0.5,nw1)
delta_cc1 = delta_cc1*np.ones(nw1)

delta_cc2 = 0.05
nw2 = 2*round(0.45/delta_cc2)
freqs_cc2 = np.linspace(mu-0.45,mu-0.05,nw2)
delta_cc2 = delta_cc2*np.ones(nw2)

delta_cc3 = 0.01
nw3 = 2*round(0.1/delta_cc3)
freqs_cc3 = np.linspace(mu-0.05,mu+0.05,nw3)
delta_cc3 = delta_cc3*np.ones(nw3)

delta_cc4 = 0.05
nw4 = 2*round(0.45/delta_cc4)
freqs_cc4 = np.linspace(mu+0.05,mu+0.5,nw4)
delta_cc4 = delta_cc4*np.ones(nw4)

delta_cc5 = 0.1
nw5 = 2*round((6-mu-0.5)/delta_cc5)
freqs_cc5 = np.linspace(mu+0.5,6,nw5)
delta_cc5 = delta_cc5*np.ones(nw5)

freqs_cc = np.concatenate((freqs_cc1,freqs_cc2,freqs_cc3,freqs_cc4,freqs_cc5))
delta_cc = np.concatenate((delta_cc1,delta_cc2,delta_cc3,delta_cc4,delta_cc5))

idx = np.argsort(freqs_cc)
freqs_cc = freqs_cc[idx]
delta_cc = delta_cc[idx]

nw = len(freqs_cc)
ao_orbs = range(1)

gmres_tol = 1e-3
gf = ccgf.CCGF(siam_cc, tol=gmres_tol)

ldos_cc = np.zeros(nw)

for iw in range(nw):
    print(iw, '/', nw)
    g_ip = gf.ipccsd_ao(ao_orbs, [freqs_cc[iw]], siam_mf.mo_coeff, delta_cc[iw]).conj()
    g_ea = gf.eaccsd_ao(ao_orbs, [freqs_cc[iw]], siam_mf.mo_coeff, delta_cc[iw])
    gf_tot = g_ip + g_ea
    ldos_cc[iw] = -1./np.pi*gf_tot[0,0,0].imag


'''
wl_cc = -6
wh_cc = 6
nw_cc = 600
freqs_cc = np.linspace(wl_cc, wh_cc, nw_cc)
gmres_tol = 1e-3
eta = 0.1

ao_orbs = range(1)

gf = ccgf.CCGF(siam_cc, tol=gmres_tol)
g_ip = gf.ipccsd_ao(ao_orbs, freqs_cc.conj(), siam_mf.mo_coeff, eta).conj()
g_ea = gf.eaccsd_ao(ao_orbs, freqs_cc, siam_mf.mo_coeff, eta)
gf = g_ip + g_ea

ldos_cc = -1./np.pi*gf[0,0,:].imag
'''

fh = h5py.File('ldos_siam_cc.h5', 'w')
fh['freqs_mf'] = freqs_mf
fh['ldos_mf'] = ldos_mf
fh['freqs_cc'] = freqs_cc
fh['ldos_cc'] = ldos_cc
fh['mu'] = mu
fh.close()

