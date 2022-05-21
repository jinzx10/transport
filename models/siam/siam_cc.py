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
mu = 0.0

#---------------- generate grid for bath discretization --------------------

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

# symmetric linear grid
wlb = -5.0
whb = 5.0
nbath = 100
# 2/5 coarse bath states
# 2/5 fine bath states near mu
# 1/5 ultrafine bath states near mu
nbath_coarse = round(nbath/5) # one side
nbath_fine = nbath_coarse
nbath_ultrafine = nbath - nbath_fine*4

# energy windows for ultrafine bath states (one-sided width)
wb = 0.05

# second energy windows for fine bath states
Wb = 0.5

grid = np.concatenate( (\
        np.linspace(wlb,mu-Wb,nbath_coarse+1), \
        np.linspace(mu-Wb,mu-wb,nbath_fine+1), \
        np.linspace(mu-wb,mu+wb,nbath_ultrafine), \
        np.linspace(mu+wb,mu+Wb,nbath_fine+1), \
        np.linspace(mu+Wb,whb,nbath_coarse+1) ) )

grid = np.array(sorted(list(set(list(grid)))))
print('grid = ', grid)
print('dgrid = ', grid[1:]-grid[:-1])

#########################################################

siam_mf = gen_siam_mf(on_site, hyb_const, U, mu, grid)
siam_mf.kernel()

print('mf mo_energy = ', siam_mf.mo_energy)
print('mf diff mo_energy = ', siam_mf.mo_energy[1:] - siam_mf.mo_energy[:-1])

#########################################################

#---------------- frequency grid for LDoS ------------------
'''
wlw = -6
whw = 6

nw_mf = 600
nw_coarse = round(nw_mf/5) # one side
nw_fine = nw_coarse
nw_ultrafine = nw_mf - nw_fine*4

# energy windows for ultrafine grid (one-sided width)
ww = 0.05

# second energy windows for fine grid
Ww = 0.5

freqs_mf = np.concatenate( (\
        np.linspace(wlw,mu-Ww,nw_coarse+1), \
        np.linspace(mu-Ww,mu-ww,nw_fine+1), \
        np.linspace(mu-ww,mu+ww,nw_ultrafine), \
        np.linspace(mu+ww,mu+Ww,nw_fine+1), \
        np.linspace(mu+Ww,whw,nw_coarse+1) ) )

freqs_mf = np.array(sorted(list(set(list(freqs)))))

delta_coarse = (mu-Ww-wlw) / nw_coarse * 10
delta_fine = (Ww-ww) / nw_fine * 10
delta_ultrafine = 2*ww/nw_ultrafine * 10

delta = np.concatenate( (\
        delta_coarse * np.ones(nw_coarse), \
        delta_fine * np.ones(nw_fine), \
        delta_ultrafine * np.ones(nw_ultrafine), \
        delta_fine * np.ones(nw_fine), \
        delta_coarse * np.ones(nw_coarse) ) )

'''

wl_mf = -6
wh_mf = 6
nw_mf = 600

# note that given the previous bath discretization, the mean-field mo_energy
# has a spacing near the Fermi level of ~0.003
delta = 0.2 * np.ones(nw_mf)
freqs_mf = np.linspace(wl_mf, wh_mf, nw_mf)

# ldos (mean-field level)
ldos_mf = np.zeros(nw_mf)
for iw in range(nw_mf):
    z = freqs_mf[iw] + 1j*delta[iw]
    gf_mo = np.diag(1./(z-siam_mf.mo_energy))
    gf = siam_mf.mo_coeff @ gf_mo @ siam_mf.mo_coeff.T
    ldos_mf[iw] = -1./np.pi*gf[0,0].imag


#plt.plot(freqs_mf, ldos_mf)
#plt.xlim([-6,6])
#plt.ylim([-0.01,0.5])
#plt.show()
#exit()

#------------ CC impurity solver ------------
siam_cc = cc.RCCSD(siam_mf)
siam_cc.conv_tol = 1e-8
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

#---------------- frequency grid for LDoS ------------------
wl = -6
wh = 6

W = 1
w = 0.1

delta_cc1 = 0.2
nw1 = 2*round((mu-W-wl)/delta_cc1)
freqs_cc1 = np.linspace(wl,mu-W,nw1)
delta_cc1 = delta_cc1*np.ones(nw1-1)

delta_cc2 = 0.1
nw2 = 2*round((W-w)/delta_cc2)
freqs_cc2 = np.linspace(mu-W,mu-w,nw2)
delta_cc2 = delta_cc2*np.ones(nw2-1)

delta_cc3 = 0.05
nw3 = 2*round(2*w/delta_cc3)
freqs_cc3 = np.linspace(mu-w,mu+w,nw3)
delta_cc3 = delta_cc3*np.ones(nw3)

delta_cc4 = 0.1
nw4 = 2*round((W-w)/delta_cc4)
freqs_cc4 = np.linspace(mu+w,mu+W,nw4)
delta_cc4 = delta_cc4*np.ones(nw4-1)

delta_cc5 = 0.2
nw5 = 2*round((wh-mu-W)/delta_cc5)
freqs_cc5 = np.linspace(mu+W,wh,nw5)
delta_cc5 = delta_cc5*np.ones(nw5-1)

freqs_cc = np.concatenate((freqs_cc1,freqs_cc2,freqs_cc3,freqs_cc4,freqs_cc5))
freqs_cc = np.array(sorted(list(set(list(freqs_cc)))))

delta_cc = np.concatenate((delta_cc1,delta_cc2,delta_cc3,delta_cc4,delta_cc5))

assert(len(freqs_cc)==len(delta_cc))

print('cc freqs grid:', freqs_cc)
print('cc freqs broadening:', delta_cc)

nw = len(freqs_cc)
ao_orbs = range(1)

gmres_tol = 1e-4
gf = ccgf.CCGF(siam_cc, tol=gmres_tol)

ldos_cc = np.zeros(nw)

for iw in range(nw):
    print(iw+1, '/', nw)
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

