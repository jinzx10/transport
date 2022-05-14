from linesearch import linesearch
from lgdm import lgdm
import numpy as np
import scipy.sparse.linalg as spl
import scipy.sparse as sp
from diis import diis

'''
def f(x):
    return -(x-0.02)**2

def df(x):
    return -2.*(x-0.02)


alpha, flag = linesearch(f, df, wolfe1=0.2, wolfe2=0.6)

print('alpha = ', alpha)
print('flag =' , flag)
'''


sz_basis = 10;
n_occ = 5;

n_imp = 3
n_bath = sz_basis - n_imp

#=============================================
#           N-impurity Anderson model
#=============================================
bath = np.linspace(-1, sz_basis/n_occ-1, n_bath)
dos = 2. / (bath[1]-bath[0])

# hybridization
Gamma = 10./dos * np.random.rand(n_imp)

# impurity on-site energy and repulsion
E_imp = 0.2*(np.random.rand(n_imp)-0.55)
U_imp = 0.2 + 0.5*np.random.rand(n_imp)


#########################################################
# a special case which overrides the above random stuff!
E_imp = np.array([ -0.005772833839200, -0.063681122658295, -0.012220451215967 ])
U_imp = np.array([ 0.512030044086845,0.539567770432874,0.397757607834297 ])
Gamma = np.array([ 0.057471045613719,0.120627322054974,0.068487690222817 ])
#########################################################

U_bath = 0.01
U = np.concatenate((U_imp, U_bath*np.ones(n_bath)))

cpl = np.sqrt(Gamma/2./np.pi/dos);

# hopping amplitude between impurity sites
t = 0.05

# one-body Hamiltonian
h = sp.diags(np.concatenate((E_imp, bath)), format='lil')

for i in range(n_imp-1):
    h[i,i+1] = t
    h[i+1,i] = t

for i in range(n_imp):
    h[n_imp:,i] = cpl[i]
    h[i,n_imp:] = cpl[i]

# generate an initial guess for orbital-based methods
e0, v0 = spl.eigsh(h, k=n_occ, which='SA')


def fockbuild(C):
    F = h.copy()
    occ = np.sum(C*C, axis=1)
    for i in range(sz_basis):
        F[i,i] += U[i] * occ[i]
    return F

# wrapper for DIIS
def iterfock(fock_in):
    e, v = spl.eigsh(fock_in, k=n_occ, which='SA')
    fock_out = fockbuild(v)

    # error vector (1-C*C^T)*FC
    FC = fock_out @ v
    err = FC - v @ (v.T @ FC)

    # important! make err insensitive to the sign of eigenvectors
    err = v * err

    return fock_out, err


#fock, flag_diis = diis(iterfock, h, max_iter=200)



# wrapper for GDM
def Etot(C):
    return 2.*np.sum(C*(h@C)) + np.dot(U, np.sum(C*C, axis=1)**2)

def dE(C):
    # 4*FC
    return 4. * fockbuild(C) @ C


C, flag_lgdm, lgdm_hist = lgdm(Etot, dE, v0, conv_thr=1e-8)

#if flag_diis == 0:
#    e, v = spl.eigsh(fock, k=n_occ, which='SA')
#    imp_occ_diis = np.sum(v*v, axis=1)
#    print('final imp occ (DIIS) = ', imp_occ_diis[:n_imp])

if flag_lgdm == 0:
    imp_occ_lgdm = np.sum(C*C, axis=1)
    print('final imp occ (LGDM) = ', imp_occ_lgdm[:n_imp])

    fock = fockbuild(C)
    e, v = spl.eigsh(fock, k=n_occ, which='SA')
    PC = C@C.T
    Pv = v@v.T
    print(np.trace(PC))
    print(np.trace(Pv))
    print('P diff = ', np.linalg.norm(PC-Pv))

