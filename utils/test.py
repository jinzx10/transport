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

#rng(1);

switch_diis_fock = 0;
switch_gdm = 1;

switch_plot = 0;

sz_basis = 5;
n_occ = 2;

#=============================================
#           N-impurity Anderson model
#=============================================
n_imp = 3
n_bath = sz_basis - n_imp
bath = np.linspace(-1, sz_basis/n_occ-1, n_bath)
dos = 2. / (bath[1]-bath[0])

# hybridization
Gamma = 10./dos * np.random.rand(n_imp)

# impurity on-site energy and repulsion
E_imp = 0.2*(np.random.rand(n_imp)-0.55)
U_imp = 0.2 + 0.5*np.random.rand(n_imp)
U_bath = 0.01

# override the above random stuff!
E_imp = np.array([ -0.005772833839200, -0.063681122658295, -0.012220451215967 ])
U_imp = np.array([ 0.512030044086845,0.539567770432874,0.397757607834297 ])
Gamma = np.array([ 0.057471045613719,0.120627322054974,0.068487690222817 ])

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

print('h = ', h.toarray())

# generate an initial guess for orbital-based methods
e, v = spl.eigsh(h, k=n_occ, which='SA')


# for DIIS
def iterfock(fock_in):
    e, v = spl.eigsh(fock_in, k=n_occ, which='SA')
    fock_out = h.copy()
    occ = np.sum(v*v,axis=1)
    for i in range(sz_basis):
        fock_out[i,i] += U[i] * occ[i]

    # error vector (1-C*C^T)*FC
    FC = fock_out @ v
    err = FC - v @ (v.T @ FC)

    print('occ = ', occ)
    print('err = ', np.max(np.abs(err)))

    return fock_out, err


f, flag = diis(iterfock, h)

print(f.toarray())
