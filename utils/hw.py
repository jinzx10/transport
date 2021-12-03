import numpy as np
from diis import diis as idiis
from pyscf.lib import diis 

def F2F(F_in, h, U, n_occ):
    val, vec = np.linalg.eigh(F_in)
    sz = np.size(F_in, 0)
    P = vec[:,:n_occ] @ vec[:,:n_occ].T

    F_out = np.copy(h)
    F_out[0,0] += U*P[0,0]
    F_out[1,1] += U*P[1,1]

    return F_out

sz = 500;
n_occ = round(np.floor(sz/2)+1);
Gamma = 0.05;
U = 0.1
bath = np.linspace(-1, 1, sz-2);
dos = 1 / (bath[1]-bath[0]);
cpl = np.sqrt(Gamma/2/np.pi/dos);

#E_imp = np.array([-0.01, 0.01])
E_imp = 0.2 * ( np.random.rand(2,1) - 0.5 )

h = np.diag(np.append(E_imp, bath), 0)
h[0,2:] = cpl
h[2:,0] = cpl
h[1,2:] = cpl/2
h[2:,1] = cpl/2

f2f = lambda F_in: F2F(F_in, h, U, n_occ)

tol = 1e-7
max_diis = 50
conv, F = idiis(f2f, h, conv_tol = tol, max_iter = max_diis, max_subspace_size=6)

print(conv)

adiis = diis.DIIS()
P = np.zeros((sz,sz))

for i in range(0,max_diis):

    F = np.copy(h)
    F[0,0] += U*P[0,0]
    F[1,1] += U*P[1,1]

    if i > 1:
        F_new = adiis.update(F)
        dF = F_new-F
        err = np.linalg.norm(dF,1)
        print('err = ', err)
        if err < tol and i > 3:
            break
        F = F_new

    val, vec = np.linalg.eigh(F)
    P = vec[:,:n_occ] @ vec[:,:n_occ].T


