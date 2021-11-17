import numpy as np
import time
from surface_green import LopezSancho1984, LopezSancho1985

def matgen(M00, M01, nb):
    '''
    Generate a block tri-diagonal Hermitian matrix from M00 (diagonal) and M01 (off-diagonal)
    '''
    sz_blk = np.size(M00, 0);
    sz = sz_blk * nb;
    
    if np.iscomplexobj(M00) or np.iscomplexobj(M01):
        M = np.zeros((sz,sz), dtype=complex)
    else:
        M = np.zeros((sz,sz))
    
    for i in range(0, nb):
        M[i*sz_blk:(i+1)*sz_blk, i*sz_blk:(i+1)*sz_blk] = M00
        if i < nb-1:
            M[i*sz_blk:(i+1)*sz_blk, (i+1)*sz_blk:(i+2)*sz_blk] = M01
            M[(i+1)*sz_blk:(i+2)*sz_blk, i*sz_blk:(i+1)*sz_blk] = M01.T.conj()
    
    return (M+M.T.conj())/2


###########################################################
#   test various surface Green's function algorithms
###########################################################

np.set_printoptions(precision=3, linewidth=200)

# (zS-H)*G=I
z = np.random.randn() + 1j * np.random.randn()

sz_blk = 3
n_blk = 4
sz_tot = sz_blk * n_blk

# generate a random Hamiltonian
H00 = np.random.randn(sz_blk, sz_blk) + 1j * np.random.randn(sz_blk, sz_blk)
H00 = (H00+H00.T.conj())/2
H01 = np.random.randn(sz_blk, sz_blk) * 0.05 + 1j * np.random.randn(sz_blk, sz_blk) * 0.05
H = matgen(H00, H01, n_blk)
print(H)

# generate a random basis overlap matrix
M = np.random.randn(sz_blk, sz_blk)
Q, R = np.linalg.qr(M)
S00 = Q @ np.diag(np.random.rand(sz_blk)) @ Q.T
S01 = np.random.rand(sz_blk, sz_blk) * 0.1 + 1j * np.random.rand(sz_blk, sz_blk) * 0.2
S = matgen(S00, S01, n_blk)
#print(S)

# exact
start = time.time()
G = np.linalg.inv(z*S-H)
G00 = G[0:sz_blk, 0:sz_blk]
print(time.time()-start)

# Lopez Sancho algorithms
start = time.time()
g84 = LopezSancho1984(z, H00, H01, S00, S01)
print(time.time()-start)

start = time.time()
g85 = LopezSancho1985(z, H00, H01, S00, S01)
print(time.time()-start)


print('exact\n', G00, '\n')
print('Lopez Sancho 1984\n', g84, '\n')
print('Lopez Sancho 1985\n', g85, '\n')


