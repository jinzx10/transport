import numpy as np
import time, sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../../../')

from transport.utils import *


def matgen(M00, M01, nb):
    '''
    Generate a block tri-diagonal Hermitian matrix from M00 and M01
    where M00 is used as the diagonal blocks and M01 the off-diagonal blocks
    M00 is supposed to be Hermitian
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

calc_exact_finite = True

np.set_printoptions(precision=3, linewidth=200)

# (zS-H)*G=I
z = np.random.randn() + 1j * np.random.randn()

# block size
if len(sys.argv) < 2:
    sz_blk = 30
else:
    sz_blk = int(sys.argv[1])

# number of blocks
if len(sys.argv) < 3:
    n_blk = 50
else:
    n_blk = int(sys.argv[2])

# number of trials (for time average)
if len(sys.argv) < 4:
    nt = 10
else:
    nt = int(sys.argv[3])

sz_tot = sz_blk * n_blk

# generate a random Hamiltonian
H00 = np.random.randn(sz_blk, sz_blk) + 1j * np.random.randn(sz_blk, sz_blk)
H00 = (H00+H00.T.conj())/2
H01 = np.random.randn(sz_blk, sz_blk) * 0.05 + 1j * np.random.randn(sz_blk, sz_blk) * 0.05

# generate a random basis overlap matrix
M = np.random.randn(sz_blk, sz_blk) + 1j * np.random.randn(sz_blk, sz_blk)
Q, _ = np.linalg.qr(M)
S00 = Q @ np.diag(np.random.rand(sz_blk)+0.5) @ Q.T.conj()
S01 = np.random.rand(sz_blk, sz_blk) * 0.1 + 1j * np.random.rand(sz_blk, sz_blk) * 0.2


# Lopez Sancho algorithms
start = time.time()
for i in range(0, nt):
    g84 = LopezSancho1984(z, H00, H01, S00, S01, conv_thr=1e-12)
print('LopezSancho1984 average elapsed time = ', (time.time()-start)/nt, ' seconds')


start = time.time()
for i in range(0, nt):
    g85 = LopezSancho1985(z, H00, H01, S00, S01, conv_thr=1e-12)
print('LopezSancho1985 average elapsed time = ', (time.time()-start)/nt, ' seconds')


# Umerski algorithm
start = time.time()
for i in range(0, nt):
    g97 = Umerski1997(z, H00, H01, S00, S01)
print('Umerski1997     average elapsed time = ', (time.time()-start)/nt, ' seconds')


if calc_exact_finite:
    H = matgen(H00, H01, n_blk)
    S = matgen(S00, S01, n_blk)
    start = time.time()
    G = np.linalg.inv(z*S-H)
    G00 = G[0:sz_blk, 0:sz_blk]
    print('exact                   elapsed time = ', time.time()-start, ' seconds\n')

    if g84 is not None:
        print('LopezSancho1984 error = ', np.linalg.norm(G00-g84,1))

    if g85 is not None:
        print('LopezSancho1985 error = ', np.linalg.norm(G00-g84,1))

    print('Umerski1997     error = ', np.linalg.norm(G00-g97,1))



