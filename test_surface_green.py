import numpy as np
from surface_green import LopezSancho1984

sz_blk = 3
n_blk = 1000

sz_tot = sz_blk * n_blk


H00 = np.random.randn(3,3)
H00 = (H00+H00.T)/2
#H00 = np.array([
#    [2, 1, 1],
#    [1, 2, 1],
#    [1, 1, 2]
#])

H01 = np.random.randn(3,3) * 0.05
#H01 = np.array([
#    [0.1, 0.2, 0.3],
#    [0.4, 0.5, 0.6],
#    [0.7, 0.8, 0.9],
#])

H = np.zeros((sz_tot, sz_tot))
for i in range(0, n_blk):
    H[i*sz_blk:(i+1)*sz_blk, i*sz_blk:(i+1)*sz_blk] = H00

    if i < n_blk-1:
        H[i*sz_blk:(i+1)*sz_blk, (i+1)*sz_blk:(i+2)*sz_blk] = H01
        H[(i+1)*sz_blk:(i+2)*sz_blk, i*sz_blk:(i+1)*sz_blk] = H01.T

np.set_printoptions(precision=3, linewidth=200)
#print(H)

E = 0
G = np.linalg.inv(E*np.eye(sz_tot) - H)
G00 = G[0:sz_blk, 0:sz_blk]

print(G00)

g = LopezSancho1984(E, H00, H01)

print(g)
