#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from pyscf import gto
from pyscf import scf


# transport calculation for a molecule sits between two gold atom chains
# ...-Au-Au-N-O-Au-Au-...

#==========================================================
#                   geometry setup
#==========================================================

#==========================================================
#               surface Green's function
#==========================================================

# distance between two gold atoms in the chain
d = 2.9

#
mol_au = gto.Mole()
mol_au.basis = 'cc-pvdz-pp'
mol_au.atom = [
        ['Au', (0,0,0)], ['Au', (0,0,d)], 
        ['Au', (0,0,2*d)], ['Au', (0,0,3*d)], 
        ['Au', (0,0,4*d)], ['Au', (0,0,5*d)],
        ['Au', (0,0,6*d)], ['Au', (0,0,7*d)],
        ['Au', (0,0,8*d)], ['Au', (0,0,9*d)],
        ['Au', (0,0,10*d)], ['Au', (0,0,11*d)],
]
mol_au.build()

S = scf.RHF(mol_au).get_ovlp()
sz_atom = np.size(S, 0) // 12

block_size = 4
sz_block = sz_atom * block_size

# remove the 0 values by replacing them with some small number
S2 = np.copy(S)
S2 = np.where(abs(S2) < 1e-16, 1e-16, S2)

# check whether S has block tri-diagonal structure
S01 = S[0:sz_block, sz_block:2*sz_block]
S02 = S[0:sz_block, 2*sz_block:3*sz_block]

print('max(abs(S02)) = ', np.max(abs(S02)) )

# visualize S
im = plt.imshow(S2, cmap=cm.rainbow, norm=colors.LogNorm())
plt.colorbar(im)

plt.savefig('Fock.png')

#plt.show()

#f = open('S.txt', 'w')
#np.savetxt(f, S, fmt='%8.5f')








