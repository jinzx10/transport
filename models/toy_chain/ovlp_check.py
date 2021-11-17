#!/usr/bin/python

#==========================================================
# transport calculation for a toy model:
# a molecule sits between two gold atom chains
#
# ...-Au-Au-N-O-Au-Au-...
#
# this script checks whether a certain block size forms 
# a good principal layer, i.e., whether there is significant
# overlap between non-neighboring blocks
#==========================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from pyscf import gto
from pyscf import scf

# number of atoms per principal layer
block_size = 3

# distance between two gold atoms in the chain
d = 2.9

mol_au = gto.Mole()
mol_au.basis = 'cc-pvdz-pp'

mol_au.atom = [ ['Au', (0,0,0)], ]
mol_au.spin = 1
mol_au.build()

# basis size of a single atom
S = scf.UHF(mol_au).get_ovlp()
sz_atom = np.size(S, 0)

# build a molecule object that contains 3 blocks
for i in range(1, 3*block_size):
    mol_au.atom.append(['Au', (0, 0, i*d)])

mol_au.spin = (3*block_size) % 2
mol_au.build()

# overlap matrix for 3 blocks
S = scf.UHF(mol_au).get_ovlp()

# check the quality of block tri-diagonal structure of S
sz_block = sz_atom * block_size
S01 = S[0:sz_block, sz_block:2*sz_block]
S02 = S[0:sz_block, 2*sz_block:3*sz_block]

print('max(abs(S02)) = ', np.max(abs(S02)) )

# replace the zeros by some finite small number (in order to visualize S with a log scale)
S2 = np.copy(S)
S2 = np.where(abs(S2) < 1e-16, 1e-16, S2)

# visualize S
im = plt.imshow(S2, cmap=cm.rainbow, norm=colors.LogNorm())
plt.colorbar(im)

plt.show()


