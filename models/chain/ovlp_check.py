#!/usr/bin/python

#==========================================================
# This script considers a 1-D gold atomic chain
#
# ...-Au-Au-Au-Au-...
#
# and checks, given a certain block size, whether there is 
# significant basis overlap between non-neighboring blocks.
#==========================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from pyscf import gto, scf
import sys, os

dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir+'/../../../')

from transport.utils import *

# number of atoms per block (principal layer)
natm_blk = int(sys.argv[1])

# distance between two gold atoms in the chain
d = 2.9

# build gto.Mole() object
mol_au_chain = ezbuild('Au', natm_blk*3, d, basis='def2-svp', ecp='def2-svp')

# overlap matrix for 3 blocks
S = scf.UHF(mol_au_chain).get_ovlp()

# check the quality of block tri-diagonal structure of S
sz_block = np.size(S,0) // 3
S00 = S[0:sz_block, 0:sz_block]
S01 = S[0:sz_block, sz_block:2*sz_block]
S02 = S[0:sz_block, 2*sz_block:3*sz_block]

print('max(abs(S02)) = ', np.max(abs(S02)) )
print('sz_block = ', sz_block)

## save S00, S01, S02
#ezsave(S00, dir+'/data/S00.txt')
#ezsave(S01, dir+'/data/S01.txt')
#ezsave(S02, dir+'/data/S02.txt')

# replace the zeros by some finite small number (in order to visualize S with a log scale)
Splot = np.copy(S)
Splot = np.where(abs(Splot) < 1e-16, 1e-16, Splot)

# visualize S
im = plt.imshow(Splot, cmap=cm.rainbow, norm=colors.LogNorm())
plt.colorbar(im)

plt.show()


