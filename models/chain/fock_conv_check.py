'''
This script does RHF calculations for finite 1-D gold atomic chains and
checks the convergence of the Fock matrix with respect to the chain length.
'''

from pyscf import gto, scf
from pyscf import scf

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors

import sys, time, os

dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir+'/../../../')

from transport.utils import *

natm_blk = 4
spacing = 2.9

for nblk in range(3, 12, 2): 
    # use a minimum of 3 blocks to enable the calculation of next-nearest neighboring coupling
    # use an odd number of blocks to keep consistency

    natm = natm_blk * nblk
    mol_chain = ezbuild('Au', natm, spacing, basis='def2-svp', ecp='def2-svp')

    rhf_chain = scf.RHF(mol_chain).newton()
    sz_blk = mol_chain.nao // nblk # basis size of a block
    rhf_chain.max_cycle = 1000

    start = time.time()
    rhf_chain.kernel()
    print('energy per atom = ' + str(rhf_chain.energy_tot()/natm) + ' Hartrees')
    print('time elapsed = ' + str(time.time()-start) + ' seconds')
    print('\n')

    F = rhf_chain.get_fock()

    im = nblk // 2
    F00 = F[    im*sz_blk:(im+1)*sz_blk,     im*sz_blk:(im+1)*sz_blk]
    F01 = F[(im-1)*sz_blk:    im*sz_blk,     im*sz_blk:(im+1)*sz_blk]
    F02 = F[(im-1)*sz_blk:    im*sz_blk, (im+1)*sz_blk:(im+2)*sz_blk]

    ezsave(F00, dir+'/data/F00_'+str(natm_blk)+'_'+str(nblk).zfill(2)+'.txt')
    ezsave(F01, dir+'/data/F01_'+str(natm_blk)+'_'+str(nblk).zfill(2)+'.txt')
    ezsave(F02, dir+'/data/F02_'+str(natm_blk)+'_'+str(nblk).zfill(2)+'.txt')

# save the basis overlap matrix
S = rhf_chain.get_ovlp()
ezsave(S[0:sz_blk,        0:  sz_blk], dir+'/data/S00.txt')
ezsave(S[0:sz_blk,   sz_blk:2*sz_blk], dir+'/data/S01.txt')
ezsave(S[0:sz_blk, 2*sz_blk:3*sz_blk], dir+'/data/S02.txt')


