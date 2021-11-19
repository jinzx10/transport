###########################################################
#
###########################################################

from pyscf import gto
from pyscf import scf

import numpy as np
import sys, time

import matplotlib.pyplot as plt
from matplotlib import cm, colors

def build_chain(N, spacing=2.9, basis='def2-svp', ecp='def2-svp'):
    '''
    return a gto.Mole() object that contains a 1-D chain of N gold atoms
    '''
    mol = gto.Mole()
    mol.basis = basis
    mol.ecp = ecp

    for i in range(0, N):
        mol.atom.append(['Au', (0,0,i*spacing)])

    mol.spin = N % 2
    mol.build()

    return mol


def save2txt(M, filename):
    f = open(filename, 'w')
    np.savetxt(f, M, fmt='%17.12f')
    f.close()


for nat in range(2, 50, 2):

    mol_chain = build_chain(nat)

    rhf_chain = scf.RHF(mol_chain)
    rhf_chain.max_cycle = 500

    start = time.time()
    rhf_chain.kernel()
    print('energy per atom = ' + str(rhf_chain.energy_tot()/nat) + ' Hartrees')
    print('time elapsed = ' + str(time.time()-start) + ' seconds')

    sz_atom = mol_chain.nao // nat # basis size of a single atom
    F = rhf_chain.get_fock()

    im = nat // 2
    F00 = F[im*sz_atom:(im+1)*sz_atom, im*sz_atom:(im+1)*sz_atom]
    F01 = F[(im-1)*sz_atom:im*sz_atom, im*sz_atom:(im+1)*sz_atom]

    save2txt(F00, 'data/F00_'+str(nat).zfill(2)+'.txt')
    save2txt(F01, 'data/F01_'+str(nat).zfill(2)+'.txt')

