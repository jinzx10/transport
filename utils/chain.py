from pyscf import gto

def ezbuild(atom, N, spacing, basis='sto-3g', ecp='sto-3g'):
    '''
    return a gto.Mole() object that contains a 1-D chain of evenly spaced N atoms
    '''
    mol = gto.Mole()
    mol.basis = basis
    mol.ecp = ecp

    for i in range(0, N):
        mol.atom.append([atom, (0,0,i*spacing)])

    mol.spin = mol.nelectron % 2
    mol.build()

    return mol
