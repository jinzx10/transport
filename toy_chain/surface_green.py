#!/usr/bin/python

#==========================================================
# transport calculation for a toy model:
# a molecule sits between two gold atom chains
#
# ...-Au-Au-N-O-Au-Au-...
#
# this script calculates the surface Green's function 
# of the gold atom chain
# 
# procedure:
# 1. do a scf calculation to determine the bulk charge density
# 2. use the bulk charge density to build 
#==========================================================

from pyscf.pbc import gto

cell_au = gto.Cell()
cell_au.dimension = 1
cell_au.atom = '''Au 0 0 0'''
cell_au.basis = ''

# number of atoms in a 'principal layer'
nat_pl = 3



