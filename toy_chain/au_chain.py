#!/usr/bin/python

from pyscf import gto
from pyscf import scf

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors

mol_au = gto.Mole()
mol_au.basis = 'def2-svp'
mol_au.ecp = 'def2-svp'
#mol_au.basis = 'cc-pvdz-pp'
#mol_au.ecp = 'cc-pvdz-pp'
d = 2.9

nat = 12

mol_au.atom = [ ['Au', (0,0,0)], ]
mol_au.spin = 1
mol_au.build()

# basis size of a single atom
S = scf.UHF(mol_au).get_ovlp()
sz_atom = np.size(S, 0)


for i in range(1,nat):
    mol_au.atom.append(['Au', (0,0,i*d)])

mol_au.spin = nat % 2
mol_au.build()


uhf_au = scf.UHF(mol_au)
ig = uhf_au.init_guess_by_minao(mol=mol_au, breaksym=True)
ig = ig + 0.5*np.random.randn(*np.shape(ig))


uhf_au.kernel(dm0=ig)

F = uhf_au.get_fock()
#print(F)

# replace the zeros by some finite small number (in order to visualize S with a log scale)
F_alpha = np.copy(F[0,:,:])
F_alpha = np.where(abs(F_alpha) < 1e-16, 1e-16, F_alpha)

im = nat // 2
F00 = F_alpha[im*sz_atom:(im+1)*sz_atom, im*sz_atom:(im+1)*sz_atom]

# visualize F
im = plt.imshow(abs(F_alpha), cmap=cm.rainbow, norm=colors.LogNorm())
plt.colorbar(im)

plt.show()


