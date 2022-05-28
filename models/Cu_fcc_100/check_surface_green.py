import numpy as np
import h5py, time, sys, os, scipy
from mpi4py import MPI

from fcdmft.solver import scf_mu

import matplotlib.pyplot as plt
from matplotlib import colors

from pyscf import gto, ao2mo, cc, scf, dft
from pyscf.lib import chkfile

from pyscf.pbc import scf as pbcscf
from pyscf.pbc import df as pbcdf
from pyscf.pbc.lib import chkfile as pbcchkfile

from utils.surface_green import *
from utils.bath_disc import *

from pyscf.scf.hf import eig as eiggen


############################################################
#               read lead's mean-field data
############################################################

bath_dir = '/home/zuxin/projects/transport/models/Cu_fcc_100/Cu/'

print('start reading lead mean field data from', bath_dir)

#------------ check Cu HOMO/LUMO energy ------------

a = 3.6
Cu_basis = 'def2-svp-bracket'

bath_cell_label = 'Cu_' + Cu_basis + '_a' + str(a)

bath_cell_fname = bath_dir + 'cell_' + bath_cell_label + '.chk'
bath_cell = pbcchkfile.load_cell(bath_cell_fname)
nat_Cu_lead = len(bath_cell.atom)

bath_gdf_fname = bath_dir + 'cderi_' + bath_cell_label + '.h5'
bath_gdf = pbcdf.GDF(bath_cell)
bath_gdf._cderi = bath_gdf_fname

method_label = 'rks'
if 'ks' in method_label:
    bath_mf = pbcscf.RKS(bath_cell).density_fit()
    xcfun = 'pbe0'
    bath_mf.xc = xcfun
    bath_method_label = 'rks_' + xcfun
else:
    bath_mf = pbcscf.RHF(bath_cell).density_fit()
    bath_method_label = 'rhf'

solver_label = 'newton'

bath_mf_fname = bath_dir + bath_cell_label + '_' + bath_method_label + '_' + solver_label + '.chk'

bath_mf.with_df = bath_gdf
bath_mf.__dict__.update( pbcchkfile.load(bath_mf_fname, 'scf') )

ihomo = 29*nat_Cu_lead//2-1
ilumo = 29*nat_Cu_lead//2

E_Cu_homo = np.asarray(bath_mf.mo_energy)[ihomo]
E_Cu_lumo = np.asarray(bath_mf.mo_energy)[ilumo]

print('ihomo = ', ihomo, '      occ = ', np.asarray(bath_mf.mo_occ)[ihomo], '      E = ', E_Cu_homo)
print('ilumo = ', ilumo, '      occ = ', np.asarray(bath_mf.mo_occ)[ilumo], '      E = ', E_Cu_lumo)

#------------ get H00 and H01 (for surface Green's function) ------------
bath_fname = bath_dir + '/data_lead_' + bath_cell_label + '_' + bath_method_label + '.h5'
fh = h5py.File(bath_fname, 'r')

hcore_lo_lead = np.asarray(fh['hcore_lo'])
JK_lo_lead = np.asarray(fh['JK_lo'])
F_lo_lead = hcore_lo_lead + JK_lo_lead

print('F_lo_lead.shape = ', F_lo_lead.shape)

# number of orbitals per atom
nao_per_Cu = 15

# number of atoms per principal layer
nat_ppl = 9

# nao per principal layer
nao_ppl = nao_per_Cu * nat_ppl

H00 = F_lo_lead[0, :nao_ppl, :nao_ppl]
H01 = F_lo_lead[0, :nao_ppl, nao_ppl:2*nao_ppl]

print('finish reading lead mean field data\n')

##################
#plt.imshow(np.abs(F_lo_lead[0]))
#plt.show()
#exit()
##################

wl = -1
wh = 1
nw = 200
freqs = np.linspace(wl,wh,nw)
delta = 0.02

ldos_surface = np.zeros(nw)
for iw in range(nw):
    z = freqs[iw] + 1j*delta
    g00 = Umerski1997(z, H00, H01)
    #ldos_surface[iw] = -1./np.pi * np.sum(g00.diagonal().imag)
    ldos_surface[iw] = -1./np.pi * np.sum(g00.diagonal().imag)

plt.plot(freqs, ldos_surface)
plt.show()


