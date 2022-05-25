import numpy as np
import os, h5py, argparse, copy

from pyscf.pbc import gto, df, scf
from pyscf.pbc.lib import chkfile

from libdmet_solid.system import lattice
from libdmet.utils import plot
from libdmet_solid.basis_transform import make_basis
from libdmet_solid.lo.iao import reference_mol

from pyscf.scf.hf import eig as eiggen

import matplotlib.pyplot as plt

# switch to 'production' for serious jobs
mode = 'production'
############################################################
#                       basis
############################################################
imp_atom = 'Fe' if mode == 'production' else 'Co'

imp_basis = 'def2-svp'
Cu_basis = 'def2-svp-bracket'

############################################################
#           directory for saving/loading data
############################################################
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default = 'data', type = str)
args = parser.parse_args()

datadir = args.datadir + '/'
print('data will be saved to:', datadir)

if not os.path.exists(datadir):
    os.mkdir(datadir)

############################################################
#                       build cell
############################################################
# transport direction fcc 100 plane
# an example of nl=4, nr=3
# |   5        4        5      4     1      4      5       4      |   5
# |-l-1.5a   -l-a   -l-0.5a   -l   0(imp)   r    r+0.5a   r+a     | r+1.5a

# x-distance between impurity atom and nearest Cu plane
l = 1.8 if mode == 'production' else 1.8
r = 1.8 if mode == 'production' else 1.8

# number of layers in the left/right lead included in contact SCF calculation
nl = 2 if mode == 'production' else 4
nr = 3 if mode == 'production' else 3

# ensure that supercell periodic boundary is properly matched
assert((nl+nr)%2 == 1)

# Cu lattice constant (for the fcc cell)
a = 3.6 if mode == 'production' else 3.6

cell_label = imp_atom + '_' + imp_basis + '_Cu_' + Cu_basis \
        + '_nl' + str(nl) + '_nr' + str(nr) \
        + '_l' + str(l) + '_r' + str(r) + '_a' + str(a)

cell_fname = datadir + '/cell_' + cell_label + '.chk'

if os.path.isfile(cell_fname):

    cell = chkfile.load_cell(cell_fname)
    print('use saved cell file:', cell_fname)

else:
    
    cell = gto.Cell()
    
    cell.unit = 'angstrom'
    cell.verbose = 4
    cell.max_memory = 50000
    cell.dimension = 3

    cell.precision = 1e-10
    #cell.ke_cutoff = 200

    # should not discard! 4s and 5s have small exponent
    #cell.exp_to_discard = 0.1

    cell.a = [[l+r+(nl+nr-1)*0.5*a,0,0], [0,10,0], [0,0,10]]
    
    cell.atom.append([imp_atom, (0, 0, 0)])

    # left lead
    for i in range(0, nl):
        x = -l - (nl-1-i) * 0.5*a
        if (i%2 == 0 and nl%2 == 1) or (i%2 == 1 and nl%2 == 0): # 4-atom layer
            cell.atom.append(['Cu', (x,      0,  0.5*a)])
            cell.atom.append(['Cu', (x,      0, -0.5*a)])
            cell.atom.append(['Cu', (x,  0.5*a,      0)])
            cell.atom.append(['Cu', (x, -0.5*a,      0)])
        else: # 5-atom layer
            cell.atom.append(['Cu', (x,      0,      0)])
            cell.atom.append(['Cu', (x,  0.5*a,  0.5*a)])
            cell.atom.append(['Cu', (x, -0.5*a,  0.5*a)])
            cell.atom.append(['Cu', (x,  0.5*a, -0.5*a)])
            cell.atom.append(['Cu', (x, -0.5*a, -0.5*a)])

    # right lead
    for i in range(0, nr):
        x = i*0.5*a + r
        if i%2 == 0: # 4-atom layer
            cell.atom.append(['Cu', (x,      0,  0.5*a)])
            cell.atom.append(['Cu', (x,      0, -0.5*a)])
            cell.atom.append(['Cu', (x,  0.5*a,      0)])
            cell.atom.append(['Cu', (x, -0.5*a,      0)])
        else: # 5-atom layer
            cell.atom.append(['Cu', (x,      0,      0)])
            cell.atom.append(['Cu', (x,  0.5*a,  0.5*a)])
            cell.atom.append(['Cu', (x, -0.5*a,  0.5*a)])
            cell.atom.append(['Cu', (x,  0.5*a, -0.5*a)])
            cell.atom.append(['Cu', (x, -0.5*a, -0.5*a)])

    cell.spin = None # let build() decides
    cell.basis = {imp_atom : imp_basis, 'Cu' : Cu_basis}
    
    cell.build()
    
    # save cell
    chkfile.save_cell(cell, cell_fname)

nat_Cu = len(cell.atom) - 1

###############
'''
# plot cell
atoms = cell.atom
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for atm in atoms:
    coor = atm[1]
    cl = 'r' if atm[0] != 'Cu' else 'b'
    ax.scatter(coor[0], coor[1], coor[2], color=cl)

plt.show()

exit()
'''
###############

############################################################
#                   mean-field method
############################################################
# This is the method used to generate the density matrix;

# For HF+DMFT, the Fock matrix for building the impurity Hamiltonian
# will use HF veff from the previous DM anyway (even though it's a KS-converged DM)

# if False, use HF instead
use_dft = True if mode == 'production' else True

# if use_dft
xcfun = 'pbe0' if mode == 'production' else 'pbe0'

do_restricted = True if mode == 'production' else True

# TODO may add support for ROHF/ROKS in the future
assert( (not do_restricted) or (cell.spin==0) )

if do_restricted:
    method_label = 'r'
else:
    method_label = 'u'

if use_dft:
    method_label += 'ks' + '_' + xcfun
else:
    method_label += 'hf'

# scf solver & addons
# if use_smearing, use Fermi smearing
# if False, scf will use newton solver to help convergence
use_smearing = True if mode == 'production' else True
smearing_sigma = 0 if mode == 'production' else 0.05

if use_smearing:
    solver_label = 'smearing' + str(smearing_sigma)
else:
    solver_label = 'newton'

############################################################
#               density fitting
############################################################

gdf_fname = datadir + '/cderi_' + cell_label + '.h5'

gdf = df.GDF(cell)

if os.path.isfile(gdf_fname):
    print('use saved gdf cderi:', gdf_fname)
    gdf._cderi = gdf_fname
else:
    gdf._cderi_to_save = gdf_fname
    gdf.auxbasis = {imp_atom : imp_basis + '-ri', 'Cu' : Cu_basis + '-ri'}
    gdf.build()

############################################################
#           low-level mean-field calculation
############################################################

if use_dft:
    if do_restricted:
        mf = scf.RKS(cell).density_fit()
    else:
        mf = scf.UKS(cell).density_fit()
    mf.xc = xcfun
else:
    if do_restricted:
        mf = scf.RHF(cell).density_fit()
    else:
        mf = scf.UHF(cell).density_fit()

mf_fname = datadir + '/' + cell_label + '_' + method_label + '_' + solver_label + '.chk'

mf.with_df = gdf

if use_smearing:
    mf = scf.addons.smearing_(mf, sigma = smearing_sigma, method = 'fermi')
else:
    mf = mf.newton()

if os.path.isfile(mf_fname):
    print('load saved mf data', mf_fname)
    mf_data = chkfile.load(mf_fname, 'scf') 
    mf.__dict__.update(mf_data)

    mf.chkfile = mf_fname
    mf.conv_tol = 1e-10
    mf.max_cycle = 50
    mf.kernel()
else:
    mf.chkfile = mf_fname
    mf.conv_tol = 1e-10
    mf.max_cycle = 300
    mf.kernel()

###############################################
#       convergence sanity check begin
###############################################
if method == 'rhf' or method == 'rks':
    S_ao_ao = mf.get_ovlp()
    hcore_ao = mf.get_hcore()
    JK_ao = mf.get_veff()
    
    e, v = eiggen(hcore_ao+JK_ao, S_ao_ao)
    
    print('sanity check: mo_energy vs. fock eigenvalue = ', np.linalg.norm(mf.mo_energy-e))
    
    DM_ao = mf.make_rdm1()
    mo_occ = mf.mo_occ
    dm_fock = (v * mo_occ) @ v.T
    print('sanity check: dm diff between make_rdm1 and fock-solved = ', np.linalg.norm(dm_fock-DM_ao))

###############################################
#       convergence sanity check end
###############################################


###############################################
#       Number of AOs
###############################################
if imp_basis == 'def2-svp':
    # 1s,2s,2p,3s,3p
    ncore_imp = 9
    
    # 3d,4s
    nval_imp = 6
    
    # 4p,4d,4f,5s
    nvirt_imp = 16

if imp_basis == 'def2-svp-bracket':
    # 1s,2s,2p,3s,3p
    ncore_imp = 9
    
    # 3d,4s
    nval_imp = 6
    
    # 4p,4d,4f,5s
    nvirt_imp = 9

nao_imp = nval_imp + nvirt_imp # core orbitals are ignored!
nao_imp_tot = nao_imp + ncore_imp

if Cu_basis == 'def2-svp-bracket':
    ncore_Cu = 9
    nval_Cu = 6
    nvirt_Cu = 9

nao_Cu = nval_Cu + nvirt_Cu
nao_Cu_tot = nao_Cu + ncore_Cu

############################################################
#               build Lattice object
############################################################
kmesh = [1,1,1]
kpts = cell.make_kpts(kmesh)
Lat = lattice.Lattice(cell, kmesh)
nao = Lat.nao
nkpts = Lat.nkpts

############################################################
#               generate a kmf object
############################################################
if use_dft:
    if do_restricted:
        kmf = scf.KRKS(cell, kpts).density_fit()
    else:
        kmf = scf.KUKS(cell, kpts).density_fit()
    kmf.xc = xcfun
else:
    if do_restricted:
        kmf = scf.KRHF(cell, kpts).density_fit()
    else:
        kmf = scf.KUHF(cell, kpts).density_fit()

kmf.with_df = gdf

# add a k axis for subsequent IAO/PAO construction
kmf.mo_coeff = mf.mo_coeff[np.newaxis,...]
kmf.mo_energy = mf.mo_energy[np.newaxis,...]
kmf.mo_occ = mf.mo_occ[np.newaxis,...]

# spin: unrestricted -> 2; restricted -> 1
if len(mf.mo_energy.shape) == 1:
    spin = 1
else:
    spin = 2

############################################################
#               Orbital Localization (IAO)
############################################################

MINAO = {imp_atom: imp_basis + '-minao', 'Cu': Cu_basis + '-minao'}

# if True, use reference core/val
use_reference_mol = True

if use_reference_mol:

    # set IAO (core+val)
    pmol = reference_mol(cell, minao=MINAO)
    
    # set valence IAO
    pmol_val = pmol.copy()
    pmol_val.basis = {imp_atom: imp_basis + '-minao-val', 'Cu': Cu_basis + '-minao-val'}
    pmol_val.build()
    basis_val = {}
    basis_val[imp_atom] = copy.deepcopy(pmol_val._basis[imp_atom])
    basis_val['Cu'] = copy.deepcopy(pmol_val._basis['Cu'])
    
    pmol_val = pmol.copy()
    pmol_val.basis = basis_val
    pmol_val.build()
    
    val_labels = pmol_val.ao_labels()
    for i in range(len(val_labels)):
        val_labels[i] = val_labels[i].replace(imp_atom + ' 1s', imp_atom+' 4s')
        val_labels[i] = val_labels[i].replace('Cu 1s', 'Cu 4s')
    pmol_val.ao_labels = lambda *args: val_labels
    
    # set core IAO
    pmol_core = pmol.copy()
    pmol_core.basis = {imp_atom: imp_basis+ '-minao-core', 'Cu': Cu_basis + '-minao-core'}
    pmol_core.build()
    basis_core = {}
    basis_core[imp_atom] = copy.deepcopy(pmol_core._basis[imp_atom])
    basis_core['Cu'] = copy.deepcopy(pmol_core._basis['Cu'])
    
    pmol_core = pmol.copy()
    pmol_core.basis = basis_core
    pmol_core.build()
    core_labels = pmol_core.ao_labels()
    
    ncore = len(pmol_core.ao_labels())
    nval = pmol_val.nao_nr()
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)
    
    print('ncore = ', ncore)
    print('nval = ', nval)
    print('nvirt = ', nvirt)
    
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt, C_ao_iao_core \
            = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=MINAO, full_return=True, \
            pmol_val=pmol_val, pmol_core=pmol_core, tol=1e-8)

else:
    # if core/val is not specified, the number of val will equal the size of MINAO
    # and the rest are virt
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=MINAO, full_return=True)

if spin == 1:
    C_ao_iao = C_ao_iao[np.newaxis,...]
    C_ao_iao_val = C_ao_iao_val[np.newaxis,...]
    C_ao_iao_virt = C_ao_iao_virt[np.newaxis,...]
    C_ao_iao_core = C_ao_iao_core[np.newaxis,...]

############################################################
#               rearrange orbitals
############################################################
# rearrange IAO/PAO according to their indices in cell.atom

if use_reference_mol:
    # C_ao_lo: transformation matrix from AO to LO (IAO) basis, all atoms, core orbitals are excluded
    # C_ao_lo_imp: transformation matrix from AO to LO (IAO) basis, only imp atom, core orbitals are excluded
    # C_ao_lo_tot: transformation matrix from AO to LO (IAO) basis, all atoms, all orbitals (core included!)
    C_ao_lo     = np.zeros((spin,nkpts,nao,nval+nvirt), dtype=C_ao_iao.dtype)
    C_ao_lo_imp = np.zeros((spin,nkpts,nao,nao_imp   ), dtype=C_ao_iao.dtype)
    C_ao_lo_tot = np.zeros((spin,nkpts,nao,nao       ), dtype=C_ao_iao.dtype)
    
    C_ao_lo[:,:,:,0:nval_imp] = C_ao_iao_val[:,:,:,0:nval_imp]
    C_ao_lo[:,:,:,nval_imp:nao_imp] = C_ao_iao_virt[:,:,:,0:nvirt_imp]

    for iat in range(nat_Cu):
        C_ao_lo[:,:,:,nao_imp+iat*nao_Cu:nao_imp+iat*nao_Cu+nval_Cu] \
                = C_ao_iao_val[:,:,:,nval_imp+iat*nval_Cu:nval_imp+(iat+1)*nval_Cu]

        C_ao_lo[:,:,:,nao_imp+iat*nao_Cu+nval_Cu:nao_imp+(iat+1)*nao_Cu] \
                = C_ao_iao_virt[:,:,:,nvirt_imp+iat*nvirt_Cu:nvirt_imp+(iat+1)*nvirt_Cu]

    C_ao_lo_imp = C_ao_lo[:,:,:,0:nao_imp]

    #----------------------------------------------------------
    C_ao_lo_tot[:,:,:,0:ncore_imp] = C_ao_iao_core[:,:,:,0:ncore_imp]
    C_ao_lo_tot[:,:,:,ncore_imp:ncore_imp+nval_imp] = C_ao_iao_val[:,:,:,0:nval_imp]
    C_ao_lo_tot[:,:,:,ncore_imp+nval_imp:nao_imp_tot] = C_ao_iao_virt[:,:,:,0:nvirt_imp]

    for iat in range(nat_Cu):
        C_ao_lo_tot[:,:,:,nao_imp_tot+iat*nao_Cu_tot:nao_imp_tot+iat*nao_Cu_tot+ncore_Cu] \
                = C_ao_iao_core[:,:,:,ncore_imp+iat*ncore_Cu:ncore_imp+(iat+1)*ncore_Cu]

        C_ao_lo_tot[:,:,:,nao_imp_tot+iat*nao_Cu_tot+ncore_Cu:nao_imp_tot+iat*nao_Cu_tot+ncore_Cu+nval_Cu] \
                = C_ao_iao_val[:,:,:,nval_imp+iat*nval_Cu:nval_imp+(iat+1)*nval_Cu]

        C_ao_lo_tot[:,:,:,nao_imp_tot+iat*nao_Cu_tot+ncore_Cu+nval_Cu:nao_imp_tot+(iat+1)*nao_Cu_tot] \
                = C_ao_iao_virt[:,:,:,nvirt_imp+iat*nvirt_Cu:nvirt_imp+(iat+1)*nvirt_Cu]
else:
    # TODO...
    print('rearranging orbitals not implemented!')

if np.max(np.abs(C_ao_lo_tot.imag)) < 1e-8:
    C_ao_lo = C_ao_lo.real
    C_ao_lo_imp = C_ao_lo_imp.real
    C_ao_lo_tot = C_ao_lo_tot.real

############################################################
#           Plot MO and LO
############################################################
plot_orb = True

if plot_orb:
    plotdir = datadir + '/plot_' + cell_label + '_' + method_label + '_' + solver_label + '.chk'
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)

    plot.plot_orb_k_all(cell, plotdir + '/lo', C_ao_lo, kpts, margin=0.0)
    plot.plot_orb_k_all(cell, plotdir + '/mo', kmf.mo_coeff, kpts, margin=0.0)

############################################################
#           Quantities in LO (IAO) basis
############################################################

data_fname = datadir + '/data_' + cell_label + '_' + method_label + '_' + solver_label + '.h5'
fh = h5py.File(data_fname, 'w')


S_ao_ao = np.asarray(kmf.get_ovlp())
DM_ao = np.asarray(kmf.make_rdm1())
hcore_ao = np.asarray(kmf.get_hcore())
JK_ao = np.asarray(kmf.get_veff())

fh['S_ao_ao'] = S_ao_ao
fh['DM_ao'] = DM_ao
fh['hcore_ao'] = hcore_ao
fh['JK_ao'] = JK_ao

fh['C_ao_lo'] = C_ao_lo
fh['C_ao_lo_imp'] = C_ao_lo_imp
fh['C_ao_lo_tot'] = C_ao_lo_tot

# add an additional axis for convenience (but this will not be stored!)
if len(DM_ao.shape) == 3:
    DM_ao = DM_ao[np.newaxis,...]

# density matrix in LO basis
DM_lo     = np.zeros((spin,nkpts,nval+nvirt,nval+nvirt), dtype=DM_ao.dtype)
DM_lo_imp = np.zeros((spin,nkpts,nao_imp   ,nao_imp   ), dtype=DM_ao.dtype)
DM_lo_tot = np.zeros((spin,nkpts,nao       ,nao       ), dtype=DM_ao.dtype)

for s in range(spin):
    for ik in range(nkpts):
        # C^\dagger*S*C=I
        Cinv = C_ao_lo[s,ik].T.conj() @ S_ao_ao[ik]
        DM_lo[s,ik] = Cinv @ DM_ao[s,ik] @ Cinv.T.conj()
    
        Cinv_imp = C_ao_lo_imp[s,ik].T.conj() @ S_ao_ao[ik]
        DM_lo_imp[s,ik] = Cinv_imp @ DM_ao[s,ik] @ Cinv_imp.T.conj()

        Cinv_tot = C_ao_lo_tot[s,ik].T.conj() @ S_ao_ao[ik]
        DM_lo_tot = Cinv_tot @ DM_ao[s,ik] @ Cinv_tot.T.conj()

fh['DM_lo'] = DM_lo
fh['DM_lo_imp'] = DM_lo_imp
fh['DM_lo_tot'] = DM_lo_tot

##########################
# sanity check starts
##########################
nelec_imp_tmp = 0
for s in range(spin):
    nelec_lo = np.trace(DM_lo[s].sum(axis=0)/nkpts)
    print ('Nelec (core excluded)', nelec_lo.real)
    
    nelec_lo_imp = np.trace(DM_lo_imp[s].sum(axis=0)/nkpts)
    print ('Nelec on Imp (core excluded)', nelec_lo_imp.real)

    nelec_imp_tmp += nelec_lo_imp

print('total number of electrons on Imp = ', nelec_imp_tmp)
##########################
# sanity check ends
##########################

# get 4-index ERI
#eri_lo = eri_transform.get_unit_eri_fast(cell, gdf, C_ao_lo=C_ao_lo, feri=gdf_fname)
eri_lo_imp = eri_transform.get_unit_eri_fast(cell, gdf, C_ao_lo=C_ao_lo_imp, feri=gdf_fname)

fh['eri_lo_imp'] = eri_lo_imp.real

assert(np.max(np.abs(eri_lo_imp.imag))<1e-8)


# get hcore & JK in LO basis
if len(JK_ao.shape) == 3:
    JK_ao = JK_ao[np.newaxis, ...]

hcore_lo     = np.zeros((spin,nkpts,nval+nvirt,nval+nvirt),dtype=hcore_ao.dtype)
hcore_lo_imp = np.zeros((spin,nkpts,nao_imp   ,nao_imp   ),dtype=hcore_ao.dtype)

JK_lo     = np.zeros((spin,nkpts,nval+nvirt,nval+nvirt),dtype=JK_ao.dtype)
JK_lo_imp = np.zeros((spin,nkpts,nao_imp   ,nao_imp   ),dtype=JK_ao.dtype)

# let C be AO-to-LO transformation matrix
# h^{LO} = \dg{C} h^{AO} C
for s in range(spin):
    for ik in range(nkpts):
        hcore_lo[s,ik] = C_ao_lo[s,ik].T.conj() @ hcore_ao[ik] @ C_ao_lo[s,ik]
        hcore_lo_imp[s,ik] = C_ao_lo_imp[s,ik].T.conj() @ hcore_ao[ik] @ C_ao_lo_imp[s,ik]
    
        JK_lo[s,ik] = C_ao_lo[s,ik].T.conj() @ JK_ao[s,ik] @ C_ao_lo[s,ik]
        JK_lo_imp[s,ik] = C_ao_lo_imp[s,ik].T.conj() @ JK_ao[s,ik] @ C_ao_lo_imp[s,ik]

fh['hcore_lo'] = hcore_lo
fh['hcore_lo_imp'] = hcore_lo_imp
fh['JK_lo'] = JK_lo
fh['JK_lo_imp'] = JK_lo_imp

# if using DFT, get HF JK term using DFT density
if use_dft:

    if do_restricted:
        kmf_hf = scf.KRHF(cell, kpts, exxdiv='ewald')
    else:
        kmf_hf = scf.KUHF(cell, kpts, exxdiv='ewald')
    kmf_hf.with_df = gdf
    kmf_hf.max_cycle = 0

    if do_restricted:
        JK_ao_hf = kmf_hf.get_veff(dm_kpts=DM_ao[0])
        JK_ao_hf = JK_ao_hf[np.newaxis,...]
    else:
        JK_ao_hf = kmf_hf.get_veff(dm_kpts=DM_ao)
    
    JK_lo_hf     = np.zeros((spin,nkpts,nval+nvirt,nval+nvirt),dtype=JK_lo.dtype)
    JK_lo_hf_imp = np.zeros((spin,nkpts,nao_imp   ,nao_imp   ),dtype=JK_lo.dtype)

    for s in range(spin):
        for ik in range(nkpts):
            JK_lo_hf[s,ik] = C_ao_lo[s,ik].T.conj() @ JK_ao_hf[s,ik] @ C_ao_lo[s,ik]
            JK_lo_hf_imp[s,ik] = C_ao_lo_imp[s,ik].T.conj() @ JK_ao_hf[s,ik] @ C_ao_lo_imp[s,ik]
    
    fh['JK_lo_hf'] = JK_lo_hf
    fh['JK_lo_hf_imp'] = JK_lo_hf_imp

fh.close()

#*********************************** 
#           data summary
#*********************************** 

#--------- data in AO basis --------
# S_ao_ao
# DM_ao
# hcore_ao, JK_ao

#--------- AO-to-LO transformation matrix --------
# C_ao_lo, C_ao_lo_imp, C_ao_lo_tot

#--------- data in LO basis ----------
# DM_lo, DM_lo_imp, DM_lo_tot
# eri_lo_imp
# hcore_lo, hcore_lo_imp
# JK_lo, JK_lo_imp

#--------- HF JK with DFT density ----------
# JK_lo_hf, JK_lo_hf_imp

#######################################
# sanity check: matrix size in data
#######################################
fh = h5py.File(data_fname, 'r')

print('S_ao_ao.shape = ', np.asarray(fh['S_ao_ao']).shape)
print('DM_ao.shape = ', np.asarray(fh['DM_ao']).shape)
print('hcore_ao.shape = ', np.asarray(fh['hcore_ao']).shape)
print('JK_ao.shape = ', np.asarray(fh['JK_ao']).shape)

print('C_ao_lo.shape = ', np.asarray(fh['C_ao_lo']).shape)
print('C_ao_lo_imp.shape = ', np.asarray(fh['C_ao_lo_imp']).shape)
print('C_ao_lo_tot.shape = ', np.asarray(fh['C_ao_lo_tot']).shape)

print('DM_lo.shape = ', np.asarray(fh['DM_lo']).shape)
print('DM_lo_imp.shape = ', np.asarray(fh['DM_lo_imp']).shape)
print('DM_lo_tot.shape = ', np.asarray(fh['DM_lo_tot']).shape)

print('eri_lo_imp.shape = ', np.asarray(fh['eri_lo_imp']).shape)

print('hcore_lo.shape = ', np.asarray(fh['hcore_lo']).shape)
print('hcore_lo_imp.shape = ', np.asarray(fh['hcore_lo_imp']).shape)
print('JK_lo.shape = ', np.asarray(fh['JK_lo']).shape)
print('JK_lo_imp.shape = ', np.asarray(fh['JK_lo_imp']).shape)

print('JK_lo_hf.shape = ', np.asarray(fh['JK_lo_hf']).shape)
print('JK_lo_hf_imp.shape = ', np.asarray(fh['JK_lo_hf_imp']).shape)

fh.close()




