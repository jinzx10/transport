import numpy as np
import os, h5py, argparse, copy

from pyscf.pbc import gto, df, scf
from pyscf.pbc.lib import chkfile

from libdmet_solid.system import lattice
from libdmet.utils import plot
from libdmet_solid.basis_transform import make_basis, eri_transform
from libdmet_solid.lo.iao import reference_mol

import matplotlib.pyplot as plt

# switch to 'production' for serious jobs
mode = 'MODE'

############################################################
#                       basis
############################################################
imp_atom = 'IMP_ATOM' if mode == 'production' else 'Co'

imp_basis = 'def2-svp'
Cu_basis = 'def2-svp-bracket'

############################################################
#           directory for saving/loading data
############################################################
datadir = 'DATADIR/' if mode == 'production' else 'Co/'
print('data will be saved to:', datadir)

if not os.path.exists(datadir):
    os.mkdir(datadir)

############################################################
#                       build cell
############################################################
# transport direction fcc 100 plane
# an example of nl=2, nr=3
# number of atoms |     5      4     1      4      5       4      |   5
# x position      | -l-0.5a   -l   0(imp)   r    r+0.5a   r+a     | r+1.5a

# x-distance between impurity atom and nearest Cu plane
l = LEFT if mode == 'production' else 1.8
r = RIGHT if mode == 'production' else 1.8

# number of layers in the left/right lead included in contact SCF calculation
# make sure total number of electrons in a unit cell is even
nl = 2
nr = 3

# Cu lattice constant (for the fcc cell)
a = LATCONST if mode == 'production' else 3.6

cell_label = imp_atom + '_' + imp_basis + '_Cu_' + Cu_basis \
        + '_l' + str(l) + '_r' + str(r) + '_a' + str(a)

cell_fname = datadir + '/cell_' + cell_label + '.chk'

if os.path.isfile(cell_fname):
    cell = chkfile.load_cell(cell_fname)
    print('use saved cell file:', cell_fname)
else:
    cell = gto.Cell()
    
    cell.unit = 'angstrom'
    cell.verbose = 4
    cell.dimension = 3

    cell.precision = 1e-10

    # should not discard! 4s and 5s have small exponent
    #cell.exp_to_discard = 0.1

    cell.a = [[l+r+(nl+nr-1)*0.5*a,0,0], [0,20,0], [0,0,20]]
    
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

    cell.spin = 0
    cell.basis = {imp_atom : imp_basis, 'Cu' : Cu_basis}
    
    cell.build()
    
    # save cell
    chkfile.save_cell(cell, cell_fname)

nat_Cu = len(cell.atom) - 1

nelec = nat_Cu * 29

if imp_atom == 'Fe':
    nelec += 26
elif imp_atom == 'Co':
    nelec += 27
elif imp_atom == 'Ni':
    nelec += 28
else:
    print('do not know nelec of imp_atom')
    exit()

nao_imp = cell.aoslice_by_atom()[0][3]
print('number of impurity AO orbitals = ', nao_imp)

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

############################################################
#               Brillouin zone sampling
############################################################
k_label = 'kKMESH' if mode == 'production' else 'k311'
kpts = cell.make_kpts([int(k_label[1]), int(k_label[2]), int(k_label[3])])
nkpts = len(kpts)
print('nkpts = ', nkpts)
print('k points:', kpts)


############################################################
#               density fitting
############################################################
gdf_fname = datadir + '/cderi_' + cell_label + '_' + k_label + '.h5'

gdf = df.GDF(cell, kpts)

if os.path.isfile(gdf_fname):
    print('use saved gdf cderi:', gdf_fname)
    gdf._cderi = gdf_fname
else:
    gdf._cderi_to_save = gdf_fname
    gdf.auxbasis = {imp_atom : imp_basis + '-ri', 'Cu' : Cu_basis + '-ri'}
    gdf.build()
    #gdf.build(j_only=True) # for LDA & GGA DFT but eri transform requires full df!


############################################################
#                   mean-field method
############################################################
# use HF veff from KS-converged DM for HF+DMFT
xcfun = 'XCFUN' if mode == 'production' else 'pbe'
method_label = 'rks_' + xcfun
mf = scf.KRKS(cell, kpts).density_fit() # pbc version
mf.with_df = gdf
mf.max_cycle = 300
mf.conv_tol = 1e-10


############################################################
#                   gate voltage
############################################################
gate = GATE if mode == 'production' else 0
gate_label = 'gate%+5.2f'%(gate)

#************** full label *****************
labels = cell_label + '_' + k_label + '_' + method_label + '_' + gate_label
#*******************************************

# save hcore at gate == 0 for reference
ref_fname = datadir + '/ref_contact_' + labels + '.h5'
print('ref_fname = ', ref_fname)

if gate == 0 and not os.path.isfile(ref_fname):
    fh = h5py.File(ref_fname, 'w')
    print('computing hcore...')
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    fh['hcore'] = hcore
    fh['ovlp'] = ovlp
    fh.close()

# read reference hcore if possible
if os.path.isfile(ref_fname):
    fh = h5py.File(ref_fname, 'r')
    print('loading hcore...')
    hcore = np.asarray(fh['hcore'])
    ovlp = np.asarray(fh['ovlp'])
    fh.close()
    
    # h -> h + \sum_{\mu,\nu \in imp}|\mu> inv(S_imp)_{\mu\nu} <\nu|
    for ik in range(nkpts):
        hcore[ik] += gate * ovlp[ik,:,0:nao_imp] @ np.linalg.solve(ovlp[ik,0:nao_imp, 0:nao_imp], ovlp[ik,0:nao_imp,:])
    
    mf.get_hcore = lambda *args: hcore
    mf.get_ovlp = lambda *args: ovlp

mf_save_fname = datadir + '/' + labels + '.chk'
mf.chkfile = mf_save_fname


############################################################
#           mean-field calculation
############################################################
mf = scf.addons.smearing_(mf, sigma=0.01, method="fermi")

mf_load_fname = 'MF_LOAD_FNAME' if mode == 'production' else ''

if mf_load_fname != '':
    mf_data = chkfile.load(mf_load_fname, 'scf')
    print('load saved mf data', mf_load_fname)
    mf.__dict__.update(mf_data)
    dm_init = mf.make_rdm1()
    mf.kernel(dm0=dm_init)
else:
    mf.kernel()


if not mf.converged:
    print('SCF fails to converge!')
    exit()


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
#                   Lattice object
############################################################
Lat = lattice.Lattice(cell, [k_label[1], k_label[2], k_label[3]])
nao = Lat.nao


############################################################
#   orbital localization (IAO) with reference core/val
############################################################
MINAO = {imp_atom: imp_basis + '-minao', 'Cu': Cu_basis + '-minao'}

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

############################################################
#               rearrange orbitals
############################################################
# rearrange IAO/PAO according to their indices in cell.atom

# C_ao_lo: transformation matrix from AO to LO (IAO) basis, all atoms, core orbitals are excluded
# C_ao_lo_imp: transformation matrix from AO to LO (IAO) basis, only imp atom, core orbitals are excluded
# C_ao_lo_tot: transformation matrix from AO to LO (IAO) basis, all atoms, all orbitals (core included!)

C_ao_lo     = np.zeros((nkpts,nao,nval+nvirt), dtype=C_ao_iao.dtype)
C_ao_lo_imp = np.zeros((nkpts,nao,nao_imp   ), dtype=C_ao_iao.dtype)
C_ao_lo_tot = np.zeros((nkpts,nao,nao       ), dtype=C_ao_iao.dtype)

C_ao_lo[:,:,       0:nval_imp] = C_ao_iao_val [:,:,0:nval_imp]
C_ao_lo[:,:,nval_imp:nao_imp]  = C_ao_iao_virt[:,:,0:nvirt_imp]

for iat in range(nat_Cu):
    C_ao_lo[:,:,nao_imp+iat*nao_Cu:nao_imp+iat*nao_Cu+nval_Cu] \
            = C_ao_iao_val[:,:,nval_imp+iat*nval_Cu:nval_imp+(iat+1)*nval_Cu]

    C_ao_lo[:,:,nao_imp+iat*nao_Cu+nval_Cu:nao_imp+(iat+1)*nao_Cu] \
            = C_ao_iao_virt[:,:,nvirt_imp+iat*nvirt_Cu:nvirt_imp+(iat+1)*nvirt_Cu]

C_ao_lo_imp = C_ao_lo[:,:,0:nao_imp]

#----------------------------------------------------------

C_ao_lo_tot[:,:,                 0:ncore_imp         ] = C_ao_iao_core[:,:,0:ncore_imp]
C_ao_lo_tot[:,:,ncore_imp         :ncore_imp+nval_imp] = C_ao_iao_val [:,:,0:nval_imp]
C_ao_lo_tot[:,:,ncore_imp+nval_imp:nao_imp_tot       ] = C_ao_iao_virt[:,:,0:nvirt_imp]

for iat in range(nat_Cu):
    C_ao_lo_tot[:,:,nao_imp_tot+iat*nao_Cu_tot:nao_imp_tot+iat*nao_Cu_tot+ncore_Cu] \
            = C_ao_iao_core[:,:,ncore_imp+iat*ncore_Cu:ncore_imp+(iat+1)*ncore_Cu]

    C_ao_lo_tot[:,:,nao_imp_tot+iat*nao_Cu_tot+ncore_Cu:nao_imp_tot+iat*nao_Cu_tot+ncore_Cu+nval_Cu] \
            = C_ao_iao_val[:,:,nval_imp+iat*nval_Cu:nval_imp+(iat+1)*nval_Cu]

    C_ao_lo_tot[:,:,nao_imp_tot+iat*nao_Cu_tot+ncore_Cu+nval_Cu:nao_imp_tot+(iat+1)*nao_Cu_tot] \
            = C_ao_iao_virt[:,:,nvirt_imp+iat*nvirt_Cu:nvirt_imp+(iat+1)*nvirt_Cu]


############################################################
#           Plot imp LO
############################################################
plot_lo = PLOT_LO if mode == 'production' else False

if plot_lo:
    plotdir = datadir + '/plot_' + labels
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)

    plot.plot_orb_k_all(cell, plotdir + '/lo', C_ao_lo_imp, kpts, margin=0.0)

############################################################
#           Quantities in LO (IAO) basis
############################################################
data_fname = datadir + '/data_contact_' + labels + '.h5'
fh = h5py.File(data_fname, 'w')

S_ao_ao  = np.asarray(kmf.get_ovlp())
DM_ao    = np.asarray(kmf.make_rdm1())
hcore_ao = np.asarray(kmf.get_hcore())
JK_ao_ks = np.asarray(kmf.get_veff())

fh['S_ao_ao']  = S_ao_ao
fh['DM_ao']    = DM_ao
fh['hcore_ao'] = hcore_ao
fh['JK_ao_ks'] = JK_ao_ks

fh['C_ao_lo']     = C_ao_lo
fh['C_ao_lo_imp'] = C_ao_lo_imp
fh['C_ao_lo_tot'] = C_ao_lo_tot

# density matrix in LO basis
DM_lo     = np.zeros((nkpts, nval+nvirt, nval+nvirt), dtype=DM_ao.dtype)
DM_lo_imp = np.zeros((nkpts, nao_imp   , nao_imp   ), dtype=DM_ao.dtype)
DM_lo_tot = np.zeros((nkpts, nao       , nao       ), dtype=DM_ao.dtype)

for ik in range(nkpts):
    # C^\dagger*S*C=I
    Cinv = C_ao_lo[ik].T.conj() @ S_ao_ao[ik]
    DM_lo[ik] = Cinv @ DM_ao[ik] @ Cinv.T.conj()

    Cinv_imp = C_ao_lo_imp[ik].T.conj() @ S_ao_ao[ik]
    DM_lo_imp[ik] = Cinv_imp @ DM_ao[ik] @ Cinv_imp.T.conj()

    Cinv_tot = C_ao_lo_tot[ik].T.conj() @ S_ao_ao[ik]
    DM_lo_tot = Cinv_tot @ DM_ao[ik] @ Cinv_tot.T.conj()

fh['DM_lo'] = DM_lo
fh['DM_lo_imp'] = DM_lo_imp
fh['DM_lo_tot'] = DM_lo_tot

#************************* sanity check starts *************************
nelec_lo = np.trace(DM_lo.sum(axis=0)/nkpts)
print ('Nelec (core excluded)', nelec_lo.real)

nelec_lo_imp = np.trace(DM_lo_imp.sum(axis=0)/nkpts)
print ('Nelec on Imp (core excluded)', nelec_lo_imp.real)

#************************* sanity check ends *************************

# 4-index ERI
eri_lo_imp = eri_transform.get_unit_eri_fast(cell, gdf, C_ao_lo=C_ao_lo_imp, feri=gdf_fname)
fh['eri_lo_imp'] = eri_lo_imp.real
assert(np.max(np.abs(eri_lo_imp.imag))<1e-8)

hcore_lo     = np.zeros((nkpts,nval+nvirt,nval+nvirt),dtype=hcore_ao.dtype)
hcore_lo_imp = np.zeros((nkpts,nao_imp   ,nao_imp   ),dtype=hcore_ao.dtype)

JK_lo_ks     = np.zeros((nkpts,nval+nvirt,nval+nvirt),dtype=JK_ao.dtype)
JK_lo_ks_imp = np.zeros((nkpts,nao_imp   ,nao_imp   ),dtype=JK_ao.dtype)

# let C be AO-to-LO transformation matrix
# h^{LO} = \dg{C} h^{AO} C
for ik in range(nkpts):
    hcore_lo    [ik] = C_ao_lo    [ik].T.conj() @ hcore_ao[ik] @ C_ao_lo    [ik]
    hcore_lo_imp[ik] = C_ao_lo_imp[ik].T.conj() @ hcore_ao[ik] @ C_ao_lo_imp[ik]

    JK_lo_ks    [ik] = C_ao_lo    [ik].T.conj() @ JK_ao[ik] @ C_ao_lo    [ik]
    JK_lo_ks_imp[ik] = C_ao_lo_imp[ik].T.conj() @ JK_ao[ik] @ C_ao_lo_imp[ik]

fh['hcore_lo']     = hcore_lo
fh['hcore_lo_imp'] = hcore_lo_imp
fh['JK_lo_ks']     = JK_lo_ks 
fh['JK_lo_ks_imp'] = JK_lo_ks_imp

# get HF JK term using DFT density
kmf_hf = scf.KRHF(cell, kpts, exxdiv='ewald')
kmf_hf.with_df = gdf
kmf_hf.max_cycle = 0
JK_ao_hf = kmf_hf.get_veff(dm_kpts=DM_ao)


JK_lo_hf     = np.zeros((nkpts,nval+nvirt,nval+nvirt),dtype=JK_lo.dtype)
JK_lo_hf_imp = np.zeros((nkpts,nao_imp   ,nao_imp   ),dtype=JK_lo.dtype)

for ik in range(nkpts):
    JK_lo_hf    [ik] = C_ao_lo    [ik].T.conj() @ JK_ao_hf[ik] @ C_ao_lo    [ik]
    JK_lo_hf_imp[ik] = C_ao_lo_imp[ik].T.conj() @ JK_ao_hf[ik] @ C_ao_lo_imp[ik]
    
fh['JK_lo_hf']     = JK_lo_hf
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
# JK_lo_ks, JK_lo_ks_imp

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
print('JK_lo_ks.shape = ', np.asarray(fh['JK_lo_ks']).shape)
print('JK_lo_ks_imp.shape = ', np.asarray(fh['JK_lo_ks_imp']).shape)

print('JK_lo_hf.shape = ', np.asarray(fh['JK_lo_hf']).shape)
print('JK_lo_hf_imp.shape = ', np.asarray(fh['JK_lo_hf_imp']).shape)

fh.close()

print('finished')




