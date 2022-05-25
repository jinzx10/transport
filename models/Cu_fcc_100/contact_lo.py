import numpy as np
import os, h5py, argparse, copy

from pyscf.pbc import gto, df, scf
from pyscf.pbc.lib import chkfile

from libdmet_solid.system import lattice
from libdmet.utils import plot
from libdmet_solid.basis_transform import make_basis
from libdmet_solid.lo.iao import reference_mol

############################################################
#                   data directory
############################################################
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default = 'data', type = str)
args = parser.parse_args()

datadir = args.datadir + '/'
print('data will be read from', datadir)

############################################################
#                   system & method info
############################################################
imp_atom = 'IMP_ATOM'
imp_basis = 'def2-svp'
Cu_basis = 'def2-svp-bracket'

# if False, use HF instead
use_dft = USE_DFT

# if use_dft
xcfun = 'XCFUN'

do_restricted = DO_RESTRICTED

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
use_smearing = USE_SMEARING
smearing_sigma = SMEARING_SIGMA

if use_smearing:
    solver_label = 'smearing' + str(smearing_sigma)
else:
    solver_label = 'newton'
# x-distance between impurity atom and nearest Cu plane
l = LEFT
r = RIGHT

# Cu lattice constant (for the fcc cell)
a = LATCONST

# number of AOs
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
#                       load cell
############################################################
cell = chkfile.load_cell(cell_fname)
print('use saved cell file:', cell_fname)

nat_Cu = len(cell.atom) - 1

############################################################
#               load density fitting
############################################################
gdf_fname = datadir + '/cderi_' + cell_label + '.h5'
gdf = df.GDF(cell)

gdf._cderi = gdf_fname
print('use saved gdf cderi:', gdf_fname)

############################################################
#               load mean-field data
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
    mf.diis_space = 15
else:
    mf = mf.newton()

print('load previous mf data', mf_fname)
mf.__dict__.update( chkfile.load(mf_fname, 'scf') )
mf.conv_tol = 1e-10
mf.max_cycle = 50
mf.kernel()

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

# make sure restricted/unrestricted quantities have the same dimension
# and set spin: unrestricted -> 2; restricted -> 1
if len(mo_energy.shape) == 2:
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
    C_ao_lo    = np.zeros((spin,nkpts,nao,nval+nvirt), dtype=C_ao_iao.dtype)
    C_ao_lo_imp = np.zeros((spin,nkpts,nao,nao_imp), dtype=C_ao_lo.dtype)
    C_ao_lo_tot = np.zeros((spin,nkpts,nao,nao), dtype=C_ao_lo.dtype)
    
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
plot_orb = False

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
DM_lo_imp  = np.zeros((spin,nkpts,nao_imp    ,nao_imp    ), dtype=DM_ao.dtype)
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

hcore_lo    = np.zeros((spin,nkpts,nval+nvirt,nval+nvirt),dtype=hcore_ao.dtype)
hcore_lo_imp = np.zeros((spin,nkpts,nao_imp    ,nao_imp    ),dtype=hcore_ao.dtype)

JK_lo    = np.zeros((spin,nkpts,nval+nvirt,nval+nvirt),dtype=JK_ao.dtype)
JK_lo_imp = np.zeros((spin,nkpts,nao_imp    ,nao_imp    ),dtype=JK_ao.dtype)

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
    
    JK_lo_hf    = np.zeros((spin,nkpts,nval+nvirt,nval+nvirt),dtype=JK_lo.dtype)
    JK_lo_hf_imp = np.zeros((spin,nkpts,nao_imp    ,nao_imp    ),dtype=JK_lo.dtype)

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



