############################################################
# This script performs a Gamma point mean-field calculation
# for a 1-D copper atomic chain
############################################################

from pyscf.pbc import gto, scf, df
from pyscf.pbc.lib import chkfile
import os, h5py, argparse, copy
import numpy as np

from libdmet_solid.system import lattice
from libdmet.utils import plot
from libdmet_solid.basis_transform import make_basis
from libdmet_solid.lo.iao import reference_mol

############################################################
#           directory for saving/loading data
############################################################
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default = 'data', type = str)

args = parser.parse_args()

if args.datadir is None:
    datadir = 'Cu_mf_data/'
else:
    datadir = args.datadir + '/'

print('data directory:', datadir)


############################################################
#                       basis
############################################################
Cu_basis = 'def2-svp-bracket'
ncore_Cu = 9
nval_Cu = 6
nvirt_Cu = 9
nao_Cu = ncore_Cu + nval_Cu + nvirt_Cu


############################################################
#                   build cell
############################################################
# total number of Cu atoms
nat = 16
assert(nat%2 == 0)


cell_fname = datadir + '/cell_Cu_' + str(nat).zfill(2) + '.chk'

if os.path.isfile(cell_fname):
    cell = chkfile.load_cell(cell_fname)
else:
    # atomic spacing
    a = 2.55
    
    cell = gto.Cell()
    
    cell.unit = 'angstrom'
    cell.verbose = 4
    cell.max_memory = 30000
    cell.dimension = 3
    cell.ke_cutoff = 500
    cell.a = [[nat*a,0,0], [0,20,0], [0,0,20]]
    
    for iat in range(0, nat):
        cell.atom.append(['Cu', (iat*a, 0, 0)])
    
    cell.spin = 0
    cell.basis = Cu_basis
    
    cell.build()
    
    # save cell
    chkfile.save_cell(cell, cell_fname)


############################################################
#               build Lattice object
############################################################
kmesh = [1,1,1]
kpts = cell.make_kpts(kmesh)
Lat = lattice.Lattice(cell, kmesh) 
nao = Lat.nao
nkpts = Lat.nkpts

label = 'Cu_' + str(nat).zfill(2) + '_' + str(kmesh[0]) + str(kmesh[1])+ str(kmesh[2])

############################################################
#               density fitting
############################################################
gdf_fname = datadir + '/cderi_' + label + '.h5'
gdf = df.GDF(cell, kpts)

if os.path.isfile(gdf_fname):
    gdf._cderi = gdf_fname
else:
    gdf._cderi_to_save = gdf_fname
    gdf.auxbasis = Cu_basis + '-ri'
    gdf.build()


############################################################
#           mean-field calculation
############################################################
# if False, use HF instead
use_pbe = True

# if False, the scf will use the Newton solver for convergence
use_smearing = False
smearing_sigma = 0.05


if use_pbe:
    kmf = scf.KRKS(cell, kpts).density_fit()
    kmf.xc = 'pbe'
    mf_fname = datadir + '/ks_' + label + '.chk'
else:
    kmf = scf.KRHF(cell, kpts).density_fit()
    mf_fname = datadir + '/hf_' + label + '.chk'

kmf.with_df = gdf

if use_smearing:
    kmf = scf.addons.smearing_(kmf, sigma = smearing_sigma, method = 'fermi')
else:
    kmf = kmf.newton()

if os.path.isfile(mf_fname):
    kmf.__dict__.update( chkfile.load(mf_fname, 'scf') )
else:
    kmf.conv_tol = 1e-12
    kmf.max_cycle = 200
    kmf.chkfile = mf_fname
    kmf.kernel()

print(kmf.mo_energy)
print(np.asarray(kmf.mo_energy)[0,29*8-1])
exit()

############################################################
#           Orbital Localization
############################################################

MINAO = Cu_basis + '-minao'

# if True, use reference core/val
use_core_val = True

if use_core_val:

    # set IAO (core+val)
    pmol = reference_mol(cell, minao=MINAO)
    
    # set valence IAO
    pmol_val = pmol.copy()
    pmol_val.basis = Cu_basis + '-minao-val'
    pmol_val.build()
    basis_val = {}
    basis_val["Cu"] = copy.deepcopy(pmol_val._basis["Cu"])
    
    pmol_val = pmol.copy()
    pmol_val.basis = basis_val
    pmol_val.build()
    
    val_labels = pmol_val.ao_labels()
    for i in range(len(val_labels)):
        val_labels[i] = val_labels[i].replace("Cu 1s", "Cu 4s")
    pmol_val.ao_labels = lambda *args: val_labels
    
    # set core IAO
    pmol_core = pmol.copy()
    pmol_core.basis = Cu_basis + '-minao-core'
    pmol_core.build()
    basis_core = {}
    basis_core["Cu"] = copy.deepcopy(pmol_core._basis["Cu"])
    
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
            pmol_val=pmol_val, pmol_core=pmol_core, tol=1e-9)
else:
    # if core/val is not specified, the number of val will equal the size of MINAO
    # and the rest are virt
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=MINAO, full_return=True)

############################################################
#               rearrange orbitals
############################################################
# rearrange IAO/PAO according to their positions

if use_core_val:
    # C_ao_lo: transformation matrix from AO to LO (IAO) basis
    # all : include core+val+virt for all atoms
    # nc  : include val+virt for all atoms
    C_ao_lo_all  = np.zeros((nkpts,nao,nao), dtype=complex)
    C_ao_lo_nc   = np.zeros((nkpts,nao,nval+nvirt), dtype=complex)

    for iat in range(nat):
        # core
        C_ao_lo_all[:,:,iat*nao_Cu:iat*nao_Cu+ncore_Cu] \
                = C_ao_iao_core[:,:,iat*ncore_Cu:(iat+1)*ncore_Cu]

        # val
        C_ao_lo_all[:,:,iat*nao_Cu+ncore_Cu:iat*nao_Cu+ncore_Cu+nval_Cu] \
                = C_ao_iao_val[:,:,iat*nval_Cu:(iat+1)*nval_Cu]
        C_ao_lo_nc[:,:,iat*(nval_Cu+nvirt_Cu):iat*(nval_Cu+nvirt_Cu)+nval_Cu] \
                = C_ao_iao_val[:,:,iat*nval_Cu:(iat+1)*nval_Cu]

        # virt
        C_ao_lo_all[:,:,iat*nao_Cu+ncore_Cu+nval_Cu:(iat+1)*nao_Cu] \
                = C_ao_iao_virt[:,:,iat*nvirt_Cu:(iat+1)*nvirt_Cu]
        C_ao_lo_nc[:,:,iat*(nval_Cu+nvirt_Cu)+nval_Cu:(iat+1)*(nval_Cu+nvirt_Cu)] \
                = C_ao_iao_virt[:,:,iat*nvirt_Cu:(iat+1)*nvirt_Cu]

else:
    # TBD...
    print('rearranging orbitals not implemented!')

############################################################
#               Plot MO and LO
############################################################
plot_orb = True

if plot_orb:
    plotdir = datadir + '/plot_' + label + '/'
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)

    plot.plot_orb_k_all(cell, plotdir + '/iao_' + str(nat).zfill(2), C_ao_iao, kpts, margin=0.0)
    plot.plot_orb_k_all(cell, plotdir + '/iao_all_' + str(nat).zfill(2), C_ao_lo_all, kpts, margin=0.0)
    plot.plot_orb_k_all(cell, plotdir + '/iao_nc_' + str(nat).zfill(2), C_ao_lo_nc, kpts, margin=0.0)

############################################################
#           Quantities in LO (IAO) basis
############################################################
hcore_ao = np.asarray(kmf.get_hcore())
JK_ao = np.asarray(kmf.get_veff())
F_ao = np.asarray(kmf.get_fock())

hcore_lo_all = np.zeros((nkpts,nao,nao), dtype=hcore_ao.dtype)
hcore_lo_nc = np.zeros((nkpts,nval+nvirt,nval+nvirt), dtype=hcore_ao.dtype)
JK_lo_all = np.zeros((nkpts,nao,nao), dtype=JK_ao.dtype)
JK_lo_nc = np.zeros((nkpts,nval+nvirt,nval+nvirt), dtype=JK_ao.dtype)
F_lo_all = np.zeros((nkpts,nao,nao), dtype=F_ao.dtype)
F_lo_nc = np.zeros((nkpts,nval+nvirt,nval+nvirt), dtype=F_ao.dtype)

for ik in range(0,nkpts):
    hcore_lo_all[ik] = np.dot(np.dot(C_ao_lo_all[ik].T.conj(), hcore_ao[ik]), C_ao_lo_all[ik])
    JK_lo_all[ik] = np.dot(np.dot(C_ao_lo_all[ik].T.conj(), JK_ao[ik]), C_ao_lo_all[ik])
    F_lo_all[ik] = np.dot(np.dot(C_ao_lo_all[ik].T.conj(), F_ao[ik]), C_ao_lo_all[ik])

    hcore_lo_nc[ik] = np.dot(np.dot(C_ao_lo_nc[ik].T.conj(), hcore_ao[ik]), C_ao_lo_nc[ik])
    JK_lo_nc[ik] = np.dot(np.dot(C_ao_lo_nc[ik].T.conj(), JK_ao[ik]), C_ao_lo_nc[ik])
    F_lo_nc[ik] = np.dot(np.dot(C_ao_lo_nc[ik].T.conj(), F_ao[ik]), C_ao_lo_nc[ik])

if use_pbe:
    fn = datadir + '/ks_ao_' + label + '.h5'
else:
    fn = datadir + '/hf_ao_' + label + '.h5'

feri = h5py.File(fn, 'w')
feri['F_ao'] = F_ao
feri['hcore_lo_all'] = hcore_lo_all
feri['JK_lo_all'] = JK_lo_all
feri['F_lo_all'] = F_lo_all
feri['hcore_lo_nc'] = hcore_lo_nc
feri['JK_lo_nc'] = JK_lo_nc
feri['F_lo_nc'] = F_lo_nc
feri.close()


