from pyscf.pbc.lib import chkfile
from pyscf.pbc import gto, scf
from libdmet_solid.system import lattice
from libdmet.utils import plot
from libdmet_solid.basis_transform import make_basis
import sys

datadir = 'data-' + sys.argv[1]
plotdir = datadir + '/plot-' + sys.argv[2]

nat_start = 5
nat_end = 19
nat_step = 2
nat_range = range(nat_start, nat_end+1, nat_step)

for nat in nat_range:

    cell = chkfile.load_cell(datadir + '/cu_au_' + str(nat).zfill(2) + '.chk')

    kmesh = [1,1,1]
    kpts = cell.make_kpts(kmesh)


    kmf = scf.KRHF(cell).density_fit()

    kmf.kpts = kpts
    kmf.with_df._cderi = datadir + '/cderi_' + str(nat).zfill(2) + '.h5'
    kmf_data = chkfile.load(datadir + '/rhf_' + str(nat).zfill(2) + '.chk', 'scf')
    kmf.__dict__.update(kmf_data)

    MINAO ='def2-svp'

    # plot raw HF orbitals
    plot.plot_orb_k_all(cell, plotdir + '/raw_' + str(nat).zfill(2), kmf.mo_coeff, kpts, margin=0.0)

    # IAO
    Lat = lattice.Lattice(cell, kmesh) 
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=MINAO, full_return=True)

    plot.plot_orb_k_all(cell, plotdir + '/iao_' + str(nat).zfill(2), C_ao_iao, kpts, margin=0.0)

    print(nat, 'done')
