from pyscf.pbc import df, scf, gto
from pyscf.pbc.lib import chkfile
from libdmet_solid.system import lattice
from libdmet_solid.basis_transform import make_basis

a = 2.9
nat = 2

cell = gto.Cell()    

for iat in range(0, nat):
    cell.atom.append(['Au', (iat*a, 0, 0)])

cell.exp_to_discard = 0.1
#cell.spin = nat % 2
cell.ke_cutoff = 200
cell.max_memory = 100000

cell.build(    
        unit = 'angstrom',    
        a = [[a*nat,0,0],[0,1,0],[0,0,1]],    
        dimension = 1,    
        basis = 'def2-svp',
        ecp = 'def2-svp',
        verbose = 4
        )    


nks = 12
kmesh = [nks,1,1]

cderi_fname = 'data-20211210181649/rhf_cderi_02_' + str(nks).zfill(2) + '.h5'
chk_fname = 'data-20211210181649/rhf_02_' + str(nks).zfill(2) + '.chk'

# Lattice object
Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts
nao = Lat.nao
nkpts = Lat.nkpts


# mean field object
ab = df.aug_etb(cell, beta=2.0)
kmf = scf.KRHF(cell).density_fit(auxbasis=ab)    
kmf.with_df._cderi = cderi_fname
data = chkfile.load(chk_fname, 'scf')
kmf.__dict__.update(data)

MINAO = {'Au':'def2-svp-minao'}
#MINAO = {'Au':'def2-svp'}
C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=MINAO, full_return=True)


