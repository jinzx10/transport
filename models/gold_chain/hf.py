from pyscf.pbc import gto, scf, df    
    

# spacing between gold atoms
a = 2.9

# number of atoms in a unit cell
nat = 2

# number of k points
nks = 2


cell = gto.Cell()    
for iat in range(0, nat):
    cell.atom.append(['Au', (iat*a, 0, 0)])
cell.exp_to_discard = 0.1
cell.build(    
        unit = 'angstrom',    
        a = [[a*nat,0,0],[0,1,0],[0,0,1]],    
        dimension = 1,    
        basis = 'def2-svp',    
        ecp = 'def2-svp'     
        )    
    
mf = scf.KRHF(cell).density_fit()    
mf.kpts = cell.make_kpts([nks,1,1], scaled_center=[0,0,0])    
chkfname = 'gold_chain_'+str(nat)+'_'+str(nks)+'.chk'
mf.chkfile = chkfname

e = mf.kernel()    
print('energy per atom = ', e/nat)    


