from pyscf.pbc import gto, scf, df    
    

# spacing between gold atoms
a = 2.9

for nat in range(2,22,2):

    cell = gto.Cell()    

    for iat in range(0, nat):
        cell.atom.append(['Au', (iat*a, 0, 0)])

    cell.exp_to_discard = 0.1
    cell.spin = cell.nelectron % 2

    cell.build(    
            unit = 'angstrom',    
            a = [[a*nat,0,0],[0,1,0],[0,0,1]],    
            dimension = 1,    
            basis = 'def2-svp',    
            ecp = 'def2-svp',
            verbose=4
            )    
        
    # number of k points
    nks = 2
    kpts = cell.make_kpts([nks,1,1], scaled_center=[0,0,0])
    print('k points = ', kpts)
    
    # unrestricted
    mf = scf.KUHF(cell).density_fit()    

    mf.kpts = kpts
    mf.chkfile = 'uhf_'+str(nat)+'_'+str(nks)+'.chk'
    mf.with_df._cderi_to_save = 'uhf_cderi_'+str(nat)+'_'+str(nks)+'.h5'
    
    e = mf.kernel()    
    print('uhf energy per atom = ', e/nat)    

    # restricted
    mf = scf.KRHF(cell).density_fit()    

    mf.kpts = kpts
    mf.chkfile = 'rhf_'+str(nat)+'_'+str(nks)+'.chk'
    mf.with_df._cderi_to_save = 'rhf_cderi_'+str(nat)+'_'+str(nks)+'.h5'
    
    e = mf.kernel()    
    print('rhf energy per atom = ', e/nat)    
