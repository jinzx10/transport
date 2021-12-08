from pyscf.pbc import gto, scf, df    

# spacing between gold atoms
a = 2.9

# spacing between Cu and left lead
l = 2.5

# spacing between Cu and right lead
r = 2.5

# total number of atoms in the lead (left+right)
for nat in range(4, 21):
    nl = nat // 2 # number of atoms in the left lead
    nr = nat - nl

    # build chain
    cell = gto.Cell()    
    for iat in range(0, nl):
        cell.atom.append(['Au', (iat*a, 0, 0)])
    cell.atom.append(['Cu', ((nl-1)*a+l, 0, 0)])
    for iat in range(0, nr):
        cell.atom.append(['Au', ((nl-1)*a+l+r+iat*a, 0, 0)])
    
    cell.exp_to_discard = 0.1
    cell.spin = (nat+1) % 2

    cell.build(    
            unit = 'angstrom',    
            a = [[l+r+(nl+nr-1)*a,0,0],[0,1,0],[0,0,1]],    
            dimension = 1,    
            basis = {'Au':'def2-svp', 'Cu':'def2-svp'},
            ecp = {'Au':'def2-svp'},
            verbose = 4,
            precision=1e-6
            ) 

    # Gamma point calculation

    # unrestricted
    mf = scf.UHF(cell).density_fit()    
    mf.chkfile = 'uhf_'+str(nat)+'.chk'
    mf.with_df._cderi_to_save = 'uhf_cderi_'+str(nat)+'.h5'
    
    e = mf.kernel()

    # restricted
    if nat % 2 == 1:
        mf = scf.RHF(cell).density_fit()    
        mf.chkfile = 'rhf_'+str(nat)+'.chk'
        mf.with_df._cderi_to_save = 'rhf_cderi_'+str(nat)+'.h5'
        
        e = mf.kernel()
        


