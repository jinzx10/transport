from pyscf.pbc import gto, scf, df    
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--savedir')
args = parser.parse_args()

savedir = args.savedir
if savedir is None:
    savedir = 'data'

print('data will be saved to ', savedir)


# spacing between gold atoms
a = 2.9

for nat in range(10,22,2):

    cell = gto.Cell()    

    for iat in range(0, nat):
        cell.atom.append(['Au', (iat*a, 0, 0)])

    cell.exp_to_discard = 0.1
    #cell.spin = cell.nelectron % 2
    cell.ke_cutoff = 200
    cell.max_memory = 100000

    cell.build(    
            unit = 'angstrom',    
            a = [[a*nat,0,0],[0,1,0],[0,0,1]],    
            dimension = 1,    
            basis = 'def2-svp',    
            ecp = 'def2-svp',
            verbose=4,
            )    
        
    # auxbasis for density fitting
    ab = df.aug_etb(cell, beta=2.0)

    # number of k points
    nks = 1
    kpts = cell.make_kpts([nks,1,1], scaled_center=[0,0,0])
    
    #================ unrestricted ================
    mf = scf.KUHF(cell).density_fit(auxbasis=ab)
    mf.kpts = kpts

    # data file
    mf.chkfile = savedir + '/uhf_'+str(nat).zfill(2)+'_'+str(nks).zfill(2)+'.chk'
    mf.with_df._cderi_to_save = savedir + '/uhf_cderi_'+str(nat).zfill(2)+'_'+str(nks).zfill(2)+'.h5'
    
    # scf solver
    mf = mf.newton()
    mf.max_cycle = 500

    # initial guess
    ig = mf.get_init_guess()
    ig[1,:,:,:] = 0

    e = mf.kernel(dm0=ig)    

    print('uhf energy per atom = ', e/nat)    

    #================ restricted ================
    mf = scf.KRHF(cell).density_fit(auxbasis=ab)    
    mf.kpts = kpts

    # data file
    mf.chkfile = savedir + '/rhf_'+str(nat).zfill(2)+'_'+str(nks).zfill(2)+'.chk'
    mf.with_df._cderi_to_save = savedir + '/rhf_cderi_'+str(nat).zfill(2)+'_'+str(nks).zfill(2)+'.h5'
    
    # scf solver
    mf = mf.newton()
    mf.max_cycle = 500

    e = mf.kernel()    

    print('rhf energy per atom = ', e/nat)    


