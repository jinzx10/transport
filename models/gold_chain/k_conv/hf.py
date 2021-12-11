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

# number of atoms in a unit cell
# need to be an even number!
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

# auxbasis for density fitting
ab = df.aug_etb(cell, beta=2.0)

# number of k points
for nks in range(2, 21):

    kpts = cell.make_kpts([nks,1,1], scaled_center=[0,0,0])    
    print('k points = ', kpts)

    # unrestricted
    mf = scf.KUHF(cell).density_fit(auxbasis=ab)    

    mf.kpts = kpts
    mf.chkfile = savedir + '/' + 'uhf_'+str(nat).zfill(2)+'_'+str(nks).zfill(2)+'.chk'
    mf.with_df._cderi_to_save = savedir + '/' + 'uhf_cderi_'+str(nat).zfill(2)+'_'+str(nks).zfill(2)+'.h5'
    
    ig = mf.get_init_guess()
    ig[1,:,:,:] = 0 # spin-symmetry-broken initial guess

    mf.max_cycle = 500

    e = mf.kernel(dm0=ig)    
    print('uhf energy per atom = ', e/nat)    

    # restricted
    mf = scf.KRHF(cell).density_fit(auxbasis=ab)    

    mf.kpts = kpts
    mf.chkfile = savedir + '/' + 'rhf_'+str(nat).zfill(2)+'_'+str(nks).zfill(2)+'.chk'
    mf.with_df._cderi_to_save = savedir + '/' + 'rhf_cderi_'+str(nat).zfill(2)+'_'+str(nks).zfill(2)+'.h5'

    mf.max_cycle = 500
    
    e = mf.kernel()    
    print('rhf energy per atom = ', e/nat)    


