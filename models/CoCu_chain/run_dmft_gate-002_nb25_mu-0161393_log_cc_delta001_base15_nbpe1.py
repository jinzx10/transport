'''
Main routine to set up DMFT parameters and run DMFT
'''

try:
    import block2
    from block2.su2 import MPICommunicator
    dmrg_ = True
except:
    dmrg_ = False
    pass
import numpy as np
import scipy, os, h5py
from fcdmft.utils import write
#from fcdmft.dmft import gwdmft
import gwdmft
from mpi4py import MPI
from dmft_solver import mf_gf

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

def dmft_abinitio():
    '''
    List of DMFT parameters

    gw_dmft : choose to run GW+DMFT (True) or HF+DMFT (False)
    opt_mu : whether to optimize chemical potential during DMFT cycles
    solver_type : choose impurity solver ('cc', 'ucc', 'dmrg', 'dmrgsz', 'fci')
    disc_type : choose bath discretization method ('opt', 'direct', 'linear', 'gauss', 'log')
    max_memory : maximum memory for DMFT calculation (per MPI process for CC, per node for DMRG)
    dmft_max_cycle : maximum number of DMFT iterations (set to 0 for one-shot DMFT)
    chkfile : chkfile for saving DMFT self-consistent quantities (hyb and self-energy)
    diag_only : choose to only fit diagonal hybridization (optional)
    orb_fit : special orbitals (e.g. 3d/4f) with x5 weight in bath optimization (optional)
    delta : broadening for discretizing hybridization (often 0.02-0.1 Ha)
    nbath : number of bath energies (can be any integer)
    nb_per_e : number of bath orbitals per bath energy (should be no greater than nval-ncore)
                total bath number = nb_per_e * nbath
    mu : initial chemical potential
    nval : number of valence (plus core) impurity orbitals (only ncore:nval orbs coupled to bath)
    ncore : number of core impurity orbitals
    nfrz : number of freezing core orbitals in CCSD/DMRG
            (NOTE: using ncore or nfrz!=0 requires putting core AOs in the beginning of IAOs)
    nelectron : electron number per cell
    gmres_tol : GMRES/GCROTMK convergence criteria for solvers in production run (often 1e-3)
    wl0, wh0: (optional) real-axis frequency range [wl0+mu, wh0+mu] for bath discretization
                in DMFT self-consistent iterations (defualt: -0.4, 0.4)
    wl, wh : real-axis frequnecy range [wl, wh] for production run
    eta : spectral broadening for production run (often 0.1-0.4 eV)
    '''
    # DMFT self-consistent loop parameters
    gw_dmft = False

    # ZJ: when this option is turned on, mu is optimized during EACH DMFT cycle,
    # and for every optimization, get_rdm_imp is called multiple times until convergence.
    opt_mu = False

    # ZJ: for Kondo problems, 'log' (or a hybrid scheme that contains 
    # fine discretization near the Fermi level) might be necessary
    #disc_type = 'nonlin2'
    disc_type = 'log'

    solver_type = 'cc'
    dmft_max_cycle = 0
    max_memory = 30000
    chkfile = 'DMFT_chk.h5'
    diag_only = False

    # ZJ: the two options below relate to the 'opt' bath discretization
    # see opt_bath & opt_bath_v_only in gwdmft.py
    orb_fit = range(1,6)
    opt_init_method = 'log'

    # ZJ: small imaginary number in the retarded Green's function (inv(z-H), z=E+i*delta)
    # delta = 0.05 is a threshold, see gwdmft.py
    delta = 0.01

    #mu = 0.623759012177
    #mu = -0.1613936097992017 # Cu chain's HOMO
    mu = -0.161393 

    gate = -0.02

    #nbath = 49
    # number of bath energies (not orbitals)
    # total number of bath orbitals equals nbath*nb_per_e
    nbath = 25

    # base for log discretization
    log_disc_base = 1.5

    nb_per_e = 1

    # bath energy range with respect to mu
    # the bath will be discretized within wl0+mu, wh0+mu
    # hybridization is significant between -0.3 and 0.3
    #wl0 = -0.25
    #wh0 = 0.4
    wl0 = -0.15
    wh0 = 0.45

    ncore = 0
    nval = 6
    nfrz = 0

    #nelectron = 7.2494509284467625
    nelectron = 9

    # Kondo specific parameters
    mf_type = 'dft'
    imp_idx = [1, 3] # t2g, eg
    # choose particular ccgf orbs
    cc_ao_orbs = None
    #cc_ao_orbs = range(ncore, nval)
    #cc_ao_orbs = imp_idx

    # DMFT production run parameters
    Ha2eV = 27.211386
    wl = 16.2/Ha2eV
    wh = 17.6/Ha2eV
    eta = 0.1/Ha2eV
    gmres_tol = 1e-3

    '''
    specific parameters for CAS treatment of impurity problem:
        cas : use CASCI or not (default: False)
        casno : natural orbital method for CASCI
                (choices: 'gw': GW@HF, 'cc': CCSD)
        composite : whether to use GW or CCSD Green's function as the low-level GF
                    for impurity problem; if False, use HF Green's function as low-level GF
        thresh : float
            Threshold on NO occupation numbers. Default is 1e-4.
        nvir_act : int
            Number of virtual NOs to keep. Default is None. If present, overrides `thresh`.
        nocc_act : int
            Number of occupied NOs to keep. Default is None. If present, overrides `thresh` and `vno_only`.
        ea_cut : float
            Energy cutoff for determining number of EA charged density matrices.
        ip_cut : float
            Energy cutoff for determining number of IP charged density matrices.
        ea_no : int
            Number of negatively charged density matrices included for making NOs.
        ip_no : int
            Number of positively charged density matrices included for making NOs.
        vno_only : bool
            Only construct virtual natural orbitals. Default is True.
    '''
    cas = False
    casno = 'ci'
    composite = False
    thresh = 5e-3
    nvir_act = 11
    nocc_act = 6
    ea_cut = None
    ip_cut = None
    ea_no = None
    ip_no = None
    vno_only = False
    save_gf = True
    read_gf = False

    # specific parameters for DMRG solvers (see fcdmft/solver/gfdmrg.py for detailed comments)
    gs_n_steps = 20
    gf_n_steps = 6
    gs_tol = 1E-8
    gf_tol = 1E-3
    gs_bond_dims = [400] * 10 + [800] * 5 + [1500] * 5
    gs_noises = [1E-3] * 5 + [1E-4] * 5 + [1E-5] * 5 + [1E-7] * 2 + [0]
    gf_bond_dims = [200] * 2 + [500] * 4
    gf_noises = [1E-4] * 1 + [1E-5] * 1 + [1E-7] * 1 + [0]
    dmrg_gmres_tol = 1E-7
    dmrg_verbose = 3
    reorder_method = 'gaopt'
    n_off_diag_cg = -2
    extra_nw = 7
    extra_dw = 0.05/Ha2eV
    #extra_delta = 0.05/Ha2eV
    extra_delta = None

    dyn_corr_method = None
    nocc_act_low = 0
    nvir_act_high = 0

    load_dir = None
    save_dir = './gs_mps'
    load_mf = False
    save_mf = True
    use_gw = False
    run_imagfreq = False

    ### Finishing parameter settings ###

    datadir = 'Co_svp_Cu_svp_bracket_pbe2/'
    label = 'CoCu_09_111'

    # read imp hcore and DFT veff
    fn = datadir + '/hcore_JK_lo_dft_' + label + '.h5'
    feri = h5py.File(fn, 'r')
    hcore_k = np.asarray(feri['hcore_lo_ncCo'])
    JK_dft_k = np.asarray(feri['JK_lo_ncCo'])
    hcore_full = np.asarray(feri['hcore_lo_nc'])
    JK_dft_full = np.asarray(feri['JK_lo_nc'])
    feri.close()

    #<===============================
    # apply gate voltage
    nao_Co = hcore_k.shape[2]
    hcore_k[0,0] = hcore_k[0,0] + gate*np.eye(nao_Co)
    hcore_full[0,0,:nao_Co,:nao_Co] = hcore_full[0,0,:nao_Co,:nao_Co] + gate*np.eye(nao_Co)
    #===============================>

    # read imp HF-JK matrix
    fn = datadir + '/JK_lo_hf_' + label + '.h5'
    feri = h5py.File(fn, 'r')
    JK_k = np.asarray(feri['JK_lo_ncCo'])
    JK_full = np.asarray(feri['JK_lo_nc'])
    feri.close()

    ## read supercell hcore and DFT veff
    #fn = 'hcore_JK_iao_k_dft_full.h5'
    #feri = h5py.File(fn, 'r')
    #hcore_full = np.asarray(feri['hcore'])
    #JK_dft_full = np.asarray(feri['JK'])
    #feri.close()

    ## read supercell HF-JK matrix
    #fn = 'hcore_JK_iao_k_hf_full.h5'
    #feri = h5py.File(fn, 'r')
    #JK_full = np.asarray(feri['JK'])
    #feri.close()

    ## read imp hcore and DFT veff
    #fn = 'hcore_JK_iao_k_dft_444.h5'
    #feri = h5py.File(fn, 'r')
    #hcore_k_band = np.asarray(feri['hcore'])
    #JK_dft_k_band = np.asarray(feri['JK'])
    #feri.close()
    hcore_k_band = np.copy(hcore_k)
    JK_dft_k_band = np.copy(JK_dft_k)

    ## read supercell hcore and DFT veff
    #fn = 'hcore_JK_iao_k_dft_full_444.h5'
    #feri = h5py.File(fn, 'r')
    #hcore_full_band = np.asarray(feri['hcore'])
    #JK_dft_full_band = np.asarray(feri['JK'])
    #feri.close()
    hcore_full_band = np.copy(hcore_full)
    JK_dft_full_band = np.copy(JK_dft_full)

    # read density matrix
    fn = datadir + '/DM_lo_' + label + '.h5'
    feri = h5py.File(fn, 'r')
    DM_k = np.asarray(feri['DM_lo_ncCo'])
    feri.close()

    # read 4-index ERI
    fn = datadir + '/eri_lo_' + label + '.h5'
    feri = h5py.File(fn, 'r')
    eri = np.asarray(feri['eri_lo_ncCo'])
    feri.close()
    eri_new = eri
    if eri_new.shape[0] == 3:
        eri_new = np.zeros_like(eri)
        eri_new[0] = eri[0]
        eri_new[1] = eri[2]
        eri_new[2] = eri[1]
    del eri

    #<=======================================================
    # ZJ: get H00 and H01 blocks for computing the surface Green's function
    # read bath Hamiltonian
    bathdir = 'Cu_svp_bracket_pbe/'
    label = 'Cu_16_111'
    fn = bathdir + '/ks_ao_' + label + '.h5'
    feri = h5py.File(fn, 'r')
    F_bath = np.asarray(feri['F_lo_nc'])

    # number of orbitals per atom
    nlo_at = 15

    # number of atoms per block
    nlo_blk = 4*nlo_at

    H00 = F_bath[0, :nlo_blk, :nlo_blk]
    H01 = F_bath[0, :nlo_blk, nlo_blk:2*nlo_blk]

    #=======================================================>

    comm.Barrier()

    # run self-consistent DMFT
    mydmft = gwdmft.DMFT(hcore_k, JK_k, DM_k, eri_new, nval, ncore, nfrz, nbath,
                       nb_per_e, disc_type=disc_type, solver_type=solver_type, 
                       H00=H00, H01=H01, log_disc_base=log_disc_base)
    mydmft.gw_dmft = gw_dmft
    mydmft.verbose = 5
    mydmft.diis = True
    mydmft.gmres_tol = gmres_tol
    mydmft.max_memory = max_memory
    mydmft.chkfile = chkfile
    mydmft.diag_only = diag_only
    mydmft.orb_fit = orb_fit
    mydmft.max_cycle = dmft_max_cycle
    mydmft.opt_init_method = opt_init_method
    mydmft.imp_idx = imp_idx
    mydmft.cc_ao_orbs = cc_ao_orbs
    mydmft.mf_type = mf_type
    if solver_type == 'dmrg' or solver_type == 'dmrgsz':
        if not dmrg_:
            raise ImportError

    # supercell and DFT Hamiltonian
    mydmft.hcore_full = hcore_full
    mydmft.JK_full = JK_full
    mydmft.JK_dft_k = JK_dft_k
    mydmft.JK_dft_full = JK_dft_full

    mydmft.JK_dft_k_band = JK_dft_k_band
    mydmft.JK_dft_full_band = JK_dft_full_band
    mydmft.hcore_k_band = hcore_k_band
    mydmft.hcore_full_band = hcore_full_band

    if cas:
        mydmft.cas = cas
        mydmft.casno = casno
        mydmft.composite = composite
        mydmft.thresh = thresh
        mydmft.nvir_act = nvir_act
        mydmft.nocc_act = nocc_act
        mydmft.ea_cut = ea_cut
        mydmft.ip_cut = ip_cut
        mydmft.ea_no = ea_no
        mydmft.ip_no = ip_no
        mydmft.vno_only = vno_only
        mydmft.read_gf = read_gf
        mydmft.save_gf = save_gf

    if solver_type == 'dmrg' or solver_type == 'dmrg_sz':
        mydmft.gs_n_steps = gs_n_steps
        mydmft.gf_n_steps = gf_n_steps
        mydmft.gs_tol = gs_tol
        mydmft.gf_tol = gf_tol
        mydmft.gs_bond_dims = gs_bond_dims
        mydmft.gs_noises = gs_noises
        mydmft.gf_bond_dims = gf_bond_dims
        mydmft.gf_noises = gf_noises
        mydmft.dmrg_gmres_tol = dmrg_gmres_tol
        mydmft.dmrg_verbose = dmrg_verbose
        mydmft.reorder_method = reorder_method
        mydmft.n_off_diag_cg = n_off_diag_cg
        mydmft.load_dir = load_dir
        mydmft.save_dir = save_dir
        mydmft.dyn_corr_method = dyn_corr_method
        mydmft.nvir_act_high = nvir_act_high
        mydmft.nocc_act_low = nocc_act_low
    mydmft.load_mf = load_mf
    mydmft.save_mf = save_mf

    mydmft.kernel(mu0=mu, wl=wl0, wh=wh0, delta=delta, occupancy=nelectron, opt_mu=opt_mu)
    occupancy = 0.
    occupancy = np.trace(mydmft.get_rdm_imp())
    if rank == 0:
        print ('At mu =', mydmft.mu, ', occupancy =', occupancy)

    calc_occ_only = False
    if calc_occ_only:
        exit()

    mydmft.verbose = 5
    mydmft._scf.mol.verbose = 5
    spin = mydmft.spin

    if not run_imagfreq:
        if extra_delta is not None:
            nw = int(round((wh-wl)/(extra_dw * extra_nw)))+1
            freqs = np.linspace(wl, wh, nw)
            extra_freqs = []
            for i in range(len(freqs)):
                freqs_tmp = []
                if extra_nw % 2 == 0:
                    for w in range(-extra_nw // 2, extra_nw // 2):
                        freqs_tmp.append(freqs[i] + extra_dw * w)
                else:
                    for w in range(-(extra_nw-1) // 2, (extra_nw+1) // 2):
                        freqs_tmp.append(freqs[i] + extra_dw * w)
                extra_freqs.append(np.array(freqs_tmp))
        else:
            nw = int(round((wh-wl)/eta))+1
            freqs = np.linspace(wl, wh, nw)
            extra_freqs = None
            extra_delta = None

        # ZJ: freqs on which ldos is computed
        freqs = np.linspace(mydmft.mu-0.1, mydmft.mu+0.1, 200)

        #<====================================================
        # ZJ: get impurity DOS

        # compute a mean-field gf for comparison
        gf_hf = mf_gf(mydmft._scf, freqs, eta)


        fn = 'ldos_gate-002_nb25_mu-0161393_log_cc_delta001_base15_nbpe1.dat'
        if rank == 0:
            f = h5py.File(fn, 'w')
            f['freqs'] = freqs
            # first 6 are valence
            for i in range(6):
                f['ldos_hf_'+str(i)] = -1./np.pi*(gf_hf[0,i,i,:].imag)

        gf = mydmft.get_gf_imp(freqs, eta, ao_orbs=range(6), 
                extra_freqs=extra_freqs, extra_delta=extra_delta, use_gw=use_gw)

        if rank == 0:
            for i in range(6):
                f['ldos_cc_'+str(i)] = -1./np.pi*(gf[0,i,i,:].imag)
            f.close()

        exit()
        #====================================================>

        # Get impurity DOS (production run)
        ldos_t2g, ldos_eg, sigma = mydmft.get_ldos_imp(freqs, eta, extra_freqs=extra_freqs,
                                                       extra_delta=extra_delta, use_gw=use_gw)

        if extra_delta is not None:
            freqs = np.array(extra_freqs).reshape(-1)
            eta = extra_delta

        # Get lattice DOS (production run)
        #ldos, ldos_gw = mydmft.get_ldos_latt(freqs, eta)

        filename = 'mu-%0.3f_n-%0.2f_%d-%d_d-%.2f_%s_t2g'%(
                    mu,occupancy,nval,nbath,delta,solver_type)
        if rank == 0:
            write.write_dos(filename, freqs, ldos_t2g, occupancy=occupancy)
 
        sigmaR = np.real(sigma[0,0,0])
        sigmaI = np.imag(sigma[0,0,0])
        if rank == 0:
            with open(filename+'_sigmaR.dat', 'w') as f:
                for n,wn in enumerate(freqs):
                    f.write('%.12g %.12g %.12g\n'%(wn, sigmaR[n], sigmaI[n]))
 
        filename = 'mu-%0.3f_n-%0.2f_%d-%d_d-%.2f_%s_eg'%(
                    mu,occupancy,nval,nbath,delta,solver_type)
        if rank == 0:
            write.write_dos(filename, freqs, ldos_eg, occupancy=occupancy)
 
        sigmaR = np.real(sigma[0,1,1])
        sigmaI = np.imag(sigma[0,1,1])
        if rank == 0:
            with open(filename+'_sigmaR.dat', 'w') as f:
                for n,wn in enumerate(freqs):
                    f.write('%.12g %.12g %.12g\n'%(wn, sigmaR[n], sigmaI[n]))

    if run_imagfreq:
        #omega_ns1 = np.linspace(0.01/27.211386, 0.1/27.211386, 10)
        #omega_ns2 = np.linspace(0.1/27.211386, 1.0/27.211386, 10)[1:]
        #omega_ns = np.concatenate([omega_ns1,omega_ns2])
        omega_ns = np.linspace(0.01/27.211386, 0.03/27.211386, 3)
        if solver_type == 'dmrg':
            sigma = np.zeros((spin,2,2,len(omega_ns)),dtype=complex)
            for iw in range(len(omega_ns)):
                ldos_t2g, ldos_eg, sigma_tmp = mydmft.get_ldos_imp([mu], omega_ns[iw], use_gw=use_gw)
                sigma[:,:,:,iw] = sigma_tmp[:,:,:,0]
        else:
            ldos_t2g, ldos_eg, sigma = mydmft.get_ldos_imp(mu+1j*omega_ns, 0., use_gw=use_gw)

        filename = 'mu-%0.3f_n-%0.2f_%d-%d_d-%.2f_%s_t2g'%(
                    mu,occupancy,nval,nbath,delta,solver_type)
        sigmaR = np.real(sigma[0,0,0])
        sigmaI = np.imag(sigma[0,0,0])
        if rank == 0:
            with open(filename+'_sigmaI.dat', 'w') as f:
                for n,wn in enumerate(omega_ns):
                    f.write('%.12g %.12g %.12g\n'%(wn, sigmaR[n], sigmaI[n]))

        filename = 'mu-%0.3f_n-%0.2f_%d-%d_d-%.2f_%s_eg'%(
                    mu,occupancy,nval,nbath,delta,solver_type)
        sigmaR = np.real(sigma[0,1,1])
        sigmaI = np.imag(sigma[0,1,1])
        if rank == 0:
            with open(filename+'_sigmaI.dat', 'w') as f:
                for n,wn in enumerate(omega_ns):
                    f.write('%.12g %.12g %.12g\n'%(wn, sigmaR[n], sigmaI[n]))


if __name__ == '__main__':
    dmft_abinitio()
