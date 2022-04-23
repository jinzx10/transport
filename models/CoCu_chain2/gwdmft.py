#!/usr/bin/python

import time, sys, os, h5py
import numpy as np
from scipy import linalg, optimize
from scipy.optimize import least_squares

from pyscf.lib import logger
from pyscf import lib
from fcdmft import solver
from fcdmft.solver import scf_mu as scf
from fcdmft.gw.pbc import krgw_gf
from fcdmft.gw.mol import gw_dc
from fcdmft.utils import write
#from fcdmft.dmft.dmft_solver import mf_kernel, mf_gf, mf_gf_withfrz, \
#                cc_gf, ucc_gf, dmrg_gf, udmrg_gf, cc_rdm, ucc_rdm, \
#                dmrg_rdm, udmrg_rdm, fci_gf, fci_rdm, get_gf, get_sigma
from dmft_solver import mf_kernel, mf_gf, mf_gf_withfrz, \
                cc_gf, ucc_gf, dmrg_gf, udmrg_gf, cc_rdm, ucc_rdm, \
                dmrg_rdm, udmrg_rdm, fci_gf, fci_rdm, get_gf, get_sigma
from mpi4py import MPI
from surface_green import *
from bath_disc import *

einsum = lib.einsum

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

# generate 
def gen_log_grid(w0, w, l, num):
    grid = w0 + (w-w0) * l**(-np.arange(num,dtype=float))
    if w > w0:
        return grid[::-1]
    else:
        return grid
# ****************************************************************************
# core routines: kernel, mu_fit
# ****************************************************************************

def kernel(dmft, mu, wl=None, wh=None, occupancy=None, delta=None,
           conv_tol=None, opt_mu=False, dump_chk=True):
    '''DMFT self-consistency cycle at fixed mu'''
    cput0 = (time.process_time(), time.time())

    # set DMFT parameters
    if delta is None:
        delta = dmft.delta
    if conv_tol is None:
        conv_tol = dmft.conv_tol

    hcore_k = dmft.hcore_k
    JK_k = dmft.JK_k
    DM_k = dmft.DM_k
    eris = dmft.eris
    nval = dmft.nval
    ncore = dmft.ncore
    nfrz = dmft.nfrz
    nb_per_e = dmft.nb_per_e
    H00 = dmft.H00
    H01 = dmft.H01
    log_disc_base = dmft.log_disc_base


    # DFT veff and supercell Hamiltonian
    hcore_full = dmft.hcore_full
    JK_full = dmft.JK_full
    JK_dft_k = dmft.JK_dft_k
    JK_dft_full = dmft.JK_dft_full

    spin, nkpts, nao, nao = hcore_k.shape
    hcore_cell = 1./nkpts * np.sum(hcore_k, axis=1)
    JK_cell = 1./nkpts * np.sum(JK_k, axis=1)
    DM_cell = 1./nkpts * np.sum(DM_k, axis=1)
    JK_dft_cell = 1./nkpts * np.sum(JK_dft_k, axis=1)

    if np.iscomplexobj(hcore_cell):
        assert (np.max(np.abs(hcore_cell.imag)) < 1e-6)
        assert (np.max(np.abs(JK_cell.imag)) < 1e-6)
        assert (np.max(np.abs(DM_cell.imag)) < 1e-6)
        hcore_cell = hcore_cell.real
        JK_cell = JK_cell.real
        DM_cell = DM_cell.real

    # JK_00: double counting term (Hartree and exchange)
    JK_00 = scf._get_veff(DM_cell, eris)
    himp_cell = hcore_cell + JK_cell - JK_00
    dmft.JK_00 = JK_00

    # ZJ: 
    if rank == 0:
        print('log_disc_base = ', log_disc_base)

    '''
    ZJ: below the code is going to construct a bath from the hyb
    note that here the variable 'hyb' is an array of self energy evaluated on x+i*delta,
    In get_bath_direct, it's imaginary part (-1/pi*hyb.imag) is used to generate bath_e & bath_v

    The first step in generating a bath is to is to generate a grid of energy points on which
    the hyb is evaluated. Those values will be used in get_bath_direct. 

    '''
    
    nw = dmft.nbath
    if wl is None and wh is None:
        wl, wh = -0.4+mu, 0.4+mu
    else:
        wl, wh = wl+mu, wh+mu

    if dmft.disc_type == 'linear':
        freqs, wts = _get_linear_freqs(wl, wh, nw)
    elif dmft.disc_type == 'gauss':
        freqs, wts = _get_scaled_legendre_roots(wl, wh, nw)
    elif dmft.disc_type == 'direct':
        nw_org = nw
        wmult = 3
        nw = nw * wmult + 1
        freqs, wts = _get_linear_freqs(wl, wh, nw)
    elif dmft.disc_type == 'nonlin':
        nw_org = nw
        wmult = 3
        nw = nw * wmult + 1
        wl2, wh2 = -0.03+mu, 0.02+mu
        nw2 = nw * 2 // 3
        if (nw2 % 2 == 1):
            nw2 += 1
        freqs = _get_nonlin_freqs(wl, wh, nw, wl2, wh2, nw2)
        wts = None
    elif dmft.disc_type == 'nonlin2':
        nw_org = nw
        wmult = 3
        nw = nw * wmult + 1
        wl2, wh2 = -0.025+mu, 0.02+mu
        nw2 = nw * 1 // 2
        if (nw2 % 2 == 1):
            nw2 += 1

        wl3, wh3 = -0.001+mu, 0.001+mu
        nw3 = nw2 * 1 // 4
        if (nw3 % 2 == 1):
            nw3 += 1

        freqs = _get_nonlin_freqs2(wl, wh, nw, wl2, wh2, nw2, wl3, wh3, nw3)
        wts = None
    elif dmft.disc_type == 'log':
        nw_org = nw
        wmult = 3
        nw = nw * wmult + 1
        
        #<============================================================
        # ZJ: log grid not necessarily symmetric
        # wl and wh is the (absolute) band edge (not distance to mu)
        nbath = dmft.nbath
        if rank == 0:
            print('gen log grid!')

        # mu to left band edge is smaller, use less orbitals
        nl = nbath // 2 - 1
        nh = nbath - nl

        freqs = np.concatenate((gen_log_grid(mu, wl, log_disc_base, nl), [mu], \
                gen_log_grid(mu, wh, log_disc_base, nh)))
        #============================================================>

        #freqs = _get_log_freqs(wl, wh, nw, expo=1.2)
        wts = None
    elif dmft.disc_type == 'opt':
        nw_org = nw
        wmult = 3
        # stop optimizing bath energies after max_opt_cycle
        max_opt_cycle = 3
        # choose initial guess ('direct' or 'log') and fitting grids
        opt_init = dmft.opt_init_method
        nw = nw * wmult + 1
        if opt_init == 'direct':
            freqs, wts = _get_linear_freqs(wl, wh, nw)
        elif opt_init == 'log':
            freqs = _get_log_freqs(wl, wh, nw, expo=1.15)
            wts = None

    dmft.freqs = freqs
    dmft.wts = wts
    if rank == 0:
        logger.info(dmft, 'bath discretization wl = %s, wh = %s', wl, wh)
        logger.info(dmft, 'discretization grids = \n %s', freqs)

    if dmft.gw_dmft:
        # Compute impurity GW self-energy (DC term)
        sigma_gw_imp = dmft.get_gw_sigma(freqs, delta)

        # Compute k-point GW self-energy at given freqs and delta
        sigma_kgw = dmft.get_kgw_sigma(freqs, delta)
        comm.Barrier()
    else:
        sigma_gw_imp = np.zeros((spin, nao, nao, nw), dtype=np.complex)
        sigma_kgw = np.zeros((spin, nkpts, nao, nao, nw), dtype=np.complex)

    # write GW self-energy (trace)
    tmpdir = 'dmft_tmp'
    if rank == 0:
        if not os.path.isdir(tmpdir):
            os.mkdir(tmpdir)
        if dmft.gw_dmft:
            write.write_sigma(tmpdir+'/sigma_gw_dc', freqs, sigma_gw_imp)
            write.write_sigma(tmpdir+'/sigma_kgw', freqs, 1./nkpts * sigma_kgw.sum(axis=1))
    comm.Barrier()

    # turn off comment to read saved DMFT quantities
    '''
    if os.path.isfile('DMFT_chk.h5'):
        with h5py.File(dmft.chkfile, 'r') as fh5:
            sigma0 = np.array(fh5['dmft/sigma'])
        assert (sigma0.shape == (spin,nao,nao,nw))
        sigma = sigma0.copy()
    else:
        sigma = np.zeros([spin,nao,nao,nw], dtype=np.complex)
    '''
    sigma = np.zeros([spin,nao,nao,nw], dtype=np.complex)
    dmft.sigma = sigma

    # get_bands hyb
    hcore_full_band = dmft.hcore_full_band
    JK_dft_full_band = dmft.JK_dft_full_band
    hcore_k_band = dmft.hcore_k_band
    JK_dft_k_band = dmft.JK_dft_k_band

    spin, nkpts_hyb, nao, nao = hcore_k_band.shape
    hcore_cell_band = 1./nkpts_hyb * np.sum(hcore_k_band, axis=1)
    JK_dft_cell_band = 1./nkpts_hyb * np.sum(JK_dft_k_band, axis=1)

    # ZJ:
    # get_gf(h, Sigma, E, delta) returns inv( (E+i*delta)*eye - h - Sigma )
    # get_sigma(G0, G) returns inv(G0) - inv(G)
    # nao = 22 is the number of Co orbitals (only val+virt)
    # nao_full =  is the total number of 1 Co + 9 Cu orbitals (only val+virt)
    # nval = 6 is the number of Co valence orbitals
    # hyb should be the size of spin x nval x nval x nw

    # gf0_cell: local GF; gf_cell: lattice GF (from supercell calc)
    _, _, nao_full, nao_full = JK_full.shape
    sigma_full = np.zeros((spin,nao_full,nao_full,nw), dtype=np.complex)
    gf0_cell = get_gf((hcore_cell_band+JK_dft_cell_band)[:,ncore:nval,ncore:nval], sigma[:,ncore:nval,ncore:nval], freqs, delta)
    gf_cell = np.zeros((spin,nao_full,nao_full,nw), dtype=np.complex)
    # ZJ: temporarily comment this out!
    #for k in range(nkpts_hyb):
    #    gf_cell += 1./nkpts_hyb * get_gf(hcore_full_band[:,k]+JK_dft_full_band[:,k], sigma_full, freqs, delta)

    # ZJ: change the hyb below
    #hyb = get_sigma(gf0_cell, gf_cell[:,ncore:nval,ncore:nval])
    #<===================================================

    # contact block of the Green's function
    def contact_Greens_function(z):
        g00 = Umerski1997(z, H00, H01)
        sz_blk, _ = g00.shape

        # compute G_C (the Green's function of the whole contact block)
        V_L = np.zeros((nao_full,sz_blk), dtype=complex)
        V_R = np.zeros((nao_full,sz_blk), dtype=complex)

        V_L[22:22+sz_blk,:] = H01.T.conj()
        V_R[-sz_blk:,:] = H01

        Sigma_L = V_L @ g00 @ V_L.T.conj()
        Sigma_R = V_R @ g00 @ V_R.T.conj()

        return np.linalg.inv(z*np.eye(nao_full)-hcore_full[0,0]-JK_dft_full[0,0]-Sigma_L-Sigma_R)

    # -1/pi*imag(Sigma(e+i*delta))
    def Gamma(e):
        z = e + 1j*delta
        G_C = contact_Greens_function(z)

        # only the 6 Co valence orbitals
        # G_imp = inv(z-h_imp-Sigma_imp)
        Sigma_imp = z*np.eye(nval) - himp_cell[0,:nval,:nval] - np.linalg.inv(G_C[:nval,:nval])
        return -1./np.pi*Sigma_imp.imag


    # -1/pi*imag(GC)
    def specden(e):
        return -1./np.pi*contact_Greens_function(e+1j*delta).imag

    hyb = np.zeros((spin,nval,nval,nw), dtype=np.complex)
    '''

    if rank == 0:
        print('nao of total contact = ', nao_full)
        print('nao of Co = ', nao)
        print('nw = ', nw)
        print('H00.shape = ', H00.shape)
        print('H01.shape = ', H01.shape)
        print('himp_cell.shape = ', himp_cell.shape)

    for iw in range(nw):
        # surface Green's function of the bath
        z = freqs[iw]+1j*delta
        g00 = Umerski1997(z, H00, H01)
        sz_blk, _ = g00.shape

        # compute G_C (the Green's function of the whole contact block)
        V_L = np.zeros((nao_full,sz_blk), dtype=complex)
        V_R = np.zeros((nao_full,sz_blk), dtype=complex)

        V_L[22:22+sz_blk,:] = H01.T.conj()
        V_R[-sz_blk:,:] = H01

        Sigma_L = V_L @ g00 @ V_L.T.conj()
        Sigma_R = V_R @ g00 @ V_R.T.conj()

        # contact block of the Green's function
        G_C = np.linalg.inv(z*np.eye(nao_full)-hcore_full[0,0]-JK_dft_full[0,0]-Sigma_L-Sigma_R)

        # here the impurity merely contains the 6 Co valence orbitals
        # G_imp = inv(z-h_imp-hyb)
        hyb[0,:,:,iw] = (freqs[iw] + 1j*delta)*np.eye(nval) - himp_cell[0,:nval,:nval] \
                - np.linalg.inv(G_C[:nval,:nval])
    
    hyb = comm.bcast(hyb,root=0)

    if rank == 0:
        write.write_sigma(tmpdir+'/dmft_hyb', freqs, hyb)
    comm.Barrier()

    #===================================================>
    '''

    if isinstance(dmft.diis, lib.diis.DIIS):
        dmft_diis = dmft.diis
    elif dmft.diis:
        dmft_diis = lib.diis.DIIS(dmft, dmft.diis_file)
        dmft_diis.space = dmft.diis_space
    else:
        dmft_diis = None
    diis_start_cycle = dmft.diis_start_cycle

    dmft_conv = False
    cycle = 0
    if rank == 0:
        cput1 = logger.timer(dmft, 'initialize DMFT', *cput0)
    while not dmft_conv and cycle < max(1, dmft.max_cycle):
        hyb_last = hyb

        if dmft.disc_type == 'direct' or dmft.disc_type == 'log' or dmft.disc_type == 'nonlin' or dmft.disc_type == 'nonlin2':
            # ZJ: bath_e has a length of nw_org*nimp, it goes through all different energies and repeats them
            #bath_v, bath_e = get_bath_direct(hyb, freqs, nw_org)

            # ZJ: direct_disc_hyb returns e as a 1-d array of nbath elements
            # and v as a 2-d array of (nbath,nimp,nb_per_e) 
            bath_e, bath_v = direct_disc_hyb(Gamma, freqs, nint=5, nbath_per_ene = nb_per_e)
            bath_e = np.tile(bath_e, nb_per_e)
            bath_e = bath_e[np.newaxis,...]
            bath_v = np.transpose(bath_v, (1,2,0))
            bath_v = bath_v.reshape((nval,nbath*nb_per_e))
            bath_v = bath_v[np.newaxis,...]

            if rank == 0:
                logger.info(dmft, 'bath energies = \n %s', bath_e[0][:nw_org])
        elif dmft.disc_type == 'opt':
            if cycle < max_opt_cycle:
                if cycle == 0:
                    bath_v, bath_e = get_bath_direct(hyb, freqs, nw_org)
                    bath_v = bath_v.reshape(spin,nval-ncore,nval-ncore,nw_org)
                    bath_v = bath_v[:,:,(nval-ncore-nb_per_e):,:].reshape(spin,nval-ncore,-1)
                    bath_e = bath_e.reshape(spin,nval-ncore,nw_org)
                    bath_e = bath_e[:,(nval-ncore-nb_per_e):,:].reshape(spin,-1)
                if rank == 0:
                    logger.info(dmft, 'initial bath energies = \n %s', bath_e[0][:nw_org])
                    bath_e, bath_v = opt_bath(bath_e, bath_v, hyb, freqs, delta, nw_org,
                                              diag_only=dmft.diag_only, orb_fit=dmft.orb_fit)
                    logger.info(dmft, 'optimized bath energies = \n %s', bath_e[0][:nw_org])
            else:
                if rank == 0:
                    bath_v = opt_bath_v_only(bath_e, bath_v, hyb, freqs, delta, nw_org,
                                            diag_only=dmft.diag_only, orb_fit=dmft.orb_fit)
            comm.Barrier()
            bath_e = comm.bcast(bath_e,root=0)
            bath_v = comm.bcast(bath_v,root=0)
        else:
            bath_v, bath_e = get_bath(hyb, freqs, wts)

        comm.Barrier()
        bath_e = comm.bcast(bath_e,root=0)
        bath_v = comm.bcast(bath_v,root=0)
        dmft.bath_e = bath_e[0][:nw_org]

        if rank == 0:
            print('bath_e.shape', bath_e.shape)
            print('bath_v.shape', bath_v.shape)
            print('freqs.shape', freqs.shape)
            print('hyb.shape', hyb.shape)

        # ZJ: himp has the same size as himp_cell (val+virt orbitals for Co)
        # but only ncore:nval orbitals are coupled to the bath

        # construct embedding Hamiltonian
        himp, eri_imp = imp_ham(himp_cell, eris, bath_v, bath_e, ncore)

        # get initial guess of impurity 1-RDM
        nimp = himp.shape[1]
        dm0 = np.zeros((spin,nimp,nimp))
        if cycle == 0:
            dm0[:,:nao,:nao] = DM_cell.copy()
        else:
            dm0[:,:nao,:nao] = dm_last[:,:nao,:nao].copy()

        # optimize chemical potential for correct number of impurity electrons
        if rank == 0:
            print('occupancy = ', occupancy)
        if opt_mu:
            mu = mu_fit(dmft, mu, occupancy, himp, eri_imp, dm0)
            comm.Barrier()
            mu = comm.bcast(mu, root=0)
            comm.Barrier()

        #if rank == 0:
        #    print('optimized mu = ', mu)
        print('rank = ', rank, 'opt_mu = ', opt_mu, 'mu = ', mu)


        # ZJ: get an intial guess for the embedding problem
        # run a HF scf or just do it one-shot
        run_hf_embed = True

        if run_hf_embed:
            if not opt_mu:
                # run HF for embedding problem
                mf = mf_kernel(himp, eri_imp, mu, nao, dm0,
                                  max_mem=dmft.max_memory, verbose=dmft.verbose, JK00=dmft.JK_00)
                dmft._scf = mf
        else:
            # Get mean-field determinant by diagonalizing imp+bath Ham just once
            dm_init = np.zeros_like(dm0)
            dm_init[:,:nao,:nao] = DM_cell[:,:nao,:nao]
            JK_impbath = scf._get_veff(dm_init, eri_imp)
            if dmft.mf_type == 'dft':
                JK_dft_tmp = JK_dft_cell - (JK_cell - JK_00)
            elif dmft.mf_type == 'hf':
                JK_dft_tmp = JK_cell - (JK_cell - JK_00)
            else:
                raise NotImplementedError

            JK_impbath[:,:nao,:nao] = JK_dft_tmp[:,:nao,:nao]
            F_impbath = himp + JK_impbath
            emo_ib, cmo_ib = linalg.eigh(F_impbath[0])
            if rank == 0:
                logger.info(dmft, 'Imp+bath mean-field mo_energy = \n %s', emo_ib)
            mo_occ_ib = np.zeros_like(emo_ib)
            mo_occ_ib[emo_ib <= mu] = 2.
            nocc_ib = int(np.sum(mo_occ_ib) // 2)
            dm_ib = 2. * np.dot(cmo_ib[:,:nocc_ib], cmo_ib[:,:nocc_ib].T)
            if rank == 0:
                logger.info(dmft, 'Imp+bath Nelec = %s', np.trace(dm_ib[:nao,:nao]))
                logger.info(dmft, 'Imp+bath 1-RDM diag = \n %s', dm_ib[:nao,:nao].diagonal())
                logger.info(dmft, 'Imp+bath Full 1-RDM diag = \n %s', dm_ib.diagonal())

            from pyscf import gto, ao2mo
            spin, n = himp.shape[0:2]
            mol = gto.M()
            mol.verbose = dmft.verbose
            mol.incore_anyway = True
            mol.build()

            mf = scf.RHF(mol, mu)
            mf.max_memory = dmft.max_memory
            mf.mo_energy = emo_ib
            mf.mo_occ = mo_occ_ib
            mf.mo_coeff = cmo_ib
            mf.max_cycle = 150
            mf.conv_tol = 1e-12
            mf.diis_space = 15

            mf.get_hcore = lambda *args: himp[0]
            mf.get_ovlp = lambda *args: np.eye(n)
            mf._eri = ao2mo.restore(8, eri_imp[0], n)
            del eri_imp

            dm = mf.make_rdm1()
            if rank == 0:
                logger.info(dmft, 'Imp+bath 1-RDM diag = \n %s', dm[:nao,:nao].diagonal())

            if dmft.save_mf:
                if rank == 0:
                    fn = 'mf.h5'
                    feri = h5py.File(fn, 'w')
                    feri['mo_coeff'] = np.asarray(mf.mo_coeff)
                    feri['mo_energy'] = np.asarray(mf.mo_energy)
                    feri['mo_occ'] = np.asarray(mf.mo_occ)
                    feri['himp'] = np.asarray(himp)
                    feri['eri'] = np.asarray(mf._eri)
                    feri.close()
                comm.Barrier()

            if dmft.load_mf:
                fn = 'mf.h5'
                feri = h5py.File(fn, 'r')
                mf.mo_coeff = np.array(feri['mo_coeff'])
                mf.mo_energy = np.array(feri['mo_energy'])
                mf.mo_occ = np.array(feri['mo_occ'])
                himp = np.array(feri['himp'])
                mf._eri = np.array(feri['eri'])
                feri.close()
                mf.get_hcore = lambda *args: himp[0]
                mf.get_ovlp = lambda *args: np.eye(n)

            mf.JK = JK_impbath[0]
            JK_hf = JK_impbath.copy()
            JK_hf[:,:nao,:nao] = JK_00[:,:nao,:nao]
            mf.JK_hf = JK_hf[0]
            mf.dm_init = dm_init[0]
            mf.nimp = nao

            dmft._scf = mf

        #print('sum(mf.mo_occ) = ', np.sum(mf.mo_occ))

        # ZJ: for one-shot DMFT (max_cycle == 0) the kernel function exit below
        if dmft.max_cycle <= 1:
            break

        dm_last = dmft._scf.make_rdm1()
        if len(dm_last.shape) == 2:
            dm_last = dm_last[np.newaxis, ...]

        # ZJ: note the if branch below is based on the broadening delta! 
        # the comment below explains why
        '''
        Run impurity solver calculation to get self-energy.
        When delta is small, CCSD imp self-energy can be non-causal if computed directly
        from imp CCSD-GF; instead, it is safer (but more expensive) to first compute
        imp+bath self-energy, then take the imp block of self-energy.
        '''
        # TODO: implement and test DMRG solver
        if delta >= 0.05:
            # imp GF -> imp sigma
            gf_imp = np.zeros((spin,nao,nao,nw),dtype=np.complex128)

            # ZJ: the line below contains the core impurity solver
            # ZJ: the code won't reach here!
            #print('ready to solve the impurity problem')
            gf_imp[:,nfrz:,nfrz:,:] = dmft.get_gf_imp(freqs, delta)
            #print('impurity problem solved!')

            if dmft.solver_type == 'cc' or dmft.solver_type == 'ucc':
                gf_imp = 0.5 * (gf_imp+gf_imp.transpose(0,2,1,3))
            # Treat frozen core in solver (core GF appoximated by HF)
            gfhf_imp, gfhf_imp_frz = mf_gf_withfrz(dmft._scf, freqs, delta, ao_orbs=range(nao), nfrz=nfrz)
            gf_imp = gf_imp - gfhf_imp_frz + gfhf_imp

            sgdum = np.zeros((spin,nimp,nimp,nw))
            gf0_imp = get_gf(himp, sgdum, freqs, delta)
            gf0_imp = gf0_imp[:,:nao,:nao,:]

            sigma_imp = get_sigma(gf0_imp, gf_imp)
        else:
            # imp+bath GF -> imp+bath sigma -> imp sigma
            #print('ready to solve the impurity problem')
            # ZJ: the line below contains the core impurity solver
            sigma_imp = dmft.get_sigma_imp(freqs, delta)
            sigma_imp = sigma_imp[:,:nao,:nao]

        if dmft.cas and cycle == 0:
            dmft.nocc_act = dmft._scf.nocc_act
            dmft.nvir_act = dmft._scf.nvir_act

        # Choose to use GW self-energy for core
        sigma_gw_imp_frz = sigma_gw_imp.copy()
        sigma_gw_imp_frz[:,nfrz:,nfrz:,:] = 0.
        sigma_imp = sigma_imp + sigma_gw_imp_frz

        # remove GW double counting term
        for w in range(nw):
            sigma[:,:,:,w] = sigma_imp[:,:,:,w] - JK_00
        sigma = sigma - sigma_gw_imp

        # update local and lattice GF
        gf0_cell = get_gf((himp_cell)[:,ncore:nval,ncore:nval], sigma_imp[:,ncore:nval,ncore:nval], freqs, delta)
        gf_cell = np.zeros([spin, nao, nao, nw], np.complex128)
        for k in range(nkpts):
            gf_cell += 1./nkpts * get_gf(hcore_k[:,k]+JK_k[:,k], sigma+sigma_kgw[:,k], freqs, delta)
        hyb_new = get_sigma(gf0_cell, gf_cell[:,ncore:nval,ncore:nval])

        # write lattice GF and sigma during self-consistent loop
        if rank == 0:
            write.write_sigma(tmpdir+'/dmft_sigma_imp_iter', freqs, sigma)
            write.write_gf_to_dos(tmpdir+'/dmft_latt_dos_iter', freqs, gf_cell)
        comm.Barrier()

        damp = dmft.damp
        if rank == 0:
            if (abs(damp) > 1e-4 and
                (0 <= cycle < diis_start_cycle-1 or dmft_diis is None)):
                hyb_new = damp*hyb_new + (1-damp)*hyb
            hyb = dmft.run_diis(hyb_new, cycle, dmft_diis)
        comm.Barrier()
        hyb = comm.bcast(hyb,root=0)
        dmft.hyb = hyb
        dmft.sigma = sigma

        norm_dhyb = np.linalg.norm(hyb-hyb_last)
        if rank == 0:
            logger.info(dmft, 'cycle= %d  |dhyb|= %4.3g', cycle+1, norm_dhyb)

        if (norm_dhyb < conv_tol):
            dmft_conv = True
        if dump_chk and dmft.chkfile:
            if rank == 0:
                dmft.dump_chk()
        comm.Barrier()

        if rank == 0:
            cput1 = logger.timer(dmft, 'cycle= %d'%(cycle+1), *cput1)
        cycle += 1

    comm.Barrier()
    if rank == 0:
        logger.timer(dmft, 'DMFT_cycle', *cput0)

    return dmft_conv, mu


def mu_fit(dmft, mu0, occupancy, himp, eri_imp, dm0, step=0.02, trust_region=0.03,
           nelec_tol=1e-5, max_cycle=50):
    '''
    Fit chemical potential to find target impurity occupancy
    '''
    mu_cycle = 0
    dmu = 0
    record = []
    if rank == 0:
        logger.info(dmft, '### Start chemical potential fitting ###')

    while mu_cycle < max_cycle:
        # run HF for embedding problem
        mu = mu0 + dmu
        dmft._scf = mf_kernel(himp, eri_imp, mu, dmft.nao, dm0,
                              max_mem=dmft.max_memory, verbose=4, JK00=dmft.JK_00)
        # run ground-state impurity solver to get 1-rdm
        rdm = dmft.get_rdm_imp()
        nelec = np.trace(rdm)
        if mu_cycle > 0:
            dnelec_old = dnelec
        dnelec = nelec - occupancy
        if abs(dnelec) < nelec_tol * occupancy:
            break
        if mu_cycle > 0:
            if abs(dnelec - dnelec_old) < 1e-5:
                if rank == 0:
                    logger.info(dmft, 'Electron number not affected by dmu, quit mu_fit')
                break
        record.append([dmu, dnelec])

        if mu_cycle == 0:
            if dnelec > 0:
                dmu = -1. * step
            else:
                dmu = step
        elif len(record) == 2:
            # linear fit
            dmu1 = record[0][0]; dnelec1 = record[0][1]
            dmu2 = record[1][0]; dnelec2 = record[1][1]
            dmu = (dmu1*dnelec2 - dmu2*dnelec1) / (dnelec2 - dnelec1)
        else:
            # linear fit
            dmu_fit = []
            dnelec_fit = []
            for rec in record:
                dmu_fit.append(rec[0])
                dnelec_fit.append(rec[1])
            dmu_fit = np.array(dmu_fit)
            dnelec_fit = np.array(dnelec_fit)
            idx = np.argsort(np.abs(dnelec_fit))[:2]
            dmu_fit = dmu_fit[idx]
            dnelec_fit = dnelec_fit[idx]
            a,b = np.polyfit(dmu_fit, dnelec_fit, deg=1)
            dmu = -b/a

        if abs(dmu) > trust_region:
            if dmu < 0:
                dmu = -trust_region
            else:
                dmu = trust_region
        if rank == 0:
            logger.info(dmft, 'mu_cycle = %s, mu = %s, nelec = %s, dmu = %s',
                        mu_cycle+1, mu, nelec, dmu)
        mu_cycle += 1

    if rank == 0:
        logger.info(dmft, 'Optimized mu = %s, Nelec = %s, Target = %s', mu, nelec, occupancy)

    return mu

# ****************************************************************************
# bath numerical optimization : opt_bath, opt_bath_v_only
# ****************************************************************************

def opt_bath(bath_e, bath_v, hyb, freqs, delta, nw_org, diag_only=False, orb_fit=None):
    '''
    Optimize bath energies and couplings for minimizing bath discretization error

    Args:
         bath_e : (spin, nb_per_e * nw_org) ndarray
         bath_v : (spin, nimp, nb_per_e * nw_org) ndarray
         hyb : (spin, nimp, nimp, nw) ndarray
         freqs : (nw) 1darray, fitting grids
         delta : float
         nw_org : interger, number of bath energies
         diag_only : bool, only fit diagonal hybridization
         orb_fit : list, orbitals with x5 weight in optimization

    Returns:
         bath_e_opt : (spin, nb_per_e * nw_org) ndarray
         bath_v_opt : (spin, nimp, nb_per_e * nw_org) ndarray
    '''
    # TODO: allow different number of bath orbitals at bath energies
    # TODO: choose to add optimization weights for different hyb elements (diag, 3d/4d)
    spin, nimp, nbath = bath_v.shape
    nb_per_e = nbath // nw_org
    v_opt = np.zeros((spin, nw_org+nimp*nbath))
    min_bound = []; max_bound = []
    for i in range(nw_org+nimp*nbath):
        if i < nw_org:
            min_bound.append(freqs[0])
            max_bound.append(freqs[-1])
        else:
            min_bound.append(-np.inf)
            max_bound.append(np.inf)

    for s in range(spin):
        if s == 0:
            v0 = np.concatenate([bath_e[s][:nw_org], bath_v[s].reshape(-1)])
            try:
                xopt = least_squares(bath_fit, v0, jac='2-point', method='trf', bounds=(min_bound,max_bound), xtol=1e-8,
                     gtol=1e-6, max_nfev=500, verbose=1, args=(hyb[s], bath_v[s], freqs, delta, nw_org, diag_only, orb_fit))
            except:
                xopt = least_squares(bath_fit, v0, jac='2-point', method='lm', xtol=1e-8,
                     gtol=1e-6, max_nfev=500, verbose=1, args=(hyb[s], bath_v[s], freqs, delta, nw_org, diag_only, orb_fit))
            v_opt[s] = xopt.x.copy()
        else:
            v0 = bath_v[s].reshape(-1)
            try:
                xopt = least_squares(bath_fit_v, v0, jac='2-point', method='trf', xtol=1e-8,
                             gtol=1e-6, max_nfev=500, verbose=1,
                             args=(v_opt[0][:nw_org], hyb[s], bath_v[s], freqs, delta, nw_org, diag_only, orb_fit))
            except:
                xopt = least_squares(bath_fit_v, v0, jac='2-point', method='lm', xtol=1e-8,
                             gtol=1e-6, max_nfev=500, verbose=1,
                             args=(v_opt[0][:nw_org], hyb[s], bath_v[s], freqs, delta, nw_org, diag_only, orb_fit))
            v_opt[s][nw_org:] = xopt.x.copy()
            v_opt[s][:nw_org] = v_opt[0][:nw_org]

    bath_e_opt = np.zeros_like(bath_e)
    bath_v_opt = np.zeros_like(bath_v)
    for s in range(spin):
        bath_v_opt[s] = v_opt[s][nw_org:].reshape(nimp, nbath)
        en = v_opt[s][:nw_org]
        for ip in range(nb_per_e):
            for iw in range(nw_org):
                bath_e_opt[s, ip*nw_org + iw] = en[iw]

    return bath_e_opt, bath_v_opt

def bath_fit(v, hyb, bath_v, omega, delta, nw_org, diag_only, orb_fit):
    '''
    Least square of hybridization fitting error
    '''
    nimp, nbath = bath_v.shape
    nb_per_e = nbath // nw_org
    en = v[:nw_org]
    v = v[nw_org:].reshape(nimp, nb_per_e, nw_org)
    w_en = 1./(omega[:,None] + 1j*delta - en[None,:])
    if not diag_only:
        J = einsum('ikn,jkn->ijn',v,v)
        hyb_now = einsum('ijn,wn->ijw',J,w_en)
        f = hyb_now - hyb
        if orb_fit:
            for i in orb_fit:
                f[i,i,:] = 20. * f[i,i,:]
    else:
        J = einsum('ikn,ikn->in',v,v)
        hyb_now = einsum('in,wn->iw',J,w_en)
        f = hyb_now - einsum('iiw->iw', hyb)
        if orb_fit:
            for i in orb_fit:
                f[i,:] = 20. * f[i,:]
    return np.array([f.real,f.imag]).reshape(-1)

def opt_bath_v_only(bath_e, bath_v, hyb, freqs, delta, nw_org, diag_only=False, orb_fit=None):
    '''
    Optimize bath couplings only for minimizing bath discretization error
    '''
    spin, nimp, nbath = bath_v.shape
    v_opt = np.zeros((spin, nimp*nbath))
    bath_v_opt = np.zeros_like(bath_v)
    for s in range(spin):
        v0 = bath_v[s].reshape(-1)
        try:
            xopt = least_squares(bath_fit_v, v0, jac='2-point', method='trf', xtol=1e-10,
                         gtol = 1e-10, max_nfev=500, verbose=1,
                         args=(bath_e[s][:nw_org], hyb[s], bath_v[s], freqs, delta, nw_org, diag_only, orb_fit))
        except:
            xopt = least_squares(bath_fit_v, v0, jac='2-point', method='lm', xtol=1e-10,
                         gtol = 1e-10, max_nfev=500, verbose=1,
                         args=(bath_e[s][:nw_org], hyb[s], bath_v[s], freqs, delta, nw_org, diag_only, orb_fit))
        v_opt[s] = xopt.x.copy()
        bath_v_opt[s] = v_opt[s].reshape(nimp, nbath)

    return bath_v_opt

def bath_fit_v(v, bath_e, hyb, bath_v, omega, delta, nw_org, diag_only, orb_fit):
    '''
    Least square of hybridization fitting error
    '''
    nval, nbath  = bath_v.shape
    nb_per_e = nbath // nw_org
    v = v.reshape(nval, nb_per_e, nw_org)
    w_en = 1./(omega[:,None] + 1j*delta - bath_e[None,:])
    if not diag_only:
        J = einsum('ikn,jkn->ijn',v,v)
        hyb_now = einsum('ijn,wn->ijw',J,w_en)
        f = hyb_now - hyb
        if orb_fit:
            for i in orb_fit:
                f[i,i,:] = 20. * f[i,i,:]
    else:
        J = einsum('ikn,ikn->in',v,v)
        hyb_now = einsum('in,wn->iw',J,w_en)
        f = hyb_now - einsum('iiw->iw', hyb)
        if orb_fit:
            for i in orb_fit:
                f[i,:] = 20. * f[i,:]
    return np.array([f.real,f.imag]).reshape(-1)

# ****************************************************************************
# bath discretization grids : legendre, linear, log 
# ****************************************************************************

def _get_scaled_legendre_roots(wl, wh, nw):
    '''
    Scale nw Legendre roots, which lie in the
    interval [-1, 1], so that they lie in [wl, wh]

    Returns:
        freqs : 1D ndarray
        wts : 1D ndarray
    '''
    freqs, wts = np.polynomial.legendre.leggauss(nw)
    freqs += 1
    freqs *= (wh - wl) / 2.
    freqs += wl
    wts *= (wh - wl) / 2.
    return freqs, wts

def _get_linear_freqs(wl, wh, nw):
    freqs = np.linspace(wl, wh, nw)
    wts = np.ones([nw]) * (wh - wl) / (nw - 1.)
    return freqs, wts

def _get_nonlin_freqs(wl, wh, nw, wl2, wh2, nw2):
    freqs_c = np.linspace(wl2, wh2, nw2)
    nw_r = (nw - nw2) // 2
    nw_l = nw - nw2 - nw_r
    freqs_l = np.linspace(wl, wl2, nw_l+1)[:-1]
    freqs_r = np.linspace(wh2, wh, nw_r+1)[1:]
    freqs = np.concatenate((freqs_l, freqs_c, freqs_r))
    return freqs

def _get_nonlin_freqs2(wl, wh, nw, wl2, wh2, nw2, wl3, wh3, nw3):
    freqs3 = np.linspace(wl3, wh3, nw3)

    nw2_r = (nw2 - nw3) // 2
    nw2_l = nw2 - nw3 - nw2_r
    freqs2_l = np.linspace(wl2, wl3, nw2_l+1)[:-1]
    freqs2_r = np.linspace(wh3, wh2, nw2_r+1)[1:]

    nw_r = (nw - nw2) // 2
    nw_l = nw - nw2 - nw_r
    freqs_l = np.linspace(wl, wl2, nw_l+1)[:-1]
    freqs_r = np.linspace(wh2, wh, nw_r+1)[1:]

    freqs = np.concatenate((freqs_l, freqs2_l, freqs3, freqs2_r, freqs_r))
    return freqs

def _get_log_freqs(wl, wh, nw, expo=1.3):
    '''
    Scale nw logorithmic roots on [wl, wh],
    with a given exponent

    Returns:
        freqs : 1D ndarray
    '''
    if (nw % 2 == 1):
        n = nw // 2
        nlist = np.arange(n)
        wpos = 1./(expo ** (nlist))
        freqs1 = (wh + wl)/2. + wpos * (wh - wl)/2.
        freqs2 = (wh + wl)/2. - wpos * (wh - wl)/2.
        freqs = np.concatenate([freqs1,freqs2,[(wh+wl)/2.]])
        freqs = np.sort(freqs)
    if (nw % 2 == 0):
        n = nw // 2
        nlist = np.arange(n-1)
        nlist = np.append(nlist,[n+n//5])
        wpos = 1./(expo ** (nlist))
        freqs1 = (wh + wl)/2. + wpos * (wh - wl)/2.
        freqs2 = (wh + wl)/2. - wpos * (wh - wl)/2.
        freqs = np.concatenate([freqs1,freqs2])
        freqs = np.sort(freqs)
    return freqs

# ****************************************************************************
# Hamiltonian routines : imp_ham, get_bath
# ****************************************************************************

def imp_ham(hcore_cell, eri_cell, bath_v, bath_e, ncore):
    '''
    Construct impurity Hamiltonian

    Args:
         hcore_cell: (spin, nimp, nimp) ndarray
         eri_cell: (spin*(spin+1)/2, nimp*4) ndarray
         bath_v: (spin, nval, nval*nw) ndarray
         bath_e: (spin, nval*nw) ndarray
         ncore: interger

    Returns:
         himp: (spin, nimp+nb, nimp+nb) ndarray
         eri_imp: (spin*(spin+1)/2, (nimp+nb)*4) ndarray
    '''
    spin, nao = hcore_cell.shape[0:2]
    nbath = bath_e.shape[-1]
    nval = bath_v.shape[1] + ncore
    himp = np.zeros([spin, nao+nbath, nao+nbath])
    himp[:,:nao,:nao] = hcore_cell
    himp[:,ncore:nval,nao:] = bath_v
    himp[:,nao:,ncore:nval] = bath_v.transpose(0,2,1)
    for s in range(spin):
        himp[s,nao:,nao:] = np.diag(bath_e[s])

    eri_imp = np.zeros([spin*(spin+1)//2, nao+nbath, nao+nbath, nao+nbath, nao+nbath])
    eri_imp[:,:nao,:nao,:nao,:nao] = eri_cell
    return himp, eri_imp

def get_bath(hyb, freqs, wts):
    '''
    Convert hybridization function
    to bath couplings and energies,
    linear or gauss discretization

    Args:
        hyb : (spin, nimp, nimp, nw) ndarray
        freqs : (nw) ndarray
        wts : (nw) ndarray, wts at freq pts

    Returns:
        bath_v : (spin, nimp, nimp*nw) ndarray
        bath_e : (spin, nimp*nw) ndarray
    '''
    nw = len(freqs)
    wh = max(freqs)
    wl = min(freqs)
    spin, nimp = hyb.shape[0:2]

    dw = (wh - wl) / (nw - 1)
    # Eq. (6), arxiv:1507.07468
    v2 = -1./np.pi * np.imag(hyb)

    # simple discretization of bath, Eq. (9), arxiv:1507.07468
    v = np.empty_like(v2)

    for s in range(spin):
        for iw in range(nw):
            eig, vec = linalg.eigh(v2[s,:,:,iw])
            # although eigs should be positive, there
            # could be numerical-zero negative eigs: check this
            neg_eigs = [e for e in eig if e < 0]
            if rank == 0:
                if not np.allclose(neg_eigs, 0):
                    log = logger.Logger(sys.stdout, 4)
                    for neg_eig in neg_eigs:
                        log.warn('hyb eval = %.8f', neg_eig)
            # set negative eigs to 0
            for k in range(len(eig)):
                if eig[k] < 0:
                    eig[k] = 0.
            v[s,:,:,iw] = np.dot(vec, np.diag(np.sqrt(np.abs(eig)))) * \
                        np.sqrt(wts[iw])

    # bath_v[p,k_n] is the coupling btw impurity site p and bath orbital k
    # (total number nw=nbath) belonging to bath n (total number nimp)
    bath_v = v.reshape([spin, nimp, nimp*nw])
    bath_e = np.zeros([spin, nimp*nw])

    # bath_e is [nimp*nw] array, with entries
    # w1, w2 ... wn, w1, w2 ... wn, ...
    for s in range(spin):
        for ip in range(nimp):
            for iw in range(nw):
                bath_e[s, ip*nw + iw] = freqs[iw]

    return bath_v, bath_e

def get_bath_direct(hyb, freqs, nw_org):
    """
    Convert hybridization function
    to bath couplings and energies,
    log or direct discretization

    Args:
        hyb : (spin, nimp, nimp, nw) ndarray
        freqs : (nw) ndarray
        nw_org: integer, number of bath energies

    Returns:
        bath_v : (spin, nimp, nimp*nw_org) ndarray
        bath_e : (spin, nimp*nw_org) ndarray
    """
    nw = len(freqs)
    wmult = nw // nw_org
    wh = max(freqs)
    wl = min(freqs)

    # Eq. (6), arxiv:1507.07468
    v2 = -1./np.pi * np.imag(hyb)

    # direct discretization of bath, Eq. (7), arxiv:2003.06062 
    spin, nimp, nimp, nw = v2.shape
    J_int = np.zeros((spin, nimp,nimp,nw_org))
    for s in range(spin):
        for iw in range(nw_org):
            for j in range(wmult):
                J_int[s,:,:,iw] += (v2[s,:,:,iw*wmult+j] + v2[s,:,:,iw*wmult+j+1]) \
                                * (freqs[iw*wmult+j+1] - freqs[iw*wmult+j]) / 2

    v = np.empty_like(J_int)
    en = np.zeros((spin, nw_org))

    for s in range(spin):
        for iw in range(nw_org):
            eig, vec = linalg.eigh(J_int[s,:,:,iw])
            # although eigs should be positive, there
            # could be numerical-zero negative eigs: check this
            neg_eigs = [e for e in eig if e < 0]
            if rank == 0:
                if not np.allclose(neg_eigs, 0):
                    log = logger.Logger(sys.stdout, 4)
                    for neg_eig in neg_eigs:
                        log.warn('hyb eval = %.8f', neg_eig)
            # set negative eigs to 0
            for k in range(len(eig)):
                if eig[k] < 0:
                    eig[k] = 0.

            v[s,:,:,iw] = np.dot(vec, np.diag(np.sqrt(np.abs(eig))))
            e_sum = 0.
            for j in range(wmult):
                e_sum += (freqs[iw*wmult+j] * np.trace(v2[s,:,:,iw*wmult+j]) \
                        + freqs[iw*wmult+j+1] * np.trace(v2[s,:,:,iw*wmult+j+1])) \
                                * (freqs[iw*wmult+j+1] - freqs[iw*wmult+j]) / 2.
            en[s,iw] = e_sum / np.trace(J_int[s,:,:,iw])

    # bath_v[p,k_n] is the coupling btw impurity site p and bath orbital k
    # (total number nw_org=nbath) belonging to bath n (total number nimp)
    bath_v = v.reshape([spin, nimp, nimp*nw_org])
    bath_e = np.zeros([spin, nimp*nw_org])

    # bath_e is [nimp*nw_org] array, with entries
    # w1, w2 ... wn, w1, w2 ... wn, ...
    for s in range(spin):
        for ip in range(nimp):
            for iw in range(nw_org):
                bath_e[s, ip*nw_org + iw] = en[s,iw]

    return bath_v, bath_e


class DMFT(lib.StreamObject):
    '''
    List of DMFT class parameters (self-consistent iterations)

    max_cycle: max number of DMFT self-consistent iterations
    conv_tol: tolerance of hybridization that controls DMFT self-consistency
    damp: damping factor for first DMFT iteration
    gmres_tol: GMRES/GCROTMK convergence tolerance for imp solvers
    '''
    max_cycle = 10
    conv_tol = 1e-3
    damp = 0.7
    gmres_tol = 1e-3
    max_memory = 8000
    n_threads = int(os.environ['OMP_NUM_THREADS'])

    # DIIS parameters for DMFT hybridization
    diis = True
    diis_space = 6
    diis_start_cycle = 1
    diis_file = None

    def __init__(self, hcore_k, JK_k, DM_k, eris, nval, ncore, nfrz,
                 nbath, nb_per_e, disc_type='opt', solver_type='cc',
                 H00=None, H01=None, log_disc_base=1.5):
        # account for both spin-restricted and spin-unrestricted cases
        if len(hcore_k.shape) == 3:
            hcore_k = hcore_k[np.newaxis, ...]
        if len(JK_k.shape) == 3:
            JK_k = JK_k[np.newaxis, ...]
        if len(DM_k.shape) == 3:
            DM_k = DM_k[np.newaxis, ...]
        if len(eris.shape) == 4:
            eris = eris[np.newaxis, ...]

        self.spin, self.nkpts, self.nao, _ = hcore_k.shape
        assert (hcore_k.shape == (self.spin, self.nkpts, self.nao, self.nao,))
        assert (JK_k.shape == (self.spin, self.nkpts, self.nao, self.nao,))
        assert (DM_k.shape == (self.spin, self.nkpts, self.nao, self.nao,))
        assert (eris.shape == (self.spin*(self.spin+1)//2, self.nao,
                               self.nao, self.nao, self.nao,))

        self.hcore_k = hcore_k
        self.JK_k = JK_k
        self.DM_k = DM_k
        self.eris = eris
        self.nbath = nbath
        self.nb_per_e = nb_per_e
        self.nval = nval
        self.ncore = ncore
        self.nfrz = nfrz
        self.solver_type = solver_type
        self.disc_type = disc_type
        self.verbose = logger.NOTE
        self.chkfile = None
        self.diag_only = False
        self.orb_fit = None
        self.gw_dmft = True
        self.opt_init_method = 'direct'

        self.mu = None
        self.JK_00 = None
        self.converged = False
        self.hyb = None
        self.sigma = None
        self.freqs = None
        self.wts = None

        #<================================
        self.H00 = H00
        self.H01 = H01
        self.log_disc_base=log_disc_base
        #================================>

        # CAS specific parameters
        self.cas = False
        self.casno = 'gw'
        self.composite = False
        self.thresh = None
        self.nvir_act = None
        self.nocc_act = None
        self.ea_cut = None
        self.ip_cut = None
        self.ea_no = None
        self.ip_no = None
        self.vno_only = True
        self.save_gf = False
        self.read_gf = False

        # DMRG specific parameters
        self.gs_n_steps = None
        self.gf_n_steps = None
        self.gs_tol = None
        self.gf_tol = None
        self.gs_bond_dims = None
        self.gs_noises = None
        self.gf_bond_dims = None
        self.gf_noises = None
        self.dmrg_gmres_tol = None
        self.dmrg_verbose = 1
        self.reorder_method = None
        self.n_off_diag_cg = 0
        self.load_dir = None
        self.save_dir = './gs_mps'

    def dump_flags(self):
        if self.verbose < logger.INFO:
            return self

        if rank == 0:
            logger.info(self, '\n')
            logger.info(self, '******** %s flags ********', self.__class__)
            logger.info(self, 'impurity solver = %s', self.solver_type)
            logger.info(self, 'discretization method = %s', self.disc_type)
            logger.info(self, 'n impurity orbitals = %d', self.nao)
            logger.info(self, 'n core orbitals = %d', self.ncore)
            logger.info(self, 'n freezing orbitals in solver = %d', self.nfrz)
            logger.info(self, 'n bath orbital energies = %d', self.nbath)
            logger.info(self, 'n bath orbitals per bath energy = %d', self.nb_per_e)
            logger.info(self, 'n bath orbitals total = %d', self.nbath*self.nb_per_e)
            logger.info(self, 'nkpts in lattice = %d', self.nkpts)
            if self.opt_mu:
                logger.info(self, 'mu will be optimized, init guess = %s, target occupancy = %s',
                            self.mu, self.occupancy)
            else:
                logger.info(self, 'mu is fixed, mu = %g', self.mu)
            logger.info(self, 'damping factor = %g', self.damp)
            logger.info(self, 'DMFT convergence tol = %g', self.conv_tol)
            logger.info(self, 'max. DMFT cycles = %d', self.max_cycle)
            logger.info(self, 'GMRES convergence tol = %g', self.gmres_tol)
            logger.info(self, 'delta for discretization = %g', self.delta)
            logger.info(self, 'using diis = %s', self.diis)
            if self.diis:
                logger.info(self, 'diis_space = %d', self.diis_space)
                logger.info(self, 'diis_start_cycle = %d', self.diis_start_cycle)
            if self.chkfile:
                logger.info(self, 'chkfile to save DMFT result = %s', self.chkfile)
        return self

    def dump_chk(self):
        if self.chkfile:
            with h5py.File(self.chkfile, 'w') as fh5:
                fh5['dmft/hyb'] = self.hyb
                fh5['dmft/sigma'] = self.sigma
                fh5['dmft/solver_type'] = self.solver_type
                fh5['dmft/disc_type'] = self.disc_type
                fh5['dmft/mu'] = self.mu
                fh5['dmft/delta'] = self.delta
                fh5['dmft/freqs'] = self.freqs
                fh5['dmft/wts'] = self.wts
        return self

    def get_kgw_sigma(self, freqs, eta):
        '''
        Get k-point GW-AC self-energy in LO basis
        '''
        fn = 'ac_coeff.h5'
        feri = h5py.File(fn, 'r')
        coeff = np.asarray(feri['coeff'])
        ef = np.asarray(feri['fermi'])
        omega_fit = np.asarray(feri['omega_fit'])
        feri.close()

        fn = 'C_mo_lo.h5'
        feri = h5py.File(fn, 'r')
        C_mo_lo = np.asarray(feri['C_mo_lo'])
        C_ao_lo = np.asarray(feri['C_ao_lo'])
        feri.close()

        nw = len(freqs)
        spin, nkpts, nao, nlo = C_mo_lo.shape
        sigma = np.zeros([spin,nkpts,nao,nao,nw], dtype=np.complex)
        if coeff.ndim == 4:
            coeff = coeff[np.newaxis, ...]
        for s in range(spin):
            for k in range(nkpts):
                for p in range(nao):
                    for q in range(nao):
                        sigma[s,k,p,q] = krgw_gf.pade_thiele(freqs-ef+1j*eta, omega_fit, coeff[s,k,:,p,q])

        sigma_lo = np.zeros([spin,nkpts,nlo,nlo,nw], dtype=np.complex)
        for s in range(spin):
            for iw in range(len(freqs)):
                for k in range(nkpts):
                    sigma_lo[s,k,:,:,iw] = np.dot(np.dot(C_mo_lo[s,k].T.conj(),
                                                   sigma[s,k,:,:,iw]), C_mo_lo[s,k])

        return sigma_lo

    def get_gw_sigma(self, freqs, eta):
        '''
        Get local GW double counting self-energy
        '''
        spin, nao, nbath = self.spin, self.nao, self.nbath
        nw = len(freqs)

        fn = 'imp_ac_coeff.h5'
        feri = h5py.File(fn, 'r')
        coeff = np.asarray(feri['coeff'])
        ef = np.asarray(feri['fermi'])
        omega_fit = np.asarray(feri['omega_fit'])
        feri.close()

        sigma = np.zeros([spin,nao,nao,nw], dtype=np.complex)
        if coeff.ndim == 3:
            coeff = coeff[np.newaxis, ...]
        for s in range(spin):
            for p in range(nao):
                for q in range(nao):
                    sigma[s,p,q] = gw_dc.pade_thiele(freqs-ef+1j*eta, omega_fit, coeff[s,:,p,q])

        return sigma

    def kernel(self, mu0, wl=None, wh=None, occupancy=None, delta=0.1,
               conv_tol=None, opt_mu=False, dump_chk=True):
        '''
        main routine for DMFT

        Args:
            mu0 : float
                Chemical potential or an initial guess if opt_mu=True

        Kwargs:
            wl, wh : None or float
                Hybridization discretization range
            occupancy : None or float
                Target average occupancy (1 is half filling)
            delta : float
                Broadening used during self-consistency
            conv_tol : float
                Convergence tolerance on the hybridization
            opt_mu : bool
                Whether to optimize the chemical potential
            dump_chk : bool
                Whether to dump DMFT chkfile
        '''

        cput0 = (time.process_time(), time.time())
        self.mu = mu0
        self.occupancy = occupancy
        self.delta = delta
        if conv_tol:
            self.conv_tol = conv_tol
        self.opt_mu = opt_mu
        if opt_mu:
            assert(self.occupancy is not None)

        self.dump_flags()

        self.converged, self.mu = kernel(self, mu0, wl=wl, wh=wh, occupancy=occupancy, delta=delta,
                                         conv_tol=conv_tol, opt_mu=opt_mu, dump_chk=dump_chk)

        if rank == 0:
            self._finalize()
            logger.timer(self, 'DMFT', *cput0)

    def dmft(self, **kwargs):
        return self.kernel(**kwargs)

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        if self.converged:
            logger.info(self, '%s converged', self.__class__.__name__)
        else:
            logger.note(self, '%s not converged', self.__class__.__name__)
        return self

    def run_diis(self, hyb, istep, adiis):
        if (adiis and istep >= self.diis_start_cycle):
            hyb = adiis.update(hyb)
            logger.debug1(self, 'DIIS for step %d', istep)
        return hyb

    def get_rdm_imp(self):
        '''Calculate the interacting local RDM from the impurity problem'''
        if self.solver_type == 'cc':
            assert(self.nfrz == 0)
            return cc_rdm(self._scf, ao_orbs=range(self.nao), cas=self.cas, casno=self.casno,
                          composite=self.composite, thresh=self.thresh, nvir_act=self.nvir_act,
                          nocc_act=self.nocc_act, ea_cut=self.ea_cut, ip_cut=self.ip_cut,
                          ea_no=self.ea_no, ip_no=self.ip_no, vno_only=self.vno_only)
        elif self.solver_type == 'ucc':
            return ucc_rdm(self._scf, self.nfrz, ao_orbs=range(self.nao))
        elif self.solver_type == 'fci':
            assert (self.nfrz == 0)
            return fci_rdm(self._scf, ao_orbs=range(self.nao))
        elif self.solver_type == 'dmrg':
            assert(self.nfrz == 0)
            return dmrg_rdm(self._scf, ao_orbs=range(self.nao), n_threads=self.n_threads,
                            cas=self.cas, casno=self.casno, composite=self.composite, thresh=self.thresh,
                            nvir_act=self.nvir_act, nocc_act=self.nocc_act, ea_cut=self.ea_cut,
                            ip_cut=self.ip_cut, ea_no=self.ea_no, ip_no=self.ip_no, vno_only=self.vno_only,
                            reorder_method=self.reorder_method, gs_n_steps=self.gs_n_steps,
                            gs_tol=self.gs_tol, gs_bond_dims=self.gs_bond_dims, gs_noises=self.gs_noises,
                            load_dir=self.load_dir, save_dir=self.save_dir, dyn_corr_method=self.dyn_corr_method,
                            ncore=self.nocc_act_low, nvirt=self.nvir_act_high)
        elif self.solver_type == 'dmrgsz':
            assert(self.nfrz == 0)
            return udmrg_rdm(self._scf, ao_orbs=range(self.nao),
                            n_threads=self.n_threads)

    def get_gf_imp(self, freqs, delta, ao_orbs=None, extra_freqs=None, extra_delta=None, use_gw=False):
        '''Calculate the interacting local GF from the impurity problem'''
        if self.solver_type == 'cc':
            assert(self.nfrz == 0)
            return cc_gf(self._scf, freqs, delta, ao_orbs=ao_orbs, gmres_tol=self.gmres_tol,
                         cas=self.cas, casno=self.casno, composite=self.composite, thresh=self.thresh,
                         nvir_act=self.nvir_act, nocc_act=self.nocc_act, ea_cut=self.ea_cut,
                         ip_cut=self.ip_cut, ea_no=self.ea_no, ip_no=self.ip_no, vno_only=self.vno_only)
        elif self.solver_type == 'ucc':
            return ucc_gf(self._scf, freqs, delta, self.nfrz, ao_orbs=range(self.nao-self.nfrz),
                         gmres_tol=self.gmres_tol)
        elif self.solver_type == 'fci':
            # TODO: CASCI
            assert (self.nfrz == 0)
            return fci_gf(self._scf, freqs, delta, ao_orbs=range(self.nao),
                          gmres_tol=self.gmres_tol)
        elif self.solver_type == 'dmrg':
            assert(self.nfrz == 0)
            return dmrg_gf(self._scf, freqs, delta, ao_orbs=ao_orbs, n_threads=self.n_threads,
                    cas=self.cas, casno=self.casno, composite=self.composite,
                    thresh=self.thresh, nvir_act=self.nvir_act, nocc_act=self.nocc_act, ea_cut=self.ea_cut,
                    ip_cut=self.ip_cut, ea_no=self.ea_no, ip_no=self.ip_no, vno_only=self.vno_only,
                    reorder_method=self.reorder_method, cc_gmres_tol=self.gmres_tol, gf_n_steps=self.gf_n_steps,
                    gs_n_steps=self.gs_n_steps, gs_tol=self.gs_tol, dmrg_verbose=self.dmrg_verbose,
                    gs_bond_dims=self.gs_bond_dims, gf_bond_dims=self.gf_bond_dims, gf_tol=self.gf_tol,
                    gmres_tol=self.dmrg_gmres_tol, gs_noises=self.gs_noises, gf_noises=self.gf_noises,
                    save_dir=self.save_dir, load_dir=self.load_dir, save_gf=self.save_gf, read_gf=self.read_gf,
                    n_off_diag_cg=self.n_off_diag_cg, extra_freqs=extra_freqs, extra_delta=extra_delta, use_gw=use_gw,
                    dyn_corr_method=self.dyn_corr_method, ncore=self.nocc_act_low, nvirt=self.nvir_act_high)
        elif self.solver_type == 'dmrgsz':
            assert(self.nfrz == 0)
            return udmrg_gf(self._scf, freqs, delta, ao_orbs=range(self.nao),
                         n_threads=self.n_threads)

    def get_gf0_imp(self, freqs, delta):
        '''Calculate the noninteracting local GF from the impurity problem'''
        himp = self._scf.get_hcore()
        if len(himp.shape) == 2:
            himp = himp[np.newaxis, ...]
        spin, nb = himp.shape[0:2]
        nw = len(freqs)
        sig_dum = np.zeros((spin,nb,nb,nw,))
        gf = get_gf(himp, sig_dum, freqs, delta)
        return gf

    def get_sigma_imp(self, freqs, delta, load_dir=None, save_dir=None, save_gf=False, read_gf=False):
        '''Calculate the local self-energy from the impurity problem'''
        spin = self.spin
        if spin == 1:
            nmo = len(self._scf.mo_energy)
        else:
            nmo = len(self._scf.mo_energy[0])
        nao = self.nao
        nfrz = self.nfrz
        gf0 = self.get_gf0_imp(freqs, delta)
        # HF GF
        gfhf_imp, gfhf_imp_frz = mf_gf_withfrz(self._scf, freqs, delta, ao_orbs=range(nmo), nfrz=self.nfrz)

        if self.solver_type == 'cc':
            assert(self.nfrz == 0)
            gf = cc_gf(self._scf, freqs, delta, ao_orbs=range(nmo), gmres_tol=self.gmres_tol,
                       nimp=self.nao, cas=self.cas, casno=self.casno, composite=self.composite,
                       thresh=self.thresh, nvir_act=self.nvir_act, nocc_act=self.nocc_act,
                       ea_cut=self.ea_cut, ip_cut=self.ip_cut, ea_no=self.ea_no,
                       ip_no=self.ip_no, vno_only=self.vno_only, save_gf=save_gf, read_gf=read_gf)
        elif self.solver_type == 'ucc':
            gf = ucc_gf(self._scf, freqs, delta, self.nfrz, ao_orbs=range(nmo-self.nfrz),
                        gmres_tol=self.gmres_tol, nimp=self.nao)
        elif self.solver_type == 'fci':
            # TODO: CASCI
            assert (self.nfrz == 0)
            gf = self.get_gf_imp(freqs, delta)
        elif self.solver_type == 'dmrg':
            assert(self.nfrz == 0)
            gf = dmrg_gf(self._scf, freqs, delta, ao_orbs=range(nmo), n_threads=self.n_threads,
                    nimp=self.nao, cas=self.cas, casno=self.casno, composite=self.composite,
                    thresh=self.thresh, nvir_act=self.nvir_act, nocc_act=self.nocc_act, ea_cut=self.ea_cut,
                    ip_cut=self.ip_cut, ea_no=self.ea_no, ip_no=self.ip_no, vno_only=self.vno_only,
                    reorder_method=self.reorder_method, cc_gmres_tol=self.gmres_tol, gf_n_steps=self.gf_n_steps,
                    gs_n_steps=self.gs_n_steps, gs_tol=self.gs_tol, dmrg_verbose=self.dmrg_verbose,
                    gs_bond_dims=self.gs_bond_dims, gf_bond_dims=self.gf_bond_dims, gf_tol=self.gf_tol,
                    gmres_tol=self.dmrg_gmres_tol, gs_noises=self.gs_noises, gf_noises=self.gf_noises,
                    save_gf=save_gf, read_gf=read_gf, load_dir=load_dir, save_dir=save_dir,
                    n_off_diag_cg=self.n_off_diag_cg, dyn_corr_method=self.dyn_corr_method,
                    ncore=self.nocc_act_low, nvirt=self.nvir_act_high)
        elif self.solver_type == 'dmrgsz':
            assert(self.nfrz == 0)
            gf = udmrg_gf(self._scf, freqs, delta,  ao_orbs=range(nmo),
                       n_threads=self.n_threads, nimp=self.nao)

        if self.solver_type == 'cc' or self.solver_type == 'ucc':
            gf = 0.5 * (gf+gf.transpose(0,2,1,3))

        # Frozen core CCSD GF (core GF from HF)
        gf = gf - gfhf_imp_frz[:,nfrz:,nfrz:]
        gf_imp = gfhf_imp.copy()
        gf_imp[:,nfrz:,nfrz:] += gf

        tmpdir = 'dmft_dos'
        if rank == 0:
            if not os.path.isdir(tmpdir):
                os.mkdir(tmpdir)
            write.write_gf_to_dos(tmpdir+'/dmft_imp_dos', freqs, gf_imp)

        return get_sigma(gf0, gf_imp)

    def get_ldos_imp(self, freqs, delta, extra_freqs=None, extra_delta=None, use_gw=False):
        '''Calculate the local DOS from the impurity problem'''
        nw = len(freqs)
        nao = self.nao
        nmo = len(self._scf.mo_energy)

        if self.solver_type == 'cc':
            if self.cc_ao_orbs is None:
                self.cc_ao_orbs = range(nmo)
            ao_orbs = self.cc_ao_orbs
        elif self.solver_type == 'dmrg':
            ao_orbs = range(nmo)

        if extra_delta is None:
            gf0_full = self.get_gf0_imp(freqs, delta)
        else:
            gf0_full = self.get_gf0_imp(np.array(extra_freqs).reshape(-1), extra_delta)
            nw = len(np.array(extra_freqs).reshape(-1))


        if self.solver_type == 'hf':
            gf = mf_gf(self._scf, freqs, delta)
        else:
            # ZJ: the function below contains the core subroutine 'cc_gf'
            gf = self.get_gf_imp(freqs, delta, ao_orbs=ao_orbs,
                                 extra_freqs=extra_freqs, extra_delta=extra_delta, use_gw=use_gw)

        if self.solver_type == 'cc':
            if ao_orbs == self.imp_idx:
                gf0 = np.zeros_like(gf)
                idx = self.imp_idx
                for ia,a in enumerate(idx):
                    for ib,b in enumerate(idx):
                        gf0[:,ia,ib,:] = gf0_full[:,a,b,:]
                sigma = get_sigma(gf0, gf)
                ldos_t2g = -1./np.pi*(gf[:,0,0,:].imag)
                ldos_eg = -1./np.pi*(gf[:,1,1,:].imag)
            elif ao_orbs == range(self.nao):
                idx = self.imp_idx
                gf0 = gf0_full[:,:self.nao,:self.nao,:]
                sigma_full = get_sigma(gf0, gf)
                sigma = np.zeros((self.spin,2,2,nw),dtype=np.complex)
                for ia,a in enumerate(idx):
                    sigma[:,ia,ia,:] = sigma_full[:,a,a,:]
                ldos_t2g = -1./np.pi*(gf[:,idx[0],idx[0],:].imag)
                ldos_eg = -1./np.pi*(gf[:,idx[1],idx[1],:].imag)
            elif ao_orbs == range(self.ncore, self.nval):
                idx = self.imp_idx
                gf0 = gf0_full[:,self.ncore:self.nval,self.ncore:self.nval]
                sigma_full = get_sigma(gf0, gf)
                sigma = np.zeros((self.spin,2,2,nw),dtype=np.complex)
                for ia,a in enumerate(idx):
                    sigma[:,ia,ia,:] = sigma_full[:,a-self.ncore,a-self.ncore,:]
                ldos_t2g = -1./np.pi*(gf[:,idx[0]-self.ncore,idx[0]-self.ncore,:].imag)
                ldos_eg = -1./np.pi*(gf[:,idx[1]-self.ncore,idx[1]-self.ncore,:].imag)
            elif ao_orbs == range(nmo):
                idx = self.imp_idx
                sigma_full = get_sigma(gf0_full, gf)
                sigma = np.zeros((self.spin,2,2,nw),dtype=np.complex)
                for ia,a in enumerate(idx):
                    sigma[:,ia,ia,:] = sigma_full[:,a,a,:]
                ldos_t2g = -1./np.pi*(gf[:,idx[0],idx[0],:].imag)
                ldos_eg = -1./np.pi*(gf[:,idx[1],idx[1],:].imag)
        elif self.solver_type == 'dmrg':
            if ao_orbs == range(self.nao):
                idx = self.imp_idx
                gf0 = gf0_full[:,:self.nao,:self.nao,:]
                sigma_full = get_sigma(gf0, gf)
                sigma = np.zeros((self.spin,2,2,nw),dtype=np.complex)
                for ia,a in enumerate(idx):
                    sigma[:,ia,ia,:] = sigma_full[:,a,a,:]
                ldos_t2g = -1./np.pi*(gf[:,idx[0],idx[0],:].imag)
                ldos_eg = -1./np.pi*(gf[:,idx[1],idx[1],:].imag)
            elif ao_orbs == range(nmo):
                idx = self.imp_idx
                sigma_full = get_sigma(gf0_full, gf)
                sigma = np.zeros((self.spin,2,2,nw),dtype=np.complex)
                for ia,a in enumerate(idx):
                    sigma[:,ia,ia,:] = sigma_full[:,a,a,:]
                ldos_t2g = -1./np.pi*(gf[:,idx[0],idx[0],:].imag)
                ldos_eg = -1./np.pi*(gf[:,idx[1],idx[1],:].imag)
        elif self.solver_type == 'hf':
            idx = self.imp_idx
            gf0 = gf0_full
            sigma_full = get_sigma(gf0, gf)
            sigma = np.zeros((self.spin,2,2,nw),dtype=np.complex)
            for ia,a in enumerate(idx):
                sigma[:,ia,ia,:] = sigma_full[:,a,a,:]
            ldos_t2g = -1./np.pi*(gf[:,idx[0],idx[0],:].imag)
            ldos_eg = -1./np.pi*(gf[:,idx[1],idx[1],:].imag)


        return ldos_t2g, ldos_eg, sigma

    def get_ldos_latt(self, freqs, delta, sigma=None):
        '''Calculate local DOS from the lattice problem'''
        nw = len(freqs)
        nao = self.nao
        nkpts = self.nkpts
        spin = self.spin
        nval = self.nval
        nfrz = self.nfrz

        if sigma is None:
            sigma = self.get_sigma_imp(freqs, delta, save_gf=self.save_gf, read_gf=self.read_gf,
                                       load_dir=self.load_dir, save_dir=self.save_dir)
        nb = self.nbath
        sigma = sigma[:,:nao,:nao,:]
        JK_00 = self.JK_00

        if self.gw_dmft:
            # Compute impurity GW self-energy (DC term)
            sigma_gw_imp = self.get_gw_sigma(freqs, delta)

            # Compute k-point GW self-energy at given freqs and delta
            sigma_kgw = self.get_kgw_sigma(freqs, delta)
        else:
            sigma_gw_imp = np.zeros((spin, nao, nao, nw), dtype=np.complex)
            sigma_kgw = np.zeros((spin, nkpts, nao, nao, nw), dtype=np.complex)

        # Choose to use GW correlation self-energy for core
        sigma_gw_imp_frz = sigma_gw_imp.copy()
        sigma_gw_imp_frz[:,nfrz:,nfrz:] = 0.
        sigma = sigma + sigma_gw_imp_frz

        # remove GW double counting
        for w in range(nw):
            sigma[:,:,:,w] = sigma[:,:,:,w] - JK_00
        sigma = sigma - sigma_gw_imp

        tmpdir = 'dmft_dos'
        if rank == 0:
            write.write_sigma(tmpdir+'/dmft_sigma_imp_prod', freqs, sigma)
            fn = 'sigma_nb-%d_eta-%0.2f_w-%.3f-%.3f.h5'%(nb, delta*27.211386,
                                                     freqs[0]*27.211386, freqs[-1]*27.211386)
            feri = h5py.File(fn, 'w')
            feri['omegas'] = np.asarray(freqs)
            feri['sigma'] = np.asarray(sigma)
            feri.close()
        comm.Barrier()

        # k-point GW GF
        gf_loc = np.zeros([spin, nao, nao, nw], np.complex128)
        for k in range(nkpts):
            gf = get_gf(self.hcore_k[:,k]+self.JK_k[:,k], sigma_kgw[:,k], freqs, delta)
            gf_loc += 1./nkpts * gf
            if rank == 0:
                if self.gw_dmft:
                    write.write_gf_to_dos(tmpdir+'/gw_dos_k-%d'%(k), freqs, gf)
                else:
                    write.write_gf_to_dos(tmpdir+'/hf_dos_k-%d'%(k), freqs, gf)
            comm.Barrier()

        ldos_gw = -1./np.pi * np.trace(gf_loc.imag,axis1=1,axis2=2)
        if rank == 0:
            for i in range(nao):
                ldos_orb = -1./np.pi * gf_loc[:,i,i,:].imag
                if self.gw_dmft:
                    write.write_dos(tmpdir+'/gw_dos_orb-%d'%(i), freqs, ldos_orb)
                else:
                    write.write_dos(tmpdir+'/hf_dos_orb-%d'%(i), freqs, ldos_orb)
        comm.Barrier()

        # DMFT GF
        gf_loc = np.zeros([spin, nao, nao, nw], np.complex128)
        for k in range(nkpts):
            gf = get_gf(self.hcore_k[:,k]+self.JK_k[:,k], sigma+sigma_kgw[:,k], freqs, delta)
            gf_loc += 1./nkpts * gf
            if rank == 0:
                write.write_gf_to_dos(tmpdir+'/dmft_dos_prod_k-%d'%(k), freqs, gf)
            comm.Barrier()

        ldos = -1./np.pi * np.trace(gf_loc.imag,axis1=1,axis2=2)
        if rank == 0:
            for i in range(nao):
                ldos_orb = -1./np.pi * gf_loc[:,i,i,:].imag
                write.write_dos(tmpdir+'/dmft_dos_prod_orb-%d'%(i), freqs, ldos_orb)
        comm.Barrier()

        return ldos, ldos_gw

