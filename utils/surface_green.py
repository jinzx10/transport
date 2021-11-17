'''
Calculate the surface Green's function of
a block tri-diagonal Hamiltonian

Given a sufficiently large (or infinite) block tri-diagonal Hamiltonian

H = [   [H00, H01, 0  , ..., ..., ..., ..., ...],
        [H10, H00, H01,  0 , ..., ..., ..., ...],
        [ 0 , H10, H00, H10,  0 , ..., ..., ...],
                    .    .    .
                         .    .    .
                              .    .    .
        [..., ..., ..., ...,  0 , H10, H00, H01],
        [..., ..., ..., ...,  0 ,  0 , H10, H00] ]

where H00=H00.H, H01 is a square matrix, H10 = H01.H,
and a similar block tri-diagonal overlap matrix S

S = [   [S00, S01, 0  , ..., ..., ..., ..., ...],
        [S10, S00, S01,  0 , ..., ..., ..., ...],
        [ 0 , S10, S00, S10,  0 , ..., ..., ...],
                    .    .    .
                         .    .    .
                              .    .    .
        [..., ..., ..., ...,  0 , S10, S00, S01],
        [..., ..., ..., ...,  0 ,  0 , S10, S00] ]

the total Green's function G(z) is given by

(zS-H) G(z) = I

The surface Green's function is the upper-left block of G.

'''

import numpy as np

def LopezSancho1984(z, H00, H01, S00 = None, S01 = None, max_iter = 50, conv_thr = 1e-8):
    '''
    See M P Lopez Sancho et al 1984 J. Phys. F: Met. Phys. 14 1205
    '''

    sz = np.size(H00,0)

    # if the basis overlap matrix is not given, assume an orthonormal basis
    if S00 is None:
        S00 = np.eye(sz)

    if S01 is None:
        S01 = np.zeros((sz,sz))

    # tt stands for \tilde{t}
    t  = -np.linalg.solve(z*S00-H00, z*S01.T.conj()-H01.T.conj())
    tt = -np.linalg.solve(z*S00-H00, z*S01-H01)

    # cumulative product of \tilde{t}
    tt_cumprod = np.copy(tt)

    # transfer matrix
    T = np.copy(t)

    for i in range(0, max_iter):
        t_new  = np.linalg.solve(np.eye(sz)-t@tt-tt@t, t) @ t
        tt_new = np.linalg.solve(np.eye(sz)-t@tt-tt@t, tt) @ tt
        T = T + tt_cumprod @ t_new

        #if np.linalg.norm(t_new) < conv_thr and np.linalg.norm(tt_new) < conv_thr:
        if np.max(abs(t_new)) < conv_thr and np.max(abs(tt_new)) < conv_thr:
            #print('convergence achieved after ', i, ' iterations')
            return np.linalg.inv(z*S00-H00+(z*S01-H01)@T)

        t = t_new
        tt = tt_new
        tt_cumprod = tt_cumprod @ tt

    print("Surface Green's function calculation fails to converge.")



def LopezSancho1985(z, H00, H01, S00 = None, S01 = None, max_iter = 50, conv_thr = 1e-8):
    '''
    See M P Lopez Sancho et al 1985 J. Phys. F: Met. Phys. 15 851
    '''

    sz = np.size(H00,0)

    # if the basis overlap matrix is not given, assume an orthonormal basis
    if S00 is None:
        S00 = np.eye(sz)

    if S01 is None:
        S01 = np.zeros((sz,sz))

    alpha = -z*S01 + H01
    beta = -z*S01.T.conj() + H01.T.conj()
    epsilon = H00
    epsilon_s = H00

    for i in range(0, max_iter):
        if np.max(abs(alpha)) < conv_thr:
        #if np.linalg.norm(alpha) < conv_thr:
            return np.linalg.inv(z*S00-epsilon_s)

        iga = np.linalg.solve(z*S00-epsilon, alpha)
        igb = np.linalg.solve(z*S00-epsilon, beta)

        epsilon_s = epsilon_s + alpha @ igb
        epsilon = epsilon + alpha @ igb + beta @ iga
        alpha = alpha @ iga
        beta = beta @ igb

    print("Surface Green's function calculation fails to converge.")



