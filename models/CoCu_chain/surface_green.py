'''
Calculate the surface Green's function of
a semi-infinite, block tri-diagonal Hamiltonian

Given a sufficiently large (or semi-infinite) block tri-diagonal Hamiltonian (H == H.T.conj())

H = [   [H00, H01, 0  , ..., ..., ..., ..., ...],
        [H10, H00, H01,  0 , ..., ..., ..., ...],
        [ 0 , H10, H00, H10,  0 , ..., ..., ...],
                    .    .    .
                         .    .    .
                              .    .    .
        [..., ..., ..., ...,  0 , H10, H00, H01],
        [..., ..., ..., ...,  0 ,  0 , H10, H00] ]

and a corresponding basis overlap matrix with similar structure:

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

This script contains functions that compute G00, the upper-left corner block of G.

'''

import numpy as np

def LopezSancho1984(z, H00, H01, S00 = None, S01 = None, max_iter = 50, conv_thr = 1e-8):
    '''
    See M. P. Lopez Sancho et al, J. Phys. F: Met. Phys. 14 1205 (1984)
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

        if np.linalg.norm(t_new,1) < conv_thr and np.linalg.norm(tt_new,1) < conv_thr:
            #print('convergence achieved after ', i, ' iterations')
            return np.linalg.inv(z*S00-H00+(z*S01-H01)@T)

        t = t_new
        tt = tt_new
        tt_cumprod = tt_cumprod @ tt

    print("Surface Green's function calculation fails to converge.")



def LopezSancho1985(z, H00, H01, S00 = None, S01 = None, max_iter = 50, conv_thr = 1e-8):
    '''
    See M. P. Lopez Sancho et al, J. Phys. F: Met. Phys. 15 851 (1985)
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
        if np.linalg.norm(alpha,1) < conv_thr:
            return np.linalg.inv(z*S00-epsilon_s)

        iga = np.linalg.solve(z*S00-epsilon, alpha)
        igb = np.linalg.solve(z*S00-epsilon, beta)

        epsilon_s = epsilon_s + alpha @ igb
        epsilon = epsilon + alpha @ igb + beta @ iga
        alpha = alpha @ iga
        beta = beta @ igb

    print("Surface Green's function calculation fails to converge.")



def Umerski1997(z, H00, H01, S00 = None, S01 = None):
    '''
    See A. Umerski, Phys. Rev. B 55, 5266 (1997)
    '''

    sz = np.size(H00,0)

    # if the basis overlap matrix is not given, assume an orthonormal basis
    if S00 is None:
        S00 = np.eye(sz)

    if S01 is None:
        S01 = np.zeros((sz,sz))

    t = z*S01.T.conj()-H01.T.conj()

    # mrdivide(A,B) returns A * inv(B)
    mrdivide = lambda A, B: np.linalg.solve(B.T, A.T).T

    X = np.block([
        [np.zeros((sz,sz)), np.linalg.inv(t)], 
        [-z*S01+H01       , mrdivide(z*S00-H00, t)] 
        ])

    # eigen-decomposition of X with eigenvalues in ascending order
    Lambda, O = np.linalg.eig(X)
    idx_sort = np.argsort(abs(Lambda))
    lambda2 = Lambda[idx_sort][sz:]
    o2 = O[:, idx_sort][:sz, sz:]

    return mrdivide(o2/lambda2, t@o2)



