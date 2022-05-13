from linesearch import linesearch
import numpy as np

def rotate(C_in, U, Sigma, V, alpha):
    return ( C_in @ V * np.cos(alpha*Sigma) + U * np.sin(alpha*Sigma) ) @ V.T


def PT2(M, U, Delta):
    # Delta2 = PT(C, U, Sigma, V, alpha, Delta) parallel transports a tangent
    # vector Delta at C along U*diag(Sigma)*V' by alpha to Delta2

    # M = -C*V.*sin(Sigma*alpha) + U.*(cos(Sigma*alpha)-1)

    return Delta + M @ ( U.T @ Delta )


def rPT2(M, U, Delta2):
    # Delta = rPT(C, U, Sigma, V, alpha, Delta2) reverses the previous
    # operation that it parallel transports Delta2 back to Delta. Note that all
    # the variables (C, U, Sigma, V, alpha) are given at where Delta is defined
    # rather than where Delta2 is defined, so that there is
    # Delta = rPT( C, U, Sigma, V, alpha, PT(C, U, Sigma, V, alpha, Delta) )
    # rPT is used in the parallel tranport of the approximate inverse Hessian.

    # M = -C*V.*sin(Sigma*alpha) + U.*(cos(Sigma*alpha)-1)

    return Delta2 - M @ ( (M+U).T @ Delta2 )


# limited-memory geometric direct minimization
def lgdm(E, dE, C0, conv_thr=1e-6, max_iter=1000, store_limit=15, wolfe1=0.2, wolfe2=0.6):
    C = C0

    # iteration history
    # each row contains [err, E, alpha]
    # where alpha is the line search step size
    iter_hist = np.zeros((max_iter,3))

    # gradient
    G = dE(C)
    G = G - C @ (C.T @ G)

    # error
    get_err = lambda G: np.max(np.abs(G))

    err = get_err(G)
    if err < conv_thr:
        print('convergence achieved before iteration!')
        return C, 0, np.array([[err, E(C), 0]])

    # sizes of the basis and occupied space
    N, L = C.shape

    # initial approximated inverse Hessian (diagonal)
    B0_diag = np.ones(N*L)

    # initial line search step size
    alpha = 1.0

    # BFGS inverse Hessian update vectors
    S = np.zeros((N*L, store_limit))
    Y = np.zeros((N*L, store_limit))
    YS = np.zeros(store_limit) # inner product Y'*S

    # stored for parallel transport purpose
    UU = np.zeros((N,L,store_limit))
    MM = np.zeros((N,L,store_limit))

    def BG2(i, n, G):
        # calculate Delta=B*G where B is the approximate inverse Hessian and G is a
        # tangent vector
        # B is stored by its initial guess (B0_diag) and update vectors (S and Y)
        #
        # in BFGS update, the original B is now replaced by (PT*B*rPT) where PT and
        # rPT are parallel transport and its inverse operator, e.g.
        # B0 = diag(B0_diag)
        # B1 = (I-(s1*y1')/(y1'*s1))*(PT1*B0*rPT1)*(I-(y1*s1')/(y1'*s1))+ (s1*s1')/(y1'*s1)
        #
        # the product is calculated recursively by using the fact that
        # Bk = (I-(sk*yk')/(yk'*sk))*(PTk*B(k-1)*rPTk)*(I-(yk*sk')/(yk'*sk)) + (sk*sk')/(yk'*sk)
        if n == 0:
            Delta = G * np.reshape(B0_diag, (N,L))
        else:
            SG = np.dot(S[:,i], G.reshape(-1))

            # tmp = (I-(yk*sk')/(yk'*sk))*G
            tmp = G - np.reshape( Y[:,i] * (SG/YS[i]), (N,L) )

            # tmp = (PTk*B(k-1)*rPTk)*tmp
            tmp = rPT2(MM[:,:,i], UU[:,:,i], tmp)
            tmp = BG2(np.mod(i-1, store_limit), n-1, tmp)
            tmp = PT2(MM[:,:,i], UU[:,:,i], tmp)

            # tmp = (I-(sk*yk')/(yk'*sk))*tmp
            tmp = tmp - np.reshape( S[:,i] * (np.dot(Y[:,i],tmp.reshape(-1))/YS[i]), (N,L) )

            # Delta = tmp + (sk*sk')/(yk'*sk)*G
            Delta = tmp + np.reshape( S[:,i] * (SG/YS[i]), (N,L) );

        return Delta


    for i in range(0, max_iter):

        # index for quantities (UU,MM,S,Y,YS) to store for this iteration
        istore = np.mod(i, store_limit)

        # get search direction Delta
        # for i=0, Delta is simply the steepest descent direction:
        # Delta(i=0) = (I-C*C^T)*dE
        # in subsequent iterations, the direction is the approximate inverse
        # Hessian times the steepest descent direction
        Delta = -BG2( np.mod(istore-1,store_limit), min(i, store_limit), G)
        Delta = Delta - C @ ( C.T @ Delta ) # optional?

        # Delta is N-by-L
        U, Sigma, VT = np.linalg.svd(Delta, full_matrices=False)
        V = VT.T
        df = lambda a: np.sum( dE( rotate(C, U, Sigma, V, a) ) \
                * ( ( C @ V * (-Sigma*np.sin(a*Sigma)) + U * (Sigma*np.cos(a*Sigma)) ) @ V.T ) )
        if df(0) > 0:
            Delta = -Delta
            V = -V
            df = lambda a: np.sum( dE( rotate(C, U, Sigma, V, a) ) \
                    * ( ( C @ V * (-Sigma*np.sin(a*Sigma)) + U * (Sigma*np.cos(a*Sigma)) ) @ V.T ) )

        f = lambda a: E( rotate(C, U, Sigma, V, a) )

        # find a step size that satisfies the strong Wolfe conditions
        alpha, ls_flag = linesearch(f, df, alpha0=alpha, wolfe1=wolfe1, wolfe2=wolfe2)

        # matrices used in parallel transport
        M = -C @ V * np.sin(Sigma*alpha) + U * (np.cos(Sigma*alpha)-1)
        UU[:,:,istore] = U
        MM[:,:,istore] = M

        # update coordinate
        C_new = rotate(C, U, Sigma, V, alpha)

        # in normal BFGS we have s = alpha*p where p is the search direction
        # here p corresponds to Delta, but Delta is define with respect to the
        # old position C, and we need to parallel tranport it to the current
        # position (C_new)
        S[:,istore] = alpha * PT2(M, U, Delta).reshape(-1)

        # update gradient
        G_new = dE(C_new)
        G_new = G_new - C_new @ (C_new.T @ G_new)

        # in normal BFGS we have y = G_new - G
        # but here G is defined with respect to the old position, so we need to
        # parallel transport it to the current position
        Y[:,istore] = (G_new - PT2(M, U, G)).reshape(-1)
        YS[istore] = np.dot(Y[:,istore], S[:,istore])
        B0_diag = np.ones(N*L) * np.abs( YS[istore] / np.dot(Y[:,istore],Y[:,istore]) )

        G = G_new
        C = C_new

        err = get_err(G)
        Enow = E(C)

        print('step: %4i    err = %8.5e    E = %17.12f    alpha = %6.4f' %(i+1, err, Enow, alpha))
        iter_hist[i,:] = np.array([err, Enow, alpha])

        if err < conv_thr:
            print('convergence achieved!')
            iter_hist = np.resize(iter_hist, (i+1,3))
            return C, 0, iter_hist

    print('lgdm failed to converge within %4i iterations' %(max_iter))
    return C, 1, iter_hist

