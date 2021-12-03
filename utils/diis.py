import numpy as np


def add_diff(f, x):
    fx = f(x)
    return [fx, fx-x]


def diis(f, x0, conv_tol = 1e-6, max_iter = 50, max_subspace_size = 15, rcond_tol = 1e-12):
    '''
    Use Pulay mixing (Direct Inversion in the Iterative Subspace) to find x such that x = f(x)
    '''
    
    try:
        x, r = f(x0)
        g = lambda y : f(y)
    except:
        x = f(x0)
        r = x - x0
        g = lambda y : add_diff(f, y)

    if np.linalg.norm(r,1) < conv_tol:
        return 0, x0

    xs = np.zeros((1, x0.size))
    rs = np.zeros((1, x0.size))

    xs[0,:] = x.flatten()
    rs[0,:] = r.flatten()

    B = rs @ rs.T.conj()

    diis_mat = lambda : np.vstack( ( np.hstack( ( B                    , np.ones((np.size(B,0),1)) ) ), 
                                     np.append(   np.ones(np.size(B,0)),          0                  ) ) )

    counter = 0
    while counter < max_iter:
        if np.size(B,0) > max_subspace_size or 1./np.linalg.cond(B) < rcond_tol:
            rs = np.delete(rs, 0, axis=0)
            xs = np.delete(xs, 0, axis=0)
            B = np.delete(B, 0, axis=0)
            B = np.delete(B, 0, axis=1)
            continue

        counter = counter + 1

        # Pulay mixing
        x_mix = np.reshape( np.linalg.solve(diis_mat(), np.append(np.zeros(np.size(B,0)), 1.0))[:-1] @ xs, np.shape(x0) )

        x, r = g(x_mix)
        print('err = ', np.linalg.norm(r))
        if np.linalg.norm(r,1) < conv_tol:
            return 0, x

        xs = np.vstack((xs, x.flatten()))
        rs = np.vstack((rs, r.flatten()))
        B = np.pad(B, (0,1), 'constant')
        B[-1,:] = rs[-1,:] @ rs.T.conj()
        B[:,-1] = B[-1,:].conj()

    print('DIIS fails to converge.')
    return 1, None

