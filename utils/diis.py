import numpy as np
import scipy.sparse as sp


def add_diff(f, x):
    fx = f(x)
    return [fx, fx-x]

def diis(f, x0, conv_tol = 1e-6, max_iter = 50, max_subspace_size = 20):
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
        return x0, 0

    szx = np.prod(x.shape)

    x_is_sparse = sp.issparse(x)

    if x_is_sparse:
        xs = sp.lil_matrix((1, szx))
        xs[0,:] = np.reshape(x,(1,szx))
    else:
        xs = np.zeros((1, szx)) # iteration history
        xs[0,:] = x.flatten()

    rs = np.zeros((1, r.size)) # error vectors
    rs[0,:] = r.flatten()

    B = rs @ rs.T.conj()

    diis_mat = lambda : np.vstack( ( np.hstack( (         B            , np.ones((np.size(B,0),1)) ) ), 
                                     np.append(   np.ones(np.size(B,0)),          0                  ) ) )

    for counter in range(0, max_iter):
        # Pulay mixing
        # use pseudo-inverse to handle possible singularity of the DIIS matrix
        #c = np.linalg.lstsq(diis_mat(), np.append(np.zeros(np.size(B,0)), 1.0), rcond=1e-12)[0]
        c = np.linalg.pinv(diis_mat()) @ np.append(np.zeros(np.size(B,0)), 1.0)
        x_mix = np.reshape( c[:-1] @ xs, np.shape(x0) )

        x, r = g(x_mix)
        #print('err = ', np.linalg.norm(r))
        if np.linalg.norm(r,1) < conv_tol:
            print('converence achieved in', counter+1, 'steps')
            return x, 0
        
        if x_is_sparse:
            xs = sp.vstack((xs, np.reshape(x, (1,szx))), format='lil')
        else:
            xs = np.vstack((xs, x.flatten()))

        rs = np.vstack((rs, r.flatten()))
        B = np.pad(B, (0,1), 'constant')
        B[-1,:] = rs[-1,:] @ rs.T.conj()
        B[:,-1] = B[-1,:].conj()

        if np.size(B,0) > max_subspace_size:
            rs = np.delete(rs, 0, axis=0)
            if x_is_sparse:
                xs = xs[1:,:]
            else:
                xs = np.delete(xs, 0, axis=0)
            B = np.delete(B, 0, axis=0)
            B = np.delete(B, 0, axis=1)
            continue

    print('DIIS fails to converge.')
    return None, 1

