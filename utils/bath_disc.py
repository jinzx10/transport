import numpy as np

def direct_disc_hyb(hyb, grid, nint=1, nbath_per_ene=None, eig_tol=1e-8):
    '''
    Direct discretization through the hybridization function
    See Nusspickel and Booth, Phys. Rev. B 102, 165107 (2020)

    The function takes a hybridization function and returns a set of bath energies (e) and couplings (v).
    On exit, e is a 1-d array, v is a 3-d array of shape (len(e), nimp, nbath_per_ene) where nimp stands
    for the number of impurity orbitals, and nbath_per_ene the number of bath orbitals per bath energy.

    The function works in two modes. In the first mode, one passes in two arrays, grid and hyb, which 
    correspond to the energies and the values of the hybridization function evaluated on those energies.
    grid should be a 1-d array, and the shape of hyb should be (len(grid), nimp, nimp).
    In this mode, nint determines how many intervals are grouped together to generate one bath energy.
    For example, if nint=3, then grid[0:4] (4 points, 3 intervals) will generate the first bath energy, 
    and grid[3:7] will generate the second bath energy, etc. In order to get M bath energies, the grid
    needs to contain nint*M+1 points.

    In the second mode, one passes in a callable hyb, and a 1-d array grid.
    In this mode, grid directly defines the intervals where each bath energy is generated, and nint
    is the number of points for numerical integration. For example, if nint=3, then 3 evenly spaced
    points between grid[0] and grid[1] will be generated. The bath energy and couplings will be
    computed by hyb values evaluated on those points.

    nbath_per_ene is the number of bath orbitals per bath energy. 
    It is by default nimp, and should be no greater than nimp.

    When nimp>1, the direct discretization method uses an eigendecomposition of \int_{In} hyb(x) dx 
    to generate the couplings, where In stands for an interval. 
    In general, one needs nimp bath orbitals per bath energy. However, very often some eigenvalues 
    are small, which means one can possibly use fewer bath orbitals per energy without significantly
    affecting the quality of bath discretization.
    '''

    ngrid = len(grid)
    # make sure grid is sorted
    idx_sort = np.argsort(grid)
    grid = grid[idx_sort]

    if callable(hyb):
        ne = ngrid - 1
        nimp = hyb(0).shape[0]
    else:
        nimp = hyb.shape[1]
        ne = (ngrid-1) // nint
        hyb = hyb[idx_sort]

    if nbath_per_ene is None:
        nbath_per_ene = nimp

    e = np.zeros(ne)
    v = np.zeros((ne, nimp, nbath_per_ene))

    for ie in range(ne):

        # \int hyb(x) dx and \int x*Tr{hyb(x)}dx for each interval
        xTrhybint = 0
        hybint = np.zeros((nimp, nimp))

        if callable(hyb):
            # generate a grid for integration between grid[ie] and grid[ie+1]
            intgrid = np.linspace(grid[ie], grid[ie+1], nint+1)
            dx = intgrid[1]-intgrid[0]
            intgrid = (intgrid[1:]+intgrid[:-1])/2

            for ii in range(nint):
                hyb_i = hyb(intgrid[ii])
                hybint += hyb_i
                xTrhybint += intgrid[ii]*hyb_i.trace()
            hybint *= dx
            xTrhybint *= dx
        else:
            i0 = ie*nint
            for ii in range(nint):
                dx = grid[i0+ii+1] - grid[i0+ii]
                hybint += (hyb[i0+ii] + hyb[i0+ii+1]) / 2 * dx
                xTrhybint += (grid[i0+ii]*hyb[i0+ii].trace() + grid[i0+ii+1]*hyb[i0+ii+1].trace()) / 2 * dx

        # compute the bath energy and couplings from the integrals
        e[ie] = xTrhybint / hybint.trace()
        val, vec = np.linalg.eigh(hybint)

        #print('val = ', val)
        idx_negval = np.where(val < 0)
        bignegval = val[idx_negval][val[idx_negval] < -eig_tol]
        if len(bignegval) != 0:
            print('warning: significantly negative eigenvalue(s) found \
                    for the integration of hybridization function:', bignegval)
        val[idx_negval] = 0

        v[ie,:,:] = vec[:,-nbath_per_ene:] * np.sqrt(val[-nbath_per_ene:])

    return e, v


# return sort( w0 + (w-w0)*l**(-i) ) where i ranges from 0 to num-1
def gen_log_grid(w0, w, l, num):
    grid = w0 + (w-w0) * l**(-np.arange(num,dtype=float))
    if w > w0:
        return grid[::-1]
    else:
        return grid

# len(grid) == nbe+1
def gen_grid(nbe, wl, wh, mu, grid_type='custom1', log_disc_base=1.6, wlog = 0.01):
    if grid_type == 'log': # log grid around mu
        wl0 = mu - wl
        wh0 = wh - mu
        
        # number of energies above/below the Fermi level
        dif = round(np.log(abs(wh0/wl0))/np.log(log_disc_base)) // 2
        nl = nbe//2 - dif
        nh = nbe - nl
        grid = np.concatenate( (gen_log_grid(mu, wl, log_disc_base, nl), [mu], \
                gen_log_grid(mu, wh, log_disc_base, nh)) )

    elif grid_type == 'linear': # linear between wl and wh
        grid = np.linspace(wl,wh,nbe+1)

    elif grid_type == 'custom1': # log near Fermi level; linear elsewhere
        # half side number of points in log disc (center point, mu, not included)
        nw_log = (nbe+1) // 4 // 2
        
        # number of points in linear disc (boundary with log disc excluded)
        nw_lin = nbe - nw_log*2
        
        # 2*nw_log+1 points
        grid_log = np.concatenate((gen_log_grid(mu, mu-wlog, log_disc_base, nw_log), [mu], \
                gen_log_grid(mu, mu+wlog, log_disc_base, nw_log)))
        
        dg_raw = (wh-wl-2*wlog)/nw_lin
        nl_lin = round( (mu-wlog-wl)/dg_raw )
        nh_lin = nw_lin - nl_lin
        grid = np.concatenate( (np.linspace(wl,mu-wlog,nl_lin+1), grid_log, np.linspace(mu+wlog,wh,nh_lin+1)) )
        grid = np.array(sorted(list(set(list(grid)))))

    return grid



