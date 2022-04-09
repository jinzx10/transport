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
    and grid[3:7] will generate the second bath energy, etc.

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

        idx_negval = np.where(val < 0)
        bignegval = val[idx_negval][val[idx_negval] < -eig_tol]
        if len(bignegval) != 0:
            print('warning: significantly negative eigenvalue(s) found \
                    for the integration of hybridization function:', bignegval)
        val[idx_negval] = 0

        v[ie,:,:] = vec[:,-nbath_per_ene:] * np.sqrt(val[-nbath_per_ene:])

    return e, v


