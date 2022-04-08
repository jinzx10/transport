import numpy as np

def direct_disc_hyb(hyb, grid, nint=2, nbath_per_ene = None):
    '''
    Direct discretization through the hybridization function
    See Nusspickel and Booth, Phys. Rev. B 102, 165107 (2020)

    The function takes a hybridization function and returns a set of bath energies (e) and couplings (v).
    On exit, e is a 1-d array, v is a 3-d array of shape (len(e), nimp, nbath_per_ene) where nimp stands
    for the number of impurity orbitals.

    The function works in two modes. In the first mode, one passes in two arrays, hyb and grid,
    which corresponds to the hybridization function and the energies where they are evaluated.
    grid should be a sorted 1-d array. The shape of hyb should be (len(grid), nimp, nimp).
    In this mode, nint determines how many intervals are grouped together to generate one bath energy.
    For example, if nint=3, then grid[0:4] (4 points, 3 intervals) will generate the first bath energy, 
    and grid[3:7] will generate the second bath energy, etc.

    In the second mode, one passes in a callable hyb, and a sorted 1-d array grid.
    In this mode, grid directly defines the intervals where each bath energy is generated, and nints
    is the number of points for numerical integration. For example, if nints=3, then 3 evenly spaced
    points between grid[0] and grid[1] will be generated. The bath energy and couplings will be
    computed by hyb values evaluated on those points.

    nbath_per_ene is the number of bath orbitals per bath energy. 
    It is by default nimp, and should be no greater than nimp.

    When nimp>1, the direct discretization method uses an eigendecomposition of \int_{In} hyb(x)dx 
    to generate the couplings where In stands for an interval. 
    In general, one needs nimp bath orbitals per bath energy. However, very often there are some small 
    eigenvalues, which means one can possibly have fewer bath orbitals per energy without significantly
    affecting the quality of bath discretization.
    '''
    ngrid = len(grid)
    nimp = hyb.shape[1]
    if nbath_per_ene is None:
        nbath_per_ene = nimp

    if callable(hyb):
        ne = ngrid - 1
    else:
        ne = ngrid // nint

    e = np.zeros(ne)
    v = np.zeros((ne, nimp, nbath_per_e))

    for ie in range(ne):
        xTrhybint = 0
        Trhybint = 0
        hybint = np.zeros((nimp, nimp))

        if callable(hyb):
            intgrid = np.linspace(grid[ie], grid[ie+1], nint+1)
            intgrid = (intgrid[1:]+intgrid[:-1])/2
            for ii in range(nint):
                hyb_i = hyb(intgrid[ii])
                hybint += hyb_i
                xTrhybint += intgrid[ii]*hyb_i.trace()
                Trhybint += hyb_i.trace()
            hybint *= intgrid[1]-intgrid[0]
        else:
            i0 = ie*nbath_per_e
            for ii in range(nint):
                dx = grid[i0+ii+1] - grid[i0+ii]
                hybint += (hyb[i0+ii] + hyb[i0+ii+1]) / 2 * dx
                xTrhybint += (grid[i0+ii]*hyb[i0+ii].trace() + grid[i0+ii+1]*hyb[i0+ii+1].trace) / 2 * dx
                Trhybint += (hyb[i0+ii].trace() + hyb[i0+ii+1].trace) / 2 * dx 

        e[ie] = xTrhybint / Trhybint
        vec, val = np.linalg.eigh(hybint)

        idx_negval = np.where(val<0)
        if idx_negval is not None:
            print('warning: negative eigenvalues', val[idx_negval], 'are set to 0')
            val[idx_negval] = 0
        v[ie,:,:] = vec[:,-nbath_per_e:] * np.sqrt(val[-nbath_per_e:])

    return e, v


