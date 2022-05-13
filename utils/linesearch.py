import numpy as np

def linesearch(f, df, alpha0=1.0, wolfe1=0.1, wolfe2=0.9, alpha_min = 1e-6, alpha_max=1e6, l=2.0):
    # alpha[i] = 0                   (i=0)
    #          = alpha0 * l**(i-1)   (i>0)
    #
    # max_iter should satisfy
    # alpha[max_iter-1] = alpha0 * l**(max_iter-2) >= alpha_max
    # alpha[max_iter-2] = alpha0 * l**(max_iter-3) < alpha_max
    # which gives 2+log(alpha_max/alpha0)/log(l) <= max_iter < 3 + log(alpha_max/alpha0)/log(l)
    max_iter = 2 + round(np.ceil(np.log(alpha_max/alpha0)/np.log(l)))

    phi = np.zeros(max_iter)
    dphi = np.zeros(max_iter)
    alpha = np.zeros(max_iter)

    # return flag
    # 0 -> success
    # 1 -> not descent
    # 2 -> failed within max_iter
    # 3 -> zoom failed

    phi[0] = f(0)
    dphi[0] = df(0)
    if dphi[0] > 0:
        print('line search direction is not descent!')
        return 0.0, 1 # alpha, flag
    
    alpha[0] = 0
    alpha[1] = alpha0

    def zoom(alpha_low, alpha_high):
        # |alpha_low-alpha_high| is halved for every iteration
        #
        # max_iter_zoom should satisfy
        # |alpha_low-alpha_high| / 2**max_iter_zoom <= alpha_min
        # |alpha_low-alpha_high| / 2**(max_iter_zoom-1) > alpha_min
        # log2(|alpha_low-alpha_high|/alpha_min) <= N < 1+log2(|alpha_low-alpha_high|/alpha_min)
        max_iter_zoom = round(np.ceil(np.log2(abs(alpha_high-alpha_low)/alpha_min)))

        for iter in range(0, max_iter_zoom):
            alpha_star = 0.5*(alpha_low+alpha_high)
            phi_star = f(alpha_star)
            if (phi_star > phi[0] + wolfe1*alpha_star*dphi[0]) or (phi_star >= f(alpha_low)):
                alpha_high = alpha_star
            else:
                dphi_star = df(alpha_star)
                if np.abs(dphi_star) <= -wolfe2*dphi[0]:
                    return alpha_star, 0
                if dphi_star*(alpha_high-alpha_low) >= 0:
                    alpha_high = alpha_low
                alpha_low = alpha_star

        print('zoom failed!')
        return alpha_star, 3
                
    for i in range(1, max_iter):
        phi[i] = f(alpha[i])

        if (phi[i] > phi[0]+wolfe1*alpha[i]*dphi[0]) or (phi[i]>=phi[i-1] and i>1):
            return zoom(alpha[i-1], alpha[i])

        dphi[i] = df(alpha[i])
        if np.abs(dphi[i]) <= -wolfe2*dphi[0]:
            return alpha[i], 0

        if dphi[i] >= 0:
            return zoom(alpha[i], alpha[i-1])

        alpha[i+1] = min(l*alpha[i], alpha_max)

    print('line search failed within', max_iter, 'iterations')
    return alpha[-1], 2

