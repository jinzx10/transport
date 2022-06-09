import numpy as np

# Broyden's root-finding method for one variable
def broydenroot(f, x0, tol=1e-8, max_iter=50, beta=0.7):
    fx = f(x0)
    x = x0
    if abs(fx) < tol:
        return x, 0

    delta = 1e-3 * max(1.0, np.sqrt(abs(x0)))
    J = (f(x+delta) - fx) / delta

    dx = 0.0
    fx_new = 0.0

    for iter in range(max_iter):
        if abs(J) < 1e-12:
            print('derivative appears to be 0')
            return x, 1

        dx = -fx / J * beta
        x += dx
        fx_new = f(x)
        if abs(fx_new) < tol:
            return x, 0

        J = (fx_new - fx) / dx;
        fx = fx_new;

    print('broydenroot fails to find the root')
    return x, 1

