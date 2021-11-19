import numpy as np

def ezsave(M, filename):
    f = open(filename, 'w')
    np.savetxt(f, M, fmt='%20.12f')
    f.close()
