import sys, os
import numpy as np

dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir+'/../../../')

from transport.utils import *

S00 = np.loadtxt('data/S00.txt')
S01 = np.loadtxt('data/S01.txt')

F00 = np.loadtxt('data/F00_4_11.txt')
F01 = np.loadtxt('data/F01_4_11.txt')


z = 2 + 0.1j

g84 = LopezSancho1984(z, F00, F01, S00, S01)
g85 = LopezSancho1984(z, F00, F01, S00, S01)


