import numpy as np
import matplotlib.pyplot as plt
import sys, os

max_len = int(sys.argv[1])

dir = os.path.dirname(os.path.abspath(__file__))

F00_00 = np.zeros(max_len//2)
F01_00 = np.zeros(max_len//2)

F00 = np.loadtxt(dir+'/F00_02.txt')
idx0 = np.unravel_index(np.argmax(F00), F00.shape)

F01 = np.loadtxt(dir+'/F01_02.txt')
idx1 = np.unravel_index(np.argmax(F01), F01.shape)

for i in range(2, max_len+1, 2):
    F00 = np.loadtxt(dir+'/F00_'+str(i).zfill(2)+'.txt')
    F01 = np.loadtxt(dir+'/F01_'+str(i).zfill(2)+'.txt')
    F00_00[i//2-1] = F00[idx0]
    F01_00[i//2-1] = F01[idx1]

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(range(2,max_len+1,2), F00_00)
ax1.set_xticks(range(2,max_len+1,4))

ax2 = fig.add_subplot(122)
ax2.plot(range(2,max_len+1,2), F01_00)
ax2.set_xticks(range(2,max_len+1,4))

plt.show()
