import numpy as np
import matplotlib.pyplot as plt
import sys, os

sz_blk = int(sys.argv[1])
nblk_max = int(sys.argv[2])

dir = os.path.dirname(os.path.abspath(__file__))

# find the index of the largest element
F00 = np.loadtxt(dir+'/F00_'+str(sz_blk)+'_03'+'.txt')
idx0 = np.unravel_index(np.argmax(F00), F00.shape)

F01 = np.loadtxt(dir+'/F01_'+str(sz_blk)+'_03'+'.txt')
idx1 = np.unravel_index(np.argmax(F01), F01.shape)

F02 = np.loadtxt(dir+'/F02_'+str(sz_blk)+'_03'+'.txt')
idx2 = np.unravel_index(np.argmax(F02), F01.shape)

# store the largest element of each size
F00_max = np.zeros((nblk_max-1)//2)
F01_max = np.zeros((nblk_max-1)//2)
F02_max = np.zeros((nblk_max-1)//2)

for i in range(3, nblk_max+1, 2):
    F00 = np.loadtxt(dir+'/F00_'+str(sz_blk)+'_'+str(i).zfill(2)+'.txt')
    F01 = np.loadtxt(dir+'/F01_'+str(sz_blk)+'_'+str(i).zfill(2)+'.txt')
    F02 = np.loadtxt(dir+'/F02_'+str(sz_blk)+'_'+str(i).zfill(2)+'.txt')
    F00_max[i//2-1] = F00[idx0]
    F01_max[i//2-1] = F01[idx1]

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax1.plot(range(sz_blk*3,sz_blk*nblk_max+1,2*sz_blk), F00_max)
ax1.set_xticks(range(sz_blk,sz_blk*nblk_max+1,sz_blk))

ax2 = fig.add_subplot(132)
ax2.plot(range(sz_blk*3,sz_blk*nblk_max+1,2*sz_blk), F01_max)
ax2.set_xticks(range(sz_blk,sz_blk*nblk_max+1,sz_blk))

ax3 = fig.add_subplot(133)
ax3.plot(range(sz_blk*3,sz_blk*nblk_max+1,2*sz_blk), F02_max)
ax3.set_xticks(range(sz_blk,sz_blk*nblk_max+1,sz_blk))

plt.show()
