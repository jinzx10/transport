#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt


lat_const = 4.171288 # gold lattice constant taken from materialsproject.org
a = lat_const / np.sqrt(2)
lat_vec = np.array([
    [1,0,0], 
    [0.5, np.sqrt(3)/2, 0],
    [0.5, 0.5/np.sqrt(3), np.sqrt(2./3)]
    ]) * a

###########################################################
#       generate a periodic unit of the electrode
###########################################################
# number of atoms in a periodic unit
n0 = 3
n1 = 3
n2 = 3 # n2 is the number of layers in a principal layer
       # must be a multiple of 3 for transport along [111] direction

nat_uc = n0 * n1 * n2 # number of atoms per unit cell

o = lambda n: np.ones(n)
v = lambda n: np.arange(0,n)

coor_uc = np.row_stack( ( \
        np.kron( o(n2), np.kron( o(n1), v(n0) ) ), \
        np.kron( o(n2), np.kron( v(n1), o(n0) ) ), \
        np.kron( v(n2), np.kron( o(n1), o(n0) ) )  ) ).T  @ lat_vec

# thickness of a principal layer
h = n2 * np.sqrt(2./3) * a

# primitive vectors for the periodic unit
prim_vec = np.zeros((3,3))
prim_vec[0,:] = n0 * lat_vec[0,:]
prim_vec[1,:] = n1 * lat_vec[1,:]
prim_vec[2,:] = np.array([0, 0, h])

############################################################
#   electrode atoms included in the contact region
#       tip geometry: a few layers + a small tip
############################################################

#------------ upper electrode ------------
# number of layers included in the contact region
nl_contact = 2  # Brandbyge2002 suggests 2 or 3

coor_contact_electrode_upper = np.zeros((n0*n1*nl_contact,3))
for i in range(0, nl_contact//n2):
    coor_contact_electrode_upper[i*nat_uc:(i+1)*nat_uc,:] = coor_uc
    coor_contact_electrode_upper[i*nat_uc:(i+1)*nat_uc,2] = coor_contact_electrode_upper[i*nat_uc:(i+1)*nat_uc,2] + i * h

nl_rem = nl_contact % n2
if nl_rem != 0:
    coor_contact_electrode_upper[-n0*n1*nl_rem:,:] = coor_uc[0:n0*n1*nl_rem,:]
    coor_contact_electrode_upper[-n0*n1*nl_rem:,2] = coor_uc[0:n0*n1*nl_rem,2] + (nl_contact//n2) * h

coor_contact_tip_upper = np.zeros((4,3))
iat_ref = 1 # index of reference atom
#coor_contact_tip_upper[0,:] = coor_contact_electrode_upper[iat_ref,:] + np.array([0.5, 0.5/np.sqrt(3), -np.sqrt(2./3)]) * a
coor_contact_tip_upper[0,:] = coor_contact_electrode_upper[iat_ref,:] + np.array([0., 1./np.sqrt(3), -np.sqrt(2./3)]) * a
coor_contact_tip_upper[1,:] = coor_contact_tip_upper[0,:] + np.array([1,0,0]) * a
coor_contact_tip_upper[2,:] = coor_contact_tip_upper[0,:] + np.array([0.5,np.sqrt(3)/2,0]) * a
coor_contact_tip_upper[3,:] = coor_contact_tip_upper[0,:] + np.array([0.5,0.5/np.sqrt(3),-np.sqrt(2./3)]) * a

coor_contact_upper = np.row_stack((coor_contact_tip_upper, coor_contact_electrode_upper))


#------------ lower electrode ------------
shift_dist = 18.
coor_contact_lower = np.copy(coor_contact_upper)
coor_contact_lower[:,2] = -coor_contact_lower[:,2] - shift_dist


#------------ molecule ------------
coor_mol = np.row_stack((coor_contact_upper[3,:], coor_contact_lower[3,:]))
coor_mol[0,2] = coor_mol[0,2] - 3
coor_mol[1,2] = coor_mol[1,2] + 3


#------------ put together ------------
coor_contact_electrode = np.row_stack((coor_contact_lower, coor_contact_upper))
coor_contact = np.row_stack((coor_contact_electrode, coor_mol))


############################################################
#       generate coordinates of the principal layer
#       which couples to the contact
############################################################

coor_pl_upper = np.zeros(np.shape(coor_uc))
coor_pl_upper[0:n0*n1*(n2-nl_rem),:] = coor_uc[n0*n1*nl_rem:,:]

if nl_rem != 0:
    coor_pl_upper[-n0*n1*nl_rem:,:] = coor_uc[0:n0*n1*nl_rem,:]
    coor_pl_upper[-n0*n1*nl_rem:,2] = coor_pl_upper[-n0*n1*nl_rem:,2] + h

coor_pl_upper[:,2] = coor_pl_upper[:,2] + (nl_contact//n2) * h

coor_pl_lower = np.copy(coor_pl_upper)
coor_pl_lower[:,2] = -coor_pl_upper[:,2] - shift_dist

coor_all = np.row_stack((coor_contact, coor_pl_lower, coor_pl_upper))

############################################################
#               plot junction
############################################################

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(coor_contact_electrode[:,0], coor_contact_electrode[:,1], coor_contact_electrode[:,2], color='red')
#ax.scatter(coor_mol[:,0], coor_mol[:,1], coor_mol[:,2], color='green')
#ax.scatter(coor_pl_upper[:,0], coor_pl_upper[:,1], coor_pl_upper[:,2], color='blue')
#ax.scatter(coor_pl_lower[:,0], coor_pl_lower[:,1], coor_pl_lower[:,2], color='blue')
#plt.show()


############################################################
#       save coor & prim_vec to file
############################################################

# wipe the file clean
filename = 'junction_tip_111.xsf'
f = open(filename, 'w')
f.close()

# open in append mode
f = open(filename, 'a')
f.write('PRIMVEC\n')

# change the third prim_vec
prim_vec[2,2] = 2*(h + (nl_contact+2)*np.sqrt(2./3)*a) + shift_dist + 10
np.savetxt(f, prim_vec, fmt='%17.12f')
f.write('PRIMCOORD\n')
f.write(str(np.shape(coor_all)[0])+' 1\n')

for i in range(0, np.size(coor_all,0)):
    f.write('Au ')
    np.savetxt(f, coor_all[i,:], fmt='%17.12f', newline=' ')
    f.write('\n')

f.close()

