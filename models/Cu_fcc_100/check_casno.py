import h5py
import numpy as np
import matplotlib.pyplot as plt

fname_cisd = 'casno_cisd.h5'
fh = h5py.File(fname_cisd, 'r')

no_coeff_v_raw = np.asarray(fh['no_coeff_v_raw'])
no_coeff_o_raw = np.asarray(fh['no_coeff_o_raw'])

no_coeff_v = np.asarray(fh['no_coeff_v_scdm'])
no_coeff_o = np.asarray(fh['no_coeff_o_scdm'])

dm_ci_mo = np.asarray(fh['dm_ci_mo'])

nocc = round(np.trace(dm_ci_mo))
print('nocc = ', nocc)
sz = dm_ci_mo.shape[0]
diag = np.zeros(sz)
diag[:nocc//2] = 2.0
dm_ref = np.diag(diag)


print(np.linalg.norm(dm_ci_mo-dm_ref))

#plt.imshow(dm_ci_mo)
#plt.show()
