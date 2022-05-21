import numpy as np

wl_freqs = 0
wh_freqs = 1
delta = 0.1
nw = 5
freqs = np.linspace(wl_freqs, wh_freqs, nw)

extra_delta = 0.02
extra_freqs = []

extra_dw = 0.02
extra_nw = 5

for i in range(nw):
    freqs_tmp = []
    if extra_nw % 2 == 0:
        for w in range(-extra_nw // 2, extra_nw // 2):
            freqs_tmp.append(freqs[i] + extra_dw * w)
    else:
        for w in range(-(extra_nw-1) // 2, (extra_nw+1) // 2):
            freqs_tmp.append(freqs[i] + extra_dw * w)
    extra_freqs.append(np.array(freqs_tmp))

print('extra_freqs = ', extra_freqs)

all_freqs = np.array(sorted(list(set(list(freqs) + \
        ([x for xx in extra_freqs for x in xx] if extra_freqs is not None else [])))))

print(all_freqs)
print([x for xx in extra_freqs for x in xx])
