from bath_disc import direct_disc_hyb
import numpy as np
import matplotlib.pyplot as plt


# reference bath
nimp = 6;
nbath = 1000;
w = 1;
Ebath_half = np.logspace(-2, 0, nbath//2);
Ebath = np.zeros(nbath)
Ebath[:nbath//2] = Ebath_half
Ebath[-nbath//2:] = -Ebath_half
Ebath = np.sort(Ebath);

Vcpl = np.zeros((nimp, nbath))
for i in range(nimp):
    Vcpl[i,:] = np.linspace(-1, 1, nbath);

Vcpl[0,:] = 0.1 * Vcpl[0,:]**2;
Vcpl[1,:] = 0.1 * np.exp(-0.1*Vcpl[1,:]**2);
Vcpl[2,:] = 0.5 * Vcpl[2,:]**3;
Vcpl[3,:] = 0.1 * Vcpl[3,:]**4;
Vcpl[4,:] = 0.1 * np.exp(-0.5*Vcpl[1,:]**2/2);
Vcpl[5,:] = 0.1 * np.exp(-0.3*Vcpl[1,:]**2);


# broadening
delta = 0.01;

# self energy
Sigma = lambda z: Vcpl / (z-Ebath) @ Vcpl.T.conj()


# reference hybridization
Gamma_ref = np.zeros((nbath, nimp, nimp));
for ib in range(nbath):
    Gamma_ref[ib,:,:] = -1./np.pi*Sigma(Ebath[ib]+1j*delta).imag;

# start bath discretization
# first pick a set of points

nb = 200;
freqs = np.linspace(-2,2,nb+1);
Gamma = np.zeros((nb+1, nimp, nimp));
for ib in range(nb+1):
    Gamma[ib,:,:] = -1./np.pi*Sigma(freqs[ib]+1j*delta).imag;

nbath_per_ene = 3

e, v = direct_disc_hyb(hyb=Gamma, grid=freqs, nbath_per_ene=nbath_per_ene)

print('e.shape', e.shape)
print('v.shape', v.shape)

gauss = lambda x,mu,sigma: 1.0/sigma/np.sqrt(2*np.pi)*np.exp(-0.5*((x-mu)/sigma)**2)

# rebuild Gamma

nz = 200
z = np.linspace(-2,2,nz)
eta = 0.05
Gamma_rebuilt = np.zeros((nz,nimp,nimp))
for iz in range(nz):
    for ib in range(nb):
        for ie in range(nbath_per_ene):
            Gamma_rebuilt[iz,:,:] += np.outer(v[ib,:,ie],v[ib,:,ie].conj()) * gauss(z[iz],e[ib],eta)


plt.plot(Ebath,Gamma_ref[:,0,0])
plt.plot(z,Gamma_rebuilt[:,0,0])
plt.show()

