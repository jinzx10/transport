from bath_disc import direct_disc_hyb
import numpy as np
import matplotlib.pyplot as plt


# reference bath
nimp = 6;
nbath = 10000;
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


# broadening used to get reference quantities
delta = 0.001;

# self energy
Sigma = lambda z: Vcpl / (z-Ebath) @ Vcpl.T.conj()


# reference hybridization
Gamma_ref = np.zeros((nbath, nimp, nimp));
for ib in range(nbath):
    Gamma_ref[ib,:,:] = -1./np.pi*Sigma(Ebath[ib]+1j*delta).imag;


# scheme #1 get a set of energies and hyb evaluated on those energies
nbath1 = 200;
nint = 3

grid = np.linspace(-2,2,nbath1*nint+1);
Gamma = np.zeros((nbath1*nint+1, nimp, nimp));
for ib in range(nbath1+1):
    Gamma[ib,:,:] = -1./np.pi*Sigma(grid[ib]+1j*delta).imag;

# scheme #2 construct a callable hyb and intervals
# callable 
hyb = lambda x: -1./np.pi*Sigma(x+1j*delta).imag
nbath2 = 200
intervals = np.linspace(-2,2,nbath2+1)

nbath_per_ene = 3

e, v = direct_disc_hyb(hyb=Gamma, grid=grid, nint=nint, nbath_per_ene=nbath_per_ene)
e2, v2 = direct_disc_hyb(hyb=hyb, grid=intervals, nbath_per_ene=nbath_per_ene)

gauss = lambda x,mu,sigma: 1.0/sigma/np.sqrt(2*np.pi)*np.exp(-0.5*((x-mu)/sigma)**2)

# rebuild Gamma on a new grid
nz = 1000
z = np.linspace(-2,2,nz)
eta = 0.01
Gamma_rebuilt = np.zeros((nz,nimp,nimp))
Gamma_rebuilt2 = np.zeros((nz,nimp,nimp))
for iz in range(nz):
    for ib in range(nbath1):
        for ie in range(nbath_per_ene):
            Gamma_rebuilt[iz,:,:] += np.outer(v[ib,:,ie],v[ib,:,ie].conj()) * gauss(z[iz],e[ib],eta)
    for ib in range(nbath2):
        for ie in range(nbath_per_ene):
            Gamma_rebuilt2[iz,:,:] += np.outer(v2[ib,:,ie],v2[ib,:,ie].conj()) * gauss(z[iz],e2[ib],eta)

a = 0
b = 0

plt.plot(Ebath,Gamma_ref[:,a,b])
plt.plot(z,Gamma_rebuilt[:,a,b])
plt.plot(z,Gamma_rebuilt2[:,a,b])
plt.show()

