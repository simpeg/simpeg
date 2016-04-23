import numpy as np
from scipy.constants import mu_0, pi

def DCAnalytic(txloc, rxlocs, sigma, flag="wholespace"):
    """
        Analytic solution for electric potential from a postive pole

        Input variables:

            txloc =  a xyz location of A (+) electrode (np.r_[xa, ya, za])

            rxlocs = [M, N]
                M: xyz locations of M (+) electrode (np.c_[xmlocs, ymlocs, zmlocs])
                N: xyz locations of N (-) electrode (np.c_[xnlocs, ynlocs, znlocs])

            sigma = conductivity (either float or complex)
            flag = "wholsespace" or "halfspace"

    """
    M = rxlocs[0]
    N = rxlocs[1]

    rM = np.sqrt( (M[:,0]-txloc[0])**2 + (M[:,1]-txloc[1])**2 + (M[:,2]-txloc[1])**2 )
    rN = np.sqrt( (N[:,0]-txloc[0])**2 + (N[:,1]-txloc[1])**2 + (N[:,2]-txloc[1])**2 )

    phiM = 1./(4*np.pi*rM*sigma)
    phiN = 1./(4*np.pi*rN*sigma)
    phi = phiM - phiN

    if flag == "halfspace":
        phi *= 2

    return phi

