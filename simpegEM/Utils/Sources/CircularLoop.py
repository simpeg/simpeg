from SimPEG import *
from scipy.special import ellipk, ellipe

def MagneticLoopVectorPotential(txLoc, obsLoc, component, radius):
    """
        Calculate the vector potential of horizontal circular loop
        at given locations

        :param numpy.ndarray txLoc: Location of the transmitter(s) (x, y, z)
        :param numpy.ndarray obsLoc: Where the potentials will be calculated (x, y, z)
        :param str component: The component to calculate - 'x', 'y', or 'z'
        :param numpy.ndarray I: Input current of the loop
        :param numpy.ndarray radius: radius of the loop
        :rtype: numpy.ndarray
        :return: The vector potential each dipole at each observation location
    """

    txLoc = np.atleast_2d(txLoc)
    obsLoc = np.atleast_2d(obsLoc)

    n = obsLoc.shape[0]
    nTx = txLoc.shape[0]

    if component=='z':
        A = np.zeros((n, nTx))
        if nTx ==1:
            return A.flatten()        
        return A

    else:

        A = np.zeros((n, nTx))
        for i in range (nTx):
            x = obsLoc[:, 0] - txLoc[i, 0]
            y = obsLoc[:, 1] - txLoc[i, 1]
            z = obsLoc[:, 2] - txLoc[i, 2]
            r = np.sqrt(x**2 + y**2)
            m = (4 * radius * r) / ((radius + r)**2 + z**2)
            m[m > 1.] = 1.
            # m might be slightly larger than 1 due to rounding errors
            # but ellipke requires 0 <= m <= 1
            K = ellipk(m)
            E = ellipe(m)
            ind = (r > 0) & (m < 1)
            # % 1/r singular at r = 0 and K(m) singular at m = 1
            Aphi = np.zeros(n)
            # % Common factor is (mu * I) / pi with I = 1 and mu = 4e-7 * pi.
            Aphi[ind] = 4e-7 / np.sqrt(m[ind])  * np.sqrt(radius / r[ind]) *((1. - m[ind] / 2.) * K[ind] - E[ind])
            if component == 'x':
                A[ind, i] = Aphi[ind] * (-y[ind] / r[ind] )
            elif component == 'y':
                A[ind, i] = Aphi[ind] * ( x[ind] / r[ind] )
            else:
                raise ValueError('Invalid component')               

        if nTx == 1:
            return A.flatten()                
        return A

if __name__ == '__main__':
    from SimPEG import Mesh
    import matplotlib.pyplot as plt
    cs = 20
    ncx, ncy, ncz = 41, 41, 40
    hx = np.ones(ncx)*cs
    hy = np.ones(ncy)*cs
    hz = np.ones(ncz)*cs
    mesh = Mesh.TensorMesh([hx, hy, hz], 'CCC')
    txLoc = np.r_[0., 0., 0.]
    Ax = MagneticLoopVectorPotential(txLoc, mesh.gridEx, 'x', 200)
    Ay = MagneticLoopVectorPotential(txLoc, mesh.gridEy, 'y', 200)
    Az = MagneticLoopVectorPotential(txLoc, mesh.gridEz, 'z', 200)
    A = np.r_[Ax, Ay, Az]
    B0 = mesh.edgeCurl*A
    J0 = mesh.edgeCurl.T*B0

    # mesh.plotImage(A, vType = 'Ex')
    # mesh.plotImage(A, vType = 'Ey')

    mesh.plotImage(B0, vType = 'Fx')
    mesh.plotImage(B0, vType = 'Fy')
    mesh.plotImage(B0, vType = 'Fz')

    # # mesh.plotImage(J0, vType = 'Ex')
    # mesh.plotImage(J0, vType = 'Ey')
    # mesh.plotImage(J0, vType = 'Ez')

    plt.show()


