from SimPEG import *
from scipy.special import ellipk, ellipe
from scipy.constants import mu_0, pi

def MagneticDipoleVectorPotential(srcLoc, obsLoc, component, moment=1., dipoleMoment=(0., 0., 1.), mu = mu_0):
    """
        Calculate the vector potential of a set of magnetic dipoles
        at given locations 'ref. <http://en.wikipedia.org/wiki/Dipole#Magnetic_vector_potential>'

        :param numpy.ndarray srcLoc: Location of the source(s) (x, y, z)
        :param numpy.ndarray,SimPEG.Mesh obsLoc: Where the potentials will be calculated (x, y, z) or a SimPEG Mesh
        :param str,list component: The component to calculate - 'x', 'y', or 'z' if an array, or grid type if mesh, can be a list
        :param numpy.ndarray dipoleMoment: The vector dipole moment
        :rtype: numpy.ndarray
        :return: The vector potential each dipole at each observation location
    """
    #TODO: break this out!

    if type(component) in [list, tuple]:
        out = range(len(component))
        for i, comp in enumerate(component):
            out[i] = MagneticDipoleVectorPotential(srcLoc, obsLoc, comp, dipoleMoment=dipoleMoment)
        return np.concatenate(out)

    if isinstance(obsLoc, Mesh.BaseMesh):
        mesh = obsLoc
        assert component in ['Ex','Ey','Ez','Fx','Fy','Fz'], "Components must be in: ['Ex','Ey','Ez','Fx','Fy','Fz']"
        return MagneticDipoleVectorPotential(srcLoc, getattr(mesh,'grid'+component), component[1], dipoleMoment=dipoleMoment)

    if component == 'x':
        dimInd = 0
    elif component == 'y':
        dimInd = 1
    elif component == 'z':
        dimInd = 2
    else:
        raise ValueError('Invalid component')

    srcLoc = np.atleast_2d(srcLoc)
    obsLoc = np.atleast_2d(obsLoc)
    dipoleMoment = np.atleast_2d(dipoleMoment)

    nEdges = obsLoc.shape[0]
    nSrc = srcLoc.shape[0]

    m = np.array(dipoleMoment).repeat(nEdges, axis=0)
    A = np.empty((nEdges, nSrc))
    for i in range(nSrc):
        dR = obsLoc - srcLoc[i, np.newaxis].repeat(nEdges, axis=0)
        mCr = np.cross(m, dR)
        r = np.sqrt((dR**2).sum(axis=1))
        A[:, i] = +(mu/(4*pi)) * mCr[:,dimInd]/(r**3)
    if nSrc == 1:
        return A.flatten()
    return A


def MagneticDipoleFields(srcLoc, obsLoc, component, moment=1., mu = mu_0):
    """
        Calculate the vector potential of a set of magnetic dipoles
        at given locations 'ref. <http://en.wikipedia.org/wiki/Dipole#Magnetic_vector_potential>'

        :param numpy.ndarray srcLoc: Location of the source(s) (x, y, z)
        :param numpy.ndarray obsLoc: Where the potentials will be calculated (x, y, z)
        :param str component: The component to calculate - 'x', 'y', or 'z'
        :param numpy.ndarray moment: The vector dipole moment (vertical)
        :rtype: numpy.ndarray
        :return: The vector potential each dipole at each observation location
    """

    if component=='x':
        dimInd = 0
    elif component=='y':
        dimInd = 1
    elif component=='z':
        dimInd = 2
    else:
        raise ValueError('Invalid component')

    srcLoc = np.atleast_2d(srcLoc)
    obsLoc = np.atleast_2d(obsLoc)
    moment = np.atleast_2d(moment)

    nFaces = obsLoc.shape[0]
    nSrc = srcLoc.shape[0]

    m = np.array(moment).repeat(nFaces, axis=0)
    B = np.empty((nFaces, nSrc))
    for i in range(nSrc):
        dR = obsLoc - srcLoc[i, np.newaxis].repeat(nFaces, axis=0)
        r = np.sqrt((dR**2).sum(axis=1))
        if dimInd == 0:
            B[:, i] = +(mu/(4*pi)) /(r**3) * (3*dR[:,2]*dR[:,0]/r**2)
        elif dimInd == 1:
            B[:, i] = +(mu/(4*pi)) /(r**3) * (3*dR[:,2]*dR[:,1]/r**2)
        elif dimInd == 2:
            B[:, i] = +(mu/(4*pi)) /(r**3) * (3*dR[:,2]**2/r**2-1)
        else:
            raise Exception("Not Implemented")
    if nSrc == 1:
        return B.flatten()
    return B



def MagneticLoopVectorPotential(srcLoc, obsLoc, component, radius, mu=mu_0):
    """
        Calculate the vector potential of horizontal circular loop
        at given locations

        :param numpy.ndarray srcLoc: Location of the source(s) (x, y, z)
        :param numpy.ndarray,SimPEG.Mesh obsLoc: Where the potentials will be calculated (x, y, z) or a SimPEG Mesh
        :param str,list component: The component to calculate - 'x', 'y', or 'z' if an array, or grid type if mesh, can be a list
        :param numpy.ndarray I: Input current of the loop
        :param numpy.ndarray radius: radius of the loop
        :rtype: numpy.ndarray
        :return: The vector potential each dipole at each observation location
    """

    if type(component) in [list, tuple]:
        out = range(len(component))
        for i, comp in enumerate(component):
            out[i] = MagneticLoopVectorPotential(srcLoc, obsLoc, comp, radius, mu)
        return np.concatenate(out)

    if isinstance(obsLoc, Mesh.BaseMesh):
        mesh = obsLoc
        assert component in ['Ex','Ey','Ez','Fx','Fy','Fz'], "Components must be in: ['Ex','Ey','Ez','Fx','Fy','Fz']"
        return MagneticLoopVectorPotential(srcLoc, getattr(mesh,'grid'+component), component[1], radius, mu)

    srcLoc = np.atleast_2d(srcLoc)
    obsLoc = np.atleast_2d(obsLoc)

    n = obsLoc.shape[0]
    nSrc = srcLoc.shape[0]

    if component=='z':
        A = np.zeros((n, nSrc))
        if nSrc ==1:
            return A.flatten()
        return A

    else:

        A = np.zeros((n, nSrc))
        for i in range (nSrc):
            x = obsLoc[:, 0] - srcLoc[i, 0]
            y = obsLoc[:, 1] - srcLoc[i, 1]
            z = obsLoc[:, 2] - srcLoc[i, 2]
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

        if nSrc == 1:
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
    srcLoc = np.r_[0., 0., 0.]
    Ax = MagneticLoopVectorPotential(srcLoc, mesh.gridEx, 'x', 200)
    Ay = MagneticLoopVectorPotential(srcLoc, mesh.gridEy, 'y', 200)
    Az = MagneticLoopVectorPotential(srcLoc, mesh.gridEz, 'z', 200)
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


