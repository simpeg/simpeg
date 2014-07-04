import numpy as np
from scipy.constants import mu_0, pi
from SimPEG import Mesh

def MagneticDipoleVectorPotential(txLoc, obsLoc, component, dipoleMoment=(0., 0., 1.)):
    """
        Calculate the vector potential of a set of magnetic dipoles
        at given locations 'ref. <http://en.wikipedia.org/wiki/Dipole#Magnetic_vector_potential>'

        :param numpy.ndarray txLoc: Location of the transmitter(s) (x, y, z)
        :param numpy.ndarray,SimPEG.Mesh obsLoc: Where the potentials will be calculated (x, y, z) or a SimPEG Mesh
        :param str,list component: The component to calculate - 'x', 'y', or 'z' if an array, or grid type if mesh, can be a list
        :param numpy.ndarray dipoleMoment: The vector dipole moment
        :rtype: numpy.ndarray
        :return: The vector potential each dipole at each observation location
    """

    if type(component) in [list, tuple]:
        out = range(len(component))
        for i, comp in enumerate(component):
            out[i] = MagneticDipoleVectorPotential(txLoc, obsLoc, comp, dipoleMoment=dipoleMoment)
        return np.concatenate(out)

    if isinstance(obsLoc, Mesh.BaseMesh):
        mesh = obsLoc
        assert component in ['Ex','Ey','Ez','Fx','Fy','Fz'], "Components must be in: ['Ex','Ey','Ez','Fx','Fy','Fz']"
        return MagneticDipoleVectorPotential(txLoc, getattr(mesh,'grid'+component), component[1], dipoleMoment=dipoleMoment)

    if component == 'x':
        dimInd = 0
    elif component == 'y':
        dimInd = 1
    elif component == 'z':
        dimInd = 2
    else:
        raise ValueError('Invalid component')

    txLoc = np.atleast_2d(txLoc)
    obsLoc = np.atleast_2d(obsLoc)
    dipoleMoment = np.atleast_2d(dipoleMoment)

    nEdges = obsLoc.shape[0]
    nTx = txLoc.shape[0]

    m = np.array(dipoleMoment).repeat(nEdges, axis=0)
    A = np.empty((nEdges, nTx))
    for i in range(nTx):
        dR = obsLoc - txLoc[i, np.newaxis].repeat(nEdges, axis=0)
        mCr = np.cross(m, dR)
        r = np.sqrt((dR**2).sum(axis=1))
        A[:, i] = +(mu_0/(4*pi)) * mCr[:,dimInd]/(r**3)
    if nTx == 1:
        return A.flatten()
    return A
