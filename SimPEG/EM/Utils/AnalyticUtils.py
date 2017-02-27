import numpy as np
from SimPEG import Mesh, Utils
from scipy.special import ellipk, ellipe
from scipy.constants import mu_0
import properties

orientationDict = {'X': np.r_[1., 0., 0.],
                   'Y': np.r_[0., 1., 0.],
                   'Z': np.r_[0., 0., 1.]}


def MagneticDipoleVectorPotential(srcLoc, obsLoc, component, moment=1.,
                                  orientation=np.r_[0., 0., 1.],
                                  mu=mu_0):
    """
        Calculate the vector potential of a set of magnetic dipoles
        at given locations 'ref. <http://en.wikipedia.org/wiki/Dipole#Magnetic_vector_potential>'

        :param numpy.ndarray srcLoc: Location of the source(s) (x, y, z)
        :param numpy.ndarray,discretize obsLoc: Where the potentials will be
                                                 calculated (x, y, z) or a
                                                 SimPEG Mesh
        :param str,list component: The component to calculate - 'x', 'y', or
                                   'z' if an array, or grid type if mesh, can
                                   be a list
        :param numpy.ndarray orientation: The vector dipole moment
        :rtype: numpy.ndarray
        :return: The vector potential each dipole at each observation location
    """
    # TODO: break this out!

    if isinstance(orientation, str):
        orientation = orientationDict[orientation]

    assert np.linalg.norm(orientation, 2) == 1., ("orientation must "
                                                            "be a unit vector")

    if type(component) in [list, tuple]:
        out = list(range(len(component)))
        for i, comp in enumerate(component):
            out[i] = MagneticDipoleVectorPotential(srcLoc, obsLoc, comp,
                                                   orientation=orientation,
                                                   mu=mu)
        return np.concatenate(out)

    if isinstance(obsLoc, Mesh.BaseMesh):
        mesh = obsLoc
        assert component in ['Ex', 'Ey', 'Ez', 'Fx', 'Fy', 'Fz'], ("Components"
                                 "must be in: ['Ex','Ey','Ez','Fx','Fy','Fz']")
        return MagneticDipoleVectorPotential(srcLoc, getattr(mesh, 'grid' +
                                                             component),
                                             component[1],
                                             orientation=orientation)

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
    orientation = np.atleast_2d(orientation)

    nObs = obsLoc.shape[0]
    nSrc = srcLoc.shape[0]

    m = moment*np.array(orientation).repeat(nObs, axis=0)
    A = np.empty((nObs, nSrc))
    for i in range(nSrc):
        dR = obsLoc - srcLoc[i, np.newaxis].repeat(nObs, axis=0)
        mCr = np.cross(m, dR)
        r = np.sqrt((dR**2).sum(axis=1))
        A[:, i] = +(mu/(4*np.pi)) * mCr[:, dimInd]/(r**3)
    if nSrc == 1:
        return A.flatten()
    return A


def MagneticDipoleFields(
    srcLoc, obsLoc, component, orientation='Z', moment=1., mu=mu_0
):
    """
        Calculate the vector potential of a set of magnetic dipoles
        at given locations 'ref. <http://en.wikipedia.org/wiki/Dipole#Magnetic_vector_potential>'

        .. math::

            B = \frac{\mu_0}{4 \pi r^3} \left( \frac{3 \vec{r} (\vec{m} \cdot
                                                                \vec{r})}{r^2})
                                                - \vec{m}
                                        \right) \cdot{\hat{rx}}

        :param numpy.ndarray srcLoc: Location of the source(s) (x, y, z)
        :param numpy.ndarray obsLoc: Where the potentials will be calculated
                                     (x, y, z)
        :param str component: The component to calculate - 'x', 'y', or 'z'
        :param numpy.ndarray moment: The vector dipole moment (vertical)
        :rtype: numpy.ndarray
        :return: The vector potential each dipole at each observation location
    """

    if isinstance(orientation, str):
        assert orientation.upper() in ['X', 'Y', 'Z'], ("orientation must be 'x', "
                                                      "'y', or 'z' or a vector"
                                                      "not {}".format(orientation)
                                                      )
    elif (not np.allclose(np.r_[1., 0., 0.], orientation) or
          not np.allclose(np.r_[0., 1., 0.], orientation) or
          not np.allclose(np.r_[0., 0., 1.], orientation)):
        warnings.warn('Arbitrary trasnmitter orientations ({}) not thouroughly tested '
                      'Pull request on a test anyone? bueller?').format(orientation)

    if isinstance(component, str):
        assert component.upper() in ['X', 'Y', 'Z'], ("component must be 'x', "
                                                      "'y', or 'z' or a vector"
                                                      "not {}".format(component)
                                                      )
    elif (not np.allclose(np.r_[1., 0., 0.], component) or
          not np.allclose(np.r_[0., 1., 0.], component) or
          not np.allclose(np.r_[0., 0., 1.], component)):
        warnings.warn('Arbitrary receiver orientations ({}) not thouroughly tested '
                      'Pull request on a test anyone? bueller?').format(component)

    if isinstance(orientation, str):
        orientation = orientationDict[orientation.upper()]

    if isinstance(component, str):
        component = orientationDict[component.upper()]

    assert np.linalg.norm(orientation, 2) == 1., ('orientation must be a unit '
                                                  'vector. Use "moment=X to '
                                                  'scale source fields')

    if np.linalg.norm(component, 2) != 1.:
        warnings.warn('The magnitude of the receiver component vector is > 1, '
                      ' it is {}. The receiver fields will be scaled.'
                      ).format(np.linalg.norm(component, 2))

    srcLoc = np.atleast_2d(srcLoc)
    component = np.atleast_2d(component)
    obsLoc = np.atleast_2d(obsLoc)
    orientation = np.atleast_2d(orientation)

    nObs = obsLoc.shape[0]
    nSrc = int(srcLoc.size / 3.)

    # use outer product to construct an array of [x_src, y_src, z_src]

    m = moment*orientation.repeat(nObs, axis=0)
    B = []

    for i in range(nSrc):
        srcLoc = srcLoc[i, np.newaxis].repeat(nObs, axis=0)
        rx = component.repeat(nObs, axis=0)
        dR = obsLoc - srcLoc
        r = np.sqrt((dR**2).sum(axis=1))
        # mult each element and sum along the axis (vector dot product)
        m_dot_dR_div_r2 = (m * dR).sum(axis=1) / (r**2)

        #multiply the scalar m_dot_dR by the 3D vector r

        rvec_m_dot_dR_div_r2 = np.vstack(
            [np.multiply(m_dot_dR_div_r2, dR[:, i]) for i in range(3)]
        ).T

        # print((3. * rvec_m_dot_dR_div_r2).shape,rvec_m_dot_dR_div_r2.shape, m.shape)
        inside = (3. * rvec_m_dot_dR_div_r2) - m

        # dot product with rx orientation
        inside_dot_rx = (inside * rx).sum(axis=1)
        front = (mu/(4.* np.pi * r**3))
        B.append(Utils.mkvc(np.multiply(front, inside_dot_rx)))

    return np.vstack(B).T


    #     if np.all(orientation == np.r_[1., 0., 0.]):

    #     elif np.all(orientation == np.r_[0., 0., 1.]):
    #         x1 = dR[:, 2]
    #         x2 = dR[:, 0]
    #         x3 = dR[:, 1]


    #     if component == 'x':
    #         B[:, i] = front * (3*x1*x2/r**2)
    #     elif component == 'y':
    #         B[:, i] = front * (3*x1*x3/r**2)
    #     elif component == 'z':
    #         B[:, i] = front * (3*x1**2/r**2-1)
    #     else:
    #         raise Exception("Not Implemented")
    # if nSrc == 1:
    #     return B.flatten()
    # return B



def MagneticLoopVectorPotential(srcLoc, obsLoc, component, radius, orientation='Z', mu=mu_0):
    """
        Calculate the vector potential of horizontal circular loop
        at given locations

        :param numpy.ndarray srcLoc: Location of the source(s) (x, y, z)
        :param numpy.ndarray,discretize obsLoc: Where the potentials will be calculated (x, y, z) or a SimPEG Mesh
        :param str,list component: The component to calculate - 'x', 'y', or 'z' if an array, or grid type if mesh, can be a list
        :param numpy.ndarray I: Input current of the loop
        :param numpy.ndarray radius: radius of the loop
        :rtype: numpy.ndarray
        :return: The vector potential each dipole at each observation location
    """

    if isinstance(orientation, str):
        if orientation.upper() != 'Z':
            raise NotImplementedError('Only Z oriented loops implemented')
    elif not np.allclose(orientation, np.r_[0., 0., 1.]):
        raise NotImplementedError('Only Z oriented loops implemented')

    if type(component) in [list, tuple]:
        out = list(range(len(component)))
        for i, comp in enumerate(component):
            out[i] = MagneticLoopVectorPotential(srcLoc, obsLoc, comp, radius,
                                                 orientation, mu)
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
            Aphi[ind] = ((mu / (np.pi * np.sqrt(m[ind])) *
                         np.sqrt(radius / r[ind]) *((1. - m[ind] / 2.) *
                         K[ind] - E[ind])))
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


