from SimPEG import Survey, Problem, Utils, np, sp
from scipy.constants import mu_0
from SimPEG.EM.Utils import *
from SimPEG.Utils import Zero

class BaseSrc(Survey.BaseSrc):
    """
    Base source class for FDEM Survey
    """

    freq = None
    # rxPair = RxFDEM
    integrate = True

    def eval(self, prob):
        """
        Evaluate the source terms.
        - :math:`S_m` : magnetic source term
        - :math:`S_e` : electric source term

        :param Problem prob: FDEM Problem
        :rtype: (numpy.ndarray, numpy.ndarray)
        :return: tuple with magnetic source term and electric source term
        """
        S_m = self.S_m(prob)
        S_e = self.S_e(prob)
        return S_m, S_e

    def evalDeriv(self, prob, v=None, adjoint=False):
        """
        Derivatives of the source terms with respect to the inversion model
        - :code:`S_mDeriv` : derivative of the magnetic source term
        - :code:`S_eDeriv` : derivative of the electric source term

        :param Problem prob: FDEM Problem
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: (numpy.ndarray, numpy.ndarray)
        :return: tuple with magnetic source term and electric source term derivatives times a vector
        """
        if v is not None:
            return self.S_mDeriv(prob, v, adjoint), self.S_eDeriv(prob, v, adjoint)
        else:
            return lambda v: self.S_mDeriv(prob, v, adjoint), lambda v: self.S_eDeriv(prob, v, adjoint)

    def bPrimary(self, prob):
        """
        Primary magnetic flux density

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: primary magnetic flux density
        """
        return Zero()

    def hPrimary(self, prob):
        """
        Primary magnetic field

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        return Zero()

    def ePrimary(self, prob):
        """
        Primary electric field

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: primary electric field
        """
        return Zero()

    def jPrimary(self, prob):
        """
        Primary current density

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: primary current density
        """
        return Zero()

    def S_m(self, prob):
        """
        Magnetic source term

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: magnetic source term on mesh
        """
        return Zero()

    def S_e(self, prob):
        """
        Electric source term

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: electric source term on mesh
        """
        return Zero()

    def S_mDeriv(self, prob, v, adjoint = False):
        """
        Derivative of magnetic source term with respect to the inversion model

        :param Problem prob: FDEM Problem
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of magnetic source term derivative with a vector
        """

        return Zero()

    def S_eDeriv(self, prob, v, adjoint = False):
        """
        Derivative of electric source term with respect to the inversion model

        :param Problem prob: FDEM Problem
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of electric source term derivative with a vector
        """
        return Zero()


class RawVec_e(BaseSrc):
    """
    RawVec electric source. It is defined by the user provided vector S_e

    :param list rxList: receiver list
    :param float freq: frequency
    :param numpy.array S_e: electric source term
    :param bool integrate: Integrate the source term (multiply by Me) [True]
    """

    def __init__(self, rxList, freq, S_e, integrate=True): #, ePrimary=None, bPrimary=None, hPrimary=None, jPrimary=None):
        self._S_e = np.array(S_e, dtype=complex)
        self.freq = float(freq)
        self.integrate = integrate

        BaseSrc.__init__(self, rxList)

    def S_e(self, prob):
        """
        Electric source term

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: electric source term on mesh
        """
        if prob._formulation is 'EB' and self.integrate is True:
            return prob.Me * self._S_e
        return self._S_e


class RawVec_m(BaseSrc):
    """
    RawVec magnetic source. It is defined by the user provided vector S_m

    :param float freq: frequency
    :param rxList: receiver list
    :param numpy.array S_m: magnetic source term
    :param bool integrate: Integrate the source term (multiply by Me) [True]
    """

    def __init__(self, rxList, freq, S_m, integrate=True):  #ePrimary=Zero(), bPrimary=Zero(), hPrimary=Zero(), jPrimary=Zero()):
        self._S_m = np.array(S_m, dtype=complex)
        self.freq = float(freq)
        self.integrate = integrate

        BaseSrc.__init__(self, rxList)

    def S_m(self, prob):
        """
        Magnetic source term

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: magnetic source term on mesh
        """
        if prob._formulation is 'HJ' and self.integrate is True:
            return prob.Me * self._S_m
        return self._S_m


class RawVec(BaseSrc):
    """
    RawVec source. It is defined by the user provided vectors S_m, S_e

    :param rxList: receiver list
    :param float freq: frequency
    :param numpy.array S_m: magnetic source term
    :param numpy.array S_e: electric source term
    :param bool integrate: Integrate the source term (multiply by Me) [True]
    """
    def __init__(self, rxList, freq, S_m, S_e, integrate=True):
        self._S_m = np.array(S_m, dtype=complex)
        self._S_e = np.array(S_e, dtype=complex)
        self.freq = float(freq)
        self.integrate = integrate
        BaseSrc.__init__(self, rxList)

    def S_m(self, prob):
        """
        Magnetic source term

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: magnetic source term on mesh
        """
        if prob._formulation is 'HJ' and self.integrate is True:
            return prob.Me * self._S_m
        return self._S_m

    def S_e(self, prob):
        """
        Electric source term

        :param Problem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: electric source term on mesh
        """
        if prob._formulation is 'EB' and self.integrate is True:
            return prob.Me * self._S_e
        return self._S_e


class MagDipole(BaseSrc):
    """
    Point magnetic dipole source calculated by taking the curl of a magnetic
    vector potential. By taking the discrete curl, we ensure that the magnetic
    flux density is divergence free (no magnetic monopoles!).

    This approach uses a primary-secondary in frequency. Here we show the
    derivation for E-B formulation noting that similar steps are followed for
    the H-J formulation.

    .. math::
        \mathbf{C} \mathbf{e} + i \omega \mathbf{b} = \mathbf{s_m} \\\\
            {\mathbf{C}^T \mathbf{M_{\mu^{-1}}^f} \mathbf{b} - \mathbf{M_{\sigma}^e} \mathbf{e} = \mathbf{s_e}}

    We split up the fields and :math:`\mu^{-1}` into primary (:math:`\mathbf{P}`) and secondary (:math:`\mathbf{S}`) components

    - :math:`\mathbf{e} = \mathbf{e^P} + \mathbf{e^S}`
    - :math:`\mathbf{b} = \mathbf{b^P} + \mathbf{b^S}`
    - :math:`\\boldsymbol{\mu}^{\mathbf{-1}} = \\boldsymbol{\mu}^{\mathbf{-1}^\mathbf{P}} + \\boldsymbol{\mu}^{\mathbf{-1}^\mathbf{S}}`

    and define a zero-frequency primary problem, noting that the source is
    generated by a divergence free electric current

    .. math::
        \mathbf{C} \mathbf{e^P} = \mathbf{s_m^P} = 0 \\\\
            {\mathbf{C}^T \mathbf{{M_{\mu^{-1}}^f}^P} \mathbf{b^P} - \mathbf{M_{\sigma}^e} \mathbf{e^P} = \mathbf{M^e} \mathbf{s_e^P}}

    Since :math:`\mathbf{e^P}` is curl-free, divergence-free, we assume that there is no constant field background, the :math:`\mathbf{e^P} = 0`, so our primary problem is

    .. math::
        \mathbf{e^P} =  0 \\\\
            {\mathbf{C}^T \mathbf{{M_{\mu^{-1}}^f}^P} \mathbf{b^P} = \mathbf{s_e^P}}

    Our secondary problem is then

    .. math::
        \mathbf{C} \mathbf{e^S} + i \omega \mathbf{b^S} = - i \omega \mathbf{b^P} \\\\
            {\mathbf{C}^T \mathbf{M_{\mu^{-1}}^f} \mathbf{b^S} - \mathbf{M_{\sigma}^e} \mathbf{e^S} = -\mathbf{C}^T \mathbf{{M_{\mu^{-1}}^f}^S} \mathbf{b^P}}

    :param list rxList: receiver list
    :param float freq: frequency
    :param numpy.ndarray loc: source location (ie: :code:`np.r_[xloc,yloc,zloc]`)
    :param string orientation: 'X', 'Y', 'Z'
    :param float moment: magnetic dipole moment
    :param float mu: background magnetic permeability
    """

    def __init__(self, rxList, freq, loc, orientation='Z', moment=1., mu=mu_0):
        self.freq = float(freq)
        self.loc = loc
        self.orientation = orientation
        assert orientation in ['X','Y','Z'], "Orientation (right now) doesn't actually do anything! The methods in SrcUtils should take care of this..."
        self.moment = moment
        self.mu = mu
        self.integrate = False
        BaseSrc.__init__(self, rxList)

    def bPrimary(self, prob):
        """
        The primary magnetic flux density from a magnetic vector potential

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        formulation = prob._formulation

        if formulation is 'EB':
            gridX = prob.mesh.gridEx
            gridY = prob.mesh.gridEy
            gridZ = prob.mesh.gridEz
            C = prob.mesh.edgeCurl

        elif formulation is 'HJ':
            gridX = prob.mesh.gridFx
            gridY = prob.mesh.gridFy
            gridZ = prob.mesh.gridFz
            C = prob.mesh.edgeCurl.T


        if prob.mesh._meshType is 'CYL':
            if not prob.mesh.isSymmetric:
                # TODO ?
                raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')
            a = MagneticDipoleVectorPotential(self.loc, gridY, 'y', mu=self.mu, moment=self.moment)

        else:
            srcfct = MagneticDipoleVectorPotential
            ax = srcfct(self.loc, gridX, 'x', mu=self.mu, moment=self.moment)
            ay = srcfct(self.loc, gridY, 'y', mu=self.mu, moment=self.moment)
            az = srcfct(self.loc, gridZ, 'z', mu=self.mu, moment=self.moment)
            a = np.concatenate((ax, ay, az))

        return C*a

    def hPrimary(self, prob):
        """
        The primary magnetic field from a magnetic vector potential

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        b = self.bPrimary(prob)
        return 1./self.mu * b 

    def S_m(self, prob):
        """
        The magnetic source term

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """

        b_p = self.bPrimary(prob)
        if prob._formulation is 'HJ':
            b_p = prob.Me * b_p 
        return -1j*omega(self.freq)*b_p

    def S_e(self, prob):
        """
        The electric source term

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """

        if all(np.r_[self.mu] == np.r_[prob.curModel.mu]):
            return Zero()
        else:
            formulation = prob._formulation

            if formulation is 'EB':
                mui_s = prob.curModel.mui - 1./self.mu
                MMui_s = prob.mesh.getFaceInnerProduct(mui_s)
                C = prob.mesh.edgeCurl
            elif formulation is 'HJ':
                mu_s = prob.curModel.mu - self.mu
                MMui_s = prob.mesh.getEdgeInnerProduct(mu_s, invMat=True)
                C = prob.mesh.edgeCurl.T

            return -C.T * (MMui_s * self.bPrimary(prob))


class MagDipole_Bfield(BaseSrc):

    """
    Point magnetic dipole source calculated with the analytic solution for the
    fields from a magnetic dipole. No discrete curl is taken, so the magnetic
    flux density may not be strictly divergence free.

    This approach uses a primary-secondary in frequency in the same fashion as the MagDipole.

    :param list rxList: receiver list
    :param float freq: frequency
    :param numpy.ndarray loc: source location (ie: :code:`np.r_[xloc,yloc,zloc]`)
    :param string orientation: 'X', 'Y', 'Z'
    :param float moment: magnetic dipole moment
    :param float mu: background magnetic permeability
    """

    def __init__(self, rxList, freq, loc, orientation='Z', moment=1., mu = mu_0):
        self.freq = float(freq)
        self.loc = loc
        assert orientation in ['X','Y','Z'], "Orientation (right now) doesn't actually do anything! The methods in SrcUtils should take care of this..."
        self.orientation = orientation
        self.moment = moment
        self.mu = mu
        BaseSrc.__init__(self, rxList)

    def bPrimary(self, prob):
        """
        The primary magnetic flux density from the analytic solution for magnetic fields from a dipole

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """

        formulation = prob._formulation

        if formulation is 'EB':
            gridX = prob.mesh.gridFx
            gridY = prob.mesh.gridFy
            gridZ = prob.mesh.gridFz
            C = prob.mesh.edgeCurl

        elif formulation is 'HJ':
            gridX = prob.mesh.gridEx
            gridY = prob.mesh.gridEy
            gridZ = prob.mesh.gridEz
            C = prob.mesh.edgeCurl.T

        srcfct = MagneticDipoleFields
        if prob.mesh._meshType is 'CYL':
            if not prob.mesh.isSymmetric:
                # TODO ?
                raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')
            bx = srcfct(self.loc, gridX, 'x', mu=self.mu, moment=self.moment)
            bz = srcfct(self.loc, gridZ, 'z', mu=self.mu, moment=self.moment)
            b = np.concatenate((bx,bz))
        else:
            bx = srcfct(self.loc, gridX, 'x', mu=self.mu, moment=self.moment)
            by = srcfct(self.loc, gridY, 'y', mu=self.mu, moment=self.moment)
            bz = srcfct(self.loc, gridZ, 'z', mu=self.mu, moment=self.moment)
            b = np.concatenate((bx,by,bz))

        return b

    def hPrimary(self, prob):
        """
        The primary magnetic field from a magnetic vector potential

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        b = self.bPrimary(prob)
        return 1/self.mu * b

    def S_m(self, prob):
        """
        The magnetic source term

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        b = self.bPrimary(prob)
        if prob._formulation is 'HJ':
            b = prob.Me * b
        return -1j*omega(self.freq)*b

    def S_e(self, prob):
        """
        The electric source term

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        if all(np.r_[self.mu] == np.r_[prob.curModel.mu]):
            return Zero()
        else:
            formulation = prob._formulation

            if formulation is 'EB':
                mui_s = prob.curModel.mui - 1./self.mu
                MMui_s = prob.mesh.getFaceInnerProduct(mui_s)
                C = prob.mesh.edgeCurl
            elif formulation is 'HJ':
                mu_s = prob.curModel.mu - self.mu
                MMui_s = prob.mesh.getEdgeInnerProduct(mu_s, invMat=True)
                C = prob.mesh.edgeCurl.T

            return -C.T * (MMui_s * self.bPrimary(prob))


class CircularLoop(BaseSrc):
    """
    Circular loop magnetic source calculated by taking the curl of a magnetic
    vector potential. By taking the discrete curl, we ensure that the magnetic
    flux density is divergence free (no magnetic monopoles!).

    This approach uses a primary-secondary in frequency in the same fashion as the MagDipole.

    :param list rxList: receiver list
    :param float freq: frequency
    :param numpy.ndarray loc: source location (ie: :code:`np.r_[xloc,yloc,zloc]`)
    :param string orientation: 'X', 'Y', 'Z'
    :param float moment: magnetic dipole moment
    :param float mu: background magnetic permeability
    """

    def __init__(self, rxList, freq, loc, orientation='Z', radius=1., mu=mu_0):
        self.freq = float(freq)
        self.orientation = orientation
        assert orientation in ['X','Y','Z'], "Orientation (right now) doesn't actually do anything! The methods in SrcUtils should take care of this..."
        self.radius = radius
        self.mu = mu
        self.loc = loc
        self.integrate = False
        BaseSrc.__init__(self, rxList)

    def bPrimary(self, prob):
        """
        The primary magnetic flux density from a magnetic vector potential

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        formulation = prob._formulation

        if formulation is 'EB':
            gridX = prob.mesh.gridEx
            gridY = prob.mesh.gridEy
            gridZ = prob.mesh.gridEz
            C = prob.mesh.edgeCurl

        elif formulation is 'HJ':
            gridX = prob.mesh.gridFx
            gridY = prob.mesh.gridFy
            gridZ = prob.mesh.gridFz
            C = prob.mesh.edgeCurl.T

        if prob.mesh._meshType is 'CYL':
            if not prob.mesh.isSymmetric:
                # TODO ?
                raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')
            a = MagneticDipoleVectorPotential(self.loc, gridY, 'y', moment=self.radius, mu=self.mu)

        else:
            srcfct = MagneticDipoleVectorPotential
            ax = srcfct(self.loc, gridX, 'x', self.radius, mu=self.mu)
            ay = srcfct(self.loc, gridY, 'y', self.radius, mu=self.mu)
            az = srcfct(self.loc, gridZ, 'z', self.radius, mu=self.mu)
            a = np.concatenate((ax, ay, az))

        return C*a

    def hPrimary(self, prob):
        """
        The primary magnetic field from a magnetic vector potential

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        b = self.bPrimary(prob)
        return 1./self.mu*b

    def S_m(self, prob):
        """
        The magnetic source term

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        b = self.bPrimary(prob)
        if prob._formulation is 'HJ':
            b =  prob.Me *  b
        return -1j*omega(self.freq)*b

    def S_e(self, prob):
        """
        The electric source term

        :param Problem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        if all(np.r_[self.mu] == np.r_[prob.curModel.mu]):
            return Zero()
        else:
            formulation = prob._formulation

            if formulation is 'EB':
                mui_s = prob.curModel.mui - 1./self.mu
                MMui_s = prob.mesh.getFaceInnerProduct(mui_s)
                C = prob.mesh.edgeCurl


            elif formulation is 'HJ':
                mu_s = prob.curModel.mu - self.mu
                MMui_s = prob.mesh.getEdgeInnerProduct(mu_s, invMat=True)
                C = prob.mesh.edgeCurl.T

            return -C.T * (MMui_s * self.bPrimary(prob))

            


