from SimPEG import Survey, Problem, Utils, np, sp
from scipy.constants import mu_0
from SimPEG.EM.Utils import (omega, MagneticDipoleFields,
                             MagneticDipoleVectorPotential,
                             MagneticLoopVectorPotential,
                             orientationDict)
from SimPEG.Utils import Zero
import warnings


class BaseSrc(Survey.BaseSrc):
    """
    Base source class for FDEM Survey
    """

    freq = None
    integrate = False
    _ePrimary = None
    _bPrimary = None
    _hPrimary = None
    _jPrimary = None

    def __init__(self, rxList, **kwargs):
        Survey.BaseSrc.__init__(self, rxList, **kwargs)

    def eval(self, prob):
        """
        - :math:`s_m` : magnetic source term
        - :math:`s_e` : electric source term

        :param BaseFDEMProblem prob: FDEM Problem
        :rtype: tuple
        :return: tuple with magnetic source term and electric source term
        """
        s_m = self.s_m(prob)
        s_e = self.s_e(prob)
        return s_m, s_e

    def evalDeriv(self, prob, v=None, adjoint=False):
        """
        Derivatives of the source terms with respect to the inversion model
        - :code:`s_mDeriv` : derivative of the magnetic source term
        - :code:`s_eDeriv` : derivative of the electric source term

        :param BaseFDEMProblem prob: FDEM Problem
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: tuple
        :return: tuple with magnetic source term and electric source term
            derivatives times a vector
        """
        if v is not None:
            return (
                self.s_mDeriv(prob, v, adjoint),
                self.s_eDeriv(prob, v, adjoint)
            )
        else:
            return (
                lambda v: self.s_mDeriv(prob, v, adjoint),
                lambda v: self.s_eDeriv(prob, v, adjoint)
            )

    def bPrimary(self, prob):
        """
        Primary magnetic flux density

        :param BaseFDEMProblem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: primary magnetic flux density
        """
        if self._bPrimary is None:
            return Zero()
        return self._bPrimary

    def bPrimaryDeriv(self, prob, v, adjoint=False):
        """
        Derivative of the primary magnetic flux density

        :param BaseFDEMProblem prob: FDEM Problem
        :param numpy.ndarray v: vector
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: primary magnetic flux density
        """
        return Zero()

    def hPrimary(self, prob):
        """
        Primary magnetic field

        :param BaseFDEMProblem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        if self._hPrimary is None:
            return Zero()
        return self._hPrimary

    def hPrimaryDeriv(self, prob, v, adjoint=False):
        """
        Derivative of the primary magnetic field

        :param BaseFDEMProblem prob: FDEM Problem
        :param numpy.ndarray v: vector
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: primary magnetic flux density
        """
        return Zero()

    def ePrimary(self, prob):
        """
        Primary electric field

        :param BaseFDEMProblem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: primary electric field
        """
        if self._ePrimary is None:
            return Zero()
        return self._ePrimary

    def ePrimaryDeriv(self, prob, v, adjoint=False):
        """
        Derivative of the primary electric field

        :param BaseFDEMProblem prob: FDEM Problem
        :param numpy.ndarray v: vector
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: primary magnetic flux density
        """
        return Zero()

    def jPrimary(self, prob):
        """
        Primary current density

        :param BaseFDEMProblem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: primary current density
        """
        if self._jPrimary is None:
            return Zero()
        return self._jPrimary

    def jPrimaryDeriv(self, prob, v, adjoint=False):
        """
        Derivative of the primary current density

        :param BaseFDEMProblem prob: FDEM Problem
        :param numpy.ndarray v: vector
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: primary magnetic flux density
        """
        return Zero()

    def s_m(self, prob):
        """
        Magnetic source term

        :param BaseFDEMProblem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: magnetic source term on mesh
        """
        return Zero()

    def s_e(self, prob):
        """
        Electric source term

        :param BaseFDEMProblem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: electric source term on mesh
        """
        return Zero()

    def s_mDeriv(self, prob, v, adjoint = False):
        """
        Derivative of magnetic source term with respect to the inversion model

        :param BaseFDEMProblem prob: FDEM Problem
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of magnetic source term derivative with a vector
        """

        return Zero()

    def s_eDeriv(self, prob, v, adjoint = False):
        """
        Derivative of electric source term with respect to the inversion model

        :param BaseFDEMProblem prob: FDEM Problem
        :param numpy.ndarray v: vector to take product with
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: product of electric source term derivative with a vector
        """
        return Zero()


class RawVec_e(BaseSrc):
    """
    RawVec electric source. It is defined by the user provided vector s_e

    :param list rxList: receiver list
    :param float freq: frequency
    :param numpy.array s_e: electric source term
    :param bool integrate: Integrate the source term (multiply by Me) [False]
    """

    def __init__(self, rxList, freq, s_e, **kwargs):
        self._s_e = np.array(s_e, dtype=complex)
        self.freq = float(freq)

        BaseSrc.__init__(self, rxList, **kwargs)

    def s_e(self, prob):
        """
        Electric source term

        :param BaseFDEMProblem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: electric source term on mesh
        """
        if prob._formulation is 'EB' and self.integrate is True:
            return prob.Me * self._s_e
        return self._s_e


class RawVec_m(BaseSrc):
    """
    RawVec magnetic source. It is defined by the user provided vector s_m

    :param float freq: frequency
    :param rxList: receiver list
    :param numpy.array s_m: magnetic source term
    :param bool integrate: Integrate the source term (multiply by Me) [False]
    """

    def __init__(self, rxList, freq, s_m, **kwargs):
        self._s_m = np.array(s_m, dtype=complex)
        self.freq = float(freq)

        BaseSrc.__init__(self, rxList, **kwargs)

    def s_m(self, prob):
        """
        Magnetic source term

        :param BaseFDEMProblem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: magnetic source term on mesh
        """
        if prob._formulation is 'HJ' and self.integrate is True:
            return prob.Me * self._s_m
        return self._s_m


class RawVec(BaseSrc):
    """
    RawVec source. It is defined by the user provided vectors s_m, s_e

    :param rxList: receiver list
    :param float freq: frequency
    :param numpy.array s_m: magnetic source term
    :param numpy.array s_e: electric source term
    :param bool integrate: Integrate the source term (multiply by Me) [False]
    """
    def __init__(self, rxList, freq, s_m, s_e, **kwargs):
        self._s_m = np.array(s_m, dtype=complex)
        self._s_e = np.array(s_e, dtype=complex)
        self.freq = float(freq)
        BaseSrc.__init__(self, rxList, **kwargs)

    def s_m(self, prob):
        """
        Magnetic source term

        :param BaseFDEMProblem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: magnetic source term on mesh
        """
        if prob._formulation is 'HJ' and self.integrate is True:
            return prob.Me * self._s_m
        return self._s_m

    def s_e(self, prob):
        """
        Electric source term

        :param BaseFDEMProblem prob: FDEM Problem
        :rtype: numpy.ndarray
        :return: electric source term on mesh
        """
        if prob._formulation is 'EB' and self.integrate is True:
            return prob.Me * self._s_e
        return self._s_e


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
            {\mathbf{C}^T \mathbf{M_{\mu^{-1}}^f} \mathbf{b} -
            \mathbf{M_{\sigma}^e} \mathbf{e} = \mathbf{s_e}}

    We split up the fields and :math:`\mu^{-1}` into primary
    (:math:`\mathbf{P}`) and secondary (:math:`\mathbf{S}`) components

    - :math:`\mathbf{e} = \mathbf{e^P} + \mathbf{e^S}`
    - :math:`\mathbf{b} = \mathbf{b^P} + \mathbf{b^S}`
    - :math:`\\boldsymbol{\mu}^{\mathbf{-1}} =
        \\boldsymbol{\mu}^{\mathbf{-1}^\mathbf{P}} +
        \\boldsymbol{\mu}^{\mathbf{-1}^\mathbf{S}}`

    and define a zero-frequency primary problem, noting that the source is
    generated by a divergence free electric current

    .. math::
        \mathbf{C} \mathbf{e^P} = \mathbf{s_m^P} = 0 \\\\
            {\mathbf{C}^T \mathbf{{M_{\mu^{-1}}^f}^P} \mathbf{b^P} -
            \mathbf{M_{\sigma}^e} \mathbf{e^P} = \mathbf{M^e} \mathbf{s_e^P}}

    Since :math:`\mathbf{e^P}` is curl-free, divergence-free, we assume that
    there is no constant field background, the :math:`\mathbf{e^P} = 0`, so our
    primary problem is

    .. math::
        \mathbf{e^P} =  0 \\\\
            {\mathbf{C}^T \mathbf{{M_{\mu^{-1}}^f}^P} \mathbf{b^P} =
            \mathbf{s_e^P}}

    Our secondary problem is then

    .. math::
        \mathbf{C} \mathbf{e^S} + i \omega \mathbf{b^S} =
        - i \omega \mathbf{b^P} \\\\
            {\mathbf{C}^T \mathbf{M_{\mu^{-1}}^f} \mathbf{b^S} -
            \mathbf{M_{\sigma}^e} \mathbf{e^S} =
            -\mathbf{C}^T \mathbf{{M_{\mu^{-1}}^f}^S} \mathbf{b^P}}

    :param list rxList: receiver list
    :param float freq: frequency
    :param numpy.ndarray loc: source location
        (ie: :code:`np.r_[xloc,yloc,zloc]`)
    :param string orientation: 'X', 'Y', 'Z'
    :param float moment: magnetic dipole moment
    :param float mu: background magnetic permeability
    """

    def __init__(self, rxList, freq, loc, orientation='Z', moment=1., mu=mu_0,
                 **kwargs):
        self.freq = float(freq)
        self.loc = loc
        if isinstance(orientation, str):
            assert orientation.upper() in ['X', 'Y', 'Z'], (
                "orientation must be in 'X', 'Y', 'Z' not {}".format(
                    orientation
                )
            )
            orientation = orientationDict[orientation.upper()]
        elif (np.linalg.norm(orientation - np.r_[1., 0., 0.]) > 1e-6 or
              np.linalg.norm(orientation - np.r_[0., 1., 0.]) > 1e-6 or
              np.linalg.norm(orientation - np.r_[0., 0., 1.]) > 1e-6):
                warnings.warn(
                    'Using orientations that are not in aligned with the mesh '
                    'axes is not thoroughly tested. PR on a test??')

        assert np.linalg.norm(orientation) == 1., (
            'Orientation must have unit length, not {}'.format(
                np.linalg.norm(orientation)
            )
        )

        self.orientation = orientation
        self.moment = moment
        self.mu = mu
        Utils.setKwargs(self, **kwargs)
        BaseSrc.__init__(self, rxList)

    def _srcFct(self, obsLoc, component):
        return MagneticDipoleVectorPotential(
            self.loc, obsLoc, component, mu=self.mu, moment=self.moment,
            orientation=self.orientation
        )

    def bPrimary(self, prob):
        """
        The primary magnetic flux density from a magnetic vector potential

        :param BaseFDEMProblem prob: FDEM problem
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
                raise NotImplementedError('Non-symmetric cyl mesh not '
                                          'implemented yet!')
            assert (np.linalg.norm(self.orientation - np.r_[0., 0., 1.]) <
                    1e-6), ('for cylindrical symmetry, the dipole must be '
                            'oriented in the Z direction')
            a = self._srcFct(gridY, 'y')

        else:
            ax = self._srcFct(gridX, 'x')
            ay = self._srcFct(gridY, 'y')
            az = self._srcFct(gridZ, 'z')
            a = np.concatenate((ax, ay, az))

        return C*a

    def hPrimary(self, prob):
        """
        The primary magnetic field from a magnetic vector potential

        :param BaseFDEMProblem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        b = self.bPrimary(prob)
        return 1./self.mu * b

    def s_m(self, prob):
        """
        The magnetic source term

        :param BaseFDEMProblem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """

        b_p = self.bPrimary(prob)
        if prob._formulation is 'HJ':
            b_p = prob.Me * b_p
        return -1j*omega(self.freq)*b_p

    def s_e(self, prob):
        """
        The electric source term

        :param BaseFDEMProblem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """

        if all(np.r_[self.mu] == np.r_[prob.mu]):
            return Zero()
        else:
            formulation = prob._formulation

            if formulation is 'EB':
                mui_s = prob.mui - 1./self.mu
                MMui_s = prob.mesh.getFaceInnerProduct(mui_s)
                C = prob.mesh.edgeCurl
            elif formulation is 'HJ':
                mu_s = prob.mu - self.mu
                MMui_s = prob.mesh.getEdgeInnerProduct(mu_s, invMat=True)
                C = prob.mesh.edgeCurl.T

            return -C.T * (MMui_s * self.bPrimary(prob))


class MagDipole_Bfield(MagDipole):

    """
    Point magnetic dipole source calculated with the analytic solution for the
    fields from a magnetic dipole. No discrete curl is taken, so the magnetic
    flux density may not be strictly divergence free.

    This approach uses a primary-secondary in frequency in the same fashion as
    the MagDipole.

    :param list rxList: receiver list
    :param float freq: frequency
    :param numpy.ndarray loc: source location (ie:
                              :code:`np.r_[xloc,yloc,zloc]`)
    :param string orientation: 'X', 'Y', 'Z'
    :param float moment: magnetic dipole moment
    :param float mu: background magnetic permeability
    """

    def __init__(self, rxList, freq, loc, orientation='Z', moment=1., mu=mu_0):
        super(MagDipole_Bfield, self).__init__(
            rxList, freq, loc, orientation=orientation, moment=moment, mu=mu
        )

    def _srcFct(self, obsLoc, component):
        return MagneticDipoleFields(
            self.srcLoc, obsLoc, component, mu=self.mu, moment=self.moment,
            orientation=self.orientation
        )

    def bPrimary(self, prob):
        """
        The primary magnetic flux density from the analytic solution for
        magnetic fields from a dipole

        :param BaseFDEMProblem prob: FDEM problem
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """

        formulation = prob._formulation

        if formulation is 'EB':
            gridX = prob.mesh.gridFx
            gridY = prob.mesh.gridFy
            gridZ = prob.mesh.gridFz

        elif formulation is 'HJ':
            gridX = prob.mesh.gridEx
            gridY = prob.mesh.gridEy
            gridZ = prob.mesh.gridEz

        srcfct = MagneticDipoleFields
        if prob.mesh._meshType is 'CYL':
            if not prob.mesh.isSymmetric:
                # TODO ?
                raise NotImplementedError(
                    'Non-symmetric cyl mesh not implemented yet!'
                )
            bx = srcfct(self.loc, gridX, 'x', mu=self.mu, moment=self.moment)
            bz = srcfct(self.loc, gridZ, 'z', mu=self.mu, moment=self.moment)
            b = np.concatenate((bx, bz))
        else:
            bx = srcfct(self.loc, gridX, 'x', mu=self.mu, moment=self.moment)
            by = srcfct(self.loc, gridY, 'y', mu=self.mu, moment=self.moment)
            bz = srcfct(self.loc, gridZ, 'z', mu=self.mu, moment=self.moment)
            b = np.concatenate((bx, by, bz))

        return Utils.mkvc(b)


class CircularLoop(MagDipole):
    """
    Circular loop magnetic source calculated by taking the curl of a magnetic
    vector potential. By taking the discrete curl, we ensure that the magnetic
    flux density is divergence free (no magnetic monopoles!).

    This approach uses a primary-secondary in frequency in the same fashion as
    the MagDipole.

    :param list rxList: receiver list
    :param float freq: frequency
    :param numpy.ndarray loc: source location
        (ie: :code:`np.r_[xloc,yloc,zloc]`)
    :param string orientation: 'X', 'Y', 'Z'
    :param float moment: magnetic dipole moment
    :param float mu: background magnetic permeability
    """

    def __init__(self, rxList, freq, loc, orientation='Z', radius=1., mu=mu_0):
        self.radius = radius
        super(CircularLoop, self).__init__(rxList, freq, loc,
                                           orientation=orientation,
                                           mu=mu)

    def _srcFct(self, obsLoc, component):
        return MagneticLoopVectorPotential(
            self.loc, obsLoc, component, mu=self.mu, radius=self.radius,
            orientation=self.orientation
    )


class PrimSecSigma(BaseSrc):

    def __init__(self, rxList, freq, sigBack, ePrimary, **kwargs):
        self.sigBack = sigBack

        BaseSrc.__init__(self, rxList, freq=freq, _ePrimary=ePrimary, **kwargs)

    def s_e(self, prob):
        return (
            prob.MeSigma -  prob.mesh.getEdgeInnerProduct(self.sigBack)
        ) * self.ePrimary(prob)

    def s_eDeriv(self, prob, v, adjoint=False):
        if adjoint:
            return prob.MeSigmaDeriv(self.ePrimary(prob)).T * v
        return prob.MeSigmaDeriv(self.ePrimary(prob)) * v


class PrimSecMappedSigma(BaseSrc):

    """
    Primary-Secondary Source in which a mapping is provided to put the current
    model onto the primary mesh. This is solved on every model update.
    There are a lot of layers to the derivatives here!

    **Required**
    :param list rxList: Receiver List
    :param float freq: frequency
    :param BaseFDEMProblem primaryProblem: FDEM primary problem
    :param SurveyFDEM primarySurvey: FDEM primary survey

    **Optional**
    :param Mapping map2meshSecondary: mapping current model to act as primary
    model on the secondary mesh
    """

    def __init__(self, rxList, freq, primaryProblem, primarySurvey,
                 map2meshSecondary=None, **kwargs):

        self.primaryProblem = primaryProblem
        self.primarySurvey = primarySurvey

        if self.primaryProblem.ispaired is False:
            self.primaryProblem.pair(self.primarySurvey)

        self.map2meshSecondary = map2meshSecondary

        BaseSrc.__init__(self, rxList, freq=freq, **kwargs)

    def _ProjPrimary(self, prob, locType, locTypeTo):
        # TODO: if meshes have not changed, store the projection
        # if getattr(self, '__ProjPrimary', None) is None:

        # TODO: implement for HJ formulation
        if prob._formulation == 'EB':
            pass
        else:
            raise NotImplementedError(
                'PrimSecMappedSigma Source has not been implemented for {} '
                'formulation'.format(prob._formulation)
                )

        # TODO: only set up for tensot meshes (Tree meshes should be easy/done)
        # but have not been tried or tested.
        assert prob.mesh._meshType in ['TENSOR'], (
            'PrimSecMappedSigma source has not been implemented for {}'.format(
                prob.mesh._meshType)
            )

        # if EB formulation, interpolate E, elif HJ interpolate J
        # if self.primaryProblem._formulation == 'EB':
        #     locType = 'E'
        # elif self.primaryProblem._formulation == 'HJ':
        #     locType = 'F'

        # get interpolation mat from primary mesh to secondary mesh
        if self.primaryProblem.mesh._meshType == 'CYL':
            return self.primaryProblem.mesh.getInterpolationMatCartMesh(
                prob.mesh, locType=locType, locTypeTo=locTypeTo
            )
        return self.primaryProblem.mesh.getInterploationMat(
            prob.mesh, locType=locType, locTypeTo=locTypeTo
        )

        # return self.__ProjPrimary

    def _primaryFields(self, prob, fieldType=None, f=None):
        # TODO: cache and check if prob.curModel has changed

        if f is None:
            f = self.primaryProblem.fields(prob.model)

        if fieldType is not None:
            return f[:, fieldType]
        return f

    def _primaryFieldsDeriv(self, prob, v, adjoint=False, f=None):
        # TODO: this should not be hard-coded for j
        # jp = self._primaryFields(prob)[:,'j']

        # TODO: pull apart Jvec so that don't have to copy paste this code in
        # A = self.primaryProblem.getA(self.freq)
        # Ainv = self.primaryProblem.Solver(A, **self.primaryProblem.solverOpts) # create the concept of Ainv (actually a solve)

        if f is None:
            f = self._primaryFields(prob.sigma, f=f)

        freq = self.freq

        A = self.primaryProblem.getA(freq)
        src = self.primarySurvey.srcList[0]
        u_src = Utils.mkvc(f[src, self.primaryProblem._solutionType])

        if adjoint is True:
            Jtv = np.zeros(prob.sigmaMap.nP, dtype=complex)
            ATinv = self.primaryProblem.Solver(
                A.T, **self.primaryProblem.solverOpts
            )
            df_duTFun = getattr(
                f, '_{0}Deriv'.format(
                    'e' if self.primaryProblem._formulation is 'EB' else 'j'
                ),
                None
            )
            df_duT, df_dmT = df_duTFun(src, None, v, adjoint=True)

            ATinvdf_duT = ATinv * df_duT

            dA_dmT = self.primaryProblem.getADeriv(
                freq, u_src, ATinvdf_duT, adjoint=True
            )
            dRHS_dmT = self.primaryProblem.getRHSDeriv(
                freq, src, ATinvdf_duT, adjoint=True
            )

            du_dmT = -dA_dmT + dRHS_dmT

            Jtv += df_dmT + du_dmT

            ATinv.clean()

            return Utils.mkvc(Jtv)

        # create the concept of Ainv (actually a solve)
        Ainv = self.primaryProblem.Solver(A, **self.primaryProblem.solverOpts)

        # for src in self.survey.getSrcByFreq(freq):
        dA_dm_v = self.primaryProblem.getADeriv(freq, u_src, v)
        dRHS_dm_v = self.primaryProblem.getRHSDeriv(freq, src, v)
        du_dm_v = Ainv * (-dA_dm_v + dRHS_dm_v)

        # if self.primaryProblem._formulation == 'EB':
        df_dmFun = getattr(
            f, '_{0}Deriv'.format(
                'e' if self.primaryProblem._formulation is 'EB' else 'j'
            ),
            None
        )
        # elif self.primaryProblem._formulation == 'HJ':
        #     df_dmFun = getattr(f, '_{0}Deriv'.format('j'), None)
        df_dm_v = df_dmFun(src, du_dm_v, v, adjoint=False)
        # Jv[src, rx] = rx.evalDeriv(src, self.mesh, f, df_dm_v)
        Ainv.clean()

        return df_dm_v

        # return self.primaryProblem.Jvec(prob.curModel, v, f=f)

    def ePrimary(self, prob, f=None):
        if f is None:
            f = self._primaryFields(prob)

        if self.primaryProblem._formulation == 'EB':
            ep = self._ProjPrimary(prob, 'E', 'E') * f[:, 'e']
        elif self.primaryProblem._formulation == 'HJ':
            ep = self._ProjPrimary(prob, 'F', 'E') * (
                    self.primaryProblem.MfI * (
                        self.primaryProblem.MfRho * f[:, 'j'])
                    )

        return Utils.mkvc(ep)

    def ePrimaryDeriv(self, prob, v, adjoint=False, f=None):

        if f is None:
            f = self._primaryFields(prob)

        # if adjoint is True:
        #     raise NotImplementedError
        if self.primaryProblem._formulation == 'EB':
            if adjoint is True:
                epDeriv = self._primaryFieldsDeriv(
                    prob, (self._ProjPrimary(prob, 'E', 'E').T * v), f=f,
                    adjoint=adjoint
                )
            else:
                epDeriv = (
                    self._ProjPrimary(prob, 'E', 'E') *
                    self._primaryFieldsDeriv(prob, v, f=f)
                )
        elif self.primaryProblem._formulation == 'HJ':
            if adjoint is True:
                PTv = (
                    self.primaryProblem.MfI.T *
                    (self._ProjPrimary(prob, 'F', 'E').T * v)
                )
                epDeriv = (
                    self.primaryProblem.MfRhoDeriv(f[:, 'j']).T * PTv +
                    self._primaryFieldsDeriv(
                        prob, self.primaryProblem.MfRho.T * PTv,
                        adjoint=adjoint, f=f
                    )
                )
                # epDeriv =(

                #     (self.primaryProblem.MfI.T * PTv)
                #     )
            else:
                epDeriv = (
                    self._ProjPrimary(prob, 'F', 'E') *
                    (
                        self.primaryProblem.MfI *
                        (
                            (self.primaryProblem.MfRhoDeriv(f[:, 'j']) * v) +
                            (
                                self.primaryProblem.MfRho *
                                self._primaryFieldsDeriv(prob, v, f=f)
                            )
                        )
                    )
                )

        return Utils.mkvc(epDeriv)

    def bPrimary(self, prob, f=None):
        if f is None:
            f = self._primaryFields(prob)

        if self.primaryProblem._formulation == 'EB':
            bp = self._ProjPrimary(prob, 'F', 'F') * f[:, 'b']
        elif self.primaryProblem._formulation == 'HJ':
            bp = (
                self._ProjPrimary(prob, 'E', 'F') *
                (
                    self.primaryProblem.MeI *
                    (
                        self.primaryProblem.MeMu * f[:, 'h']
                    )
                )
            )

        return Utils.mkvc(bp)

    def s_e(self, prob, f=None):
        sigmaPrimary = self.map2meshSecondary * prob.model

        return Utils.mkvc(
            (prob.MeSigma - prob.mesh.getEdgeInnerProduct(sigmaPrimary)) *
            self.ePrimary(prob, f=f)
        )

    def s_eDeriv(self, prob, v, adjoint=False):

        sigmaPrimary = self.map2meshSecondary * prob.model
        sigmaPrimaryDeriv = self.map2meshSecondary.deriv(
                prob.model)

        f = self._primaryFields(prob)
        ePrimary = self.ePrimary(prob, f=f)

        if adjoint is True:
            return (
                prob.MeSigmaDeriv(ePrimary).T * v -
                (
                    sigmaPrimaryDeriv.T * prob.mesh.getEdgeInnerProductDeriv(
                        sigmaPrimary
                    )(ePrimary).T * v
                ) +
                self.ePrimaryDeriv(prob, (
                    prob.MeSigma - prob.mesh.getEdgeInnerProduct(
                        sigmaPrimary)).T * v, adjoint=adjoint, f=f)
            )

        return(
            prob.MeSigmaDeriv(ePrimary) * v -
            prob.mesh.getEdgeInnerProductDeriv(sigmaPrimary)(ePrimary) *
            (sigmaPrimaryDeriv * v) +
            (prob.MeSigma - prob.mesh.getEdgeInnerProduct(sigmaPrimary)) *
            self.ePrimaryDeriv(prob, v, adjoint=adjoint, f=f)
        )



