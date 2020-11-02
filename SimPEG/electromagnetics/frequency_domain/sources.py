import properties
import numpy as np
from scipy.constants import mu_0
import warnings

from geoana.em.static import MagneticDipoleWholeSpace, CircularLoopWholeSpace

from ...props import LocationVector
from ...utils import mkvc, Zero
from ...utils.code_utils import deprecate_property

from ..utils import omega
from ..base import BaseEMSrc


class BaseFDEMSrc(BaseEMSrc):
    """
    Base source class for FDEM Survey
    """

    frequency = properties.Float("frequency of the source", min=0, required=True)

    _ePrimary = None
    _bPrimary = None
    _hPrimary = None
    _jPrimary = None

    def __init__(self, receiver_list=None, frequency=None, **kwargs):
        super(BaseFDEMSrc, self).__init__(receiver_list=receiver_list, **kwargs)
        if frequency is not None:
            self.frequency = frequency

    def bPrimary(self, simulation):
        """
        Primary magnetic flux density

        :param BaseFDEMSimulation simulation: FDEM simulation
        :rtype: numpy.ndarray
        :return: primary magnetic flux density
        """
        if self._bPrimary is None:
            return Zero()
        return self._bPrimary

    def bPrimaryDeriv(self, simulation, v, adjoint=False):
        """
        Derivative of the primary magnetic flux density

        :param BaseFDEMSimulation simulation: FDEM simulation
        :param numpy.ndarray v: vector
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: primary magnetic flux density
        """
        return Zero()

    def hPrimary(self, simulation):
        """
        Primary magnetic field

        :param BaseFDEMSimulation simulation: FDEM simulation
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        if self._hPrimary is None:
            return Zero()
        return self._hPrimary

    def hPrimaryDeriv(self, simulation, v, adjoint=False):
        """
        Derivative of the primary magnetic field

        :param BaseFDEMSimulation simulation: FDEM simulation
        :param numpy.ndarray v: vector
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: primary magnetic flux density
        """
        return Zero()

    def ePrimary(self, simulation):
        """
        Primary electric field

        :param BaseFDEMSimulation simulation: FDEM simulation
        :rtype: numpy.ndarray
        :return: primary electric field
        """
        if self._ePrimary is None:
            return Zero()
        return self._ePrimary

    def ePrimaryDeriv(self, simulation, v, adjoint=False):
        """
        Derivative of the primary electric field

        :param BaseFDEMSimulation simulation: FDEM simulation
        :param numpy.ndarray v: vector
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: primary magnetic flux density
        """
        return Zero()

    def jPrimary(self, simulation):
        """
        Primary current density

        :param BaseFDEMSimulation simulation: FDEM simulation
        :rtype: numpy.ndarray
        :return: primary current density
        """
        if self._jPrimary is None:
            return Zero()
        return self._jPrimary

    def jPrimaryDeriv(self, simulation, v, adjoint=False):
        """
        Derivative of the primary current density

        :param BaseFDEMSimulation simulation: FDEM simulation
        :param numpy.ndarray v: vector
        :param bool adjoint: adjoint?
        :rtype: numpy.ndarray
        :return: primary magnetic flux density
        """
        return Zero()

    freq = deprecate_property(
        frequency, "freq", new_name="frequency", removal_version="0.15.0"
    )


class RawVec_e(BaseFDEMSrc):
    """
    RawVec electric source. It is defined by the user provided vector s_e

    :param list receiver_list: receiver list
    :param float freq: frequency
    :param numpy.ndarray s_e: electric source term
    :param bool integrate: Integrate the source term (multiply by Me) [False]
    """

    def __init__(self, receiver_list=None, frequency=None, s_e=None, **kwargs):
        self._s_e = np.array(s_e, dtype=complex)

        super(RawVec_e, self).__init__(receiver_list, frequency=frequency, **kwargs)

    def s_e(self, simulation):
        """
        Electric source term

        :param BaseFDEMSimulation simulation: FDEM simulation
        :rtype: numpy.ndarray
        :return: electric source term on mesh
        """
        if simulation._formulation == "EB" and self.integrate is True:
            return simulation.Me * self._s_e
        return self._s_e


class RawVec_m(BaseFDEMSrc):
    """
    RawVec magnetic source. It is defined by the user provided vector s_m

    :param float freq: frequency
    :param receiver_list: receiver list
    :param numpy.ndarray s_m: magnetic source term
    :param bool integrate: Integrate the source term (multiply by Me) [False]
    """

    def __init__(self, receiver_list=None, frequency=None, s_m=None, **kwargs):
        self._s_m = np.array(s_m, dtype=complex)
        super(RawVec_m, self).__init__(
            receiver_list=receiver_list, frequency=frequency, ** kwargs
        )

    def s_m(self, simulation):
        """
        Magnetic source term

        :param BaseFDEMSimulation simulation: FDEM simulation
        :rtype: numpy.ndarray
        :return: magnetic source term on mesh
        """
        if simulation._formulation == "HJ" and self.integrate is True:
            return simulation.Me * self._s_m
        return self._s_m


class RawVec(BaseFDEMSrc):
    """
    RawVec source. It is defined by the user provided vectors s_m, s_e

    :param receiver_list: receiver list
    :param float freq: frequency
    :param numpy.ndarray s_m: magnetic source term
    :param numpy.ndarray s_e: electric source term
    :param bool integrate: Integrate the source term (multiply by Me) [False]
    """

    def __init__(
        self, receiver_list=None, frequency=None, s_m=None, s_e=None, **kwargs
    ):
        self._s_m = np.array(s_m, dtype=complex)
        self._s_e = np.array(s_e, dtype=complex)
        super(RawVec, self).__init__(
            receiver_list=receiver_list, frequency=frequency, **kwargs
        )

    def s_m(self, simulation):
        """
        Magnetic source term

        :param BaseFDEMSimulation simulation: FDEM simulation
        :rtype: numpy.ndarray
        :return: magnetic source term on mesh
        """
        if simulation._formulation == "HJ" and self.integrate is True:
            return simulation.Me * self._s_m
        return self._s_m

    def s_e(self, simulation):
        """
        Electric source term

        :param BaseFDEMSimulation simulation: FDEM simulation
        :rtype: numpy.ndarray
        :return: electric source term on mesh
        """
        if simulation._formulation == "EB" and self.integrate is True:
            return simulation.Me * self._s_e
        return self._s_e


class MagDipole(BaseFDEMSrc):
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

    and define a zero-frequency primary simulation, noting that the source is
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

    :param list receiver_list: receiver list
    :param float freq: frequency
    :param numpy.ndarray location: source location
        (ie: :code:`np.r_[xloc,yloc,zloc]`)
    :param string orientation: 'X', 'Y', 'Z'
    :param float moment: magnetic dipole moment
    :param float mu: background magnetic permeability

    """

    moment = properties.Float("dipole moment of the transmitter", default=1.0, min=0.0)
    mu = properties.Float("permeability of the background", default=mu_0, min=0.0)
    orientation = properties.Vector3(
        "orientation of the source", default="Z", length=1.0, required=True
    )
    location = LocationVector(
        "location of the source", default=np.r_[0.0, 0.0, 0.0], shape=(3,)
    )
    loc = deprecate_property(
        location, "loc", new_name="location", removal_version="0.15.0"
    )

    def __init__(self, receiver_list=None, frequency=None, location=None, **kwargs):
        super(MagDipole, self).__init__(receiver_list, frequency=frequency, **kwargs)
        if location is not None:
            self.location = location

    def _srcFct(self, obsLoc, coordinates="cartesian"):
        if getattr(self, "_dipole", None) is None:
            self._dipole = MagneticDipoleWholeSpace(
                mu=self.mu,
                orientation=self.orientation,
                location=self.location,
                moment=self.moment,
            )
        return self._dipole.vector_potential(obsLoc, coordinates=coordinates)

    def bPrimary(self, simulation):
        """
        The primary magnetic flux density from a magnetic vector potential

        :param BaseFDEMSimulation simulation: FDEM simulation
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        formulation = simulation._formulation
        coordinates = "cartesian"

        if formulation == "EB":
            gridX = simulation.mesh.gridEx
            gridY = simulation.mesh.gridEy
            gridZ = simulation.mesh.gridEz
            C = simulation.mesh.edgeCurl

        elif formulation == "HJ":
            gridX = simulation.mesh.gridFx
            gridY = simulation.mesh.gridFy
            gridZ = simulation.mesh.gridFz
            C = simulation.mesh.edgeCurl.T

        if simulation.mesh._meshType == "CYL":
            coordinates = "cylindrical"

            if simulation.mesh.isSymmetric is True:
                if not (np.linalg.norm(self.orientation - np.r_[0.0, 0.0, 1.0]) < 1e-6):
                    raise AssertionError(
                        "for cylindrical symmetry, the dipole must be oriented"
                        " in the Z direction"
                    )
                a = self._srcFct(gridY)[:, 1]

                return C * a

        ax = self._srcFct(gridX, coordinates)[:, 0]
        ay = self._srcFct(gridY, coordinates)[:, 1]
        az = self._srcFct(gridZ, coordinates)[:, 2]
        a = np.concatenate((ax, ay, az))

        return C * a

    def hPrimary(self, simulation):
        """
        The primary magnetic field from a magnetic vector potential

        :param BaseFDEMSimulation simulation: FDEM simulation
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """
        b = self.bPrimary(simulation)
        return 1.0 / self.mu * b

    def s_m(self, simulation):
        """
        The magnetic source term

        :param BaseFDEMSimulation simulation: FDEM simulation
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """

        b_p = self.bPrimary(simulation)
        if simulation._formulation == "HJ":
            b_p = simulation.Me * b_p
        return -1j * omega(self.frequency) * b_p

    def s_e(self, simulation):
        """
        The electric source term

        :param BaseFDEMSimulation simulation: FDEM simulation
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """

        if all(np.r_[self.mu] == np.r_[simulation.mu]):
            return Zero()
        else:
            formulation = simulation._formulation

            if formulation == "EB":
                mui_s = simulation.mui - 1.0 / self.mu
                MMui_s = simulation.mesh.getFaceInnerProduct(mui_s)
                C = simulation.mesh.edgeCurl
            elif formulation == "HJ":
                mu_s = simulation.mu - self.mu
                MMui_s = simulation.mesh.getEdgeInnerProduct(mu_s, invMat=True)
                C = simulation.mesh.edgeCurl.T

            return -C.T * (MMui_s * self.bPrimary(simulation))

    def s_eDeriv(self, simulation, v, adjoint=False):
        if not hasattr(simulation, "muMap") or not hasattr(simulation, "muiMap"):
            return Zero()
        else:
            formulation = simulation._formulation

            if formulation == "EB":
                mui_s = simulation.mui - 1.0 / self.mu
                MMui_sDeriv = (
                    simulation.mesh.getFaceInnerProductDeriv(mui_s)(
                        self.bPrimary(simulation)
                    )
                    * simulation.muiDeriv
                )
                C = simulation.mesh.edgeCurl

                if adjoint:
                    return -MMui_sDeriv.T * (C * v)

                return -C.T * (MMui_sDeriv * v)

            elif formulation == "HJ":
                return Zero()
                # raise NotImplementedError
                mu_s = simulation.mu - self.mu
                MMui_s = simulation.mesh.getEdgeInnerProduct(mu_s, invMat=True)
                C = simulation.mesh.edgeCurl.T

                return -C.T * (MMui_s * self.bPrimary(simulation))


class MagDipole_Bfield(MagDipole):

    """
    Point magnetic dipole source calculated with the analytic solution for the
    fields from a magnetic dipole. No discrete curl is taken, so the magnetic
    flux density may not be strictly divergence free.

    This approach uses a primary-secondary in frequency in the same fashion as
    the MagDipole.

    :param list receiver_list: receiver list
    :param float freq: frequency
    :param numpy.ndarray loc: source location (ie:
                              :code:`np.r_[xloc,yloc,zloc]`)
    :param string orientation: 'X', 'Y', 'Z'
    :param float moment: magnetic dipole moment
    :param float mu: background magnetic permeability
    """

    def __init__(self, receiver_list=None, frequency=None, location=None, **kwargs):
        super(MagDipole_Bfield, self).__init__(
            receiver_list=receiver_list,
            frequency=frequency,
            location=location,
            **kwargs
        )

    def _srcFct(self, obsLoc, coordinates="cartesian"):
        if getattr(self, "_dipole", None) is None:
            self._dipole = MagneticDipoleWholeSpace(
                mu=self.mu,
                orientation=self.orientation,
                location=self.location,
                moment=self.moment,
            )
        return self._dipole.magnetic_flux_density(obsLoc, coordinates=coordinates)

    def bPrimary(self, simulation):
        """
        The primary magnetic flux density from the analytic solution for
        magnetic fields from a dipole

        :param BaseFDEMSimulation simulation: FDEM simulation
        :rtype: numpy.ndarray
        :return: primary magnetic field
        """

        formulation = simulation._formulation
        coordinates = "cartesian"

        if formulation == "EB":
            gridX = simulation.mesh.gridFx
            gridY = simulation.mesh.gridFy
            gridZ = simulation.mesh.gridFz

        elif formulation == "HJ":
            gridX = simulation.mesh.gridEx
            gridY = simulation.mesh.gridEy
            gridZ = simulation.mesh.gridEz

        if simulation.mesh._meshType == "CYL":
            coordinates = "cylindrical"
            if simulation.mesh.isSymmetric:
                bx = self._srcFct(gridX)[:, 0]
                bz = self._srcFct(gridZ)[:, 2]
                b = np.concatenate((bx, bz))
        else:
            bx = self._srcFct(gridX, coordinates=coordinates)[:, 0]
            by = self._srcFct(gridY, coordinates=coordinates)[:, 1]
            bz = self._srcFct(gridZ, coordinates=coordinates)[:, 2]
            b = np.concatenate((bx, by, bz))

        return mkvc(b)


class CircularLoop(MagDipole):
    """
    Circular loop magnetic source calculated by taking the curl of a magnetic
    vector potential. By taking the discrete curl, we ensure that the magnetic
    flux density is divergence free (no magnetic monopoles!).

    This approach uses a primary-secondary in frequency in the same fashion as
    the MagDipole.

    :param list receiver_list: receiver list
    :param float freq: frequency
    :param numpy.ndarray loc: source location
        (ie: :code:`np.r_[xloc,yloc,zloc]`)
    :param string orientation: 'X', 'Y', 'Z'
    :param float moment: magnetic dipole moment
    :param float mu: background magnetic permeability
    """

    radius = properties.Float("radius of the loop", default=1.0, min=0.0)

    current = properties.Float("current in the loop", default=1.0)

    def __init__(self, receiver_list=None, frequency=None, location=None, **kwargs):
        super(CircularLoop, self).__init__(receiver_list, frequency, location, **kwargs)

    @property
    def moment(self):
        return np.pi * self.radius ** 2 * self.current

    def _srcFct(self, obsLoc, coordinates="cartesian"):
        if getattr(self, "_loop", None) is None:
            self._loop = CircularLoopWholeSpace(
                mu=self.mu,
                location=self.location,
                orientation=self.orientation,
                radius=self.radius,
                current=self.current,
            )
        return self._loop.vector_potential(obsLoc, coordinates)


class PrimSecSigma(BaseFDEMSrc):
    def __init__(
        self, receiver_list=None, frequency=None, sigBack=None, ePrimary=None, **kwargs
    ):
        self.sigBack = sigBack

        BaseFDEMSrc.__init__(
            self,
            receiver_list=receiver_list,
            frequency=frequency,
            _ePrimary=ePrimary,
            **kwargs
        )

    def s_e(self, simulation):
        return (
            simulation.MeSigma - simulation.mesh.getEdgeInnerProduct(self.sigBack)
        ) * self.ePrimary(simulation)

    def s_eDeriv(self, simulation, v, adjoint=False):
        if adjoint:
            return simulation.MeSigmaDeriv(self.ePrimary(simulation), v, adjoint)
        return simulation.MeSigmaDeriv(self.ePrimary(simulation), v, adjoint)


class PrimSecMappedSigma(BaseFDEMSrc):

    """
    Primary-Secondary Source in which a mapping is provided to put the current
    model onto the primary mesh. This is solved on every model update.
    There are a lot of layers to the derivatives here!

    **Required**
    :param list receiver_list: Receiver List
    :param float freq: frequency
    :param BaseFDEMSimulation primarySimulation: FDEM psimulation
    :param SurveyFDEM primarySurvey: FDEM primary survey

    **Optional**
    :param Mapping map2meshSecondary: mapping current model to act as primary
    model on the secondary mesh
    """

    def __init__(
        self,
        receiver_list=None,
        frequency=None,
        primarySimulation=None,
        primarySurvey=None,
        map2meshSecondary=None,
        **kwargs
    ):

        self.primarySimulation = primarySimulation
        self.primarySurvey = primarySurvey

        if getattr(self.primarySimulation, "survey", None) is None:
            self.primarySimulation.survey = self.primarySurvey

        self.map2meshSecondary = map2meshSecondary

        BaseFDEMSrc.__init__(
            self, receiver_list=receiver_list, frequency=frequency, **kwargs
        )

    def _ProjPrimary(self, simulation, locType, locTypeTo):
        # TODO: if meshes have not changed, store the projection
        # if getattr(self, '__ProjPrimary', None) is None:

        # TODO: implement for HJ formulation
        if simulation._formulation == "EB":
            pass
        else:
            raise NotImplementedError(
                "PrimSecMappedSigma Source has not been implemented for {} "
                "formulation".format(simulation._formulation)
            )

        # TODO: only set up for tensot meshes (Tree meshes should be easy/done)
        # but have not been tried or tested.
        assert simulation.mesh._meshType in [
            "TENSOR"
        ], "PrimSecMappedSigma source has not been implemented for {}".format(
            simulation.mesh._meshType
        )

        # if EB formulation, interpolate E, elif HJ interpolate J
        # if self.primarySimulation._formulation == 'EB':
        #     locType = 'E'
        # elif self.primarySimulation._formulation == 'HJ':
        #     locType = 'F'

        # get interpolation mat from primary mesh to secondary mesh
        if self.primarySimulation.mesh._meshType == "CYL":
            return self.primarySimulation.mesh.getInterpolationMatCartMesh(
                simulation.mesh, locType=locType, locTypeTo=locTypeTo
            )
        return self.primarySimulation.mesh.getInterploationMat(
            simulation.mesh, locType=locType, locTypeTo=locTypeTo
        )

        # return self.__ProjPrimary

    def _primaryFields(self, simulation, fieldType=None, f=None):
        # TODO: cache and check if simulation.curModel has changed

        if f is None:
            f = self.primarySimulation.fields(simulation.model)

        if fieldType is not None:
            return f[:, fieldType]
        return f

    def _primaryFieldsDeriv(self, simulation, v, adjoint=False, f=None):
        # TODO: this should not be hard-coded for j
        # jp = self._primaryFields(simulation)[:,'j']

        # TODO: pull apart Jvec so that don't have to copy paste this code in
        # A = self.primarySimulation.getA(self.frequency)
        # Ainv = self.primarySimulation.Solver(A, **self.primarySimulation.solver_opts) # create the concept of Ainv (actually a solve)

        if f is None:
            f = self._primaryFields(simulation.sigma, f=f)

        freq = self.frequency

        A = self.primarySimulation.getA(freq)
        src = self.primarySurvey.source_list[0]
        u_src = mkvc(f[src, self.primarySimulation._solutionType])

        if adjoint is True:
            Jtv = np.zeros(simulation.sigmaMap.nP, dtype=complex)
            ATinv = self.primarySimulation.Solver(
                A.T, **self.primarySimulation.solver_opts
            )
            df_duTFun = getattr(
                f,
                "_{0}Deriv".format(
                    "e" if self.primarySimulation._formulation == "EB" else "j"
                ),
                None,
            )
            df_duT, df_dmT = df_duTFun(src, None, v, adjoint=True)

            ATinvdf_duT = ATinv * df_duT

            dA_dmT = self.primarySimulation.getADeriv(
                freq, u_src, ATinvdf_duT, adjoint=True
            )
            dRHS_dmT = self.primarySimulation.getRHSDeriv(
                freq, src, ATinvdf_duT, adjoint=True
            )

            du_dmT = -dA_dmT + dRHS_dmT

            Jtv += df_dmT + du_dmT

            ATinv.clean()

            return mkvc(Jtv)

        # create the concept of Ainv (actually a solve)
        Ainv = self.primarySimulation.Solver(A, **self.primarySimulation.solver_opts)

        # for src in self.survey.getSrcByFreq(freq):
        dA_dm_v = self.primarySimulation.getADeriv(freq, u_src, v)
        dRHS_dm_v = self.primarySimulation.getRHSDeriv(freq, src, v)
        du_dm_v = Ainv * (-dA_dm_v + dRHS_dm_v)

        # if self.primarySimulation._formulation == 'EB':
        df_dmFun = getattr(
            f,
            "_{0}Deriv".format(
                "e" if self.primarySimulation._formulation == "EB" else "j"
            ),
            None,
        )
        # elif self.primarySimulation._formulation == 'HJ':
        #     df_dmFun = getattr(f, '_{0}Deriv'.format('j'), None)
        df_dm_v = df_dmFun(src, du_dm_v, v, adjoint=False)
        # Jv[src, rx] = rx.evalDeriv(src, self.mesh, f, df_dm_v)
        Ainv.clean()

        return df_dm_v

        # return self.primarySimulation.Jvec(simulation.curModel, v, f=f)

    def ePrimary(self, simulation, f=None):
        if f is None:
            f = self._primaryFields(simulation)

        if self.primarySimulation._formulation == "EB":
            ep = self._ProjPrimary(simulation, "E", "E") * f[:, "e"]
        elif self.primarySimulation._formulation == "HJ":
            ep = self._ProjPrimary(simulation, "F", "E") * (
                self.primarySimulation.MfI * (self.primarySimulation.MfRho * f[:, "j"])
            )

        return mkvc(ep)

    def ePrimaryDeriv(self, simulation, v, adjoint=False, f=None):

        if f is None:
            f = self._primaryFields(simulation)

        # if adjoint is True:
        #     raise NotImplementedError
        if self.primarySimulation._formulation == "EB":
            if adjoint is True:
                epDeriv = self._primaryFieldsDeriv(
                    simulation,
                    (self._ProjPrimary(simulation, "E", "E").T * v),
                    f=f,
                    adjoint=adjoint,
                )
            else:
                epDeriv = self._ProjPrimary(
                    simulation, "E", "E"
                ) * self._primaryFieldsDeriv(simulation, v, f=f)
        elif self.primarySimulation._formulation == "HJ":
            if adjoint is True:
                PTv = self.primarySimulation.MfI.T * (
                    self._ProjPrimary(simulation, "F", "E").T * v
                )
                epDeriv = self.primarySimulation.MfRhoDeriv(
                    f[:, "j"], PTv, adjoint
                ) + self._primaryFieldsDeriv(
                    simulation,
                    self.primarySimulation.MfRho.T * PTv,
                    adjoint=adjoint,
                    f=f,
                )
            else:
                epDeriv = self._ProjPrimary(simulation, "F", "E") * (
                    self.primarySimulation.MfI
                    * (
                        self.primarySimulation.MfRhoDeriv(f[:, "j"], v, adjoint)
                        + (
                            self.primarySimulation.MfRho
                            * self._primaryFieldsDeriv(simulation, v, f=f)
                        )
                    )
                )

        return mkvc(epDeriv)

    def bPrimary(self, simulation, f=None):
        if f is None:
            f = self._primaryFields(simulation)

        if self.primarySimulation._formulation == "EB":
            bp = self._ProjPrimary(simulation, "F", "F") * f[:, "b"]
        elif self.primarySimulation._formulation == "HJ":
            bp = self._ProjPrimary(simulation, "E", "F") * (
                self.primarySimulation.MeI * (self.primarySimulation.MeMu * f[:, "h"])
            )

        return mkvc(bp)

    def s_e(self, simulation, f=None):
        sigmaPrimary = self.map2meshSecondary * simulation.model

        return mkvc(
            (simulation.MeSigma - simulation.mesh.getEdgeInnerProduct(sigmaPrimary))
            * self.ePrimary(simulation, f=f)
        )

    def s_eDeriv(self, simulation, v, adjoint=False):

        sigmaPrimary = self.map2meshSecondary * simulation.model
        sigmaPrimaryDeriv = self.map2meshSecondary.deriv(simulation.model)

        f = self._primaryFields(simulation)
        ePrimary = self.ePrimary(simulation, f=f)

        if adjoint is True:
            return (
                simulation.MeSigmaDeriv(ePrimary, v, adjoint)
                - (
                    sigmaPrimaryDeriv.T
                    * simulation.mesh.getEdgeInnerProductDeriv(sigmaPrimary)(ePrimary).T
                    * v
                )
                + self.ePrimaryDeriv(
                    simulation,
                    (
                        simulation.MeSigma
                        - simulation.mesh.getEdgeInnerProduct(sigmaPrimary)
                    ).T
                    * v,
                    adjoint=adjoint,
                    f=f,
                )
            )

        return (
            simulation.MeSigmaDeriv(ePrimary, v, adjoint)
            - simulation.mesh.getEdgeInnerProductDeriv(sigmaPrimary)(ePrimary)
            * (sigmaPrimaryDeriv * v)
            + (simulation.MeSigma - simulation.mesh.getEdgeInnerProduct(sigmaPrimary))
            * self.ePrimaryDeriv(simulation, v, adjoint=adjoint, f=f)
        )
