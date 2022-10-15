import numpy as np
from scipy.constants import mu_0
import warnings
from scipy.special import roots_legendre

from geoana.em.static import MagneticDipoleWholeSpace, CircularLoopWholeSpace

from ...utils import (
    mkvc,
    Zero,
    validate_float,
    validate_location_property,
    validate_ndarray_with_shape,
    validate_type,
    validate_direction,
    validate_integer,
)
from ...utils.code_utils import deprecate_property

from ..utils import omega
from ..utils import segmented_line_current_source_term, line_through_faces
from ..base import BaseEMSrc


class BaseFDEMSrc(BaseEMSrc):
    """Base FDEM source class

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.frequency_domain.receivers.BaseRx
        A list of FDEM receivers
    frequency : float
        Source frequency
    location : (dim) numpy.ndarray, default: ``None``
        Source location.
    """

    # frequency = properties.Float("frequency of the source", min=0, required=True)

    _ePrimary = None
    _bPrimary = None
    _hPrimary = None
    _jPrimary = None

    def __init__(self, receiver_list, frequency, location=None, **kwargs):

        super().__init__(receiver_list=receiver_list, location=location, **kwargs)
        self.frequency = frequency

    @property
    def frequency(self):
        """Source frequency

        Returns
        -------
        float
            Source frequency
        """
        return self._frequency

    @frequency.setter
    def frequency(self, freq):
        freq = validate_float("frequency", freq, min_val=0.0)
        self._frequency = freq

    def bPrimary(self, simulation):
        """Compute primary magnetic flux density

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            Primary magnetic flux density
        """
        if self._bPrimary is None:
            return Zero()
        return self._bPrimary

    def bPrimaryDeriv(self, simulation, v, adjoint=False):
        """Compute derivative of primary magnetic flux density times a vector

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation
        v : numpy.ndarray
            A vector
        adjoint : bool
            If ``True``, return the adjoint

        Returns
        -------
        numpy.ndarray
            Derivative of primary magnetic flux density times a vector
        """
        return Zero()

    def hPrimary(self, simulation):
        """Compute primary magnetic field

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            Primary magnetic field
        """
        if self._hPrimary is None:
            return Zero()
        return self._hPrimary

    def hPrimaryDeriv(self, simulation, v, adjoint=False):
        """Compute derivative of primary magnetic field times a vector

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation
        v : numpy.ndarray
            A vector
        adjoint : bool
            If ``True``, return the adjoint

        Returns
        -------
        numpy.ndarray
            Derivative of primary magnetic field times a vector
        """
        return Zero()

    def ePrimary(self, simulation):
        """Compute primary electric field

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            Primary electric field
        """
        if self._ePrimary is None:
            return Zero()
        return self._ePrimary

    def ePrimaryDeriv(self, simulation, v, adjoint=False):
        """Compute derivative of primary electric field times a vector

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation
        v : numpy.ndarray
            A vector
        adjoint : bool
            If ``True``, return the adjoint

        Returns
        -------
        numpy.ndarray
            Derivative of primary eletric field times a vector
        """
        return Zero()

    def jPrimary(self, simulation):
        """Compute primary current density

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            Primary current density
        """
        if self._jPrimary is None:
            return Zero()
        return self._jPrimary

    def jPrimaryDeriv(self, simulation, v, adjoint=False):
        """Compute derivative of primary current density times a vector

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation
        v : numpy.ndarray
            A vector
        adjoint : bool
            If ``True``, return the adjoint

        Returns
        -------
        numpy.ndarray
            Derivative of primary current density times a vector
        """
        return Zero()


class RawVec_e(BaseFDEMSrc):
    """User-provided electric source term (s_e) class.

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.frequency_domain.receivers.BaseRx
        A list of FDEM receivers
    frequency : float
        Source frequency
    s_e: numpy.ndarray
        Electric source term
    integrate : bool, default: ``False``
        If ``True``, integrate the source term; i.e. multiply by Me matrix
    """

    def __init__(self, receiver_list, frequency, s_e, **kwargs):
        self._s_e = np.asarray(s_e, dtype=complex)

        super().__init__(receiver_list, frequency=frequency, **kwargs)

    def s_e(self, simulation):
        """Electric source term (s_e)

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            electric source term on mesh.
        """
        if simulation._formulation == "EB" and self.integrate is True:
            return simulation.Me * self._s_e
        return self._s_e


class RawVec_m(BaseFDEMSrc):
    """User-provided magnetic source term (s_m) class.

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.frequency_domain.receivers.BaseRx
        A list of FDEM receivers
    frequency : float
        Source frequency
    s_m: numpy.ndarray
        Magnetic source term
    integrate : bool, default: ``False``
        If ``True``, integrate the source term; i.e. multiply by Me matrix
    """

    def __init__(self, receiver_list, frequency, s_m, **kwargs):
        self._s_m = np.asarray(s_m, dtype=complex)
        super().__init__(receiver_list=receiver_list, frequency=frequency, **kwargs)

    def s_m(self, simulation):
        """Magnetic source term (s_m)

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            magnetic source term on mesh.
        """
        if simulation._formulation == "HJ" and self.integrate is True:
            return simulation.Me * self._s_m
        return self._s_m


class RawVec(RawVec_e, RawVec_m):
    """User-provided electric (s_e) and magnetic (s_m) source terms.

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.frequency_domain.receivers.BaseRx
        A list of FDEM receivers
    frequency : float
        Source frequency
    s_m: numpy.ndarray
        Magnetic source term
    s_e: numpy.ndarray
        Electric source term
    integrate : bool, default: ``False``
        If ``True``, integrate the source terms; i.e. multiply by Me matrix
    """

    def __init__(self, receiver_list, frequency, s_m, s_e, **kwargs):
        super().__init__(
            receiver_list=receiver_list,
            frequency=frequency,
            s_m=s_m,
            s_e=s_e,
            **kwargs,
        )


class MagDipole(BaseFDEMSrc):
    r"""
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

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.frequency_domain.receivers.BaseRx
        A list of FDEM receivers
    frequency : float
        Source frequency
    location : (dim) numpy.ndarray, default: numpy.r_[0., 0., 0.]
        Source location.
    moment : float
        Magnetic dipole moment amplitude
    orientation : {'z', x', 'y'} or (dim) numpy.ndarray
        Orientation of the dipole.
    mu : float
        Background magnetic permeability
    """

    def __init__(
        self,
        receiver_list,
        frequency,
        location=None,
        moment=1.0,
        orientation="z",
        mu=mu_0,
        **kwargs,
    ):
        if location is None:
            location = np.r_[0.0, 0.0, 0.0]

        super().__init__(
            receiver_list=receiver_list,
            frequency=frequency,
            location=location,
            **kwargs,
        )

        self.moment = moment
        self.orientation = orientation
        self.mu = mu

    # moment = properties.Float("dipole moment of the transmitter", default=1.0, min=0.0)
    # mu = properties.Float("permeability of the background", default=mu_0, min=0.0)
    # orientation = properties.Vector3(
    #     "orientation of the source", default="Z", length=1.0, required=True
    # )
    # location = LocationVector(
    #     "location of the source", default=np.r_[0.0, 0.0, 0.0], shape=(3,)
    # )

    # def __init__(self, receiver_list=None, frequency=None, location=None, **kwargs):
    #     super(MagDipole, self).__init__(receiver_list, frequency=frequency, **kwargs)
    #     if location is not None:
    #         self.location = location

    @property
    def location(self):
        """Location of the dipole

        Returns
        -------
        (3) numpy.ndarray of float
            xyz dipole location
        """
        return self._location

    @location.setter
    def location(self, vec):
        self._location = validate_location_property("location", vec, 3)

    @property
    def moment(self):
        """Amplitude of the dipole moment of the magnetic dipole (:math:`A/m^2`)

        Returns
        -------
        float
            Amplitude of the dipole moment of the magnetic dipole (:math:`A/m^2`)
        """
        return self._moment

    @moment.setter
    def moment(self, value):
        self._moment = validate_float("moment", value, min_val=0.0)

    @property
    def orientation(self):
        """Orientation of the dipole as a normalized vector

        Returns
        -------
        (3) numpy.ndarray of float
            dipole orientation, normalized to unit magnitude
        """
        return self._orientation

    @orientation.setter
    def orientation(self, var):
        self._orientation = validate_direction("orientation", var, dim=3)

    @property
    def mu(self):
        """Magnetic permeability in H/m

        Returns
        -------
        float
            Magnetic permeability in H/m
        """
        return self._mu

    @mu.setter
    def mu(self, value):
        value = validate_float("mu", value, min_val=mu_0)
        self._mu = value

    @property
    def _dipole(self):
        if getattr(self, "__dipole", None) is None:
            self.__dipole = MagneticDipoleWholeSpace(
                mu=self.mu,
                orientation=self.orientation,
                location=self.location,
                moment=self.moment,
            )
        return self.__dipole

    def _srcFct(self, obsLoc, coordinates="cartesian"):
        return self._dipole.vector_potential(obsLoc, coordinates=coordinates)

    def bPrimary(self, simulation):
        """Compute primary magnetic flux density.

        Note that we compute analytic vector potential and take numerical
        curl do it is divergence free on the mesh.

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            Primary magnetic flux density
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
        """Compute primary magnetic field.

        Note that we compute analytic vector potential and take numerical
        curl so that B is divergence-free.

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            A SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            Primary magnetic field
        """
        if simulation._formulation == "1D":
            if getattr(self, "_1d_h", None) is None:
                dipole = self._dipole
                out = []
                for rx in self.receiver_list:
                    if rx.use_source_receiver_offset:
                        locs = rx.locations + self.location
                    else:
                        locs = rx.locations
                    h_rx = dipole.magnetic_field(locs)
                    out.append(h_rx @ rx.orientation)
                self._1d_h = out
            return self._1d_h
        b = self.bPrimary(simulation)
        return 1.0 / self.mu * b

    def s_m(self, simulation):
        """Magnetic source term (s_m)

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            Magnetic source term on mesh.
        """

        b_p = self.bPrimary(simulation)
        if simulation._formulation == "HJ":
            b_p = simulation.Me * b_p
        return -1j * omega(self.frequency) * b_p

    def s_e(self, simulation):
        """Electric source term (s_e)

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            Electric source term on mesh.
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

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.frequency_domain.receivers.BaseRx
        A list of FDEM receivers
    frequency : float
        Source frequency
    location : (dim) numpy.ndarray, default: np.r_[0., 0., 0.]
        Source location.
    moment : float
        Magnetic dipole moment amplitude
    orientation : {'z', x', 'y'} or (dim) numpy.ndarray
        Orientation of the dipole.
    mu : float
        Background magnetic permeability
    """

    def __init__(self, receiver_list, frequency, location=None, **kwargs):
        super().__init__(
            receiver_list=receiver_list,
            frequency=frequency,
            location=location,
            **kwargs,
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

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.frequency_domain.receivers.BaseRx
        A list of FDEM receivers
    frequency : float
        Source frequency
    location : (dim) np.ndarray, default: np.r_[0., 0., 0.]
        Source location.
    moment : float
        Magnetic dipole moment amplitude
    orientation : {'z', x', 'y'} or (dim) numpy.ndarray
        Orientation of the dipole.
    mu : float
        Background magnetic permeability
    orientation : str, default: 'z'
        Loop orientation. One of ('x', 'y', 'z')
    radius : float, default: 1.0
        Loop radius
    current : float, default: 1.0
        Source current
    mu : float
        Background magnetic permeability
    """

    def __init__(
        self,
        receiver_list,
        frequency,
        location=None,
        orientation="z",
        radius=1.0,
        current=1.0,
        n_turns=1,
        mu=mu_0,
        **kwargs,
    ):
        kwargs.pop("moment", None)
        N = kwargs.pop("N", None)
        if N is not None:
            self.N = N
        else:
            self.n_turns = n_turns
        super().__init__(
            receiver_list=receiver_list,
            frequency=frequency,
            location=location,
            **kwargs,
        )

        self.orientation = orientation
        self.mu = mu
        self.radius = radius
        self.current = current

    # n_turns = properties.Integer("number of turns in the loop", default=1)

    @property
    def radius(self):
        """Loop radius

        Returns
        -------
        float
            Loop radius
        """
        return self._radius

    @radius.setter
    def radius(self, rad):
        rad = validate_float("radius", rad, min_val=0, inclusive_min=False)
        self._radius = rad

    # current = properties.Float("current in the loop", default=1.0)

    @property
    def current(self):
        """Source current

        Returns
        -------
        float
            Source current
        """
        return self._current

    @current.setter
    def current(self, I):
        I = validate_float("current", I)
        if np.abs(I) == 0.0:
            raise ValueError("current must be non-zero.")
        self._current = I

    # def __init__(self, receiver_list=None, frequency=None, location=None, **kwargs):
    #     super(CircularLoop, self).__init__(receiver_list, frequency, location, **kwargs)

    @property
    def moment(self):
        """Dipole moment of the loop.

        The dipole moment is given by :math:`I\\pi r^2`

        Returns
        -------
        float
            Dipole moment of the loop
        """
        return np.pi * self.radius ** 2 * np.abs(self.current) * self.n_turns

    @moment.setter
    def moment(self, value):
        warnings.warn(
            "Moment is not set as a property. I is the product"
            "of the loop radius and transmitter current"
        )
        pass

    @property
    def n_turns(self):
        """Number of turns in the loop.

        Returns
        -------
        int
        """
        return self._n_turns

    @n_turns.setter
    def n_turns(self, value):
        self._n_turns = validate_integer("n_turns", value, min_val=1)


    def _srcFct(self, obsLoc, coordinates="cartesian"):
        if getattr(self, "_loop", None) is None:
            self._loop = CircularLoopWholeSpace(
                mu=self.mu,
                location=self.location,
                orientation=self.orientation,
                radius=self.radius,
                current=self.current,
            )
        return self.n_turns * self._loop.vector_potential(obsLoc, coordinates)

    N = deprecate_property(n_turns, "N", "n_turns", removal_version="0.19.0")

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
            **kwargs,
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

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.frequency_domain.receiver.BaseRx
        List of FDEM receivers
    frequency : float
        Frequency
    primarySimulation : BaseFDEMSimulation
        Base simulation
    primarySurvey : BaseEMSimulation
        Primary FDEM survey
    map2meshSecondary : maps.BaseMap
        Mapping current model to act as primary model on the secondary mesh
    """

    def __init__(
        self,
        receiver_list=None,
        frequency=None,
        primarySimulation=None,
        primarySurvey=None,
        map2meshSecondary=None,
        **kwargs,
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
        # Ainv = self.primarySimulation.solver(A, **self.primarySimulation.solver_opts) # create the concept of Ainv (actually a solve)

        if f is None:
            f = self._primaryFields(simulation.sigma, f=f)

        freq = self.frequency

        A = self.primarySimulation.getA(freq)
        src = self.primarySurvey.source_list[0]
        u_src = mkvc(f[src, self.primarySimulation._solutionType])

        if adjoint is True:
            Jtv = np.zeros(simulation.sigmaMap.nP, dtype=complex)
            ATinv = self.primarySimulation.solver(
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
        Ainv = self.primarySimulation.solver(A, **self.primarySimulation.solver_opts)

        # for src in self.survey.get_sources_by_frequency(freq):
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
        """Electric source term (s_m)

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            SimPEG FDEM simulation
        f : SimPEG.electromagnetics.frequency_domain.field.FieldsFDEM
            A SimPEG FDEM fields object

        Returns
        -------
        numpy.ndarray
            Electric source term on mesh.
        """
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


class LineCurrent(BaseFDEMSrc):
    """Line current source.

    Given the wire path provided by the (n_loc, 3) locations array,
    the cells intersected by the wire path are identified and integrated
    source terms are computed.

    Parameters
    ----------
    receiver_list : list of SimPEG.electromagnetics.frequency_domain.receivers.BaseRx
        List of FDEM receivers
    frequency : float
        Source frequency
    locations : (n,3) numpy.ndarray
        Array defining the node locations for the wire path. For inductive sources,
        you must close the loop.
    """

    def __init__(
        self,
        receiver_list=None,
        frequency=None,
        location=None,
        current=1.0,
        mu=mu_0,
        **kwargs,
    ):

        BaseFDEMSrc.__init__(
            self,
            receiver_list=receiver_list,
            frequency=frequency,
            location=location,
            **kwargs,
        )

        self.current = current
        self.mu = mu

    # location = properties.Array("location of the source", shape=("*", 3))

    @property
    def location(self):
        """Line current nodes locations

        Returns
        -------
        (n, 3) np.ndarray
            Line current node locations.
        """
        return self._location

    @location.setter
    def location(self, loc):
        loc = validate_ndarray_with_shape("location", loc, shape=("*", 3))
        self._location = loc

    # current = properties.Float("current in the line", default=1.0)

    @property
    def current(self):
        """Source current

        Returns
        -------
        float
            Source current
        """
        return self._current

    @current.setter
    def current(self, I):
        I = validate_float("current", I)
        if np.abs(I) == 0.0:
            raise ValueError("current must be non-zero.")
        self._current = I

    def Mejs(self, simulation):
        """Integrated electrical source term on edges

        Parameters
        ----------
        simulation : SimPEG.electromagnetics.frequency_domain.simulation.BaseFDEMSimulation
            Base FDEM simulation

        Returns
        -------
        numpy.ndarray of length (mesh.nE)
            Contains the source term for all x, y, and z edges of the mesh.
        """
        if getattr(self, "_Mejs", None) is None:
            mesh = simulation.mesh
            locs = self.location
            self._Mejs = self.current * segmented_line_current_source_term(mesh, locs)
        return self.current * self._Mejs

    def Mfjs(self, simulation):
        """Integrated electrical source term on faces

        Parameters
        ----------
        simulation : SimPEG.electromagnetics.frequency_domain.simulation.BaseFDEMSimulation
            Base FDEM simulation

        Returns
        -------
        numpy.ndarray of length (mesh.nF)
            Contains the source term for all x, y, and z faces of the mesh.
        """
        if getattr(self, "_Mfjs", None) is None:
            self._Mfjs = line_through_faces(
                simulation.mesh, self.location, normalize_by_area=True
            )
        return self.current * self._Mfjs

    def getRHSdc(self, simulation):
        """Right-hand side for galvanic source term

        Parameters
        ----------
        simulation : SimPEG.electromagnetics.frequency_domain.simulation.BaseFDEMSimulation
            Base FDEM simulation

        Returns
        -------
        numpy.ndarray
            Right-hand side of galvanic source term. On edges for 'EB' formulation,
            and on faces for 'HJ' formulation.
        """
        if simulation._formulation == "EB":
            Grad = simulation.mesh.nodalGrad
            return Grad.T * self.Mejs(simulation)
        elif simulation._formulation == "HJ":
            Div = sdiag(simulation.mesh.vol) * simulation.mesh.faceDiv
            return Div * self.Mfjs(simulation)

    def s_m(self, simulation):
        """Magnetic source term (s_m)

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            Magnetic source term on mesh.
        """
        return Zero()

    def s_e(self, simulation):
        """Electric source term (s_m)

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            Electric source term on mesh.
        """

        if simulation._formulation == "EB":
            return self.Mejs(simulation)
        elif simulation._formulation == "HJ":
            return self.Mfjs(simulation)


class LineCurrent1D(LineCurrent):

    # n_points_per_path = properties.Integer(
    #     "number of quadrature points per linear wire path", default=3
    # )

    def __init__(
        self, receiver_list, frequency, locations, n_points_per_path=3, **kwargs
    ):
        super().__init__(
            receiver_list, frequency=frequency, location=locations, **kwargs
        )
        self.n_points_per_path = n_points_per_path
        # calculate lateral dipole locations
        x, w = roots_legendre(self.n_points_per_path)
        xy_src_path = self.location[:, :2]
        n_path = len(xy_src_path) - 1
        xyks = []
        thetas = []
        weights = []
        for i_path in range(n_path):
            dx = xy_src_path[i_path + 1, 0] - xy_src_path[i_path, 0]
            dy = xy_src_path[i_path + 1, 1] - xy_src_path[i_path, 1]
            dl = np.sqrt(dx ** 2 + dy ** 2)
            theta = np.arctan2(dy, dx)
            lk = np.c_[(x + 1) * dl / 2, np.zeros(self.n_points_per_path)]

            R = np.array([[dx, -dy], [dy, dx]]) / dl
            xyk = lk.dot(R.T) + xy_src_path[i_path, :]

            xyks.append(xyk)
            thetas.append(theta * np.ones(xyk.shape[0]))
            weights.append(w * dl / 2)
        # store these for future evalution of integrals
        self._xyks = np.vstack(xyks)
        self._weights = np.hstack(weights)
        self._thetas = np.hstack(thetas)

    @property
    def n_quad_points(self):
        self._n_quad_points = len(self._weights)
        return self._n_quad_points

    @property
    def n_points_per_path(self):
        """The number of integration points for each line segment.

        Returns
        -------
        int
        """
        return self._n_points_per_path

    @n_points_per_path.setter
    def n_points_per_path(self, val):
        self._n_points_per_path = validate_type("n_points_per_path", val, int)

    def hPrimary(self, simulation):
        raise NotImplementedError(
            "Primary field calculation for LineCurrent1D has not been implemented"
        )
