from scipy.constants import mu_0
import numpy as np
from scipy import sparse as sp
from scipy.special import roots_legendre
from empymod.transform import get_dlf_points

from ..simulation import BaseSimulation

# from .time_domain.sources import MagDipole as t_MagDipole, CircularLoop as t_CircularLoop
# from .frequency_domain.sources import MagDipole as f_MagDipole, CircularLoop as f_CircularLoop

from .. import utils
from ..utils import (
    validate_string,
    validate_ndarray_with_shape,
    validate_type,
    validate_integer,
)
from .. import props
from empymod.utils import check_hankel

__all__ = ["BaseEM1DSimulation"]

###############################################################################
#                                                                             #
#                             Base EM1D Simulation                            #
#                                                                             #
###############################################################################


class BaseEM1DSimulation(BaseSimulation):
    """
    Base simulation class for simulating the EM response over a 1D layered Earth
    for a single sounding. The simulation computes the fields by solving the
    Hankel transform solutions from Electromagnetic Theory for Geophysical
    Applications: Chapter 4 (Ward and Hohmann, 1988).
    """

    _formulation = "1D"
    _coefficients_set = False

    # Properties for electrical conductivity/resistivity
    sigma, sigmaMap, sigmaDeriv = props.Invertible(
        "Electrical conductivity at infinite frequency (S/m)"
    )
    rho, rhoMap, rhoDeriv = props.Invertible("Electrical resistivity (Ohm m)")
    props.Reciprocal(sigma, rho)

    eta = props.PhysicalProperty("Intrinsic chargeability (V/V), 0 <= eta < 1")
    tau = props.PhysicalProperty("Time constant for Cole-Cole model (s)")
    c = props.PhysicalProperty("Frequency Dependency for Cole-Cole model, 0 < c < 1")

    # Properties for magnetic susceptibility
    mu, muMap, muDeriv = props.Invertible(
        "Magnetic permeability at infinite frequency (SI)"
    )
    dchi = props.PhysicalProperty(
        "DC magnetic susceptibility for viscous remanent magnetization contribution (SI)"
    )
    tau1 = props.PhysicalProperty(
        "Lower bound for log-uniform distribution of time-relaxation constants for viscous remanent magnetization (s)"
    )
    tau2 = props.PhysicalProperty(
        "Upper bound for log-uniform distribution of time-relaxation constants for viscous remanent magnetization (s)"
    )

    # Additional properties
    h, hMap, hDeriv = props.Invertible("Receiver Height (m), h > 0")

    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "layer thicknesses (m)"
    )

    def __init__(
        self,
        sigma=None,
        sigmaMap=None,
        rho=None,
        rhoMap=None,
        thicknesses=None,
        thicknessesMap=None,
        mu=mu_0,
        muMap=None,
        h=None,
        hMap=None,
        eta=0.0,
        tau=1.0,
        c=0.5,
        dchi=0.0,
        tau1=1.0e-10,
        tau2=10.0,
        hankel_filter="key_101_2009",
        fix_Jmatrix=False,
        topo=None,
        n_points_per_path=3,
        **kwargs,
    ):
        super().__init__(mesh=None, **kwargs)
        self.sigma = sigma
        self.rho = rho
        self.sigmaMap = sigmaMap
        self.rhoMap = rhoMap
        self.mu = mu
        self.muMap = muMap
        self.h = h
        self.hMap = hMap
        if thicknesses is None:
            thicknesses = np.array([])
        self.thicknesses = thicknesses
        self.thicknessesMap = thicknessesMap
        self.eta = eta
        self.tau = tau
        self.c = c
        self.dchi = dchi
        self.tau1 = tau1
        self.tau2 = tau2
        self.n_points_per_path = n_points_per_path

        if topo is None:
            topo = np.r_[0.0, 0.0, 0.0]
        self.topo = topo

        for src in self.survey.source_list:
            if np.any(src.location[..., 2] < self.topo[2]):
                raise ValueError("Source must be located above the topography")
            for rx in src.receiver_list:
                if rx.use_source_receiver_offset:
                    if np.any(src.location[2] + rx.locations[:, 2] < self.topo[2]):
                        raise ValueError(
                            "Receiver must be located above the topography"
                        )
                else:
                    if np.any(rx.locations[:, 2] < self.topo[2]):
                        raise ValueError(
                            "Receiver must be located above the topography"
                        )

        self.hankel_filter = hankel_filter
        self.fix_Jmatrix = fix_Jmatrix
        # Check input arguments. If self.hankel_filter is not a valid filter,
        # it will set it to the default (key_201_2009).
        ht, htarg = check_hankel(
            "dlf", {"dlf": self.hankel_filter, "pts_per_dec": 0}, 1
        )

        self._fhtfilt = htarg["dlf"]  # Store filter
        # self.hankel_pts_per_dec = htarg["pts_per_dec"]  # Store pts_per_dec
        if self.verbose:
            print(">> Use " + self.hankel_filter + " filter for Hankel Transform")

    @property
    def hankel_filter(self):
        """The hankely filter to use.

        Returns
        -------
        str
        """
        return self._hankel_filter

    @hankel_filter.setter
    def hankel_filter(self, value):
        self._hankel_filter = validate_string("hankel_filter", value)

    _hankel_pts_per_dec = 0  # Default: Standard DLF

    @property
    def fix_Jmatrix(self):
        """Whether to fix the sensitivity matrix.

        Returns
        -------
        bool
        """
        return self._fix_Jmatrix

    @fix_Jmatrix.setter
    def fix_Jmatrix(self, value):
        self._fix_Jmatrix = validate_type("fix_Jmatrix", value, bool)

    @property
    def topo(self):
        """Topography.

        Returns
        -------
        numpy.ndarray of float
        """
        return self._topo

    @topo.setter
    def topo(self, value):
        self._topo = validate_ndarray_with_shape("topo", value, shape=("*",))

    @property
    def n_layer(self):
        """number of layers"""
        return int(self.thicknesses.size + 1)

    @property
    def n_filter(self):
        """Length of filter"""
        return self._fhtfilt.base.size

    @property
    def depth(self):
        """layer depths"""
        if self.thicknesses is not None:
            return np.r_[0.0, -np.cumsum(self.thicknesses)]
        return None

    @property
    def n_points_per_path(self):
        """The number of integration points for each segment of line current sources.

        Returns
        -------
        int
        """
        return self._n_points_per_path

    @n_points_per_path.setter
    def n_points_per_path(self, val):
        self._n_points_per_path = validate_integer("n_points_per_path", val, min_val=1)

    def compute_complex_sigma(self, frequencies):
        r"""
        Computes the complex conductivity matrix using Pelton's Cole-Cole model:

        .. math ::

            \sigma (\omega ) = \sigma \Bigg [
                1 - \eta \Bigg ( \frac{1}{1 + (1-\eta ) (1 + i\omega \tau)^c} \Bigg )
            \Bigg ]

        :param numpy.array frequencies: np.array(N,) containing frequencies
        :rtype: numpy.ndarray: np.array(n_layer, n_frequency)
        :return: complex conductivity matrix

        """
        n_layer = self.n_layer
        n_frequency = len(frequencies)
        # n_filter = self.n_filter

        sigma = np.tile(self.sigma.reshape([-1, 1]), (1, n_frequency))

        # No IP effect
        if np.all(self.eta) == 0.0:
            return sigma

        # IP effect
        else:
            if np.isscalar(self.eta):
                eta = self.eta
                tau = self.tau
                c = self.c
            else:
                eta = np.tile(self.eta.reshape([-1, 1]), (1, n_frequency))
                tau = np.tile(self.tau.reshape([-1, 1]), (1, n_frequency))
                c = np.tile(self.c.reshape([-1, 1]), (1, n_frequency))

            w = np.tile(2 * np.pi * frequencies, (n_layer, 1))

            sigma_complex = np.empty(
                [n_layer, n_frequency], dtype=np.complex128, order="F"
            )
            sigma_complex[:, :] = sigma - sigma * eta / (
                1 + (1 - eta) * (1j * w * tau) ** c
            )

            return sigma_complex

    def compute_complex_mu(self, frequencies):
        r"""
        Computes the complex magnetic permeability matrix assuming a log-uniform
        distribution of time-relaxation constants:

        .. math::

            \chi (\omega ) = \chi + \Delta \chi \Bigg [
                1 - \Bigg ( \frac{1}{ln (\tau_2 / \tau_1 )} \Bigg )
                ln \Bigg ( \frac{1 + i\omega \tau_2}{1 + i\omega tau_1} ) \Bigg )
            \Bigg ]

        :param numpy.array frequencies: np.array(N,) containing frequencies
        :rtype: numpy.ndarray: np.array(n_layer, n_frequency)
        :return: complex magnetic susceptibility matrix
        """

        if np.isscalar(self.mu):
            mu = np.ones_like(self.sigma) * self.mu
        else:
            mu = self.mu

        n_layer = self.n_layer
        n_frequency = len(frequencies)
        # n_filter = self.n_filter

        mu = np.tile(mu.reshape([-1, 1]), (1, n_frequency))

        # No magnetic viscosity
        if np.all(self.dchi) == 0.0:
            return mu

        # Magnetic viscosity
        else:
            if np.isscalar(self.dchi):
                dchi = self.dchi * np.ones_like(self.mu)
                tau1 = self.tau1 * np.ones_like(self.mu)
                tau2 = self.tau2 * np.ones_like(self.mu)
            else:
                dchi = np.tile(self.dchi.reshape([-1, 1]), (1, n_frequency))
                tau1 = np.tile(self.tau1.reshape([-1, 1]), (1, n_frequency))
                tau2 = np.tile(self.tau2.reshape([-1, 1]), (1, n_frequency))

            w = np.tile(2 * np.pi * frequencies, (n_layer, 1))

            mu_complex = mu + mu_0 * dchi * (
                1
                - np.log((1 + 1j * w * tau2) / (1 + 1j * w * tau1))
                / np.log(tau2 / tau1)
            )

            return mu_complex

    def Jvec(self, m, v, f=None):
        Js = self.getJ(m, f=f)
        out = 0.0
        if self.hMap is not None:
            out = out + Js["dh"] @ (self.hDeriv @ v)
        if self.sigmaMap is not None:
            out = out + Js["ds"] @ (self.sigmaDeriv @ v)
        if self.muMap is not None:
            out = out + Js["dmu"] @ (self.muDeriv @ v)
        if self.thicknessesMap is not None:
            out = out + Js["dthick"] @ (self.thicknessesDeriv @ v)
        return out

    def Jtvec(self, m, v, f=None):
        Js = self.getJ(m, f=f)
        out = 0.0
        if self.hMap is not None:
            out = out + self.hDeriv.T @ (Js["dh"].T @ v)
        if self.sigmaMap is not None:
            out = out + self.sigmaDeriv.T @ (Js["ds"].T @ v)
        if self.muMap is not None:
            out = out + self.muDeriv.T @ (Js["dmu"].T @ v)
        if self.thicknessesMap is not None:
            out = out + self.thicknessesDeriv.T @ (Js["dthick"].T @ v)
        return out

    def _compute_hankel_coefficients(self):
        survey = self.survey
        C0s = []
        C1s = []
        lambs = []
        Is = []
        n_w_past = 0
        i_count = 0
        for src in survey.source_list:
            # doing the check for source type by checking its name
            # to avoid importing and checking "isinstance"
            class_name = type(src).__name__
            is_circular_loop = class_name == "CircularLoop"
            is_mag_dipole = class_name == "MagDipole"
            is_wire_loop = class_name == "LineCurrent"

            if is_circular_loop:
                if np.any(src.orientation[:-1] != 0.0):
                    raise ValueError("Can only simulate horizontal circular loops")
            if self.hMap is not None:
                h = 0  # source height above topo
            else:
                h = src.location[2] - self.topo[-1]

            if is_circular_loop or is_mag_dipole:
                src_x, src_y, src_z = src.orientation * src.moment / (4 * np.pi)
                # src.moment is pi * radius**2 * I for circular loop
            if is_wire_loop:
                x, w = roots_legendre(self.n_points_per_path)
                xy_src_path = src.location[:, :2]
                xyks = []
                thetas = []
                weights = []
                for i_path in range(src.n_segments):
                    dx = xy_src_path[i_path + 1, 0] - xy_src_path[i_path, 0]
                    dy = xy_src_path[i_path + 1, 1] - xy_src_path[i_path, 1]
                    dl = np.sqrt(dx**2 + dy**2)
                    theta = np.arctan2(dy, dx)
                    lk = np.c_[(x + 1) * dl / 2, np.zeros(self.n_points_per_path)]

                    R = np.array([[dx, -dy], [dy, dx]]) / dl
                    xyk = lk.dot(R.T) + xy_src_path[i_path, :]

                    xyks.append(xyk)
                    thetas.append(theta * np.ones(xyk.shape[0]))
                    weights.append(w * dl / 2)
                # store these for future evalution of integrals
                xyks = np.vstack(xyks)
                weights = np.hstack(weights) * src.current
                thetas = -np.hstack(thetas)

            for rx in src.receiver_list:
                #######
                # Hankel Transform coefficients
                ######
                rx_x, rx_y, rx_z = rx.orientation

                # Compute receiver height
                if rx.use_source_receiver_offset:
                    dxyz = rx.locations
                    z = h + rx.locations[:, 2]
                else:
                    dxyz = rx.locations - src.location
                    z = h + rx.locations[:, 2] - src.location[2]

                if is_wire_loop:
                    dxy = rx.locations[:, :2] - xyks
                    h = src.location.mean(axis=0)[2] - self.topo[-1]
                    z = h + rx.locations[:, 2] - src.location.mean(axis=0)[2]
                    offsets = np.linalg.norm(dxy, axis=-1)
                else:
                    offsets = np.linalg.norm(dxyz[:, :-1], axis=-1)

                if is_circular_loop:
                    if np.any(offsets != 0.0):
                        raise ValueError(
                            "Can only simulate central loop receivers with circular loop source"
                        )
                    offsets = src.radius * np.ones(rx.locations.shape[0])

                # computations for hankel transform...
                lambd, _ = get_dlf_points(
                    self._fhtfilt, offsets, self._hankel_pts_per_dec
                )
                # calculate the source-rx coefficients for the hankel transform
                C0 = 0.0
                C1 = 0.0
                if is_circular_loop:
                    # I * a/ 2 * (lambda **2 )/ (lambda)
                    C1 += src_z * rx_z * (2 / src.radius) * lambd
                    n_w = 1

                elif is_mag_dipole:
                    n_w = 1
                    if src_x != 0.0:
                        if rx_x != 0.0:
                            C0 += (
                                src_x
                                * rx_x
                                * (dxyz[:, 0] ** 2 / offsets**2)[:, None]
                                * lambd**2
                            )
                            C1 += (
                                src_x
                                * rx_x
                                * (1 / offsets - 2 * dxyz[:, 0] ** 2 / offsets**3)[
                                    :, None
                                ]
                                * lambd
                            )
                        if rx_y:
                            C0 += (
                                src_x
                                * rx_y
                                * (dxyz[:, 0] * dxyz[:, 1] / offsets**2)[:, None]
                                * lambd**2
                            )
                            C1 -= (
                                src_x
                                * rx_y
                                * (2 * dxyz[:, 0] * dxyz[:, 1] / offsets**3)[:, None]
                                * lambd
                            )
                        if rx_z != 0.0:
                            # C0 += 0.0
                            C1 -= (src_x * rx_z * dxyz[:, 0] / offsets)[
                                :, None
                            ] * lambd**2
                    if src_y != 0.0:
                        if rx_x != 0.0:
                            C0 += (
                                src_y
                                * rx_x
                                * rx_x
                                * (dxyz[:, 0] * dxyz[:, 1] / offsets**2)[:, None]
                                * lambd**2
                            )
                            C1 -= (
                                src_y
                                * rx_x
                                * (2 * dxyz[:, 0] * dxyz[:, 1] / offsets**3)[:, None]
                                * lambd
                            )
                        if rx_y != 0.0:
                            C0 += (
                                src_y
                                * rx_y
                                * (dxyz[:, 1] ** 2 / offsets**2)[:, None]
                                * lambd**2
                            )
                            C1 += (
                                src_y
                                * rx_y
                                * (1 / offsets - 2 * dxyz[:, 1] ** 2 / offsets**3)[
                                    :, None
                                ]
                                * lambd
                            )
                        if rx_z != 0.0:
                            # C0 += 0.0
                            C1 -= (src_y * rx_z * dxyz[:, 1] / offsets)[
                                :, None
                            ] * lambd**2
                    if src_z != 0.0:
                        if rx_x != 0.0:
                            # C0 += 0.0
                            C1 += (src_z * rx_x * dxyz[:, 0] / offsets)[
                                :, None
                            ] * lambd**2
                        if rx_y != 0.0:
                            # C0 += 0.0
                            C1 += (src_z * rx_y * dxyz[:, 1] / offsets)[
                                :, None
                            ] * lambd**2
                        if rx_z != 0.0:
                            C0 += src_z * rx_z * lambd**2
                elif is_wire_loop:
                    R = np.stack(
                        [
                            [np.cos(thetas), -np.sin(thetas)],
                            [np.sin(thetas), np.cos(thetas)],
                        ]
                    )
                    dxy_rot = np.einsum("...i,ji...", dxy, R)
                    C1 = (1 / (4 * np.pi) * (dxy_rot[:, 1] / offsets * weights))[
                        :, None
                    ] * lambd
                    # Assume
                    # 1) source_list only includes wire_loop sources
                    # 1) rx.locations.shape = (1,3)
                    n_w = weights.size
                else:
                    raise TypeError(
                        f"Unsupported source type of {type(src)}. Must be a CircularLoop or MagDipole"
                    )
                # divide by offsets to pre-do that part from the dft (1 less item to store)
                C0s.append(np.exp(-lambd * (z + h)[:, None]) * C0 / offsets[:, None])
                C1s.append(np.exp(-lambd * (z + h)[:, None]) * C1 / offsets[:, None])
                lambs.append(lambd)
                n_w_past += n_w
                Is.append(np.ones(n_w, dtype=int) * i_count)
                i_count += 1

        # Store these on the simulation for faster future executions
        self._lambs = np.vstack(lambs)
        self._unique_lambs, inv_lambs = np.unique(self._lambs, return_inverse=True)
        self._inv_lambs = inv_lambs.reshape(self._lambs.shape)
        self._C0s = np.vstack(C0s)
        self._C1s = np.vstack(C1s)
        Is = np.hstack(Is)
        n_row = Is.size
        n_col = Is.max() + 1
        Js = np.arange(n_row)
        data = np.ones(n_row, dtype=int)
        self._W = sp.coo_matrix((data, (Is, Js)), shape=(n_col, n_row))
        self._W = self._W.tocsr()

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = super().deleteTheseOnModelUpdate
        if self.fix_Jmatrix is False:
            toDelete += ["_J", "_gtgdiag"]
        return toDelete

    def depth_of_investigation_christiansen_2012(self, std, thres_hold=0.8):
        pred = self.survey._pred.copy()
        delta_d = std * np.log(abs(self.survey.dobs))
        J = self.getJ(self.model)
        J_sum = abs(utils.sdiag(1 / delta_d / pred) * J).sum(axis=0)
        S = np.cumsum(J_sum[::-1])[::-1]
        active = S - thres_hold > 0.0
        doi = abs(self.depth[active]).max()
        return doi, active

    def get_threshold(self, uncert):
        _, active = self.depth_of_investigation(uncert)
        JtJdiag = self.get_JtJdiag(uncert)
        delta = JtJdiag[active].min()
        return delta

    def getJtJdiag(self, m, W=None, f=None):
        if getattr(self, "_gtgdiag", None) is None:
            Js = self.getJ(m, f=f)
            if W is None:
                W = np.ones(self.survey.nD)
            else:
                W = W.diagonal() ** 2
            out = 0.0
            if self.hMap is not None:
                J = Js["dh"] @ self.hDeriv
                out = out + np.einsum("i,ij,ij->j", W, J, J)
            if self.sigmaMap is not None:
                J = Js["ds"] @ self.sigmaDeriv
                out = out + np.einsum("i,ij,ij->j", W, J, J)
            if self.muMap is not None:
                J = Js["dmu"] @ self.muDeriv
                out = out + np.einsum("i,ij,ij->j", W, J, J)
            if self.thicknessesMap is not None:
                J = Js["dthick"] @ self.thicknessesDeriv
                out = out + np.einsum("i,ij,ij->j", W, J, J)
            self._gtgdiag = out
        return self._gtgdiag
