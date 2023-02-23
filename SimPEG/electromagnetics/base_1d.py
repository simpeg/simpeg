import properties
from scipy.constants import mu_0
import numpy as np
from scipy import sparse as sp
from empymod.transform import get_dlf_points

from ..data import Data
from ..simulation import BaseSimulation

from .. import utils
from .. import props
from empymod.utils import check_hankel

from multiprocessing import Pool
from sys import platform

__all__ = ["BaseEM1DSimulation", "BaseStitchedEM1DSimulation"]

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

    hankel_filter = "key_101_2009"  # Default: Hankel filter
    _hankel_pts_per_dec = 0  # Default: Standard DLF
    verbose = False
    fix_Jmatrix = False
    _formulation = "1D"
    _coefficients_set = False
    gtgdiag = None

    # Properties for electrical conductivity/resistivity
    sigma, sigmaMap, sigmaDeriv = props.Invertible(
        "Electrical conductivity at infinite frequency (S/m)"
    )

    rho, rhoMap, rhoDeriv = props.Invertible("Electrical resistivity (Ohm m)")

    props.Reciprocal(sigma, rho)

    eta, etaMap, etaDeriv = props.Invertible(
        "Intrinsic chargeability (V/V), 0 <= eta < 1", default=0.0
    )
    tau, tauMap, tauDeriv = props.Invertible("Time constant for Cole-Cole model (s)", default=1.0)

    c, cMap, cDeriv = props.Invertible(
        "Frequency Dependency for Cole-Cole model, 0 < c < 1", default=0.5
    )

    # Properties for magnetic susceptibility
    mu, muMap, muDeriv = props.Invertible(
        "Magnetic permeability at infinite frequency (SI)", default=mu_0
    )
    dchi = props.PhysicalProperty(
        "DC magnetic susceptibility for viscous remanent magnetization contribution (SI)",
        default=0.0,
    )
    tau1 = props.PhysicalProperty(
        "Lower bound for log-uniform distribution of time-relaxation constants for viscous remanent magnetization (s)",
        default=1e-10,
    )
    tau2 = props.PhysicalProperty(
        "Upper bound for log-uniform distribution of time-relaxation constants for viscous remanent magnetization (s)",
        default=10.0,
    )

    # Additional properties
    h, hMap, hDeriv = props.Invertible("Receiver Height (m), h > 0",)

    topo = properties.Array("Topography (x, y, z)", dtype=float)

    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "layer thicknesses (m)", default=np.array([])
    )

    def __init__(self, **kwargs):
        BaseSimulation.__init__(self, **kwargs)

        # Check input arguments. If self.hankel_filter is not a valid filter,
        # it will set it to the default (key_201_2009).
        ht, htarg = check_hankel(
            "dlf", {"dlf": self.hankel_filter, "pts_per_dec": 0}, 1
        )

        self.fhtfilt = htarg["dlf"]  # Store filter
        self.hankel_pts_per_dec = htarg["pts_per_dec"]  # Store pts_per_dec
        if self.verbose:
            print(">> Use " + self.hankel_filter + " filter for Hankel Transform")

    @property
    def n_layer(self):
        """number of layers"""
        return int(self.thicknesses.size + 1)

    @property
    def n_filter(self):
        """ Length of filter """
        return self.fhtfilt.base.size

    @property
    def depth(self):
        """layer depths"""
        if self.thicknesses is not None:
            return np.r_[0.0, -np.cumsum(self.thicknesses)]
        return None

    def compute_complex_sigma(self, frequencies):
        """
        Computes the complex conductivity matrix using Pelton's Cole-Cole model:

        .. math ::
            \\sigma (\\omega ) = \\sigma \\Bigg [
            1 - \\eta \\Bigg ( \\frac{1}{1 + (1-\\eta ) (1 + i\\omega \\tau)^c} \\Bigg )
            \\Bigg ]

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
                1 + (1j * w * tau) ** c
            )

            return sigma_complex

    def compute_dcomplex_sigma_dsigma_inf(self, frequencies):
        n_layer = self.n_layer
        n_frequency = len(frequencies)

        sigma = np.tile(self.sigma.reshape([-1, 1]), (1, n_frequency))

        if np.isscalar(self.eta):
            eta = self.eta
            tau = self.tau
            c = self.c
        else:
            eta = np.tile(self.eta.reshape([-1, 1]), (1, n_frequency))
            tau = np.tile(self.tau.reshape([-1, 1]), (1, n_frequency))
            c = np.tile(self.c.reshape([-1, 1]), (1, n_frequency))

        w = np.tile(2 * np.pi * frequencies, (n_layer, 1))

        dsigma_dsigma_inf = np.empty(
            [n_layer, n_frequency], dtype=np.complex128, order="F"
        )
        dsigma_dsigma_inf[:, :] = 1 - 1 * eta / (
            1 + (1j * w * tau) ** c
        )
        return dsigma_dsigma_inf

    def compute_dcomplex_sigma_deta(self, frequencies):
        n_layer = self.n_layer
        n_frequency = len(frequencies)

        sigma = np.tile(self.sigma.reshape([-1, 1]), (1, n_frequency))

        if np.isscalar(self.eta):
            eta = self.eta
            tau = self.tau
            c = self.c
        else:
            eta = np.tile(self.eta.reshape([-1, 1]), (1, n_frequency))
            tau = np.tile(self.tau.reshape([-1, 1]), (1, n_frequency))
            c = np.tile(self.c.reshape([-1, 1]), (1, n_frequency))

        w = np.tile(2 * np.pi * frequencies, (n_layer, 1))

        dsigma_deta = np.empty(
            [n_layer, n_frequency], dtype=np.complex128, order="F"
        )
        dsigma_deta[:, :] = - sigma / (
            1 + (1j * w * tau) ** c
        )
        return dsigma_deta

    def compute_dcomplex_sigma_dtau(self, frequencies):
        n_layer = self.n_layer
        n_frequency = len(frequencies)

        sigma = np.tile(self.sigma.reshape([-1, 1]), (1, n_frequency))

        if np.isscalar(self.eta):
            eta = self.eta
            tau = self.tau
            c = self.c
        else:
            eta = np.tile(self.eta.reshape([-1, 1]), (1, n_frequency))
            tau = np.tile(self.tau.reshape([-1, 1]), (1, n_frequency))
            c = np.tile(self.c.reshape([-1, 1]), (1, n_frequency))

        w = np.tile(2 * np.pi * frequencies, (n_layer, 1))

        dsigma_tau = np.empty(
            [n_layer, n_frequency], dtype=np.complex128, order="F"
        )
        S = (1j * w * tau) ** c
        dsigma_tau[:, :] = sigma*eta*c*S / (
            tau*(S+1)**2
        )
        return dsigma_tau

    def compute_dcomplex_sigma_dc(self, frequencies):
        n_layer = self.n_layer
        n_frequency = len(frequencies)

        sigma = np.tile(self.sigma.reshape([-1, 1]), (1, n_frequency))

        if np.isscalar(self.eta):
            eta = self.eta
            tau = self.tau
            c = self.c
        else:
            eta = np.tile(self.eta.reshape([-1, 1]), (1, n_frequency))
            tau = np.tile(self.tau.reshape([-1, 1]), (1, n_frequency))
            c = np.tile(self.c.reshape([-1, 1]), (1, n_frequency))

        w = np.tile(2 * np.pi * frequencies, (n_layer, 1))

        S = (1j * w * tau) ** c
        dsigma_c = np.empty(
            [n_layer, n_frequency], dtype=np.complex128, order="F"
        )
        dsigma_c[:, :] = sigma*eta*S*np.log(1j * w * tau) / (
            (S+1)**2
        )
        return dsigma_c

    def compute_complex_mu(self, frequencies):
        """
        Computes the complex magnetic permeability matrix assuming a log-uniform
        distribution of time-relaxation constants:

        .. math::
            \\chi (\\omega ) = \\chi + \\Delta \\chi \\Bigg [
            1 - \\Bigg ( \\frac{1}{ln (\\tau_2 / \\tau_1 )} \\Bigg )
            ln \\Bigg ( \\frac{1 + i\\omega \\tau_2}{1 + i\\omega tau_1} ) \\Bigg )
            \\Bigg ]

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
        if self.etaMap is not None:
            out = out + Js["deta"] @ (self.etaDeriv @ v)
        if self.tauMap is not None:
            out = out + Js["dtau"] @ (self.tauDeriv @ v)
        if self.cMap is not None:
            out = out + Js["dc"] @ (self.cDeriv @ v)
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
        if self.etaMap is not None:
            out = out + self.etaDeriv.T @ (Js["deta"].T @ v)
        if self.tauMap is not None:
            out = out + self.tauDeriv.T @ (Js["dtau"].T @ v)
        if self.cMap is not None:
            out = out + self.cDeriv.T @ (Js["dc"].T @ v)
        if self.muMap is not None:
            out = out + self.muDeriv.T @ (Js["dmu"].T @ v)
        if self.thicknessesMap is not None:
            out = out + self.thicknessesDeriv.T @ (Js["dthick"].T @ v)
        return out

    def _compute_hankel_coefficients(self):
        survey = self.survey
        if self.hMap is not None:
            h_vector = np.zeros(len(survey.source_list))  # , self.h
            # if it has an hMap, do not include the height in the
            # pre-computed coefficients
        else:
            h_vector = np.array(
                [src.location[2] - self.topo[-1] for src in self.survey.source_list]
            )
        C0s = []
        C1s = []
        lambs = []
        Is = []
        n_w_past = 0
        i_count = 0
        for i_src, src in enumerate(survey.source_list):
            # doing the check for source type by checking its name
            # to avoid importing and checking "isinstance"
            class_name = type(src).__name__
            is_circular_loop = class_name == "CircularLoop"
            is_mag_dipole = class_name == "MagDipole"
            is_wire_loop = class_name == "PiecewiseWireLoop"

            if is_circular_loop:
                if np.any(src.orientation[:-1] != 0.0):
                    raise ValueError("Can only simulate horizontal circular loops")
            h = h_vector[i_src]  # source height above topo
            if is_circular_loop or is_mag_dipole:
                src_x, src_y, src_z = src.orientation * src.moment / (4 * np.pi)
                # src.moment is pi * radius**2 * I for circular loop
            for i_rx, rx in enumerate(src.receiver_list):
                #######
                # Hankel Transform coefficients
                ######

                # Compute receiver height
                if rx.use_source_receiver_offset:
                    dxyz = rx.locations
                    z = h + rx.locations[:, 2]
                else:
                    dxyz = rx.locations - src.location
                    z = h + rx.locations[:, 2] - src.location[2]

                if is_wire_loop:
                    dxy = rx.locations[:,:2] - src._xyks
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
                    self.fhtfilt, offsets, self._hankel_pts_per_dec
                )
                # calculate the source-rx coefficients for the hankel transform
                C0 = 0.0
                C1 = 0.0
                if is_circular_loop:
                    # I * a/ 2 * (lambda **2 )/ (lambda)
                    C1 += src_z * (2 / src.radius) * lambd
                    n_w = 1

                elif is_mag_dipole:
                    n_w = 1
                    if src_x != 0.0:
                        if rx.orientation == "x":
                            C0 += (
                                src_x
                                * (dxyz[:, 0] ** 2 / offsets ** 2)[:, None]
                                * lambd ** 2
                            )
                            C1 += (
                                src_x
                                * (1 / offsets - 2 * dxyz[:, 0] ** 2 / offsets ** 3)[
                                    :, None
                                ]
                                * lambd
                            )
                        elif rx.orientation == "y":
                            C0 += (
                                src_x
                                * (dxyz[:, 0] * dxyz[:, 1] / offsets ** 2)[:, None]
                                * lambd ** 2
                            )
                            C1 -= (
                                src_x
                                * (2 * dxyz[:, 0] * dxyz[:, 1] / offsets ** 3)[:, None]
                                * lambd
                            )
                        elif rx.orientation == "z":
                            # C0 += 0.0
                            C1 -= (src_x * dxyz[:, 0] / offsets)[:, None] * lambd ** 2
                    if src_y != 0.0:
                        if rx.orientation == "x":
                            C0 += (
                                src_y
                                * (dxyz[:, 0] * dxyz[:, 1] / offsets ** 2)[:, None]
                                * lambd ** 2
                            )
                            C1 -= (
                                src_y
                                * (2 * dxyz[:, 0] * dxyz[:, 1] / offsets ** 3)[:, None]
                                * lambd
                            )
                        elif rx.orientation == "y":
                            C0 += (
                                src_y
                                * (dxyz[:, 1] ** 2 / offsets ** 2)[:, None]
                                * lambd ** 2
                            )
                            C1 += (
                                src_y
                                * (1 / offsets - 2 * dxyz[:, 1] ** 2 / offsets ** 3)[
                                    :, None
                                ]
                                * lambd
                            )
                        elif rx.orientation == "z":
                            # C0 += 0.0
                            C1 -= (src_y * dxyz[:, 1] / offsets)[:, None] * lambd ** 2
                    if src_z != 0.0:
                        if rx.orientation == "x":
                            # C0 += 0.0
                            C1 += (src_z * dxyz[:, 0] / offsets)[:, None] * lambd ** 2
                        elif rx.orientation == "y":
                            # C0 += 0.0
                            C1 += (src_z * dxyz[:, 1] / offsets)[:, None] * lambd ** 2
                        elif rx.orientation == "z":
                            C0 += src_z * lambd ** 2
                elif is_wire_loop:
                    weights = src._weights
                    dxy_rot = src.rotate_points_xy_var_theta(dxy, -src._thetas)
                    C1 = (1 /(4*np.pi)*(dxy_rot[:,1]/offsets * weights))[:,None] * lambd
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
        n_col = Is.max()+1
        Js = np.arange(n_row)
        data = np.ones(n_row, dtype=int)
        self._W = sp.coo_matrix((data, (Is, Js)), shape=(n_col, n_row))
        self._W = self._W.tocsr()


    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.fix_Jmatrix is False:
            toDelete += ["_J"]
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

    def getJtJdiag(self, m, W=None):
        if self.gtgdiag is None:
            Js = self.getJ(m)
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
            self.gtgdiag = out
        return self.gtgdiag


class BaseStitchedEM1DSimulation(BaseSimulation):
    """
    Base class for the stitched 1D simulation. This simulation models the EM
    response for a set of 1D EM soundings.
    """

    _Jmatrix_sigma = None
    _Jmatrix_height = None
    _coefficients = []
    _coefficients_set = False
    run_simulation = None
    n_cpu = None
    parallel = False
    parallel_jvec_jtvec = False
    verbose = False
    fix_Jmatrix = False
    invert_height = None
    n_sounding_for_chunk = None
    use_sounding = True

    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "thicknesses of the layers",
        default=np.array([])
    )

    sigma, sigmaMap, sigmaDeriv = props.Invertible(
        "Electrical conductivity (S/m)"
    )

    h, hMap, hDeriv = props.Invertible(
        "Receiver Height (m), h > 0",
    )

    eta = props.PhysicalProperty(
        "Electrical chargeability (V/V), 0 <= eta < 1"
    )

    tau = props.PhysicalProperty(
        "Time constant (s)"
    )

    c = props.PhysicalProperty(
        "Frequency Dependency, 0 < c < 1"
    )

    chi = props.PhysicalProperty(
        "Magnetic susceptibility (SI)"
    )

    dchi = props.PhysicalProperty(
        "DC magnetic susceptibility attributed to magnetic viscosity (SI)"
    )

    tau1 = props.PhysicalProperty(
        "Lower bound for log-uniform distribution of time-relaxation constants (s)"
    )

    tau2 = props.PhysicalProperty(
        "Lower bound for log-uniform distribution of time-relaxation constants (s)"
    )

    topo = properties.Array("Topography (x, y, z)", dtype=float, shape=('*', 3))

    n_layer = properties.Integer("Number of layers", default=None)


    def __init__(self, **kwargs):
        utils.setKwargs(self, **kwargs)

        if self.parallel:
            if self.verbose:
                print(">> Use multiprocessing for parallelization")
                if self.n_cpu is None:
                    self.n_cpu = multiprocessing.cpu_count()
                print((">> n_cpu: %i") % (self.n_cpu))
        else:
            if self.verbose:
                print(">> Serial version is used")

        if self.hMap is None:
            self.invert_height = False
        else:
            self.invert_height = True

    # ------------- For survey ------------- #
    # @property
    # def dz(self):
    #     if self.mesh.dim==2:
    #         return self.mesh.dy
    #     elif self.mesh.dim==3:
    #         return self.mesh.dz

    @property
    def halfspace_switch(self):
        """True = halfspace, False = layered Earth"""
        if (self.thicknesses is None) | (len(self.thicknesses)==0):
            return True
        else:
            return False

    # @property
    # def n_layer(self):
    #     if self.thicknesses is None:
    #         return 1
    #     else:
    #         return len(self.thicknesses) + 1

    @property
    def n_sounding(self):
        return len(self.survey.source_location_by_sounding_dict)


    @property
    def data_index(self):
        return self.survey.data_index


    # ------------- For physical properties ------------- #
    @property
    def Sigma(self):
        if getattr(self, '_Sigma', None) is None:
            # Ordering: first z then x
            self._Sigma = self.sigma.reshape((self.n_sounding, self.n_layer))
        return self._Sigma

    @property
    def Thicknesses(self):
        if getattr(self, '_Thicknesses', None) is None:
            # Ordering: first z then x
            if len(self.thicknesses) == int(self.n_sounding * (self.n_layer-1)):
                self._Thicknesses = self.thicknesses.reshape((self.n_sounding, self.n_layer-1))
            else:
                self._Thicknesses = np.tile(self.thicknesses, (self.n_sounding, 1))
        return self._Thicknesses

    @property
    def Eta(self):
        if getattr(self, '_Eta', None) is None:
            # Ordering: first z then x
            if self.eta is None:
                self._Eta = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order='C'
                )
            else:
                self._Eta = self.eta.reshape((self.n_sounding, self.n_layer))
        return self._Eta

    @property
    def Tau(self):
        if getattr(self, '_Tau', None) is None:
            # Ordering: first z then x
            if self.tau is None:
                self._Tau = 1e-3*np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order='C'
                )
            else:
                self._Tau = self.tau.reshape((self.n_sounding, self.n_layer))
        return self._Tau

    @property
    def C(self):
        if getattr(self, '_C', None) is None:
            # Ordering: first z then x
            if self.c is None:
                self._C = np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order='C'
                )
            else:
                self._C = self.c.reshape((self.n_sounding, self.n_layer))
        return self._C

    @property
    def Chi(self):
        if getattr(self, '_Chi', None) is None:
            # Ordering: first z then x
            if self.chi is None:
                self._Chi = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order='C'
                )
            else:
                self._Chi = self.chi.reshape((self.n_sounding, self.n_layer))
        return self._Chi

    @property
    def dChi(self):
        if getattr(self, '_dChi', None) is None:
            # Ordering: first z then x
            if self.dchi is None:
                self._dChi = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order='C'
                )
            else:
                self._dChi = self.dchi.reshape((self.n_sounding, self.n_layer))
        return self._dChi

    @property
    def Tau1(self):
        if getattr(self, '_Tau1', None) is None:
            # Ordering: first z then x
            if self.tau1 is None:
                self._Tau1 = 1e-10 * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order='C'
                )
            else:
                self._Tau1 = self.tau1.reshape((self.n_sounding, self.n_layer))
        return self._Tau1

    @property
    def Tau2(self):
        if getattr(self, '_Tau2', None) is None:
            # Ordering: first z then x
            if self.tau2 is None:
                self._Tau2 = 100. * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order='C'
                )
            else:
                self._Tau2 = self.tau2.reshape((self.n_sounding, self.n_layer))
        return self._Tau2

    @property
    def JtJ_sigma(self):
        return self._JtJ_sigma

    def JtJ_height(self):
        return self._JtJ_height

    @property
    def H(self):
        if self.hMap is None:
            return np.ones(self.n_sounding)
        else:
            return self.h


    # ------------- Etcetra .... ------------- #
    @property
    def IJLayers(self):
        if getattr(self, '_IJLayers', None) is None:
            # Ordering: first z then x
            self._IJLayers = self.set_ij_n_layer()
        return self._IJLayers

    @property
    def IJHeight(self):
        if getattr(self, '_IJHeight', None) is None:
            # Ordering: first z then x
            self._IJHeight = self.set_ij_n_layer(n_layer=1)
        return self._IJHeight

    # ------------- For physics ------------- #

    def get_uniq_soundings(self):
            self._sounding_types_uniq, self._ind_sounding_uniq = np.unique(
                self.survey._sounding_types, return_index=True
            )

    def input_args(self, i_sounding, output_type='forward'):
        output = (
            self.survey.get_sources_by_sounding_number(i_sounding),
            self.topo[i_sounding, :],
            self.Thicknesses[i_sounding,:],
            self.Sigma[i_sounding, :],
            self.Eta[i_sounding, :],
            self.Tau[i_sounding, :],
            self.C[i_sounding, :],
            self.Chi[i_sounding, :],
            self.dChi[i_sounding, :],
            self.Tau1[i_sounding, :],
            self.Tau2[i_sounding, :],
            self.H[i_sounding],
            output_type,
            self.invert_height,
            False,
            self._coefficients[i_sounding],
        )
        return output

    # This is the most expensive process, but required once
    # May need to find unique
    def input_args_for_coeff(self, i_sounding):
        output = (
            self.survey.get_sources_by_sounding_number(i_sounding),
            self.topo[i_sounding, :],
            self.Thicknesses[i_sounding,:],
            self.Sigma[i_sounding, :],
            self.Eta[i_sounding, :],
            self.Tau[i_sounding, :],
            self.C[i_sounding, :],
            self.Chi[i_sounding, :],
            self.dChi[i_sounding, :],
            self.Tau1[i_sounding, :],
            self.Tau2[i_sounding, :],
            self.H[i_sounding],
            'forward',
            self.invert_height,
            True,
            [],
        )
        return output

    def fields(self, m):
        if self.verbose:
            print("Compute fields")

        return self.forward(m)

    def dpred(self, m, f=None):
        """
            Return predicted data.
            Predicted data, (`_pred`) are computed when
            self.fields is called.
        """
        if f is None:
            f = self.fields(m)

        return f

    @property
    def sounding_number(self):
        self._sounding_number = [key for key in self.survey.source_location_by_sounding_dict.keys()]
        return self._sounding_number

    @property
    def sounding_number_chunks(self):
        self._sounding_number_chunks = list(self.chunks(self.sounding_number, self.n_sounding_for_chunk))
        return self._sounding_number_chunks

    @property
    def n_chunk(self):
        self._n_chunk = len(self.sounding_number_chunks)
        return self._n_chunk

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def input_args_by_chunk(self, i_chunk, output_type):
        args_by_chunks = []
        for i_sounding in self.sounding_number_chunks[i_chunk]:
            args_by_chunks.append(self.input_args(i_sounding, output_type))
        return args_by_chunks

    def set_null_topography(self):
        self.topo = np.vstack(
            [np.c_[src.location[0], src.location[1], 0.] for i, src in enumerate(self.survey.source_list)]
        )


    def set_ij_n_layer(self, n_layer=None):
        """
        Compute (I, J) indicies to form sparse sensitivity matrix
        This will be used in GlobalEM1DSimulation when after sensitivity matrix
        for each sounding is computed
        """
        I = []
        J = []
        shift_for_J = 0
        shift_for_I = 0
        if n_layer is None:
            m = self.n_layer
        else:
            m = n_layer
        source_location_by_sounding_dict = self.survey.source_location_by_sounding_dict
        for i_sounding in range(self.n_sounding):
            n = self.survey.vnD_by_sounding_dict[i_sounding]
            J_temp = np.tile(np.arange(m), (n, 1)) + shift_for_J
            I_temp = (
                np.tile(np.arange(n), (1, m)).reshape((n, m), order='F') +
                shift_for_I
            )
            J.append(utils.mkvc(J_temp))
            I.append(utils.mkvc(I_temp))
            shift_for_J += m
            shift_for_I = I_temp[-1, -1] + 1
        J = np.hstack(J).astype(int)
        I = np.hstack(I).astype(int)
        return (I, J)

    def set_ij_height(self):
        """
        Compute (I, J) indicies to form sparse sensitivity matrix
        This will be used in GlobalEM1DSimulation when after sensitivity matrix
        for each sounding is computed
        """
        I = []
        J = []
        shift_for_J = 0
        shift_for_I = 0
        m = self.n_layer
        for i_sounding in range(self.n_sounding):
            n = self.survey.vnD_by_sounding_dict[i_sounding]
            J_temp = np.tile(np.arange(m), (n, 1)) + shift_for_J
            I_temp = (
                np.tile(np.arange(n), (1, m)).reshape((n, m), order='F') +
                shift_for_I
            )
            J.append(utils.mkvc(J_temp))
            I.append(utils.mkvc(I_temp))
            shift_for_J += m
            shift_for_I = I_temp[-1, -1] + 1
        J = np.hstack(J).astype(int)
        I = np.hstack(I).astype(int)
        return (I, J)

    def Jvec(self, m, v, f=None):
        J_sigma = self.getJ_sigma(m)
        J_height = self.getJ_height(m)
        Jv = J_sigma*(utils.sdiag(1./self.sigma)*(self.sigmaDeriv * v))
        if self.hMap is not None:
            Jv += J_height*(self.hDeriv * v)
        return Jv

    def Jtvec(self, m, v, f=None):
        J_sigma = self.getJ_sigma(m)
        J_height = self.getJ_height(m)
        Jtv = self.sigmaDeriv.T * (utils.sdiag(1./self.sigma) * (J_sigma.T*v))
        if self.hMap is not None:
            Jtv += self.hDeriv.T*(J_height.T*v)
        return Jtv

    def getJtJdiag(self, m, W=None, threshold=1e-8):
        """
        Compute diagonal component of JtJ or
        trace of sensitivity matrix (J)
        """
        J_sigma = self.getJ_sigma(m)
        J_matrix = J_sigma*(utils.sdiag(1./self.sigma)*(self.sigmaDeriv))

        if self.hMap is not None:
            J_height = self.getJ_height(m)
            J_matrix += J_height*self.hDeriv

        if W is None:
            W = utils.speye(J_matrix.shape[0])

        J_matrix = W*J_matrix
        JtJ_diag = (J_matrix.T*J_matrix).diagonal()
        JtJ_diag /= JtJ_diag.max()
        JtJ_diag += threshold
        return JtJ_diag

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.sigmaMap is not None:
            toDelete += ['_Sigma']
        if self.fix_Jmatrix is False:
            if self._Jmatrix_sigma is not None:
                toDelete += ['_Jmatrix_sigma']
            if self._Jmatrix_height is not None:
                toDelete += ['_Jmatrix_height']
        return toDelete

    def _run_simulation_by_chunk(self, args_chunk):
        """
        This method simulates the EM response or computes the sensitivities for
        a single sounding. The method allows for parallelization of
        the stitched 1D problem.
        """
        n = len(args_chunk)
        results = [
                    self.run_simulation(args_chunk[i_sounding]) for i_sounding in range(n)
        ]
        return results
