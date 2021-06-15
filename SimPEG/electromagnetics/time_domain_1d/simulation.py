from ... import maps, utils
from ..base_1d import BaseEM1DSimulation, BaseStitchedEM1DSimulation
from ..time_domain.sources import StepOffWaveform
from ..time_domain.sources import MagDipole, CircularLoop
from ..time_domain.receivers import (
    PointMagneticFluxDensity,
    PointMagneticField,
    PointMagneticFluxTimeDerivative,
)
import numpy as np

# from .survey import EM1DSurveyTD
from ..time_domain.survey import Survey
from scipy.constants import mu_0
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
from scipy.special import roots_legendre

from empymod import filters
from empymod.transform import get_dlf_points

from geoana.kernels.tranverse_electric_reflections import rTE_forward, rTE_gradient

import properties


class EM1DTMSimulation(BaseEM1DSimulation):
    """
    Simulation class for simulating the TEM response over a 1D layered Earth
    for a single sounding.
    """

    _coefficients_set = False
    time_filter = "key_81_CosSin_2009"

    survey = properties.Instance("a survey object", Survey, required=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.time_filter == "key_81_CosSin_2009":
            self.fftfilt = filters.key_81_CosSin_2009()
        elif self.time_filter == "key_201_CosSin_2012":
            self.fftfilt = filters.key_201_CosSin_2012()
        elif self.time_filter == "key_601_CosSin_2012":
            self.fftfilt = filters.key_601_CosSin_2009()
        else:
            raise Exception()

        if self.topo is None:
            self.topo = np.array([0, 0, 0], dtype=float)

        for i_src, src in enumerate(self.survey.source_list):
            if src.location[2] < self.topo[2]:
                raise Exception("Source must be located above the topography")
            for i_rx, rx in enumerate(src.receiver_list):
                if rx.locations[0, 2] < self.topo[2]:
                    raise Exception("Receiver must be located above the topography")

    def _compute_coefficients(self):
        if self._coefficients_set:
            return
        survey = self.survey
        if self.hMap is not None:
            h_vector = np.zeros(len(survey.source_list))  # , self.h
            # if it has an hMap, do not include the height in the
            # pre-computed coefficients
        else:
            h_vector = np.array(
                [src.location[2] - self.topo[-1] for src in self.survey.source_list]
            )

        t_min = np.infty
        t_max = -np.infty
        # loop through source and receiver lists to create necessary times (and offsets)
        for i_src, src in enumerate(survey.source_list):
            if isinstance(src, CircularLoop):
                if np.any(src.orientation[:-1] != 0.0):
                    raise ValueError("Can only simulate horizontal circular loops")
            for i_rx, rx in enumerate(src.receiver_list):
                # find the minimum and maximum time span...
                wave = src.waveform
                if isinstance(wave, StepOffWaveform):
                    t_min = min(rx.times.min(), t_min)
                    t_max = max(rx.times.max(), t_max)
                else:
                    try:
                        times = rx.times - wave.time_nodes[:, None]
                        t_min = min(times.min(), t_min)
                        t_max = max(times.max(), t_max)
                    except AttributeError:
                        raise TypeError(
                            f"Unsupported source waveform object of {src.waveform}"
                        )

        t_min = max(t_min, 1e-8)  # make it slightly above zero if it happens...
        omegas, t_spline_points = get_dlf_points(self.fftfilt, np.r_[t_min, t_max], -1)
        omegas = omegas.reshape(-1)
        n_omega = len(omegas)
        n_t = len(t_spline_points)

        n_base = len(self.fftfilt.base)
        A_dft = np.zeros((n_t, n_omega))
        for i in range(n_t):
            A_dft[i, i : i + n_base] = (
                self.fftfilt.cos / t_spline_points[i] * (-2.0 / np.pi)
            )
        A_dft = A_dft[::-1]  # shuffle these back

        # Calculate the interpolating spline basis functions for each spline point
        splines = []
        for i in range(n_t):
            e = np.zeros(n_t)
            e[i] = 1.0
            sp = iuSpline(np.log(t_spline_points[::-1]), e, k=5)
            splines.append(sp)
        # As will go from frequency to time domain
        As = []

        # loop through source and receiver lists to create offsets
        # get unique times for each receiver
        # Compute coefficients for Hankel transform
        C0s = []
        C1s = []
        lambs = []
        for i_src, src in enumerate(survey.source_list):
            h = h_vector[i_src]  # source height above topo
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

                offsets = np.linalg.norm(dxyz[:, :-1], axis=-1)
                if isinstance(src, CircularLoop):
                    if np.any(offsets != 0.0):
                        raise ValueError(
                            "Can only simulate central loop receivers with circular loop source"
                        )
                    offsets = src.radius * np.ones(rx.locations.shape[0])

                # computations for hankel transform...
                lambd, _ = get_dlf_points(
                    self.fhtfilt, offsets, self.hankel_pts_per_dec
                )
                # calculate the source-rx coefficients for the hankel transform
                C0 = 0.0
                C1 = 0.0
                if isinstance(src, CircularLoop):
                    # I * a/ 2 * (lambda **2 )/ (lambda)
                    C1 += src_z * (2 / src.radius) * lambd
                elif isinstance(src, MagDipole):
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

                # divide by offsets to pre-do that part from the dft (1 less item to store)
                C0s.append(np.exp(-lambd * (z + h)[:, None]) * C0 / offsets[:, None])
                C1s.append(np.exp(-lambd * (z + h)[:, None]) * C1 / offsets[:, None])
                lambs.append(lambd)

                #######
                # Fourier Transform coefficients
                ######
                wave = src.waveform

                def func(t, i):
                    out = np.zeros_like(t)
                    out[t > 0.0] = splines[i](np.log(t[t > 0.0]))
                    return out

                # Then calculate the values at each time
                A = np.zeros((len(rx.times), n_t))
                if isinstance(wave, StepOffWaveform):
                    # do not need to do too much fancy here, just need to interpolate
                    # from t_spline_points to rx.times...
                    for i in range(n_t):
                        A[:, i] = func(rx.times, i)
                else:
                    # loop over pairs of nodes and use gaussian quadrature to integrate
                    x, w = roots_legendre(251)

                    time_nodes = wave.time_nodes
                    n_interval = len(time_nodes) - 1
                    quad_times = []
                    for i in range(n_interval):
                        b = rx.times - time_nodes[i]
                        a = rx.times - time_nodes[i + 1]
                        quad_times = (b - a)[:, None] * (x + 1) / 2.0 + a[:, None]
                        quad_scale = (b - a) / 2
                        wave_eval = wave.evalDeriv(rx.times[:, None] - quad_times)
                        for i in range(n_t):
                            A[:, i] -= np.sum(
                                quad_scale[:, None]
                                * w
                                * wave_eval
                                * func(quad_times, i),
                                axis=-1,
                            )
                if isinstance(rx, (PointMagneticFluxDensity, PointMagneticField)):
                    As.append(A @ (A_dft / omegas))
                else:
                    As.append(A @ A_dft)
                if isinstance(
                    rx, (PointMagneticFluxTimeDerivative, PointMagneticFluxDensity)
                ):
                    As[-1] *= mu_0

        # Store these on the simulation for faster future executions
        self._lambs = np.vstack(lambs)
        self._unique_lambs, inv_lambs = np.unique(self._lambs, return_inverse=True)
        self._inv_lambs = inv_lambs.reshape(self._lambs.shape)
        self._C0s = np.vstack(C0s)
        self._C1s = np.vstack(C1s)
        self._frequencies = omegas / (2 * np.pi)
        self._As = As
        self._coefficients_set = True

    def dpred(self, m, f=None):
        """
        This method evaluates the Hankel transform for each source and
        receiver and outputs it as a list. Used for computing response
        or sensitivities.
        """
        self._compute_coefficients()

        self.model = m

        C0s = self._C0s
        C1s = self._C1s
        lambs = self._lambs
        if self.hMap is not None:
            # Grab a copy
            C0s = C0s.copy()
            C1s = C1s.copy()
            h_vec = self.h
            i = 0
            for i_src, src in self.survey.source_list:
                h = h_vec[i_src]
                nD = sum(rx.locations.shape[0] for rx in src.receiver_list)
                ip1 = i + nD
                v = np.exp(-lambs[i:ip1] * h)
                C0s[i:ip1] *= v
                C1s[i:ip1] *= v
                i = ip1

        frequencies = self._frequencies
        unique_lambs = self._unique_lambs
        inv_lambs = self._inv_lambs

        sig = self.compute_sigma_matrix(frequencies)
        mu = self.compute_mu_matrix(frequencies)

        rTE = rTE_forward(frequencies, unique_lambs, sig, mu, self.thicknesses)
        rTE = rTE[:, inv_lambs]
        v = (C0s * rTE) @ self.fhtfilt.j0 + (C1s * rTE) @ self.fhtfilt.j1

        return self._project_to_data(v)

    def _project_to_data(self, v):
        As = self._As
        out = np.zeros(self.survey.nD)
        i_dat = 0
        i = 0
        for i_src, src in enumerate(self.survey.source_list):
            for i_rx, rx in enumerate(src.receiver_list):
                i_datp1 = i_dat + rx.nD
                if isinstance(rx, (PointMagneticFluxDensity, PointMagneticField)):
                    d = As[i] @ (v[:, i].imag)
                else:
                    d = As[i] @ (v[:, i].real)
                out[i_dat:i_datp1] = d
                i_dat = i_datp1
                i += 1
        return out


#######################################################################
#       STITCHED 1D SIMULATION CLASS AND GLOBAL FUNCTIONS
#######################################################################


class StitchedEM1DTMSimulation(BaseStitchedEM1DSimulation):
    def run_simulation(self, args):
        if self.verbose:
            print(">> Time-domain")
        return self._run_simulation(args)

    def _run_simulation(self, args):
        """
        This method simulates the EM response or computes the sensitivities for
        a single sounding. The method allows for parallelization of
        the stitched 1D problem.

        :param src: a EM1DTM source object
        :param topo: Topographic location (x, y, z)
        :param np.array thicknesses: np.array(N-1,) layer thicknesses for a single sounding
        :param np.array sigma: np.array(N,) layer conductivities for a single sounding
        :param np.array eta: np.array(N,) intrinsic chargeabilities for a single sounding
        :param np.array tau: np.array(N,) Cole-Cole time constant for a single sounding
        :param np.array c: np.array(N,) Cole-Cole frequency distribution constant for a single sounding
        :param np.array chi: np.array(N,) magnetic susceptibility for a single sounding
        :param np.array dchi: np.array(N,) DC susceptibility for magnetic viscosity for a single sounding
        :param np.array tau1: np.array(N,) lower time-relaxation constant for magnetic viscosity for a single sounding
        :param np.array tau2: np.array(N,) upper time-relaxation constant for magnetic viscosity for a single sounding
        :param float h: source height for a single sounding
        :param string output_type: "response", "sensitivity_sigma", "sensitivity_height"
        :param bool invert_height: boolean switch for inverting for source height
        :return: response or sensitivities

        """

        (
            source_list,
            topo,
            thicknesses,
            sigma,
            eta,
            tau,
            c,
            chi,
            dchi,
            tau1,
            tau2,
            h,
            output_type,
            invert_height,
        ) = args

        n_layer = len(thicknesses) + 1
        local_survey = Survey(source_list)
        exp_map = maps.ExpMap(nP=n_layer)

        if not invert_height:
            # Use Exponential Map
            # This is hard-wired at the moment
            sim = EM1DTMSimulation(
                survey=local_survey,
                thicknesses=thicknesses,
                sigmaMap=exp_map,
                eta=eta,
                tau=tau,
                c=c,
                chi=chi,
                dchi=dchi,
                tau1=tau1,
                tau2=tau2,
                topo=topo,
                hankel_filter="key_101_2009",
                use_sounding=True,
            )

            if output_type == "sensitivity_sigma":
                drespdsig = sim.getJ_sigma(np.log(sigma))
                return utils.mkvc(drespdsig * sim.sigmaDeriv)
            else:
                resp = sim.dpred(np.log(sigma))
                return resp
        else:

            wires = maps.Wires(("sigma", n_layer), ("h", 1))
            sigma_map = exp_map * wires.sigma
            sim = EM1DTMSimulation(
                survey=local_survey,
                thicknesses=thicknesses,
                sigmaMap=sigma_map,
                hMap=wires.h,
                topo=topo,
                eta=eta,
                tau=tau,
                c=c,
                chi=chi,
                dchi=dchi,
                tau1=tau1,
                tau2=tau2,
                hankel_filter="key_101_2009",
                use_sounding=True,
            )

            m = np.r_[np.log(sigma), h]
            if output_type == "sensitivity_sigma":
                drespdsig = sim.getJ_sigma(m)
                return utils.mkvc(drespdsig * utils.sdiag(sigma))
            elif output_type == "sensitivity_height":
                drespdh = sim.getJ_height(m)
                return utils.mkvc(drespdh)
            else:
                resp = sim.dpred(m)
                return resp
