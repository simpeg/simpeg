from ... import maps, utils
from ..base_1d import BaseEM1DSimulation, BaseStitchedEM1DSimulation
from .sources import StepOffWaveform
from .receivers import (
    PointMagneticFluxDensity,
    PointMagneticField,
    PointMagneticFluxTimeDerivative,
)
import numpy as np

from .survey import Survey
from scipy.constants import mu_0
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
from scipy.special import roots_legendre

from empymod import filters
from empymod.transform import get_dlf_points

from geoana.kernels.tranverse_electric_reflections import rTE_forward, rTE_gradient

import properties


class Simulation1DLayered(BaseEM1DSimulation):
    """
    Simulation class for simulating the TEM response over a 1D layered Earth
    for a single sounding.
    """

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
        self._compute_hankel_coefficients()
        survey = self.survey

        t_min = np.infty
        t_max = -np.infty
        # loop through source and receiver lists to create necessary times (and offsets)
        for i_src, src in enumerate(survey.source_list):
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
        for i_src, src in enumerate(survey.source_list):
            for i_rx, rx in enumerate(src.receiver_list):
                #######
                # Fourier Transform coefficients
                ######
                wave = src.waveform

                def func(t, i):
                    out = np.zeros_like(t)
                    t = t.copy()
                    t[(t >= 0.0) and (t <= t_min)] = t_min  # constant at very low ts
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

        sig = self.compute_complex_sigma(frequencies)
        mu = self.compute_complex_mu(frequencies)

        rTE = rTE_forward(frequencies, unique_lambs, sig, mu, self.thicknesses)
        rTE = rTE[:, inv_lambs]

        v = (C0s * rTE) @ self.fhtfilt.j0 + (C1s * rTE) @ self.fhtfilt.j1

        return self._project_to_data(v.T)

    def getJ(self, m, f=None):
        if getattr(self, "_J", None) is None:
            self._J = {}
            self._compute_coefficients()

            self.model = m

            C0s = self._C0s
            C1s = self._C1s
            lambs = self._lambs
            frequencies = self._frequencies
            unique_lambs = self._unique_lambs
            inv_lambs = self._inv_lambs

            sig = self.compute_complex_sigma(frequencies)
            mu = self.compute_complex_mu(frequencies)

            if self.hMap is not None:
                # Grab a copy
                C0s_dh = C0s.copy()
                C1s_dh = C1s.copy()
                h_vec = self.h
                i = 0
                for i_src, src in self.survey.source_list:
                    h = h_vec[i_src]
                    nD = sum(rx.locations.shape[0] for rx in src.receiver_list)
                    ip1 = i + nD
                    v = np.exp(-lambs[i:ip1] * h)
                    C0s_dh[i:ip1] *= v * -lambs[i:ip1]
                    C1s_dh[i:ip1] *= v * -lambs[i:ip1]
                    i = ip1
                    # J will be n_d * n_src (each source has it's own h)...

                rTE = rTE_forward(frequencies, unique_lambs, sig, mu, self.thicknesses)
                rTE = rTE[:, inv_lambs]
                v_dh_temp = (C0s_dh * rTE) @ self.fhtfilt.j0 + (
                    C1s_dh * rTE
                ) @ self.fhtfilt.j1
                # need to re-arange v_dh as it's currently (n_data x n_freqs)
                # however it already contains all the relevant information...
                # just need to map it from the rx index to the source index associated..
                v_dh = np.zeros((self.survey.nSrc, *v_dh_temp.shape))

                i = 0
                for i_src, src in enumerate(self.survey.source_list):
                    nD = sum(rx.locations.shape[0] for rx in src.receiver_list)
                    ip1 = i + nD
                    v_dh[i_src, i:ip1] = v_dh_temp[i:ip1]
                    i = ip1
                v_dh = np.transpose(v_dh, (1, 2, 0))
                self._J["dh"] = self._project_to_data(v_dh)

            if (
                self.sigmaMap is not None
                or self.muMap is not None
                or self.thicknessesMap is not None
            ):
                rTE_ds, rTE_dh, rTE_dmu = rTE_gradient(
                    frequencies, unique_lambs, sig, mu, self.thicknesses
                )
                if self.sigmaMap is not None:
                    rTE_ds = rTE_ds[..., inv_lambs]
                    v_ds = (
                        (C0s * rTE_ds) @ self.fhtfilt.j0
                        + (C1s * rTE_ds) @ self.fhtfilt.j1
                    ).T
                    self._J["ds"] = self._project_to_data(v_ds)
                if self.muMap is not None:
                    rTE_dmu = rTE_dmu[..., inv_lambs]
                    v_dmu = (
                        (C0s * rTE_ds) @ self.fhtfilt.j0
                        + (C1s * rTE_ds) @ self.fhtfilt.j1
                    ).T
                    self._J["dmu"] = self._project_to_data(v_dmu)
                if self.thicknessesMap is not None:
                    rTE_dh = rTE_dh[..., inv_lambs]
                    v_dthick = (
                        (C0s * rTE_dh) @ self.fhtfilt.j0
                        + (C1s * rTE_dh) @ self.fhtfilt.j1
                    ).T
                    self._J["dthick"] = self._project_to_data(v_dthick)
        return self._J

    def _project_to_data(self, v):
        As = self._As
        if v.ndim == 3:
            out = np.empty((self.survey.nD, v.shape[-1]))
        else:
            out = np.empty((self.survey.nD))
        i_dat = 0
        i_A = 0
        i = 0
        for i_src, src in enumerate(self.survey.source_list):
            for i_rx, rx in enumerate(src.receiver_list):
                i_datp1 = i_dat + rx.nD
                n_locs = rx.locations.shape[0]
                i_p1 = i + n_locs
                v_slice = v[np.arange(i, i_p1)]
                # this should order it as location changing faster than time
                # i.e. loc_1 t_1, loc_2 t_1, loc1 t2, loc2 t2
                if v.ndim == 3:
                    if isinstance(rx, (PointMagneticFluxDensity, PointMagneticField)):
                        d = np.einsum("ij,...jk->...ik", As[i_A], v_slice.imag)
                    else:
                        d = np.einsum("ij,...jk->...ik", As[i_A], v_slice.real)
                    out[i_dat:i_datp1] = d.reshape((-1, v.shape[-1]), order="F")
                else:
                    if isinstance(rx, (PointMagneticFluxDensity, PointMagneticField)):
                        d = np.einsum("ij,...j->...i", As[i_A], v_slice.imag)
                    else:
                        d = np.einsum("ij,...j->...i", As[i_A], v_slice.real)
                    out[i_dat:i_datp1] = d.reshape(-1, order="F")
                i_dat = i_datp1
                i = i_p1
                i_A += 1
        return out


#######################################################################
#       STITCHED 1D SIMULATION CLASS AND GLOBAL FUNCTIONS
#######################################################################


class Simulation1DLayeredStitched(BaseStitchedEM1DSimulation):

    survey = properties.Instance("a survey object", Survey, required=True)

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
