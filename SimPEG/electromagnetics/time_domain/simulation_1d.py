from ... import maps, utils
from ..base_1d import BaseEM1DSimulation, BaseStitchedEM1DSimulation
from .sources import StepOffWaveform
from .receivers import (
    PointMagneticFluxDensity,
    PointMagneticField,
    PointMagneticFluxTimeDerivative,
)
import numpy as np
from scipy import sparse as sp

from .survey import Survey
from scipy.constants import mu_0
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
from scipy.special import roots_legendre

from empymod import filters
from empymod.transform import get_dlf_points

from geoana.kernels.tranverse_electric_reflections import rTE_forward, rTE_gradient

import properties

try:
    from multiprocessing import Pool
    from sys import platform
except ImportError:
    print("multiprocessing is not available")
    PARALLEL = False
else:
    PARALLEL = True
    import multiprocessing



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

    def get_coefficients(self):
        if self._coefficients_set is False:
            self._compute_coefficients()
        return (
            self._As,
            self._frequencies,
            self._lambs,
            self._unique_lambs,
            self._inv_lambs,
            self._C0s,
            self._C1s,
            self._W
        )

    def _set_coefficients(self, coefficients):
        self._As = coefficients[0]
        self._frequencies = coefficients[1]
        self._lambs = coefficients[2]
        self._unique_lambs = coefficients[3]
        self._inv_lambs = coefficients[4]
        self._C0s = coefficients[5]
        self._C1s = coefficients[6]
        self._W = coefficients[7]
        self._coefficients_set = True
        return

    def _compute_coefficients(self):
        if self._coefficients_set:
            return
        self._compute_hankel_coefficients()
        survey = self.survey
        t_min = np.infty
        t_max = -np.infty
        x, w = roots_legendre(251)
        # loop through source and receiver lists to find the minimum and maximum
        # evaluation times for the step response
        for i_src, src in enumerate(survey.source_list):
            for i_rx, rx in enumerate(src.receiver_list):
                wave = src.waveform
                if isinstance(wave, StepOffWaveform):
                    times = rx.times[rx.times > 0]
                    t_min = min(times.min(), t_min)
                    t_max = max(times.max(), t_max)
                else:
                    try:
                        times = rx.times - wave.time_nodes[:, None]
                        times[times < 0.0] = 0.0
                        quad_points = (times[:-1] - times[1:])[..., None] * (
                            x + 1
                        ) + times[1:, :, None]
                        t_min = min(quad_points[quad_points > 0].min(), t_min)
                        t_max = max(quad_points[quad_points > 0].max(), t_max)
                    except AttributeError:
                        raise TypeError(
                            f"Unsupported source waveform object of {src.waveform}"
                        )

        omegas, t_spline_points = get_dlf_points(self.fftfilt, np.r_[t_min, t_max], -1)
        omegas = omegas.reshape(-1)
        n_omega = len(omegas)
        n_t = len(t_spline_points)

        n_base = len(self.fftfilt.base)
        A_dft = np.zeros((n_t, n_omega))
        for i in range(n_t):
            A_dft[i, i : i + n_base] = self.fftfilt.cos * (-2.0 / np.pi)
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
                    t[
                        (t > 0.0) & (t <= t_spline_points.min())
                    ] = t_spline_points.min()  # constant at very low ts
                    out[t > 0.0] = splines[i](np.log(t[t > 0.0])) / t[t > 0.0]
                    return out

                # Then calculate the values at each time
                A = np.zeros((len(rx.times), n_t))
                if isinstance(wave, StepOffWaveform):
                    # do not need to do too much fancy here, just need to interpolate
                    # from t_spline_points to rx.times (at positive times)...
                    for i in range(n_t):
                        A[:, i] = func(rx.times, i)
                else:
                    # loop over pairs of nodes and use gaussian quadrature to integrate

                    time_nodes = wave.time_nodes
                    n_interval = len(time_nodes) - 1
                    quad_times = []
                    for i in range(n_interval):
                        b = rx.times - time_nodes[i]
                        b = np.maximum(b, 0.0)
                        a = rx.times - time_nodes[i + 1]
                        a = np.maximum(a, 0.0)
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
            Return predicted data.
            Predicted data, (`_pred`) are computed when
            self.fields is called.
        """
        if f is None:
            f = self.fields(m)

        return f

    def fields(self, m):
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

        frequencies = self._frequencies
        unique_lambs = self._unique_lambs
        inv_lambs = self._inv_lambs
        W = self._W

        sig = self.compute_complex_sigma(frequencies)
        mu = self.compute_complex_mu(frequencies)

        rTE = rTE_forward(frequencies, unique_lambs, sig, mu, self.thicknesses)
        rTE = rTE[:, inv_lambs]
        v = ((C0s * rTE) @ self.fhtfilt.j0 + (C1s * rTE) @ self.fhtfilt.j1)@W.T
        return self._project_to_data(v.T)

    def getJ(self, m, f=None):
        self.model = m
        if getattr(self, "_J", None) is None:
            self._J = {}
            self._compute_coefficients()

            C0s = self._C0s
            C1s = self._C1s
            lambs = self._lambs
            frequencies = self._frequencies
            unique_lambs = self._unique_lambs
            inv_lambs = self._inv_lambs
            # Had to convert this to an array for a tensor operation (?)
            W = self._W.toarray()

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
                v_dh_temp = W@((C0s_dh * rTE) @ self.fhtfilt.j0 + (
                    C1s_dh * rTE
                ) @ self.fhtfilt.j1)
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
                    v_ds = ((
                        (C0s * rTE_ds) @ self.fhtfilt.j0
                        + (C1s * rTE_ds) @ self.fhtfilt.j1
                    )@W.T).T
                    self._J["ds"] = self._project_to_data(v_ds)
                    if(
                        self.etaMap is None
                        and self.tauMap is None
                        and self.cMap is None
                    ):
                        # No IP effect
                        self._J["ds"] = self._project_to_data(v_ds)
                    else:
                        # With IP effect
                        ds_dsigma_inf = self.compute_dcomplex_sigma_dsigma_inf(frequencies)
                        v_ds_inf = v_ds * ds_dsigma_inf.T
                        self._J["ds"] = self._project_to_data(v_ds_inf)
                        if self.etaMap is not None:
                            ds_deta = self.compute_dcomplex_sigma_deta(frequencies)
                            v_deta = v_ds * ds_deta.T
                            self._J["deta"] = self._project_to_data(v_deta)
                        if self.tauMap is not None:
                            ds_dtau = self.compute_dcomplex_sigma_dtau(frequencies)
                            v_tau = v_ds * ds_dtau.T
                            self._J["dtau"] = self._project_to_data(v_tau)
                        if self.cMap is not None:
                            ds_dc = self.compute_dcomplex_sigma_dc(frequencies)
                            v_c = v_ds * ds_dc.T
                            self._J["dc"] = self._project_to_data(v_c)
                if self.muMap is not None:
                    rTE_dmu = rTE_dmu[..., inv_lambs]
                    v_dmu = ((
                        (C0s * rTE_ds) @ self.fhtfilt.j0
                        + (C1s * rTE_ds) @ self.fhtfilt.j1
                    )@W.T).T
                    self._J["dmu"] = self._project_to_data(v_dmu)
                if self.thicknessesMap is not None:
                    rTE_dh = rTE_dh[..., inv_lambs]
                    v_dthick = ((
                        (C0s * rTE_dh) @ self.fhtfilt.j0
                        + (C1s * rTE_dh) @ self.fhtfilt.j1
                    )@W.T).T
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
                frequencies = self._frequencies
                h = (1+1j*(frequencies/rx.lp_cutoff_frequency_1))**-rx.lp_power_1   # First order type
                h *= (1+1j*frequencies/rx.lp_cutoff_frequency_2)**-rx.lp_power_2 # First order type

                if v.ndim == 3:
                    v_slice *= h[None,:,None]
                    if isinstance(rx, (PointMagneticFluxDensity, PointMagneticField)):
                        d = np.einsum("ij,...jk->...ik", As[i_A], v_slice.imag)
                    else:
                        d = np.einsum("ij,...jk->...ik", As[i_A], v_slice.real)
                    out[i_dat:i_datp1] = d.reshape((-1, v.shape[-1]), order="F")
                else:
                    v_slice *= h[None,:]
                    if isinstance(rx, (PointMagneticFluxDensity, PointMagneticField)):
                        d = np.einsum("ij,...j->...i", As[i_A], v_slice.imag)
                    else:
                        d = np.einsum("ij,...j->...i", As[i_A], v_slice.real)
                    out[i_dat:i_datp1] = d.reshape(-1, order="F")
                i_dat = i_datp1
                i = i_p1
                i_A += 1
        return out


def run_simulation_time_domain(args):
    import os
    os.environ["MKL_NUM_THREADS"] = "1"     
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
        return_projection,
        coefficients
    ) = args

    n_layer = len(thicknesses) + 1
    local_survey = Survey(source_list)
    exp_map = maps.ExpMap(nP=n_layer)

    # Deprecate at the moment
    # if not invert_height:
    # Use Exponential Map
    # This is hard-wired at the moment
    sim = Simulation1DLayered(
        survey=local_survey,
        thicknesses=thicknesses,
        sigmaMap=exp_map,
        eta=eta,
        tau=tau,
        c=c,
        # chi=chi,
        # dchi=dchi,
        # tau1=tau1,
        # tau2=tau2,
        topo=topo,
        hankel_filter="key_101_2009",
    )

    if return_projection:
        return sim.get_coefficients()

    sim._set_coefficients(coefficients)

    if output_type == "sensitivity_sigma":
        J = sim.getJ(np.log(sigma))
        return utils.mkvc(J['ds'] * sim.sigmaDeriv)
    else:
        resp = sim.dpred(np.log(sigma))
        return resp
    # else:

    #     wires = maps.Wires(("sigma", n_layer), ("h", 1))
    #     sigma_map = exp_map * wires.sigma
    #     sim = Simulation1DLayered(
    #         survey=local_survey,
    #         thicknesses=thicknesses,
    #         sigmaMap=sigma_map,
    #         hMap=wires.h,
    #         topo=topo,
    #         eta=eta,
    #         tau=tau,
    #         c=c,
    #         # chi=chi,
    #         # dchi=dchi,
    #         # tau1=tau1,
    #         # tau2=tau2,
    #         hankel_filter="key_101_2009",
    #         use_sounding=True,
    #     )

    #     m = np.r_[np.log(sigma), h]
    #     if output_type == "sensitivity_sigma":
    #         J['ds'] = sim.getJ(m)
    #         return utils.mkvc(J['ds'] * utils.sdiag(sigma))
    #     elif output_type == "sensitivity_height":
    #         J = sim.getJ(m)
    #         return utils.mkvc(drespdh)
    #     else:
    #         resp = sim.dpred(m)
    #         return resp


#######################################################################
#       STITCHED 1D SIMULATION CLASS AND GLOBAL FUNCTIONS
#######################################################################


class Simulation1DLayeredStitched(BaseStitchedEM1DSimulation):

    survey = properties.Instance("a survey object", Survey, required=True)
    _simulation_type = 'time'

    def run_simulation(self, args):
        if self.verbose:
            print(">> Time-domain")
        return self._run_simulation(args)

    # def get_uniq_soundings(self):
    #         self._sounding_types_uniq, self._ind_sounding_uniq = np.unique(
    #             self.survey._sounding_types, return_index=True
    #         )
    # def get_coefficients(self):
    #     if self.verbose:
    #         print(">> Calculate coefficients")

    #     self.get_uniq_soundings()

    #     run_simulation = run_simulation_time_domain

    #     self._coefficients = {}
    #     for kk, ii in enumerate(self._ind_sounding_uniq):
    #         name = self._sounding_types_uniq[kk]
    #         self._coefficients[name] = run_simulation(self.input_args_for_coeff(ii))
    #     self._coefficients_set = True

    def get_coefficients(self):

        run_simulation = run_simulation_time_domain

        if self.verbose:
            print(">> Calculate coefficients")
        if self.parallel:
            pool = Pool(self.n_cpu)
            self._coefficients = pool.map(
                run_simulation,
                [
                    self.input_args_for_coeff(i) for i in range(self.n_sounding)
                ]
             )
            self._coefficients_set = True
            pool.close()
            pool.join()
        else:
            self._coefficients = [
                run_simulation(self.input_args_for_coeff(i)) for i in range(self.n_sounding)
            ]

    def forward(self, m):
        self.model = m

        if self.verbose:
            print(">> Compute response")

        # Set flat topo at zero
        if self.topo is None:
            self.set_null_topography()

        # TODO: Need to pull separate hankel coeffcients
        #       and A matrix for convolution
        #       hankel coefficients vary with variable height!

        if self._coefficients_set is False:
            self.get_coefficients()

        run_simulation = run_simulation_time_domain

        if self.parallel:
            if self.verbose:
                print ('parallel')
            #This assumes the same # of layers for each of sounding
            # if self.n_sounding_for_chunk is None:
            pool = Pool(self.n_cpu)
            result = pool.map(
                run_simulation,
                [
                    self.input_args(i, output_type='forward') for i in range(self.n_sounding)
                ]
            )

            pool.close()
            pool.join()
        else:

            result = [
                run_simulation(self.input_args(i, output_type='forward')) for i in range(self.n_sounding)
            ]

        return np.hstack(result)

    def getJ_sigma(self, m):
        """
             Compute d F / d sigma
        """
        if self._Jmatrix_sigma is not None:
            return self._Jmatrix_sigma

        self.model = m

        if self.verbose:
            print(">> Compute J sigma")

        if self._coefficients_set is False:
            self.get_coefficients()

        run_simulation = run_simulation_time_domain

        if self.parallel:
            if self.verbose:
                print(">> Start pooling")

            pool = Pool(self.n_cpu)

            # Deprecate this for now, but revisit later
            # It is an idea of chunking for parallelization
            # if self.n_sounding_for_chunk is None:
            self._Jmatrix_sigma = pool.map(
                run_simulation,
                [
                    self.input_args(i, output_type='sensitivity_sigma') for i in range(self.n_sounding)
                ]
            )


            if self.verbose:
                print(">> End pooling and form J matrix")

            self._Jmatrix_sigma = np.hstack(self._Jmatrix_sigma)
            # else:
            # self._Jmatrix_sigma = pool.map(
            #     self._run_simulation_by_chunk,
            #     [
            #         self.input_args_by_chunk(i, output_type='sensitivity_sigma') for i in range(self.n_chunk)
            #     ]
            # )
            self._Jmatrix_sigma = np.r_[self._Jmatrix_sigma].ravel()
            self._Jmatrix_sigma = sp.coo_matrix(
                (self._Jmatrix_sigma, self.IJLayers), dtype=float
            ).tocsr()
            if self.verbose:
                print(">> End forming J matrix")


        else:
            self._Jmatrix_sigma = [
                    run_simulation(self.input_args(i, output_type='sensitivity_sigma')) for i in range(self.n_sounding)
            ]
            self._Jmatrix_sigma = np.hstack(self._Jmatrix_sigma)
            self._Jmatrix_sigma = sp.coo_matrix(
                (self._Jmatrix_sigma, self.IJLayers), dtype=float
            ).tocsr()

        return self._Jmatrix_sigma

    def getJ_height(self, m):
        """
             Compute d F / d height
        """
        if self.hMap is None:
            return utils.Zero()

        if self._Jmatrix_height is not None:
            return self._Jmatrix_height
        if self.verbose:
            print(">> Compute J height")

        self.model = m

        run_simulation = run_simulation_time_domain

        if (self.parallel) & (__name__=='__main__'):
            pool = Pool(self.n_cpu)
            # if self.n_sounding_for_chunk is None:
            self._Jmatrix_height = pool.map(
                run_simulation,
                [
                    self.input_args(i, output_type="sensitivity_height") for i in range(self.n_sounding)
                ]
            )
            # else:
            # self._Jmatrix_height = pool.map(
            #     self._run_simulation_by_chunk,
            #     [
            #         self.input_args_by_chunk(i, output_type='sensitivity_height') for i in range(self.n_chunk)
            #     ]
            # )
            pool.close()
            pool.join()
            if self.parallel_jvec_jtvec is False:
                # self._Jmatrix_height = sp.block_diag(self._Jmatrix_height).tocsr()
                self._Jmatrix_height = np.hstack(self._Jmatrix_height)
                self._Jmatrix_height = sp.coo_matrix(
                    (self._Jmatrix_height, self.IJHeight), dtype=float
                ).tocsr()
        else:
            self._Jmatrix_height = [
                    run_simulation(self.input_args(i, output_type='sensitivity_height')) for i in range(self.n_sounding)
            ]
            self._Jmatrix_height = np.hstack(self._Jmatrix_height)
            self._Jmatrix_height = sp.coo_matrix(
                (self._Jmatrix_height, self.IJHeight), dtype=float
            ).tocsr()

        return self._Jmatrix_height
