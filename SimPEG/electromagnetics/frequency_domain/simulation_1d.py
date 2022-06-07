from ... import maps, utils
from ..base_1d import BaseEM1DSimulation, BaseStitchedEM1DSimulation
from .receivers import PointMagneticFieldSecondary, PointMagneticField
from .survey import Survey
import numpy as np
from scipy import sparse as sp
import properties

from geoana.kernels.tranverse_electric_reflections import rTE_forward, rTE_gradient

try:
    from multiprocessing import Pool
    from sys import platform
except ImportError:
    print("multiprocessing is not available")
    PARALLEL = False
else:
    PARALLEL = True
    import multiprocessing


#######################################################################
#               SIMULATION FOR A SINGLE SOUNDING
#######################################################################


class Simulation1DLayered(BaseEM1DSimulation):
    """
    Simulation class for simulating the FEM response over a 1D layered Earth
    for a single sounding.
    """

    survey = properties.Instance("a survey object", Survey, required=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.topo is None:
            self.topo = np.array([0, 0, 0], dtype=float)
        self._coefficients_set = False

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
            self._i_freq,
            self._lambs,
            self._unique_lambs,
            self._inv_lambs,
            self._C0s,
            self._C1s,
            self._W
        )

    def _set_coefficients(self, coefficients):
        self._i_freq = coefficients[0]
        self._lambs = coefficients[1]
        self._unique_lambs = coefficients[2]
        self._inv_lambs = coefficients[3]
        self._C0s = coefficients[4]
        self._C1s = coefficients[5]
        self._W = coefficients[6]
        self._coefficients_set = True

    def _compute_coefficients(self):
        if self._coefficients_set:
            return

        self._compute_hankel_coefficients()
        survey = self.survey
        # loop through source and receiver lists to create offsets
        # get unique source-receiver offsets
        frequencies = np.array(survey.frequencies)
        # Compute coefficients for Hankel transform
        i_freq = []
        for i_src, src in enumerate(survey.source_list):
            class_name = type(src).__name__
            is_wire_loop = class_name == "PiecewiseWireLoop"
            i_f = np.searchsorted(frequencies, src.frequency)
            for i_rx, rx in enumerate(src.receiver_list):
                if is_wire_loop:
                    i_freq.append([i_f] * rx.locations.shape[0] * src.n_quad_points)
                else:
                    i_freq.append([i_f] * rx.locations.shape[0])
        self._i_freq = np.hstack(i_freq)
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
        W = self._W

        frequencies = np.array(self.survey.frequencies)
        unique_lambs = self._unique_lambs
        i_freq = self._i_freq
        inv_lambs = self._inv_lambs

        sig = self.compute_complex_sigma(frequencies)
        mu = self.compute_complex_mu(frequencies)

        rTE = rTE_forward(frequencies, unique_lambs, sig, mu, self.thicknesses)
        rTE = rTE[i_freq]
        rTE = np.take_along_axis(rTE, inv_lambs, axis=1)
        v = W@((C0s * rTE) @ self.fhtfilt.j0 + (C1s * rTE) @ self.fhtfilt.j1)

        return self._project_to_data(v)

    def getJ(self, m, f=None):
        self.model = m
        if getattr(self, "_J", None) is None:
            self._J = {}
            self._compute_coefficients()

            C0s = self._C0s
            C1s = self._C1s
            lambs = self._lambs
            frequencies = np.array(self.survey.frequencies)
            unique_lambs = self._unique_lambs
            i_freq = self._i_freq
            inv_lambs = self._inv_lambs
            W = self._W

            sig = self.compute_complex_sigma(frequencies)
            mu = self.compute_complex_mu(frequencies)

            if self.hMap is not None:
                # Grab a copy
                C0s_dh = C0s.copy()
                C1s_dh = C1s.copy()
                h_vec = self.h
                i = 0
                for i_src, src in enumerate(self.survey.source_list):
                    class_name = type(src).__name__
                    is_wire_loop = class_name == "PiecewiseWireLoop"

                    h = h_vec[i_src]
                    if is_wire_loop:
                        nD = sum(rx.locations.shape[0] * src.n_quad_points for rx in src.receiver_list)
                    else:
                        nD = sum(rx.locations.shape[0] for rx in src.receiver_list)
                    ip1 = i + nD
                    v = np.exp(-lambs[i:ip1] * h)
                    C0s_dh[i:ip1] *= v * -lambs[i:ip1]
                    C1s_dh[i:ip1] *= v * -lambs[i:ip1]
                    i = ip1
                    # J will be n_d * n_src (each source has it's own h)...

                rTE = rTE_forward(frequencies, unique_lambs, sig, mu, self.thicknesses)
                rTE = rTE[i_freq]
                rTE = np.take_along_axis(rTE, inv_lambs, axis=1)
                v_dh_temp = (C0s_dh * rTE) @ self.fhtfilt.j0 + (
                    C1s_dh * rTE
                ) @ self.fhtfilt.j1
                v_dh_temp += W@v_dh_temp
                # need to re-arange v_dh as it's currently (n_data x 1)
                # however it already contains all the relevant information...
                # just need to map it from the rx index to the source index associated..
                v_dh = np.zeros((self.survey.nSrc, v_dh_temp.shape[0]))

                i = 0
                for i_src, src in enumerate(self.survey.source_list):
                    class_name = type(src).__name__
                    is_circular_loop = class_name == "CircularLoop"
                    if is_wire_loop:
                        nD = sum(rx.locations.shape[0] * src.n_quad_points for rx in src.receiver_list)
                    else:
                        nD = sum(rx.locations.shape[0] for rx in src.receiver_list)
                    ip1 = i + nD
                    v_dh[i_src, i:ip1] = v_dh_temp[i:ip1]
                    i = ip1
                v_dh = v_dh.T
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
                    rTE_ds = rTE_ds[:, i_freq]
                    rTE_ds = np.take_along_axis(rTE_ds, inv_lambs[None, ...], axis=-1)
                    tmp = (
                        (C0s * rTE_ds) @ self.fhtfilt.j0
                        + (C1s * rTE_ds) @ self.fhtfilt.j1
                    )
                    v_ds = ((
                        (C0s * rTE_ds) @ self.fhtfilt.j0
                        + (C1s * rTE_ds) @ self.fhtfilt.j1
                    )@W.T).T
                    self._J["ds"] = self._project_to_data(v_ds)
                if self.muMap is not None:
                    rTE_dmu = rTE_dmu[:, i_freq]
                    rTE_dmu = np.take_along_axis(rTE_dmu, inv_lambs[None, ...], axis=-1)
                    v_dmu = ((
                        (C0s * rTE_ds) @ self.fhtfilt.j0
                        + (C1s * rTE_ds) @ self.fhtfilt.j1
                    )@W.T).T
                    self._J["dmu"] = self._project_to_data(v_dmu)
                if self.thicknessesMap is not None:
                    rTE_dh = rTE_dh[:, i_freq]
                    rTE_dh = np.take_along_axis(rTE_dh, inv_lambs[None, ...], axis=-1)
                    v_dthick = ((
                        (C0s * rTE_dh) @ self.fhtfilt.j0
                        + (C1s * rTE_dh) @ self.fhtfilt.j1
                    )@W.T).T
                    self._J["dthick"] = self._project_to_data(v_dthick)
        return self._J

    def _project_to_data(self, v):
        i_dat = 0
        i_v = 0
        if v.ndim == 1:
            out = np.zeros(self.survey.nD)
        else:
            out = np.zeros((self.survey.nD, v.shape[1]))
        for i_src, src in enumerate(self.survey.source_list):
            class_name = type(src).__name__
            is_wire_loop = class_name == "PiecewiseWireLoop"
            for i_rx, rx in enumerate(src.receiver_list):
                i_dat_p1 = i_dat + rx.nD
                i_v_p1 = i_v + rx.locations.shape[0]
                v_slice = v[i_v:i_v_p1]

                if isinstance(rx, PointMagneticFieldSecondary):
                    if rx.data_type == "ppm":
                        if is_wire_loop:
                            raise NotImplementedError(
                                "Primary field for PiecewiseWireLoop has not been implemented"
                            )
                        if v_slice.ndim == 2:
                            v_slice /= src.hPrimary(self)[i_rx][:, None]
                        else:
                            v_slice /= src.hPrimary(self)[i_rx]
                        v_slice *= 1e6
                elif isinstance(rx, PointMagneticField):
                    if is_wire_loop:
                        raise NotImplementedError(
                            "Primary field for PiecewiseWireLoop has not been implemented"
                        )
                    if v_slice.ndim == 2:
                        pass
                        # here because it was called on sensitivity (so don't add)
                    else:
                        v_slice += src.hPrimary(self)[i_rx]

                if rx.component == "both":
                    out[i_dat:i_dat_p1:2] = v_slice.real
                    out[i_dat + 1 : i_dat_p1 : 2] = v_slice.imag
                elif rx.component == "real":
                    out[i_dat:i_dat_p1] = v_slice.real
                elif rx.component == "imag":
                    out[i_dat:i_dat_p1] = v_slice.imag
                i_dat = i_dat_p1
                i_v = i_v_p1
        return out


def run_simulation_frequency_domain(args):
    """
    This method simulates the EM response or computes the sensitivities for
    a single sounding. The method allows for parallelization of
    the stitched 1D problem.
    :param src: a EM1DFM source object
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
        src_list,
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
    local_survey = Survey(src_list)
    exp_map = maps.ExpMap(nP=n_layer)

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
    #     )

    #     m = np.r_[np.log(sigma), h]
    #     if output_type == "sensitivity_sigma":
    #         J = sim.getJ(m)
    #         return utils.mkvc(J['ds'] * utils.sdiag(sigma))
    #     elif output_type == "sensitivity_height":
    #         J = sim.getJ(m)
    #         return utils.mkvc(J['dh'])
    #     else:
    #         resp = sim.dpred(m)
    #         return resp

#######################################################################
#       STITCHED 1D SIMULATION CLASS AND GLOBAL FUNCTIONS
#######################################################################


class Simulation1DLayeredStitched(BaseStitchedEM1DSimulation):
    _simulation_type = 'frequency'
    survey = properties.Instance("a survey object", Survey, required=True)

    def run_simulation(self, args):
        if self.verbose:
            print(">> Frequency-domain")
        return self._run_simulation(args)

    def dot(self, args):
        return np.dot(args[0], args[1])

    def forward(self, m):
        self.model = m

        if self.verbose:
            print(">> Compute response")

        # Set flat topo at zero
        if self.topo is None:
            self.set_null_topography()

        run_simulation = run_simulation_frequency_domain

        if self.parallel:
            if self.verbose:
                print ('parallel')

            #This assumes the same # of layers for each of sounding
            if self._coefficients_set is False:
                if self.verbose:
                    print(">> Calculate coefficients")
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

            # if self.n_sounding_for_chunk is None:
            pool = Pool(self.n_cpu)
            result = pool.map(
                run_simulation,
                [
                    self.input_args(i, output_type='forward') for i in range(self.n_sounding)
                ]
            )
            # else:
            #     result = pool.map(
            #         self._run_simulation_by_chunk,
            #         [
            #             self.input_args_by_chunk(i, output_type='forward') for i in range(self.n_chunk)
            #         ]
            #     )
            #     return np.r_[result].ravel()

            pool.close()
            pool.join()
        else:
            if self._coefficients_set is False:
                if self.verbose:
                    print(">> Calculate coefficients")

                self._coefficients = [
                    run_simulation(self.input_args_for_coeff(i)) for i in range(self.n_sounding)
                ]
                self._coefficients_set = True

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
        if self.verbose:
            print(">> Compute J sigma")
        self.model = m

        run_simulation = run_simulation_frequency_domain

        if self.parallel:

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
            self._Jmatrix_sigma = np.hstack(self._Jmatrix_sigma)
            # else:
            # self._Jmatrix_sigma = pool.map(
            #     self._run_simulation_by_chunk,
            #     [
            #         self.input_args_by_chunk(i, output_type='sensitivity_sigma') for i in range(self.n_chunk)
            #     ]
            # )
            self._Jmatrix_sigma = np.r_[self._Jmatrix_sigma].ravel()
            pool.close()
            pool.join()

            self._Jmatrix_sigma = sp.coo_matrix(
                (self._Jmatrix_sigma, self.IJLayers), dtype=float
            ).tocsr()

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

        run_simulation = run_simulation_frequency_domain

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