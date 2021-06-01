from ... import maps, utils
from ..base_1d import BaseEM1DSimulation, BaseStitchedEM1DSimulation, Sensitivity
from ..frequency_domain.sources import MagDipole, CircularLoop
from ..frequency_domain.receivers import PointMagneticFieldSecondary
from ..frequency_domain.survey import Survey
import numpy as np
import properties

# from .sources import *
# from .survey import EM1DSurveyFD
from .supporting_functions.kernels import *
from .supporting_functions.kernels_by_sounding import (
    magnetic_dipole_response_by_sounding,
    horizontal_loop_response_by_sounding,
)
from SimPEG import Data
from empymod.utils import check_time
from empymod import filters
from empymod.transform import dlf, fourier_dlf, get_dlf_points
from empymod.utils import check_hankel

from geoana.kernels.tranverse_electric_reflections import rTE_forward, rTE_gradient
from scipy.constants import mu_0

#######################################################################
#               SIMULATION FOR A SINGLE SOUNDING
#######################################################################


class EM1DFMSimulation(BaseEM1DSimulation):
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

        # loop through source and receiver lists to create offsets
        # get unique source-receiver offsets
        frequencies = np.array(survey.frequencies)
        # Compute coefficients for Hankel transform
        C0s = []
        C1s = []
        i_freq = []
        lambs = []
        for i_src, src in enumerate(self.survey.source_list):
            if isinstance(src, CircularLoop):
                if np.any(src.orientation[:-1] != 0.0):
                    raise ValueError("Can only simulate horizontal circular loops")
            i_f = np.searchsorted(frequencies, src.frequency)

            h = h_vector[i_src]  # source height above topo
            src_x, src_y, src_z = src.orientation * src.moment / (4 * np.pi)
            # src.moment is pi * radius**2 * I for circular loop
            for i_rx, rx in enumerate(src.receiver_list):
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

                lambd, _ = get_dlf_points(
                    self.fhtfilt, offsets, self.hankel_pts_per_dec
                )
                C0 = 0.0
                C1 = 0.0
                if isinstance(src, CircularLoop):
                    # I * a/ 2 * (lambda **2 )/ (lambda)
                    C0 += src_z * (2 / src.radius) * lambd
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
                C0s.append(np.exp(-lambd * (z + h)[:, None]) * C0 / offsets)
                C1s.append(np.exp(-lambd * (z + h)[:, None]) * C1 / offsets)
                lambs.append(lambd)
                i_freq.append([i_f] * rx.locations.shape[0])

        # Store these on the simulation for faster future executions
        self._i_freq = np.hstack(i_freq)
        self._lambs = np.vstack(lambs)
        self._unique_lambs, inv_lambs = np.unique(lambs, return_inverse=True)
        self._inv_lambs = inv_lambs.reshape(lambs.shape)
        self._C0s = np.vstack(C0s)
        self._C1s = np.vstack(C1s)
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

        frequencies = self.survey.frequencies
        unique_lambs = self._unique_lambs
        i_freq = self._i_freq
        inv_lambs = self._inv_lambs

        sig = self.compute_sigma_matrix(frequencies)
        chi = self.compute_chi_matrix(frequencies)

        rTE = rTE_forward(
            frequencies, unique_lambs, sig, (chi + 1) * mu_0, self.thicknesses
        )
        rTE = rTE[i_freq]
        rTE = np.take_along_axis(rTE, inv_lambs, axis=1)
        v = (C0s * rTE) @ self.fhtfilt.j0 + (C1s * rTE) @ self.fhtfilt.j1

        i_dat = 0
        i_v = 0
        out = np.zeros(self.survey.nD)
        for i_src, src in enumerate(self.survey.source_list):
            for i_rx, rx in enumerate(src.receiver_list):
                i_dat_p1 = i_dat + rx.nD
                i_v_p1 = i_v + rx.locations.shape[0]
                v_slice = v[i_v:i_v_p1]
                if rx.component == "both":
                    out[i_dat:i_dat_p1] = v_slice.view(np.float64)
                elif rx.component == "real":
                    out[i_dat:i_dat_p1] = v_slice[i_v:i_v_p1].real()
                elif rx.component == "complex":
                    out[i_dat:i_dat_p1] = v_slice[i_v:i_v_p1].complex()
        return out

    def Jvec(self, m, v, f=None):
        self._compute_coefficients()

        self.model = m

        C0s = self._C0s
        C1s = self._C1s
        lambs = self._lambs
        frequencies = self.survey.frequencies
        unique_lambs = self._unique_lambs
        i_freq = self._i_freq
        inv_lambs = self._inv_lambs

        sig = self.compute_sigma_matrix(frequencies)
        chi = self.compute_chi_matrix(frequencies)

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

            rTE = rTE_forward(
                frequencies, unique_lambs, sig, (chi + 1) * mu_0, self.thicknesses
            )
            rTE = rTE[i_freq]
            rTE = np.take_along_axis(rTE, inv_lambs, axis=1)
            v_dh_temp = (C0s_dh * rTE) @ self.fhtfilt.j0 + (
                C1s_dh * rTE
            ) @ self.fhtfilt.j1
            # need to re-arange v_dh as it's currently (n_data x 1)
            # however it already contains all the relevant information...
            # just need to map it from the rx index to the source index associated..
            v_dh = np.zeros(self.survey.nSrc, v_dh_temp.shape[0])

            i = 0
            for i_src, src in enumerate(self.survey.source_list):
                nD = sum(rx.locations.shape[0] for rx in src.receiver_list)
                ip1 = i + nD
                v_dh[i_src, i:ip1] = v_dh_temp[i:ip1]
                i = ip1

        if (
            self.sigmaMap is not None
            or self.chiMap is not None
            or self.thicknessesMap is not None
        ):
            rTE_ds, rTE_dh, rTE_dmu = rTE_gradient(
                frequencies, unique_lambs, sig, (chi + 1) * mu_0, self.thicknesses
            )
            if self.sigmaMap is not None:
                rTE_ds = rTE_ds[:, i_freq]
                rTE_ds = np.take_along_axis(rTE_ds, inv_lambs[None, ...], axis=-1)
                v_ds = (
                    (C0s * rTE_ds) @ self.fhtfilt.j0 + (C1s * rTE_ds) @ self.fhtfilt.j1
                ).T
            if self.chiMap is not None:
                rTE_dmu = rTE_dmu[:, i_freq]
                rTE_dmu = np.take_along_axis(rTE_dmu, inv_lambs[None, ...], axis=-1)
                v_dmu = (
                    (C0s * rTE_ds) @ self.fhtfilt.j0 + (C1s * rTE_ds) @ self.fhtfilt.j1
                ).T
                v_dchi = mu_0 * (v_dmu @ self.chiMap)
            if self.thicknessesMap is not None:
                rTE_dh = rTE_dh[:, i_freq]
                rTE_dh = np.take_along_axis(rTE_dh, inv_lambs[None, ...], axis=-1)
                v_dthick = (
                    (C0s * rTE_dh) @ self.fhtfilt.j0 + (C1s * rTE_dh) @ self.fhtfilt.j1
                ).T

    def project_fields(self, u, output_type="response"):
        """
        Project from the list of Hankel transform evaluations to the data or sensitivities.
        Data can be real or imaginary component of: total field, secondary field or ppm.

        :param list u: list containing Hankel transform outputs for each unique
        source-receiver pair.
        :rtype: list: list containing predicted data for each unique
        source-receiver pair.
        :return: predicted data or sensitivities by source-receiver
        """

        COUNT = 0
        for i_src, src in enumerate(self.survey.source_list):
            for i_rx, rx in enumerate(src.receiver_list):

                u_temp = u[COUNT]
                if rx.component == "real":
                    u_temp = np.real(u_temp)
                elif rx.component == "imag":
                    u_temp = np.imag(u_temp)
                elif rx.component == "both":
                    u_temp_r = np.real(u_temp)
                    u_temp_i = np.imag(u_temp)
                    if output_type == "sensitivity_sigma":
                        u_temp = np.vstack((u_temp_r, u_temp_i))
                    else:
                        u_temp = np.r_[u_temp_r, u_temp_i]
                else:
                    raise Exception()

                if isinstance(rx, PointMagneticFieldSecondary):

                    if rx.data_type == "ppm":
                        u_primary = src.hPrimary1D(
                            rx.locations, rx.use_source_receiver_offset
                        )
                        k = [comp == rx.orientation for comp in ["x", "y", "z"]]
                        u_temp = 1e6 * u_temp / u_primary[0, k]

                elif isinstance(rx, PointMagneticField):
                    u_primary = src.hPrimary1D(
                        rx.locations, rx.use_source_receiver_offset
                    )
                    if rx.component == "both":
                        if output_type == "sensitivity_sigma":
                            u_temp = np.vstack((u_temp_r + u_primary, u_temp_i))
                        else:
                            u_temp = np.r_[u_temp_r + u_primary, u_temp_i]

                    else:
                        u_temp = +u_primary
                else:
                    raise Exception()

                u[COUNT] = u_temp
                COUNT = COUNT + 1

        return u

    def compute_integral_by_sounding(self, m, output_type="response"):
        """
        This method evaluates the Hankel transform for each source and
        receiver and outputs it as a list. Used for computing response
        or sensitivities.
        """

        self.model = m
        n_layer = self.n_layer
        n_filter = self.n_filter

        # Define source height above topography by mapping or from sources and receivers.
        # Issue: this only works for a single source.

        integral_output_list = []

        source_location_by_sounding_dict = self.survey.source_location_by_sounding_dict
        if output_type == "sensitivity_sigma":
            data_or_sensitivity = Sensitivity(self.survey, M=n_layer)
        else:
            data_or_sensitivity = Data(self.survey)

        for i_sounding in source_location_by_sounding_dict:
            src_locations = np.vstack(source_location_by_sounding_dict[i_sounding])
            rx_locations = self.survey.receiver_location_by_sounding_dict[i_sounding]
            rx_use_offset = self.survey.receiver_use_offset_by_sounding_dict[i_sounding]

            n_filter = self.n_filter
            n_rx = self.survey.vnrx_by_sounding_dict[i_sounding]
            frequencies = self.survey.frequency_by_sounding_dict[i_sounding]

            # Create globally, not for each receiver in the future
            sig = self.compute_sigma_matrix(frequencies)
            chi = self.compute_chi_matrix(frequencies)

            # Compute receiver height
            # Assume all sources in i-th sounding have the same src.location
            if self.hMap is not None:
                h = self.h
            else:
                h = src_locations[0, 2] - self.topo[-1]

            # Assume all receivers in i-th sounding have the same receiver height
            if rx_use_offset[0]:
                z = h + rx_locations[0, 2]
            else:
                z = h + rx_locations[0, 2] - src_locations[0, 2]

            # Assume all receivers in i-th sounding have the same rx.use_source_receiver_offset.
            # But, their Radial distance can be different.
            if rx_use_offset[0]:
                r = rx_locations[:, 0:2]
            else:
                r = rx_locations[:, 0:2] - src_locations[0, 0:2]

            r = np.sqrt(np.sum(r ** 2, axis=1))
            radial_distance = np.unique(r)
            if len(radial_distance) > 1:
                raise Exception(
                    "The receiver offsets in the sounding should be the same."
                )
            # Assume all sources in i-th sounding have the same type
            source_list = self.survey.get_sources_by_sounding_number(i_sounding)
            src = source_list[0]

            if isinstance(src, CircularLoop):
                # Assume all sources in i-th sounding have the same radius
                a = np.array([src.radius])
                # Use function from empymod to define Hankel coefficients.
                # Size of lambd is (1 x n_filter)
                lambd = np.empty([1, n_filter], order="F")
                lambd[:, :], _ = get_dlf_points(
                    self.fhtfilt, a, self.hankel_pts_per_dec
                )

                data_or_sensitivity = horizontal_loop_response_by_sounding(
                    self,
                    lambd,
                    frequencies,
                    n_layer,
                    sig,
                    chi,
                    a,
                    h,
                    z,
                    source_list,
                    data_or_sensitivity,
                    output_type=output_type,
                )

            elif isinstance(src, MagDipole):

                # Use function from empymod to define Hankel coefficients.
                # Size of lambd is (1 x n_filter)
                lambd = np.empty([1, n_filter], order="F")
                lambd[:, :], _ = get_dlf_points(
                    self.fhtfilt, radial_distance, self.hankel_pts_per_dec
                )
                data_or_sensitivity = magnetic_dipole_response_by_sounding(
                    self,
                    lambd,
                    frequencies,
                    n_layer,
                    sig,
                    chi,
                    h,
                    z,
                    source_list,
                    data_or_sensitivity,
                    radial_distance,
                    output_type=output_type,
                )
            return data_or_sensitivity

    def project_fields_src_rx(self, u, i_sounding, src, rx, output_type="response"):
        """
        Project from the list of Hankel transform evaluations to the data or sensitivities.
        Data can be real or imaginary component of: total field, secondary field or ppm.

        :param list u: list containing Hankel transform outputs for each unique
        source-receiver pair.
        :rtype: list: list containing predicted data for each unique
        source-receiver pair.
        :return: predicted data or sensitivities by source-receiver
        """

        if rx.component == "real":
            data = np.atleast_1d(np.real(u))
        elif rx.component == "imag":
            data = np.atleast_1d(np.imag(u))
        elif rx.component == "both":
            data_r = np.real(u)
            data_i = np.imag(u)
            if output_type == "sensitivity_sigma":
                data = np.vstack((data_r, data_i))
            else:
                data = np.r_[data_r, data_i]
        else:
            raise Exception()

        if isinstance(rx, PointMagneticFieldSecondary):

            if rx.data_type == "ppm":
                data_primary = src.hPrimary1D(
                    rx.locations, rx.use_source_receiver_offset
                )
                k = [comp == rx.orientation for comp in ["x", "y", "z"]]
                data = 1e6 * data / data_primary[0, k]

        elif isinstance(rx, PointMagneticField):
            data_primary = src.hPrimary1D(rx.locations, rx.use_source_receiver_offset)
            if rx.component == "both":
                if output_type == "sensitivity_sigma":
                    data = np.vstack((data_r + data_primary, data_i))
                else:
                    data = np.r_[data_r + data_primary, data_i]
            else:
                data = +data_primary
        else:
            raise Exception()

        return data


#######################################################################
#       STITCHED 1D SIMULATION CLASS AND GLOBAL FUNCTIONS
#######################################################################


class StitchedEM1DFMSimulation(BaseStitchedEM1DSimulation):
    def run_simulation(self, args):
        if self.verbose:
            print(">> Frequency-domain")
        return self._run_simulation(args)

    def dot(self, args):
        return np.dot(args[0], args[1])

    def _run_simulation(self, args):
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
        ) = args

        n_layer = len(thicknesses) + 1
        local_survey = Survey(src_list)
        exp_map = maps.ExpMap(nP=n_layer)

        if not invert_height:
            # Use Exponential Map
            # This is hard-wired at the moment

            sim = EM1DFMSimulation(
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

            sim = EM1DFMSimulation(
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
                # return utils.mkvc(drespdsig)
            elif output_type == "sensitivity_height":
                drespdh = sim.getJ_height(m)
                return utils.mkvc(drespdh)
            else:
                resp = sim.dpred(m)
                return resp
