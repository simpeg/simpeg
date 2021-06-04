from ... import maps, utils
from ..base_1d import BaseEM1DSimulation, BaseStitchedEM1DSimulation, Sensitivity
from ..frequency_domain.sources import MagDipole, CircularLoop
from ..frequency_domain.receivers import PointMagneticFieldSecondary, PointMagneticField
from ..frequency_domain.survey import Survey
import numpy as np
import properties

# from .sources import *
# from .survey import EM1DSurveyFD
# from .supporting_functions.kernels import *
from .supporting_functions.kernels_by_sounding import (
    magnetic_dipole_response_by_sounding,
    horizontal_loop_response_by_sounding,
)
from SimPEG import Data
from empymod.transform import get_dlf_points

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
        for i_src, src in enumerate(survey.source_list):
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
                C0s.append(np.exp(-lambd * (z + h)[:, None]) * C0 / offsets[:, None])
                C1s.append(np.exp(-lambd * (z + h)[:, None]) * C1 / offsets[:, None])
                lambs.append(lambd)
                i_freq.append([i_f] * rx.locations.shape[0])

        # Store these on the simulation for faster future executions
        self._i_freq = np.hstack(i_freq)
        self._lambs = np.vstack(lambs)
        self._unique_lambs, inv_lambs = np.unique(lambs, return_inverse=True)
        self._inv_lambs = inv_lambs.reshape(self._lambs.shape)
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

        frequencies = np.array(self.survey.frequencies)
        unique_lambs = self._unique_lambs
        i_freq = self._i_freq
        inv_lambs = self._inv_lambs

        sig = self.compute_sigma_matrix(frequencies)
        mu = self.compute_mu_matrix(frequencies)

        rTE = rTE_forward(frequencies, unique_lambs, sig, mu, self.thicknesses)
        rTE = rTE[i_freq]
        rTE = np.take_along_axis(rTE, inv_lambs, axis=1)
        v = (C0s * rTE) @ self.fhtfilt.j0 + (C1s * rTE) @ self.fhtfilt.j1

        return self._project_to_data(v)

    def getJ(self, m, f=None):
        if getattr(self, "_J", None) is None:
            self._J = {}
            self._compute_coefficients()

            self.model = m

            C0s = self._C0s
            C1s = self._C1s
            lambs = self._lambs
            frequencies = np.array(self.survey.frequencies)
            unique_lambs = self._unique_lambs
            i_freq = self._i_freq
            inv_lambs = self._inv_lambs

            sig = self.compute_sigma_matrix(frequencies)
            mu = self.compute_mu_matrix(frequencies)

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
                rTE = rTE[i_freq]
                rTE = np.take_along_axis(rTE, inv_lambs, axis=1)
                v_dh_temp = (C0s_dh * rTE) @ self.fhtfilt.j0 + (
                    C1s_dh * rTE
                ) @ self.fhtfilt.j1
                # need to re-arange v_dh as it's currently (n_data x 1)
                # however it already contains all the relevant information...
                # just need to map it from the rx index to the source index associated..
                v_dh = np.zeros((self.survey.nSrc, v_dh_temp.shape[0]))

                i = 0
                for i_src, src in enumerate(self.survey.source_list):
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
                    v_ds = (
                        (C0s * rTE_ds) @ self.fhtfilt.j0
                        + (C1s * rTE_ds) @ self.fhtfilt.j1
                    ).T
                    self._J["ds"] = self._project_to_data(v_ds)
                if self.muMap is not None:
                    rTE_dmu = rTE_dmu[:, i_freq]
                    rTE_dmu = np.take_along_axis(rTE_dmu, inv_lambs[None, ...], axis=-1)
                    v_dmu = (
                        (C0s * rTE_ds) @ self.fhtfilt.j0
                        + (C1s * rTE_ds) @ self.fhtfilt.j1
                    ).T
                    self._J["dmu"] = self._project_to_data(v_dmu)
                if self.thicknessesMap is not None:
                    rTE_dh = rTE_dh[:, i_freq]
                    rTE_dh = np.take_along_axis(rTE_dh, inv_lambs[None, ...], axis=-1)
                    v_dthick = (
                        (C0s * rTE_dh) @ self.fhtfilt.j0
                        + (C1s * rTE_dh) @ self.fhtfilt.j1
                    ).T
                    self._J["dthick"] = self._project_to_data(v_dthick)
        return self._J

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

    def JTvec(self, m, v, f=None):
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

    def _project_to_data(self, v):
        i_dat = 0
        i_v = 0
        if v.ndim == 1:
            out = np.zeros(self.survey.nD)
        else:
            out = np.zeros((self.survey.nD, v.shape[1]))
        for i_src, src in enumerate(self.survey.source_list):
            for i_rx, rx in enumerate(src.receiver_list):
                i_dat_p1 = i_dat + rx.nD
                i_v_p1 = i_v + rx.locations.shape[0]
                v_slice = v[i_v:i_v_p1]

                if isinstance(rx, PointMagneticFieldSecondary):
                    if rx.data_type == "ppm":
                        if v_slice.ndim == 2:
                            v_slice /= src.hPrimary(self)[i_rx][:, None]
                        else:
                            v_slice /= src.hPrimary(self)[i_rx]
                        v_slice *= 1e6
                elif isinstance(rx, PointMagneticField):
                    if v_slice.ndim == 2:
                        pass
                        # here because it was called on sensitivity (so don't add)
                    else:
                        v_slice += src.hPrimary(self)[i_rx]

                if rx.component == "both":
                    out[i_dat:i_dat_p1:2] = v_slice.real
                    out[i_dat + 1 : i_dat_p1 : 2] = v_slice.imag
                elif rx.component == "real":
                    out[i_dat:i_dat_p1] = v_slice.real()
                elif rx.component == "complex":
                    out[i_dat:i_dat_p1] = v_slice.complex()
                i_dat = i_dat_p1
                i_v = i_v_p1
        return out


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
