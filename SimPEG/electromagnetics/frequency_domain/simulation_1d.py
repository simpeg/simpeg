from ... import maps, utils
from ..base_1d import BaseEM1DSimulation
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
            if np.any(src.location[2] < self.topo[2]):
                raise ValueError("Source must be located above the topography")
            for i_rx, rx in enumerate(src.receiver_list):
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
            self._W,
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
            is_wire_loop = class_name == "LineCurrent1D"
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
        v = W @ ((C0s * rTE) @ self.fhtfilt.j0 + (C1s * rTE) @ self.fhtfilt.j1)

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
                    is_wire_loop = class_name == "LineCurrent1D"

                    h = h_vec[i_src]
                    if is_wire_loop:
                        nD = sum(
                            rx.locations.shape[0] * src.n_quad_points
                            for rx in src.receiver_list
                        )
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
                v_dh_temp += W @ v_dh_temp
                # need to re-arange v_dh as it's currently (n_data x 1)
                # however it already contains all the relevant information...
                # just need to map it from the rx index to the source index associated..
                v_dh = np.zeros((self.survey.nSrc, v_dh_temp.shape[0]))

                i = 0
                for i_src, src in enumerate(self.survey.source_list):
                    class_name = type(src).__name__
                    is_wire_loop = class_name == "LineCurrent1D"
                    if is_wire_loop:
                        nD = sum(
                            rx.locations.shape[0] * src.n_quad_points
                            for rx in src.receiver_list
                        )
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
                    v_ds = (
                        (
                            (C0s * rTE_ds) @ self.fhtfilt.j0
                            + (C1s * rTE_ds) @ self.fhtfilt.j1
                        )
                        @ W.T
                    ).T
                    self._J["ds"] = self._project_to_data(v_ds)
                if self.muMap is not None:
                    rTE_dmu = rTE_dmu[:, i_freq]
                    rTE_dmu = np.take_along_axis(rTE_dmu, inv_lambs[None, ...], axis=-1)
                    v_dmu = (
                        (
                            (C0s * rTE_dmu) @ self.fhtfilt.j0
                            + (C1s * rTE_dmu) @ self.fhtfilt.j1
                        )
                        @ W.T
                    ).T
                    self._J["dmu"] = self._project_to_data(v_dmu)
                if self.thicknessesMap is not None:
                    rTE_dh = rTE_dh[:, i_freq]
                    rTE_dh = np.take_along_axis(rTE_dh, inv_lambs[None, ...], axis=-1)
                    v_dthick = (
                        (
                            (C0s * rTE_dh) @ self.fhtfilt.j0
                            + (C1s * rTE_dh) @ self.fhtfilt.j1
                        )
                        @ W.T
                    ).T
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
            is_wire_loop = class_name == "LineCurrent1D"
            for i_rx, rx in enumerate(src.receiver_list):
                i_dat_p1 = i_dat + rx.nD
                i_v_p1 = i_v + rx.locations.shape[0]
                v_slice = v[i_v:i_v_p1]

                if isinstance(rx, PointMagneticFieldSecondary):
                    if rx.data_type == "ppm":
                        if is_wire_loop:
                            raise NotImplementedError(
                                "Primary field for LineCurrent1D has not been implemented"
                            )
                        if v_slice.ndim == 2:
                            v_slice /= src.hPrimary(self)[i_rx][:, None]
                        else:
                            v_slice /= src.hPrimary(self)[i_rx]
                        v_slice *= 1e6
                elif isinstance(rx, PointMagneticField):
                    if is_wire_loop:
                        raise NotImplementedError(
                            "Primary field for LineCurrent1D has not been implemented"
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
