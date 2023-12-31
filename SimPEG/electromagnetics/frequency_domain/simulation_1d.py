from ...utils import validate_type
from ..base_1d import BaseEM1DSimulation
from .receivers import PointMagneticFieldSecondary, PointMagneticField
from .sources import LineCurrent
from .survey import Survey
import numpy as np

from geoana.kernels.tranverse_electric_reflections import rTE_forward, rTE_gradient


#######################################################################
#               SIMULATION FOR A SINGLE SOUNDING
#######################################################################


class Simulation1DLayered(BaseEM1DSimulation):
    """
    Simulation class for simulating the FEM response over a 1D layered Earth
    for a single sounding.
    """

    def __init__(self, survey=None, **kwargs):
        super().__init__(survey=survey, **kwargs)
        self._frequency_map: np.ndarray | None = None

    @property
    def survey(self):
        """The simulations survey.

        Returns
        -------
        SimPEG.electromagnetics.frequency_domain.survey.Survey
        """
        if self._survey is None:
            raise AttributeError("Simulation must have a survey set")
        return self._survey

    @survey.setter
    def survey(self, value):
        if value is not None:
            value = validate_type("survey", value, Survey, cast=False)
        self._survey = value

    @property
    def frequency_map(self) -> np.ndarray:
        """Map between the receivers and frequency indices."""
        if self._frequency_map is None:
            survey = self.survey
            # loop through source and receiver lists to create offsets
            # get unique source-receiver offsets
            frequencies = np.array(survey.frequencies)
            # Compute coefficients for Hankel transform
            i_freq = []
            for src in survey.source_list:
                i_f = np.searchsorted(frequencies, src.frequency)

                for rx in src.receiver_list:
                    if isinstance(src, LineCurrent):
                        n_quad_points = src.n_segments * self.n_points_per_path
                        i_freq.append([i_f] * rx.locations.shape[0] * n_quad_points)
                    else:
                        i_freq.append([i_f] * rx.locations.shape[0])

            self._frequency_map = np.hstack(i_freq)

        return self._frequency_map

    def dpred(self, m=None, f=None):
        """
        Return predicted data.
        Predicted data, (`_pred`) are computed when
        self.fields is called.
        """
        if f is None:
            f = self.fields(m)

        return f

    def fields(self, m=None):
        """
        This method evaluates the Hankel transform for each source and
        receiver and outputs it as a list. Used for computing response
        or sensitivities.
        """
        self.model = m
        frequencies = np.array(self.survey.frequencies)
        i_freq = self.frequency_map
        sig = self.compute_complex_sigma(frequencies)
        mu = self.compute_complex_mu(frequencies)
        rTE = rTE_forward(
            frequencies,
            self.hankel_coefficients.unique_lambs,
            sig,
            mu,
            self.thicknesses,
        )
        rTE = rTE[i_freq]
        rTE = np.take_along_axis(rTE, self.hankel_coefficients.inv_lambs, axis=1)
        v = self.hankel_coefficients.W @ (
            (self.hankel_coefficients.C0s * rTE) @ self._fhtfilt.j0
            + (self.hankel_coefficients.C1s * rTE) @ self._fhtfilt.j1
        )

        return self._project_to_data(v)

    def getJ(self, m, f=None):
        self.model = m
        if getattr(self, "_J", None) is None:
            self._J = {}
            frequencies = np.array(self.survey.frequencies)
            i_freq = self.frequency_map
            sig = self.compute_complex_sigma(frequencies)
            mu = self.compute_complex_mu(frequencies)

            if self.hMap is not None:
                # It seems to be the 2 * lambs to be multiplied, but had to drop factor of 2
                C0s_dh = -self.hankel_coefficients.lambs * self.hankel_coefficients.C0s
                C1s_dh = -self.hankel_coefficients.lambs * self.hankel_coefficients.C1s
                rTE = rTE_forward(
                    frequencies,
                    self.hankel_coefficients.unique_lambs,
                    sig,
                    mu,
                    self.thicknesses,
                )
                rTE = rTE[i_freq]
                rTE = np.take_along_axis(
                    rTE, self.hankel_coefficients.inv_lambs, axis=1
                )
                v_dh_temp = (C0s_dh * rTE) @ self._fhtfilt.j0 + (
                    C1s_dh * rTE
                ) @ self._fhtfilt.j1
                v_dh_temp += self.hankel_coefficients.W @ v_dh_temp
                # need to re-arange v_dh as it's currently (n_data x 1)
                # however it already contains all the relevant information...
                # just need to map it from the rx index to the source index associated..
                v_dh = np.zeros((self.survey.nSrc, v_dh_temp.shape[0]), dtype=complex)

                i = 0
                for i_src, src in enumerate(self.survey.source_list):
                    class_name = type(src).__name__
                    is_wire_loop = class_name == "LineCurrent"
                    if is_wire_loop:
                        n_quad_points = src.n_segments * self.n_points_per_path
                        nD = sum(
                            rx.locations.shape[0] * n_quad_points
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
                    frequencies,
                    self.hankel_coefficients.unique_lambs,
                    sig,
                    mu,
                    self.thicknesses,
                )
                if self.sigmaMap is not None:
                    rTE_ds = rTE_ds[:, i_freq]
                    rTE_ds = np.take_along_axis(
                        rTE_ds, self.hankel_coefficients.inv_lambs[None, ...], axis=-1
                    )
                    v_ds = (
                        (
                            (self.hankel_coefficients.C0s * rTE_ds) @ self._fhtfilt.j0
                            + (self.hankel_coefficients.C1s * rTE_ds) @ self._fhtfilt.j1
                        )
                        @ self.hankel_coefficients.W.T
                    ).T
                    self._J["ds"] = self._project_to_data(v_ds)
                if self.muMap is not None:
                    rTE_dmu = rTE_dmu[:, i_freq]
                    rTE_dmu = np.take_along_axis(
                        rTE_dmu, self.hankel_coefficients.inv_lambs[None, ...], axis=-1
                    )
                    v_dmu = (
                        (
                            (self.hankel_coefficients.C0s * rTE_dmu) @ self._fhtfilt.j0
                            + (self.hankel_coefficients.C1s * rTE_dmu)
                            @ self._fhtfilt.j1
                        )
                        @ self.hankel_coefficients.W.T
                    ).T
                    self._J["dmu"] = self._project_to_data(v_dmu)
                if self.thicknessesMap is not None:
                    rTE_dh = rTE_dh[:, i_freq]
                    rTE_dh = np.take_along_axis(
                        rTE_dh, self.hankel_coefficients.inv_lambs[None, ...], axis=-1
                    )
                    v_dthick = (
                        (
                            (self.hankel_coefficients.C0s * rTE_dh) @ self._fhtfilt.j0
                            + (self.hankel_coefficients.C1s * rTE_dh) @ self._fhtfilt.j1
                        )
                        @ self.hankel_coefficients.W.T
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
        for src in self.survey.source_list:
            class_name = type(src).__name__
            is_wire_loop = class_name == "LineCurrent"
            for i_rx, rx in enumerate(src.receiver_list):
                i_dat_p1 = i_dat + rx.nD
                i_v_p1 = i_v + rx.locations.shape[0]
                v_slice = v[i_v:i_v_p1]

                if isinstance(rx, PointMagneticFieldSecondary):
                    if rx.data_type == "ppm":
                        if is_wire_loop:
                            raise NotImplementedError(
                                "Primary field for LineCurrent has not been implemented"
                            )
                        if v_slice.ndim == 2:
                            v_slice /= src.hPrimary(self)[i_rx][:, None]
                        else:
                            v_slice /= src.hPrimary(self)[i_rx]
                        v_slice *= 1e6
                elif isinstance(rx, PointMagneticField):
                    if is_wire_loop:
                        raise NotImplementedError(
                            "Primary field for LineCurrent has not been implemented"
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
