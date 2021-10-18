from ... import maps, utils
from ..base_1d import BaseEM1DSimulation, BaseStitchedEM1DSimulation
from .receivers import PointMagneticFieldSecondary, PointMagneticField
from .survey import Survey
import numpy as np
import properties

from geoana.kernels.tranverse_electric_reflections import rTE_forward, rTE_gradient

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
            i_f = np.searchsorted(frequencies, src.frequency)
            for i_rx, rx in enumerate(src.receiver_list):
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
        if self.hMap is not None:
            # Grab a copy
            C0s = C0s.copy()
            C1s = C1s.copy()
            h_vec = self.h
            i = 0
            for i_src, src in enumerate(self.survey.source_list):
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

        sig = self.compute_complex_sigma(frequencies)
        mu = self.compute_complex_mu(frequencies)

        rTE = rTE_forward(frequencies, unique_lambs, sig, mu, self.thicknesses)
        rTE = rTE[i_freq]
        rTE = np.take_along_axis(rTE, inv_lambs, axis=1)
        v = (C0s * rTE) @ self.fhtfilt.j0 + (C1s * rTE) @ self.fhtfilt.j1

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

            sig = self.compute_complex_sigma(frequencies)
            mu = self.compute_complex_mu(frequencies)

            if self.hMap is not None:
                # Grab a copy
                C0s_dh = C0s.copy()
                C1s_dh = C1s.copy()
                h_vec = self.h
                i = 0
                for i_src, src in enumerate(self.survey.source_list):
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

    def _project_to_data(self, v):
        i_dat = 0
        i_v = 0
        sign = -1

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
                            v_slice /= sign*src.hPrimary(self)[i_rx][:, None]
                        else:
                            v_slice /= sign*src.hPrimary(self)[i_rx]
                        v_slice *= 1e6
                elif isinstance(rx, PointMagneticField):
                    if v_slice.ndim == 2:
                        pass
                        # here because it was called on sensitivity (so don't add)
                    else:
                        v_slice += sign*src.hPrimary(self)[i_rx]

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


#######################################################################
#       STITCHED 1D SIMULATION CLASS AND GLOBAL FUNCTIONS
#######################################################################


class Simulation1DLayeredStitched(BaseStitchedEM1DSimulation):

    survey = properties.Instance("a survey object", Survey, required=True)

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

            if output_type == "sensitivity_sigma":
                J = sim.getJ(np.log(sigma))
                return utils.mkvc(J['ds'] * sim.sigmaDeriv)
            else:
                resp = sim.dpred(np.log(sigma))
                return resp

        else:

            wires = maps.Wires(("sigma", n_layer), ("h", 1))
            sigma_map = exp_map * wires.sigma

            sim = Simulation1DLayered(
                survey=local_survey,
                thicknesses=thicknesses,
                sigmaMap=sigma_map,
                hMap=wires.h,
                topo=topo,
                eta=eta,
                tau=tau,
                c=c,
                # chi=chi,
                # dchi=dchi,
                # tau1=tau1,
                # tau2=tau2,
                hankel_filter="key_101_2009",
            )

            m = np.r_[np.log(sigma), h]
            if output_type == "sensitivity_sigma":
                J = sim.getJ(m)
                return utils.mkvc(J['ds'] * utils.sdiag(sigma))
            elif output_type == "sensitivity_height":
                J = sim.getJ(m)
                return utils.mkvc(J['dh'])
            else:
                resp = sim.dpred(m)
                return resp
