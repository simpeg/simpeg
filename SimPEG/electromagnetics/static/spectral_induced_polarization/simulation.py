import numpy as np
import sys
import gc

from .... import props
from .data import Data
from ....utils import sdiag, validate_type, validate_active_indices
import scipy.sparse as sp

from ..induced_polarization.simulation import BaseIPSimulation
from ..induced_polarization import (
    Simulation3DCellCentered as BaseSimulation3DCellCentered,
)
from ..induced_polarization import Simulation3DNodal as BaseSimulation3DNodal
from .survey import Survey


class BaseSIPSimulation(BaseIPSimulation):
    tau, tauMap, tauDeriv = props.Invertible("Time constant (s)")
    taui, tauiMap, tauiDeriv = props.Invertible("Inverse of time constant (1/s)")
    props.Reciprocal(tau, taui)

    c, cMap, cDeriv = props.Invertible("Frequency dependency")

    Ainv = None
    _f = None
    _Jmatrix = None
    _dc_voltage_set = False

    _eta_store = None
    _taui_store = None
    _c_store = None
    _pred = None

    def __init__(
        self,
        mesh,
        survey=None,
        tau=0.1,
        tauMap=None,
        taui=None,
        tauiMap=None,
        c=0.5,
        cMap=None,
        storeJ=False,
        actinds=None,
        storeInnerProduct=True,
        **kwargs,
    ):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.tau = tau
        self.taui = taui
        self.tauMap = tauMap
        self.tauiMap = tauiMap
        self.c = c
        self.cMap = cMap
        self.storeJ = storeJ
        self.storeInnerProduct = storeInnerProduct
        self.actinds = actinds

    @property
    def survey(self):
        """The SIP survey object.

        Returns
        -------
        simpeg.electromagnetics.static.spectral_induced_polarization.survey.Survey
        """
        if self._survey is None:
            raise AttributeError("Simulation must have a survey")
        return self._survey

    @survey.setter
    def survey(self, value):
        if value is not None:
            value = validate_type("survey", value, Survey, cast=False)
        self._survey = value

    @property
    def actinds(self):
        """Active indices when storing J.

        Returns
        -------
        (mesh.n_cells) numpy.ndarray of bool
        """
        return self._actinds

    @actinds.setter
    def actinds(self, value):
        if self.storeJ:
            if value is None:
                value = np.ones(self.mesh.n_cells, dtype=bool)
            value = validate_active_indices("actinds", value, self.mesh.n_cells)
            self._actinds = value
            self._P = sp.eye(self.mesh.n_cells, format="csr")[:, value]

    @property
    def storeInnerProduct(self):
        """Whether to store inner product matrices

        Returns
        -------
        bool
        """
        return self._storeInnerProduct

    @storeInnerProduct.setter
    def storeInnerProduct(self, value):
        self._storeInnerProduct = validate_type("storeInnerProduct", value, bool)

    @property
    def storeJ(self):
        """Whether to store the sensitivity matrix

        Returns
        -------
        bool
        """
        return self._storeJ

    @storeJ.setter
    def storeJ(self, value):
        self._storeJ = validate_type("storeJ", value, bool)

    @property
    def n(self):
        return self.mesh.n_nodes

    @property
    def sigmaDeriv(self):
        if self.storeJ:
            dsigma_dlogsigma = sdiag(self.sigma) * self._P
        else:
            dsigma_dlogsigma = sdiag(self.sigma)
        return -dsigma_dlogsigma

    @property
    def rhoDeriv(self):
        if self.storeJ:
            drho_dlogrho = sdiag(self.rho) * self._P
        else:
            drho_dlogrho = sdiag(self.rho)
        return drho_dlogrho

    @property
    def etaDeriv_store(self):
        if getattr(self, "_etaDeriv_store", None) is None:
            self._etaDeriv_store = self.etaDeriv
        return self._etaDeriv_store

    @property
    def tauiDeriv_store(self):
        if getattr(self, "_tauiDeriv_store", None) is None:
            self._tauiDeriv_store = self.tauiDeriv
        return self._tauiDeriv_store

    @property
    def tauDeriv_store(self):
        if getattr(self, "_tauDeriv_store", None) is None:
            self._tauDeriv_store = self.tauDeriv
        return self._tauDeriv_store

    @property
    def cDeriv_store(self):
        if getattr(self, "_cDeriv_store", None) is None:
            self._cDeriv_store = self.cDeriv
        return self._cDeriv_store

    def get_t_over_tau(self, t):
        taui = self._taui_store
        return t * taui

    def get_exponent(self, t):
        c = self._c_store
        t_over_tau = self.get_t_over_tau(t)
        return (t_over_tau) ** c

    def get_peta_step_off(self, exponent):
        eta = self._eta_store
        peta = eta * np.exp(-exponent)
        return peta

    def get_peta_pulse_off(self, t):
        """
        Compute pseudo-chargeability from a single pulse waveform
        """
        T = self.survey.T
        exponent_0 = self.get_exponent(t)
        exponent_1 = self.get_exponent(t + T / 4.0)
        peta = self.get_peta_step_off(exponent_0) - self.get_peta_step_off(exponent_1)
        return peta

    def get_multi_pulse_response(self, t, pulse_func):
        n_pulse = self.survey.n_pulse
        T = self.survey.T
        peta = np.zeros(self._eta_store.shape[0], float, order="C")
        for i_pulse in range(n_pulse):
            factor = (-1) ** i_pulse * (n_pulse - i_pulse)
            peta += pulse_func(t + T / 2 * i_pulse) * factor
        return peta / n_pulse

    def get_peta(self, t):
        n_pulse = self.survey.n_pulse
        if n_pulse == 0:
            exponent = self.get_exponent(t)
            return self.get_peta_step_off(exponent)
        else:
            return self.get_multi_pulse_response(t, self.get_peta_pulse_off)

    def get_peta_eta_deriv_step_off(self, exponent):
        return np.exp(-exponent)

    def get_peta_eta_deriv_pulse_off(self, t):
        """
        Compute derivative of pseudo-chargeability w.r.t eta from a single pulse waveform
        """
        T = self.survey.T
        exponent_0 = self.get_exponent(t)
        exponent_1 = self.get_exponent(t + T / 4.0)
        peta_eta_deriv = self.get_peta_eta_deriv_step_off(
            exponent_0
        ) - self.get_peta_eta_deriv_step_off(exponent_1)
        return peta_eta_deriv

    def get_peta_eta_deriv(self, t):
        n_pulse = self.survey.n_pulse
        if n_pulse == 0:
            exponent = self.get_exponent(t)
            return self.get_peta_eta_deriv_step_off(exponent)
        else:
            return self.get_multi_pulse_response(t, self.get_peta_eta_deriv_pulse_off)

    def PetaEtaDeriv(self, t, v, adjoint=False):
        etaDeriv = self.etaDeriv_store

        v = np.array(v, dtype=float)
        dpetadeta = self.get_peta_eta_deriv(t)

        if adjoint:
            if v.ndim == 1:
                return etaDeriv.T * (dpetadeta * v)
            else:
                return etaDeriv.T * (sdiag(dpetadeta) * v)
        else:
            return dpetadeta * (etaDeriv * v)

    def get_peta_taui_deriv_step_off(self, exponent):
        eta = self._eta_store
        taui = self._taui_store
        c = self._c_store
        peta_taui_deriv = -c * eta / taui * exponent * np.exp(-exponent)
        return peta_taui_deriv

    def get_peta_taui_deriv_pulse_off(self, t):
        """
        Compute derivative of pseudo-chargeability w.r.t eta from a single pulse waveform
        """
        T = self.survey.T
        exponent_0 = self.get_exponent(t)
        exponent_1 = self.get_exponent(t + T / 4.0)
        peta_taui_deriv = self.get_peta_taui_deriv_step_off(
            exponent_0
        ) - self.get_peta_taui_deriv_step_off(exponent_1)
        return peta_taui_deriv

    def get_peta_taui_deriv(self, t):
        n_pulse = self.survey.n_pulse
        if n_pulse == 0:
            exponent = self.get_exponent(t)
            return self.get_peta_taui_deriv_step_off(exponent)
        else:
            return self.get_multi_pulse_response(t, self.get_peta_taui_deriv_pulse_off)

    def PetaTauiDeriv(self, t, v, adjoint=False):
        v = np.array(v, dtype=float)
        tauiDeriv = self.tauiDeriv_store
        dpetadtaui = self.get_peta_taui_deriv(t)

        if adjoint:
            if v.ndim == 1:
                return tauiDeriv.T * (dpetadtaui * v)
            else:
                return tauiDeriv.T * (sdiag(dpetadtaui) * v)
        else:
            return dpetadtaui * (tauiDeriv * v)

    def get_peta_c_deriv_step_off(self, exponent, t_over_tau):
        eta = self._eta_store
        peta_c_deriv = -eta * (exponent) * np.exp(-exponent) * np.log(t_over_tau)
        return peta_c_deriv

    def get_peta_c_deriv_pulse_off(self, t):
        """
        Compute derivative of pseudo-chargeability w.r.t eta from a single pulse waveform
        """
        T = self.survey.T
        exponent_0 = self.get_exponent(t)
        exponent_1 = self.get_exponent(t + T / 4.0)
        t_over_tau_0 = self.get_t_over_tau(t)
        t_over_tau_1 = self.get_t_over_tau(t + T / 4.0)

        peta_c_deriv = self.get_peta_c_deriv_step_off(
            exponent_0, t_over_tau_0
        ) - self.get_peta_c_deriv_step_off(exponent_1, t_over_tau_1)
        return peta_c_deriv

    def get_peta_c_deriv(self, t):
        n_pulse = self.survey.n_pulse
        if n_pulse == 0:
            exponent = self.get_exponent(t)
            t_over_tau = self.get_t_over_tau(t)
            return self.get_peta_c_deriv_step_off(exponent, t_over_tau)
        else:
            return self.get_multi_pulse_response(t, self.get_peta_c_deriv_pulse_off)

    def PetaCDeriv(self, t, v, adjoint=False):
        v = np.array(v, dtype=float)
        cDeriv = self.cDeriv_store
        dpetadc = self.get_peta_c_deriv(t)
        if adjoint:
            if v.ndim == 1:
                return cDeriv.T * (dpetadc * v)
            else:
                return cDeriv.T * (sdiag(dpetadc) * v)
        else:
            return dpetadc * (cDeriv * v)

    def fields(self, m):
        if self.verbose:
            print(">> Compute DC fields")
        if self._f is None:
            # re-uses the DC simulation's fields method
            # This grabs the class just above BasIPSimulation
            self._f = super(BaseIPSimulation, self).fields(None)

        if not self._dc_voltage_set:
            try:
                f = self.fields_to_space(self._f)
            except AttributeError:
                f = self._f
            # loop through receievers to check if they need to set the _dc_voltage
            for src in self.survey.source_list:
                for rx in src.receiver_list:
                    if rx.data_type == "apparent_chargeability":
                        rx.data_type = "volt"
                        rx._dc_voltage = rx.eval(src, self.mesh, f)
                        if rx.storeProjections:
                            P = rx._Ps[self.mesh]
                            rx._Ps[self.mesh] = sdiag(1.0 / rx.dc_voltage) * P
                        rx.data_type = "apparent_chargeability"

        self._pred = self.forward(m, f=self._f)

        return self._f

    # @profile
    def getJ(self, m, f=None):
        """
        Generate Full sensitivity matrix
        """

        if self._Jmatrix is not None:
            return self._Jmatrix
        else:
            if self.verbose:
                print("Calculating J and storing")
            if f is None:
                f = self.fields(m)

            Jt = np.zeros(
                (self._P.shape[1], int(self.survey.nD / self.survey.unique_times.size)),
                order="F",
            )
            istrt = int(0)
            iend = int(0)

            for isrc, src in enumerate(self.survey.source_list):
                if self.verbose:
                    sys.stdout.write(("\r %d / %d") % (isrc + 1, self.survey.nSrc))
                    sys.stdout.flush()
                u_src = f[src, self._solutionType]
                for rx in src.receiver_list:
                    P = rx.getP(self.mesh, rx.projGLoc(f)).toarray()
                    ATinvdf_duT = self.Ainv * (P.T)
                    dA_dmT = self.getADeriv(u_src, ATinvdf_duT, adjoint=True)
                    iend = istrt + rx.nD
                    if rx.nD == 1:
                        Jt[:, istrt] = -dA_dmT
                    else:
                        Jt[:, istrt:iend] = -dA_dmT
                    istrt += rx.nD

            self._Jmatrix = Jt.T
            collected = gc.collect()
            if self.verbose:
                collected = gc.collect()
                print("Garbage collector: collected %d objects." % (collected))
            # clean all factorization
            if self.Ainv is not None:
                self.Ainv.clean()

            return self._Jmatrix

    def getJtJdiag(self, m, Wd, f=None):
        """
        Compute JtJ using adjoint problem. Still we never form
        JtJ
        """
        if self.verbose:
            print(">> Compute trace(JtJ)")
        ntime = len(self.survey.unique_times)
        JtJdiag = np.zeros_like(m)
        J = self.getJ(m, f=f)
        wd = Wd.diagonal().reshape((self.survey.locations_n, ntime), order="F")
        for tind in range(ntime):
            t = self.survey.unique_times[tind]
            Jtv = self._P * J.T * sdiag(wd[:, tind])
            JtJdiag += (
                (self.PetaEtaDeriv(t, Jtv, adjoint=True) ** 2).sum(axis=1)
                + (self.PetaTauiDeriv(t, Jtv, adjoint=True) ** 2).sum(axis=1)
                + (self.PetaCDeriv(t, Jtv, adjoint=True) ** 2).sum(axis=1)
            )
        return JtJdiag

    # @profile
    def forward(self, m, f=None):
        if self.verbose:
            print(">> Compute predicted data")

        self.model = m

        self._eta_store = self.eta
        self._taui_store = self.taui
        self._c_store = self.c

        Jv = []

        # When sensitivity matrix is stored
        if self.storeJ:
            J = self.getJ(m, f=f)

            ntime = len(self.survey.unique_times)

            self.model = m
            for tind in range(ntime):
                Jv.append(
                    J.dot(self._P.T * self.get_peta(self.survey.unique_times[tind]))
                )
            return np.hstack(Jv)

        # Do not store sensitivity matrix (memory-wise efficient)
        else:
            if f is None:
                f = self.fields(m)

            # A = self.getA()
            for tind in range(len(self.survey.unique_times)):
                # Pseudo-chareability
                t = self.survey.unique_times[tind]
                v = self.get_peta(t)
                for src in self.survey.source_list:
                    u_src = f[src, self._solutionType]  # solution vector
                    dA_dm_v = self.getADeriv(u_src, v)
                    dRHS_dm_v = self.getRHSDeriv(src, v)
                    du_dm_v = self.Ainv * (-dA_dm_v + dRHS_dm_v)
                    for rx in src.receiver_list:
                        timeindex = rx.getTimeP(self.survey.unique_times)
                        if timeindex[tind]:
                            df_dmFun = getattr(
                                f, "_{0!s}Deriv".format(rx.projField), None
                            )
                            df_dm_v = df_dmFun(src, du_dm_v, v, adjoint=False)
                            Jv.append(rx.evalDeriv(src, self.mesh, f, df_dm_v))

            return np.hstack(Jv)

    def dpred(self, m, f=None):
        r"""
        Predicted data.

        .. math::

            d_\text{pred} = Pf(m)

        """
        if f is None:
            f = self.fields(m)
        return self._pred
        # return self.forward(m, f=f)

    def Jvec(self, m, v, f=None):
        self.model = m

        Jv = []

        # When sensitivity matrix is stored
        if self.storeJ:
            J = self.getJ(m, f=f)
            ntime = len(self.survey.unique_times)

            for tind in range(ntime):
                t = self.survey.unique_times[tind]
                v0 = self.PetaEtaDeriv(t, v)
                v1 = self.PetaTauiDeriv(t, v)
                v2 = self.PetaCDeriv(t, v)
                PTv = self._P.T * (v0 + v1 + v2)
                Jv.append(J.dot(PTv))

            return np.hstack(Jv)

        # Do not store sensitivity matrix (memory-wise efficient)
        else:
            if f is None:
                f = self.fields(m)

            for tind in range(len(self.survey.unique_times)):
                t = self.survey.unique_times[tind]
                v0 = self.PetaEtaDeriv(t, v)
                v1 = self.PetaTauiDeriv(t, v)
                v2 = self.PetaCDeriv(t, v)

                for src in self.survey.source_list:
                    u_src = f[src, self._solutionType]  # solution vector
                    dA_dm_v = self.getADeriv(u_src, v0 + v1 + v2)
                    dRHS_dm_v = self.getRHSDeriv(src, v0 + v1 + v2)
                    du_dm_v = self.Ainv * (-dA_dm_v + dRHS_dm_v)

                    for rx in src.receiver_list:
                        # Assume same # of time
                        # timeindex = rx.getTimeP(self.survey.unique_times)
                        # if timeindex[tind]:
                        Jv_temp = rx.evalDeriv(src, self.mesh, f, du_dm_v)
                        Jv.append(Jv_temp)

            return np.hstack(Jv)

    def Jtvec(self, m, v, f=None):
        self.model = m

        # When sensitivity matrix is stored
        if self.storeJ:
            J = self.getJ(m, f=f)
            ntime = len(self.survey.unique_times)
            Jtvec = np.zeros(m.size)
            v = v.reshape((int(self.survey.nD / ntime), ntime), order="F")

            for tind in range(ntime):
                t = self.survey.unique_times[tind]
                Jtv = self._P * J.T.dot(v[:, tind])
                Jtvec += (
                    self.PetaEtaDeriv(t, Jtv, adjoint=True)
                    + self.PetaTauiDeriv(t, Jtv, adjoint=True)
                    + self.PetaCDeriv(t, Jtv, adjoint=True)
                )

            return Jtvec

        # Do not store sensitivity matrix (memory-wise efficient)
        else:
            if f is None:
                f = self.fields(m)

            # Ensure v is a data object.
            if not isinstance(v, Data):
                v = Data(self.survey, v)

            Jtv = np.zeros(m.size, dtype=float)
            n_time = len(self.survey.unique_times)
            du_dmT = np.zeros((self.mesh.nC, n_time), dtype=float, order="F")

            for tind in range(n_time):
                t = self.survey.unique_times[tind]
                for src in self.survey.source_list:
                    u_src = f[src, self._solutionType]
                    for rx in src.receiver_list:
                        # Ignore case when each rx has different # of times

                        # timeindex = rx.getTimeP(self.survey.unique_times)
                        # if timeindex[tind]:
                        # wrt f, need possibility wrt m
                        PTv = rx.evalDeriv(
                            src, self.mesh, f, v[src, rx, t], adjoint=True
                        )
                        df_duTFun = getattr(f, "_{0!s}Deriv".format(rx.projField), None)
                        df_duT, _ = df_duTFun(src, None, PTv, adjoint=True)
                        ATinvdf_duT = self.Ainv * df_duT
                        du_dmT[:, tind] += -self.getADeriv(
                            u_src, ATinvdf_duT, adjoint=True
                        )

                Jtv += (
                    self.PetaEtaDeriv(
                        self.survey.unique_times[tind], du_dmT[:, tind], adjoint=True
                    )
                    + self.PetaTauiDeriv(
                        self.survey.unique_times[tind], du_dmT[:, tind], adjoint=True
                    )
                    + self.PetaCDeriv(
                        self.survey.unique_times[tind], du_dmT[:, tind], adjoint=True
                    )
                )
            return Jtv

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = [
            "_etaDeriv_store",
            "_tauiDeriv_store",
            "_cDeriv_store",
            "_tauDeriv_store",
        ]
        return toDelete


class Simulation3DCellCentered(BaseSIPSimulation, BaseSimulation3DCellCentered):
    """
    3D cell centered Spectral IP problem
    """


class Simulation3DNodal(BaseSIPSimulation, BaseSimulation3DNodal):
    """
    3D nodal Spectral IP problem
    """


Simulation3DCellCentred = Simulation3DCellCentered
