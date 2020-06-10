import numpy as np
import sys
import gc
import warnings
from ....utils.code_utils import deprecate_class
import properties

from .... import props
from .... import maps
from .data import Data
from ....utils import sdiag

from ...base import BaseEMSimulation
from ..resistivity.fields import FieldsDC, Fields3DCellCentered, Fields3DNodal
from ..induced_polarization import (
    Simulation3DCellCentered as BaseSimulation3DCellCentered,
)
from ..induced_polarization import Simulation3DNodal as BaseSimulation3DNodal
from .survey import Survey


class BaseSIPSimulation(BaseEMSimulation):

    sigma = props.PhysicalProperty("Electrical conductivity (S/m)")

    rho = props.PhysicalProperty("Electrical resistivity (Ohm m)")

    props.Reciprocal(sigma, rho)

    eta, etaMap, etaDeriv = props.Invertible("Electrical Chargeability (V/V)")

    tau, tauMap, tauDeriv = props.Invertible("Time constant (s)", default=0.1)

    taui, tauiMap, tauiDeriv = props.Invertible("Inverse of time constant (1/s)")

    c, cMap, cDeriv = props.Invertible("Frequency dependency", default=0.5)

    props.Reciprocal(tau, taui)

    survey = properties.Instance("an SIP survey object", Survey, required=True)

    fieldsPair = FieldsDC
    Ainv = None
    _f = None
    actinds = None
    storeJ = False
    _Jmatrix = None
    actMap = None
    n_pulse = 1

    _eta_store = None
    _taui_store = None
    _c_store = None
    _pred = None

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
                return etaDeriv.T * (Utils.sdiag(dpetadeta) * v)
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
                return tauiDeriv.T * (Utils.sdiag(dpetadtaui) * v)
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
                return cDeriv.T * (Utils.sdiag(dpetadc) * v)
        else:
            return dpetadc * (cDeriv * v)

    def fields(self, m):

        if self._f is None:

            if self.verbose:
                print(">> Compute DC fields")

            self._f = self.fieldsPair(self)

            if self.Ainv is None:
                A = self.getA()
                self.Ainv = self.Solver(A, **self.solver_opts)
            RHS = self.getRHS()
            u = self.Ainv * RHS
            Srcs = self.survey.source_list
            self._f[Srcs, self._solutionType] = u

            # Compute DC voltage
            if self.data_type == "apparent_chargeability":
                if self.verbose is True:
                    print(">> Data type is apparaent chargeability")
                for src in self.survey.source_list:
                    for rx in src.receiver_list:
                        rx._dc_voltage = rx.eval(src, self.mesh, self._f)
                        rx.data_type = self.data_type
                        rx._Ps = {}

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
                (self.actMap.nP, int(self.survey.nD / self.survey.unique_times.size)),
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
            # clean field object
            self._f = []
            # clean all factorization
            if self.Ainv is not None:
                self.Ainv.clean()

            return self._Jmatrix

    def getJtJdiag(self, m, Wd):
        """
        Compute JtJ using adjoint problem. Still we never form
        JtJ
        """
        if self.verbose:
            print(">> Compute trace(JtJ)")
        ntime = len(self.survey.unique_times)
        JtJdiag = np.zeros_like(m)
        J = self.getJ(m, f=None)
        wd = (Wd.diagonal()).reshape((self.survey.n_locations, ntime), order="F")
        for tind in range(ntime):
            t = self.survey.unique_times[tind]
            Jtv = self.actMap.P * J.T * Utils.sdiag(wd[:, tind])
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
                    J.dot(
                        self.actMap.P.T * self.get_peta(self.survey.unique_times[tind])
                    )
                )
            return self.sign * np.hstack(Jv)

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

            return self.sign * np.hstack(Jv)

    def dpred(self, m, f=None):
        """
            Predicted data.

            .. math::

                d_\\text{pred} = Pf(m)

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
                PTv = self.actMap.P.T * (v0 + v1 + v2)
                Jv.append(J.dot(PTv))

            return self.sign * np.hstack(Jv)

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

            return self.sign * np.hstack(Jv)

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
                Jtv = self.actMap.P * J.T.dot(v[:, tind])
                Jtvec += (
                    self.PetaEtaDeriv(t, Jtv, adjoint=True)
                    + self.PetaTauiDeriv(t, Jtv, adjoint=True)
                    + self.PetaCDeriv(t, Jtv, adjoint=True)
                )

            return self.sign * Jtvec

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
                        dA_dmT = self.getADeriv(u_src, ATinvdf_duT, adjoint=True)
                        # Unecessary at the moment

                        # dRHS_dmT = self.getRHSDeriv(
                        #     src, ATinvdf_duT, adjoint=True
                        # )
                        # du_dmT[:, tind] = -dA_dmT + dRHS_dmT

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

            return self.sign * Jtv

    def getSourceTerm(self):
        """
        takes concept of source and turns it into a matrix
        """
        """
        Evaluates the sources, and puts them in matrix form

        :rtype: (numpy.ndarray, numpy.ndarray)
        :return: q (nC or nN, nSrc)
        """

        Srcs = self.survey.source_list

        if self._formulation == "EB":
            n = self.mesh.nN
            # return NotImplementedError

        elif self._formulation == "HJ":
            n = self.mesh.nC

        q = np.zeros((n, len(Srcs)))

        for i, src in enumerate(Srcs):
            q[:, i] = src.eval(self)
        return q

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = [
            "_etaDeriv_store",
            "_tauiDeriv_store",
            "_cDeriv_store",
            "_tauDeriv_store",
        ]
        return toDelete

    @property
    def MfRhoDerivMat(self):
        """
        Derivative of MfRho with respect to the model
        """
        if getattr(self, "_MfRhoDerivMat", None) is None:
            if self.storeJ:
                drho_dlogrho = sdiag(self.rho) * self.actMap.P
            else:
                drho_dlogrho = sdiag(self.rho)
            self._MfRhoDerivMat = (
                self.mesh.getFaceInnerProductDeriv(np.ones(self.mesh.nC))(
                    np.ones(self.mesh.nF)
                )
                * drho_dlogrho
            )
        return self._MfRhoDerivMat

    def MfRhoIDeriv(self, u, v, adjoint=False):
        """
            Derivative of :code:`MfRhoI` with respect to the model.
        """
        dMfRhoI_dI = -self.MfRhoI ** 2

        if self.storeInnerProduct:
            if adjoint:
                return self.MfRhoDerivMat.T * (sdiag(u) * (dMfRhoI_dI.T * v))
            else:
                return dMfRhoI_dI * (sdiag(u) * (self.MfRhoDerivMat * v))
        else:
            if self.storeJ:
                drho_dlogrho = sdiag(self.rho) * self.actMap.P
            else:
                drho_dlogrho = sdiag(self.rho)
            dMf_drho = self.mesh.getFaceInnerProductDeriv(self.rho)(u)
            if adjoint:
                return drho_dlogrho.T * (dMf_drho.T * (dMfRhoI_dI.T * v))
            else:
                return dMfRhoI_dI * (dMf_drho * (drho_dlogrho * v))

    @property
    def MeSigmaDerivMat(self):
        """
        Derivative of MeSigma with respect to the model
        """
        if getattr(self, "_MeSigmaDerivMat", None) is None:
            if self.storeJ:
                dsigma_dlogsigma = sdiag(self.sigma) * self.actMap.P
            else:
                dsigma_dlogsigma = sdiag(self.sigma)
            self._MeSigmaDerivMat = (
                self.mesh.getEdgeInnerProductDeriv(np.ones(self.mesh.nC))(
                    np.ones(self.mesh.nE)
                )
                * dsigma_dlogsigma
            )
        return self._MeSigmaDerivMat

    # TODO: This should take a vector
    def MeSigmaDeriv(self, u, v, adjoint=False):
        """
        Derivative of MeSigma with respect to the model times a vector (u)
        """
        if self.storeInnerProduct:
            if adjoint:
                return self.MeSigmaDerivMat.T * (sdiag(u) * v)
            else:
                return sdiag(u) * (self.MeSigmaDerivMat * v)
        else:
            if self.storeJ:
                dsigma_dlogsigma = sdiag(self.sigma) * self.actMap.P
            else:
                dsigma_dlogsigma = sdiag(self.sigma)
            if adjoint:
                return dsigma_dlogsigma.T * (
                    self.mesh.getEdgeInnerProductDeriv(self.sigma)(u).T * v
                )
            else:
                return self.mesh.getEdgeInnerProductDeriv(self.sigma)(u) * (
                    dsigma_dlogsigma * v
                )


class Simulation3DCellCentered(BaseSIPSimulation, BaseSimulation3DCellCentered):

    _solutionType = "phiSolution"
    _formulation = "HJ"  # CC potentials means J is on faces
    fieldsPair = Fields3DCellCentered
    sign = 1.0
    bc_type = "Neumann"

    def __init__(self, mesh, **kwargs):
        BaseSIPSimulation.__init__(self, mesh, **kwargs)
        self.setBC()
        self.n = self.mesh.nC
        if self.storeJ:
            if self.actinds is None:
                print("You did not put Active indices")
                print("So, set actMap = IdentityMap(mesh)")
                self.actinds = np.ones(mesh.nC, dtype=bool)

            self.actMap = maps.InjectActiveCells(mesh, self.actinds, 0.0)


class Simulation3DNodal(BaseSIPSimulation, BaseSimulation3DNodal):

    _solutionType = "phiSolution"
    _formulation = "EB"  # N potentials means B is on faces
    fieldsPair = Fields3DNodal
    sign = -1.0

    def __init__(self, mesh, **kwargs):
        BaseSIPSimulation.__init__(self, mesh, **kwargs)
        self.n = self.mesh.nN

        if self.storeJ:
            if self.actinds is None:
                print("You did not put Active indices")
                print("So, set actMap = IdentityMap(mesh)")
                self.actinds = np.ones(mesh.nC, dtype=bool)

            self.actMap = maps.InjectActiveCells(mesh, self.actinds, 0.0)


Simulation3DCellCentred = Simulation3DCellCentered


############
# Deprecated
############


@deprecate_class(removal_version="0.15.0")
class Problem3D_N(Simulation3DNodal):
    pass


@deprecate_class(removal_version="0.15.0")
class Problem3D_CC(Simulation3DCellCentered):
    pass
