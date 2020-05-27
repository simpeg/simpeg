import numpy as np
import properties
from ....utils.code_utils import deprecate_class

from .... import props
from ....data import Data
from ....utils import mkvc, sdiag
from ...base import BaseEMSimulation

from ..resistivity.fields import FieldsDC, Fields3DCellCentered, Fields3DNodal
from ..resistivity import Simulation3DCellCentered as BaseSimulation3DCellCentered
from ..resistivity import Simulation3DNodal as BaseSimulation3DNodal


class BaseIPSimulation(BaseEMSimulation):

    sigma = props.PhysicalProperty("Electrical conductivity (S/m)")

    rho = props.PhysicalProperty("Electrical resistivity (Ohm m)")

    props.Reciprocal(sigma, rho)

    eta, etaMap, etaDeriv = props.Invertible("Electrical Chargeability")

    data_type = properties.StringChoice(
        "IP data type",
        default='volt',
        choices=['volt', 'apparent_chargeability'],
    )

    fieldsPair = FieldsDC
    Ainv = None
    _f = None
    storeJ = False
    _Jmatrix = None
    gtgdiag = None
    sign = None
    _pred = None
    gtgdiag = None

    def fields(self, m=None):

        if m is not None:
            self.model = m
            # sensitivity matrix is fixed
            # self._Jmatrix = None

        if self._f is None:

            if self.verbose is True:
                print(">> Solve DC problem")

            self._f = self.fieldsPair(self)
            A = self.getA()
            self.Ainv = self.solver(A, **self.solver_opts)
            RHS = self.getRHS()
            Srcs = self.survey.source_list
            self._f[Srcs, self._solutionType] = self.Ainv * RHS

            if self.data_type == "apparent_chargeability":
                if self.verbose is True:
                    print(">> Data type is apparaent chargeability")
                for src in self.survey.source_list:
                    for rx in src.receiver_list:
                        rx._dc_voltage = rx.eval(src, self.mesh, self._f)
                        rx.data_type = self.data_type
                        rx._Ps = {}

        if self.verbose is True:
            print(">> Compute predicted data")

        self._pred = self.forward(m, f=self._f)

        # if not self.storeJ:
        #     self.Ainv.clean()

        return self._f

    def dpred(self, m=None, f=None):
        """
            Predicted data.

            .. math::

                d_\\text{pred} = Pf(m)

        """
        if f is None:
            f = self.fields(m)

        return self._pred

    def getJtJdiag(self, m, W=None):
        """
            Return the diagonal of JtJ
        """

        if self.gtgdiag is None:
            J = self.getJ(m)

            # Need to check if multiplying weights makes sense
            if W is None:
                W = np.ones(J.shape[0])
            else:
                W = W.diagonal() ** 2

            diag = np.zeros(J.shape[1])
            for i in range(J.shape[0]):
                diag += (W[i]) * (J[i] * J[i])

            self.gtgdiag = diag

        return self.gtgdiag

    def getJ(self, m, f=None):
        """
            Generate Full sensitivity matrix
        """
        if self._Jmatrix is None:
            self._Jmatrix = self._Jtvec(m, v=None, f=f).T
        return self._Jmatrix

    # @profile
    def Jvec(self, m, v, f=None):

        self.model = m

        if f is None:
            f = self.fields(m)

        # When sensitivity matrix J is stored
        if self.storeJ:
            J = self.getJ(m, f=f)
            Jv = J.dot(v)
            return self.sign * Jv
        Jv = []

        for src in self.survey.source_list:
            # solution vector
            u_src = f[src, self._solutionType]
            dA_dm_v = self.getADeriv(u_src.flatten(), v, adjoint=False)
            dRHS_dm_v = self.getRHSDeriv(src, v)
            du_dm_v = self.Ainv * (-dA_dm_v + dRHS_dm_v)

            for rx in src.receiver_list:
                df_dmFun = getattr(f, "_{0!s}Deriv".format(rx.projField), None)
                df_dm_v = df_dmFun(src, du_dm_v, v, adjoint=False)
                Jv.append(rx.evalDeriv(src, self.mesh, f, df_dm_v))

        # Conductivity (d u / d log sigma) - EB form
        # Resistivity (d u / d log rho) - HJ form
        return self.sign * np.hstack(Jv)

    def forward(self, m, f=None):
        return np.asarray(self.Jvec(m, m, f=f))

    def Jtvec(self, m, v, f=None):
        """
            Compute adjoint sensitivity matrix (J^T) and vector (v) product.

        """
        if f is None:
            f = self.fields(m)

        # When sensitivity matrix J is stored
        if self.storeJ:
            J = self.getJ(m, f=f)
            Jtv = np.asarray(J.T.dot(v))
            return self.sign * Jtv

        else:
            self.model = m

            if f is None:
                f = self.fields(m)
            return self._Jtvec(m, v=v, f=f)

    def _Jtvec(self, m, v=None, f=None):
        """
            Compute adjoint sensitivity matrix (J^T) and vector (v) product.
            Full J matrix can be computed by inputing v=None
        """

        if v is not None:
            # Ensure v is a data object.
            if not isinstance(v, Data):
                v = Data(self.survey, v)
            Jtv = np.zeros(m.size)
        else:
            # This is for forming full sensitivity matrix
            Jtv = np.zeros((self.model.size, self.survey.nD), order="F")
            istrt = int(0)
            iend = int(0)

        for isrc, src in enumerate(self.survey.source_list):
            u_src = f[src, self._solutionType]
            # if self.storeJ:
            #     # TODO: use logging package
            #     sys.stdout.write(("\r %d / %d") % (isrc+1, self.survey.nSrc))
            #     sys.stdout.flush()

            for rx in src.receiver_list:
                if v is not None:
                    PTv = rx.evalDeriv(
                        src, self.mesh, f, v[src, rx], adjoint=True
                    )  # wrt f, need possibility wrt m
                    df_duTFun = getattr(f, "_{0!s}Deriv".format(rx.projField), None)
                    df_duT, df_dmT = df_duTFun(src, None, PTv, adjoint=True)
                    ATinvdf_duT = self.Ainv * df_duT
                    dA_dmT = self.getADeriv(u_src.flatten(), ATinvdf_duT, adjoint=True)
                    # dA_dmT = da.from_delayed(dA_dmT, shape=self.model.shape, dtype=float)
                    dRHS_dmT = self.getRHSDeriv(src, ATinvdf_duT, adjoint=True)
                    # dRHS_dmT = da.from_delayed(dRHS_dmT, shape=self.model.shape, dtype=float)
                    du_dmT = -dA_dmT + dRHS_dmT
                    Jtv += (df_dmT + du_dmT).astype(float)
                else:
                    P = rx.getP(self.mesh, rx.projGLoc(f)).toarray()
                    ATinvdf_duT = self.Ainv * (P.T)
                    dA_dmT = self.getADeriv(u_src, ATinvdf_duT, adjoint=True)

                    iend = istrt + rx.nD
                    if rx.nD == 1:
                        Jtv[:, istrt] = -dA_dmT
                    else:
                        Jtv[:, istrt:iend] = -dA_dmT
                    istrt += rx.nD

        # Conductivity ((d u / d log sigma).T) - EB form
        # Resistivity ((d u / d log rho).T) - HJ form

        if v is not None:
            return self.sign * mkvc(Jtv)
        else:
            return Jtv
        return

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

    def delete_these_for_sensitivity(self):
        del self._Jmatrix, self._MfRhoI, self._MeSigma

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        return toDelete

    @property
    def MfRhoDerivMat(self):
        """
        Derivative of MfRho with respect to the model
        """
        if getattr(self, "_MfRhoDerivMat", None) is None:
            drho_dlogrho = sdiag(self.rho) * self.etaDeriv
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
            dMf_drho = self.mesh.getFaceInnerProductDeriv(self.rho)(u)
            drho_dlogrho = sdiag(self.rho) * self.etaDeriv
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
            dsigma_dlogsigma = sdiag(self.sigma) * self.etaDeriv
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
            dsigma_dlogsigma = sdiag(self.sigma) * self.etaDeriv
            if adjoint:
                return dsigma_dlogsigma.T * (
                    self.mesh.getEdgeInnerProductDeriv(self.sigma)(u).T * v
                )
            else:
                return self.mesh.getEdgeInnerProductDeriv(self.sigma)(u) * (
                    dsigma_dlogsigma * v
                )


class Simulation3DCellCentered(BaseIPSimulation, BaseSimulation3DCellCentered):

    _solutionType = "phiSolution"
    _formulation = "HJ"  # CC potentials means J is on faces
    fieldsPair = Fields3DCellCentered
    sign = 1.0
    bc_type = "Dirichlet"

    def __init__(self, mesh, **kwargs):
        super(Simulation3DCellCentered, self).__init__(mesh, **kwargs)
        self.setBC()


class Simulation3DNodal(BaseIPSimulation, BaseSimulation3DNodal):

    _solutionType = "phiSolution"
    _formulation = "EB"  # N potentials means B is on faces
    fieldsPair = Fields3DNodal
    sign = -1.0

    def __init__(self, mesh, **kwargs):
        super(Simulation3DNodal, self).__init__(mesh, **kwargs)


############
# Deprecated
############


@deprecate_class(removal_version="0.15.0")
class Problem3D_N(Simulation3DNodal):
    pass


@deprecate_class(removal_version="0.15.0")
class Problem3D_CC(Simulation3DCellCentered):
    pass
