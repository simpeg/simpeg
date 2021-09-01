import numpy as np
import properties
from ....utils.code_utils import deprecate_class, deprecate_property

from .... import props
from ....data import Data
from ....utils import sdiag

from ..resistivity.simulation import BaseDCSimulation
from ..resistivity.fields import Fields3DCellCentered, Fields3DNodal
from ..resistivity import Simulation3DCellCentered as BaseSimulation3DCellCentered
from ..resistivity import Simulation3DNodal as BaseSimulation3DNodal


class BaseIPSimulation(BaseDCSimulation):

    eta, etaMap, etaDeriv = props.Invertible("Electrical Chargeability")

    _data_type = properties.StringChoice(
        "IP data type", default="volt", choices=["volt", "apparent_chargeability"],
    )

    data_type = deprecate_property(
        _data_type, "data_type", new_name="receiver.data_type", removal_version="0.16.0"
    )

    Ainv = None
    _f = None
    _Jmatrix = None
    gtgdiag = None
    _sign = None
    _pred = None
    _scale = None

    def fields(self, m=None):

        if m is not None:
            self.model = m
            # sensitivity matrix is fixed
            # self._Jmatrix = None

        if self._f is None:
            if self.verbose is True:
                print(">> Solve DC problem")
            self._f = super().fields(m=None)

        if self._scale is None:
            scale = Data(self.survey, np.full(self.survey.nD, self._sign))
            # loop through receievers to check if they need to set the _dc_voltage
            for src in self.survey.source_list:
                for rx in src.receiver_list:
                    if (
                        rx.data_type == "apparent_chargeability"
                        or self._data_type == "apparent_chargeability"
                    ):
                        scale[src, rx] = self._sign / rx.eval(src, self.mesh, self._f)
            self._scale = scale.dobs

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
            if W is None:
                W = self._scale ** 2
            else:
                W = (self._scale * W.diagonal()) ** 2

            self.gtgdiag = np.einsum("i,ij,ij->j", W, J, J)

        return self.gtgdiag

    # @profile
    def Jvec(self, m, v, f=None):
        return self._scale * super().Jvec(m, v, f)

    def forward(self, m, f=None):
        return np.asarray(self.Jvec(m, m, f=f))

    def Jtvec(self, m, v, f=None):
        return super().Jtvec(m, v * self._scale, f)

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
    _sign = 1.0


class Simulation3DNodal(BaseIPSimulation, BaseSimulation3DNodal):

    _solutionType = "phiSolution"
    _formulation = "EB"  # N potentials means B is on faces
    fieldsPair = Fields3DNodal
    _sign = -1.0


############
# Deprecated
############


@deprecate_class(removal_version="0.16.0", future_warn=True)
class Problem3D_N(Simulation3DNodal):
    pass


@deprecate_class(removal_version="0.16.0", future_warn=True)
class Problem3D_CC(Simulation3DCellCentered):
    pass
