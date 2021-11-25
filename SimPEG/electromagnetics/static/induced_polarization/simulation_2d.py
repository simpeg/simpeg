import numpy as np
import properties
from ....utils.code_utils import deprecate_class, deprecate_property

from .... import props
from ....utils import sdiag
from ....data import Data

from ..resistivity.fields_2d import Fields2D, Fields2DCellCentered, Fields2DNodal

from ..resistivity.simulation_2d import BaseDCSimulation2D
from ..resistivity import Simulation2DCellCentered as BaseSimulation2DCellCentered
from ..resistivity import Simulation2DNodal as BaseSimulation2DNodal


class BaseIPSimulation2D(BaseDCSimulation2D):

    eta, etaMap, etaDeriv = props.Invertible("Electrical Chargeability (V/V)")

    _data_type = properties.StringChoice(
        "IP data type", default="volt", choices=["volt", "apparent_chargeability"],
    )

    data_type = deprecate_property(
        _data_type,
        "data_type",
        new_name="receiver.data_type",
        removal_version="0.17.0",
        future_warn=True,
    )

    fieldsPair = Fields2D
    _Jmatrix = None
    _f = None  # the DC fields
    _sign = 1.0
    _pred = None
    _scale = None
    gtgdiag = None

    def fields(self, m):
        if self.verbose:
            print(">> Compute DC fields")
        if self._f is None:
            # re-uses the DC simulation's fields method
            self._f = super().fields(None)

        if self._scale is None:
            scale = Data(self.survey, np.full(self.survey.nD, self._sign))
            f = self.fields_to_space(self._f)
            # loop through receievers to check if they need to set the _dc_voltage
            for src in self.survey.source_list:
                for rx in src.receiver_list:
                    if (
                        rx.data_type == "apparent_chargeability"
                        or self._data_type == "apparent_chargeability"
                    ):
                        scale[src, rx] = self._sign / rx.eval(src, self.mesh, f)
            self._scale = scale.dobs

        self._pred = self.forward(m, f=self._f)

        return self._f

    def dpred(self, m=None, f=None):
        """
        Predicted data.

        .. math::

            d_\\text{pred} = Pf(m)

        """
        # return self.Jvec(m, m, f=f)
        if f is None:
            f = self.fields(m)

        return self._pred

    def getJtJdiag(self, m, W=None):
        if self.gtgdiag is None:
            J = self.getJ(m)
            if W is None:
                W = self._scale ** 2
            else:
                W = (self._scale * W.diagonal()) ** 2

            self.gtgdiag = np.einsum("i,ij,ij->j", W, J, J)

        return self.gtgdiag

    def Jvec(self, m, v, f=None):
        return self._scale * super().Jvec(m, v, f)

    def forward(self, m, f=None):
        return self.Jvec(m, m, f=f)

    def Jtvec(self, m, v, f=None):
        return super().Jtvec(m, v * self._scale, f)

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        return toDelete

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

    @property
    def MccRhoiDerivMat(self):
        """
        Derivative of MccRho with respect to the model
        """
        if getattr(self, "_MccRhoiDerivMat", None) is None:
            rho = self.rho
            vol = self.mesh.vol
            drho_dlogrho = sdiag(rho) * self.etaDeriv
            self._MccRhoiDerivMat = sdiag(vol * (-1.0 / rho ** 2)) * drho_dlogrho
        return self._MccRhoiDerivMat

    def MccRhoiDeriv(self, u, v, adjoint=False):
        """
        Derivative of :code:`MccRhoi` with respect to the model.
        """
        if len(self.rho.shape) > 1:
            if self.rho.shape[1] > self.mesh.dim:
                raise NotImplementedError(
                    "Full anisotropy is not implemented for MccRhoiDeriv."
                )
        if self.storeInnerProduct:
            if adjoint:
                return self.MccRhoiDerivMat.T * (sdiag(u) * v)
            else:
                return sdiag(u) * (self.MccRhoiDerivMat * v)
        else:
            vol = self.mesh.vol
            rho = self.rho
            drho_dlogrho = sdiag(rho) * self.etaDeriv
            if adjoint:
                return drho_dlogrho.T * (sdiag(u * vol * (-1.0 / rho ** 2)) * v)
            else:
                return sdiag(u * vol * (-1.0 / rho ** 2)) * (drho_dlogrho * v)

    @property
    def MnSigmaDerivMat(self):
        """
        Derivative of MnSigma with respect to the model
        """
        if getattr(self, "_MnSigmaDerivMat", None) is None:
            sigma = self.sigma
            vol = self.mesh.vol
            dsigma_dlogsigma = sdiag(sigma) * self.etaDeriv
            self._MnSigmaDerivMat = self.mesh.aveN2CC.T * sdiag(vol) * dsigma_dlogsigma
        return self._MnSigmaDerivMat

    def MnSigmaDeriv(self, u, v, adjoint=False):
        """
        Derivative of MnSigma with respect to the model times a vector (u)
        """
        if self.storeInnerProduct:
            if adjoint:
                return self.MnSigmaDerivMat.T * (sdiag(u) * v)
            else:
                return sdiag(u) * (self.MnSigmaDerivMat * v)
        else:
            sigma = self.sigma
            vol = self.mesh.vol
            dsigma_dlogsigma = sdiag(sigma) * self.etaDeriv
            if adjoint:
                return dsigma_dlogsigma.T * (vol * (self.mesh.aveN2CC * (u * v)))
            else:
                dsig_dm_v = dsigma_dlogsigma * v
                return u * (self.mesh.aveN2CC.T * (vol * dsig_dm_v))


class Simulation2DCellCentered(BaseIPSimulation2D, BaseSimulation2DCellCentered):
    """
    2.5D cell centered IP problem
    """

    _solutionType = "phiSolution"
    _formulation = "HJ"  # CC potentials means J is on faces
    fieldsPair = Fields2DCellCentered
    bc_type = "Mixed"

    def delete_these_for_sensitivity(self, sigma=None, rho=None):
        if self._Jmatrix is not None:
            del self._Jmatrix
        self._MfrhoI = None
        if sigma is not None:
            self.sigma = sigma
        elif rho is not None:
            self.sigma = 1.0 / rho
        else:
            raise Exception("Either sigma or rho should be provided")

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


class Simulation2DNodal(BaseIPSimulation2D, BaseSimulation2DNodal):
    """
    2.5D nodal IP problem
    """

    _solutionType = "phiSolution"
    _formulation = "EB"  # CC potentials means J is on faces
    fieldsPair = Fields2DNodal
    _sign = -1.0

    def delete_these_for_sensitivity(self, sigma=None, rho=None):
        if self._Jmatrix is not None:
            del self._Jmatrix
        self._MeSigma = None
        self._MnSigma = None
        if sigma is not None:
            self.sigma = sigma
        elif rho is not None:
            self.sigma = 1.0 / rho
        else:
            raise Exception("Either sigma or rho should be provided")


Simulation2DCellCentred = Simulation2DCellCentered


############
# Deprecated
############


@deprecate_class(removal_version="0.16.0", error=True)
class Problem2D_N(Simulation2DNodal):
    pass


@deprecate_class(removal_version="0.16.0", error=True)
class Problem2D_CC(Simulation2DCellCentered):
    pass
