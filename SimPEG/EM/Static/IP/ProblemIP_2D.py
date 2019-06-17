import numpy as np

from .... import props
from ....utils import sdiag

from ..DC.FieldsDC_2D import (
    Fields_ky, Fields_ky_CC, Fields_ky_N
)

from ..DC import BaseDCProblem_2D
from ..DC import Problem2D_CC as BaseProblem2D_CC
from ..DC import Problem2D_N as BaseProblem2D_N

from .SurveyIP import Survey


class BaseIPProblem_2D(BaseDCProblem_2D):

    sigma = props.PhysicalProperty(
        "Electrical conductivity (S/m)"
    )

    rho = props.PhysicalProperty(
        "Electrical resistivity (Ohm m)"
    )

    props.Reciprocal(sigma, rho)

    eta, etaMap, etaDeriv = props.Invertible(
        "Electrical Chargeability (V/V)"
    )

    surveyPair = Survey
    fieldsPair = Fields_ky
    _Jmatrix = None
    _f = None
    sign = None

    def fields(self, m):
        if self.verbose:
            print(">> Compute DC fields")

        if self._f is None:
            self._f = self.fieldsPair(self.mesh, self.survey)
            Srcs = self.survey.srcList
            for iky in range(self.nky):
                ky = self.kys[iky]
                A = self.getA(ky)
                self.Ainv[iky] = self.Solver(A, **self.solverOpts)
                RHS = self.getRHS(ky)
                u = self.Ainv[iky] * RHS
                self._f[Srcs, self._solutionType, iky] = u
        return self._f

    def Jvec(self, m, v, f=None):
        self.model = m
        J = self.getJ(m, f=f)
        Jv = J.dot(v)
        return self.sign * Jv

    def Jtvec(self, m, v, f=None):
        self.model = m
        J = self.getJ(m, f=f)
        Jtv = J.T.dot(v)
        return self.sign * Jtv

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        return toDelete

    @property
    def MeSigmaDerivMat(self):
        """
        Derivative of MeSigma with respect to the model
        """
        if getattr(self, '_MeSigmaDerivMat', None) is None:
            dsigma_dlogsigma = sdiag(self.sigma)*self.etaDeriv
            self._MeSigmaDerivMat = self.mesh.getEdgeInnerProductDeriv(
                np.ones(self.mesh.nC)
            )(np.ones(self.mesh.nE)) * dsigma_dlogsigma
        return self._MeSigmaDerivMat

    # TODO: This should take a vector
    def MeSigmaDeriv(self, u, v, adjoint=False):
        """
        Derivative of MeSigma with respect to the model times a vector (u)
        """
        if self.storeInnerProduct:
            if adjoint:
                return self.MeSigmaDerivMat.T * (sdiag(u)*v)
            else:
                return sdiag(u)*(self.MeSigmaDerivMat * v)
        else:
            dsigma_dlogsigma = sdiag(self.sigma)*self.etaDeriv
            if adjoint:
                return (
                    dsigma_dlogsigma.T * (
                        self.mesh.getEdgeInnerProductDeriv(self.sigma)(u).T * v
                    )
                )
            else:
                return (
                    self.mesh.getEdgeInnerProductDeriv(self.sigma)(u) *
                    (dsigma_dlogsigma * v)
                )

    @property
    def MccRhoiDerivMat(self):
        """
            Derivative of MccRho with respect to the model
        """
        if getattr(self, '_MccRhoiDerivMat', None) is None:
            rho = self.rho
            vol = self.mesh.vol
            drho_dlogrho = sdiag(rho)*self.etaDeriv
            self._MccRhoiDerivMat = (
                sdiag(vol*(-1./rho**2))*drho_dlogrho
            )
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
            drho_dlogrho = sdiag(rho)*self.etaDeriv
            if adjoint:
                return drho_dlogrho.T * (sdia(u*vol*(-1./rho**2)) * v)
            else:
                return sdiag(u*vol*(-1./rho**2))*(drho_dlogrho * v)

    @property
    def MnSigmaDerivMat(self):
        """
            Derivative of MnSigma with respect to the model
        """
        if getattr(self, '_MnSigmaDerivMat', None) is None:
            sigma = self.sigma
            vol = self.mesh.vol
            dsigma_dlogsigma = sdiag(sigma)*self.etaDeriv
            self._MnSigmaDerivMat = (
                self.mesh.aveN2CC.T * sdiag(vol) * dsigma_dlogsigma
                )
        return self._MnSigmaDerivMat

    def MnSigmaDeriv(self, u, v, adjoint=False):
        """
            Derivative of MnSigma with respect to the model times a vector (u)
        """
        if self.storeInnerProduct:
            if adjoint:
                return self.MnSigmaDerivMat.T * (sdiag(u)*v)
            else:
                return sdiag(u)*(self.MnSigmaDerivMat * v)
        else:
            sigma = self.sigma
            vol = self.mesh.vol
            dsigma_dlogsigma = sdiag(sigma)*self.etaDeriv
            if adjoint:
                return dsigma_dlogsigma.T * (vol * (self.mesh.aveN2CC * (u*v)))
            else:
                dsig_dm_v = dsigma_dlogsigma * v
                return (
                    u * (self.mesh.aveN2CC.T * (vol * dsig_dm_v))
                )


class Problem2D_CC(BaseIPProblem_2D, BaseProblem2D_CC):
    """
    2.5D cell centered IP problem
    """

    _solutionType = 'phiSolution'
    _formulation = 'HJ'  # CC potentials means J is on faces
    fieldsPair = Fields_ky_CC
    bc_type = 'Mixed'
    sign = 1.

    def __init__(self, mesh, **kwargs):
        BaseIPProblem_2D.__init__(self, mesh, **kwargs)

    def delete_these_for_sensitivity(self, sigma=None, rho=None):
        if self._Jmatrix is not None:
            del self._Jmatrix
        self._MfrhoI = None
        if sigma is not None:
            self.sigma = sigma
        elif rho is not None:
            self.sigma = 1./rho
        else:
            raise Exception("Either sigma or rho should be provided")
    @property
    def MfRhoDerivMat(self):
        """
        Derivative of MfRho with respect to the model
        """
        if getattr(self, '_MfRhoDerivMat', None) is None:
            self._MfRhoDerivMat = self.mesh.getFaceInnerProductDeriv(
                np.ones(self.mesh.nC)
            )(np.ones(self.mesh.nF)) * sdiag(self.rho) * self.etaDeriv
        return self._MfRhoDerivMat

    def MfRhoIDeriv(self, u, v, adjoint=False):
        """
            Derivative of :code:`MfRhoI` with respect to the model.
        """
        dMfRhoI_dI = -self.MfRhoI**2
        if self.storeInnerProduct:
            if adjoint:
                return self.MfRhoDerivMat.T * (
                    sdiag(u) * (dMfRhoI_dI.T * v)
                )
            else:
                return dMfRhoI_dI * (sdiag(u) * (self.MfRhoDerivMat*v))
        else:
            dMf_drho = self.mesh.getFaceInnerProductDeriv(self.rho)(u)
            drho_dlogrho = sdiag(self.rho)*self.etaDeriv
            if adjoint:
                return drho_dlogrho.T * (dMf_drho.T * (dMfRhoI_dI.T*v))
            else:
                return dMfRhoI_dI * (dMf_drho * (drho_dlogrho*v))


class Problem2D_N(BaseIPProblem_2D, BaseProblem2D_N):
    """
    2.5D nodal IP problem
    """

    _solutionType = 'phiSolution'
    _formulation = 'EB'  # CC potentials means J is on faces
    fieldsPair = Fields_ky_N
    sign = -1.

    def __init__(self, mesh, **kwargs):
        BaseIPProblem_2D.__init__(self, mesh, **kwargs)

    def delete_these_for_sensitivity(self, sigma=None, rho=None):
        if self._Jmatrix is not None:
            del self._Jmatrix
        self._MeSigma = None
        self._MnSigma = None
        if sigma is not None:
            self.sigma = sigma
        elif rho is not None:
            self.sigma = 1./rho
        else:
            raise Exception("Either sigma or rho should be provided")
