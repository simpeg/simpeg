from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import SimPEG
from SimPEG.Utils import Identity, Zero, sdiag
import numpy as np
from scipy.constants import epsilon_0


class FieldsDC(SimPEG.Problem.Fields):
    knownFields = {}
    dtype = float

    def _phiDeriv(self, src, du_dm_v, v, adjoint=False):
        if (
            getattr(self, '_phiDeriv_u', None) is None or
            getattr(self, '_phiDeriv_m', None) is None
        ):
            raise NotImplementedError(
                'Getting phiDerivs from {0!s} is not '
                'implemented'.format(self.knownFields.keys()[0])
            )

        if adjoint:
            return (self._phiDeriv_u(src, v, adjoint=adjoint),
                    self._phiDeriv_m(src, v, adjoint=adjoint))

        return (np.array(self._phiDeriv_u(src, du_dm_v, adjoint) +
                         self._phiDeriv_m(src, v, adjoint), dtype=float))

    def _eDeriv(self, src, du_dm_v, v, adjoint=False):
        if (
            getattr(self, '_eDeriv_u', None) is None or
            getattr(self, '_eDeriv_m', None) is None
        ):
            raise NotImplementedError(
                'Getting eDerivs from {0!s} is not '
                'implemented'.format(self.knownFields.keys()[0])
            )

        if adjoint:
            return (self._eDeriv_u(src, v, adjoint),
                    self._eDeriv_m(src, v, adjoint))
        return (np.array(self._eDeriv_u(src, du_dm_v, adjoint) +
                         self._eDeriv_m(src, v, adjoint), dtype=float))

    def _jDeriv(self, src, du_dm_v, v, adjoint=False):
        if (
            getattr(self, '_jDeriv_u', None) is None or
            getattr(self, '_jDeriv_m', None) is None
        ):
            raise NotImplementedError(
                'Getting jDerivs from {0!s} is not '
                'implemented'.format(self.knownFields.keys()[0])
            )

        if adjoint:
            return (self._jDeriv_u(src, v, adjoint),
                    self._jDeriv_m(src, v, adjoint))
        return (np.array(self._jDeriv_u(src, du_dm_v, adjoint) +
                         self._jDeriv_m(src, v, adjoint), dtype=float))


class Fields_CC(FieldsDC):
    knownFields = {'phiSolution': 'CC'}
    aliasFields = {
        'phi': ['phiSolution', 'CC', '_phi'],
        'j': ['phiSolution', 'F', '_j'],
        'e': ['phiSolution', 'F', '_e'],
        'charge': ['phiSolution', 'CC', '_charge'],
    }
    # primary - secondary
    # CC variables

    def __init__(self, mesh, survey, **kwargs):
        FieldsDC.__init__(self, mesh, survey, **kwargs)

        if getattr(self.survey.prob, 'bc_type', None) == 'Dirichlet':
            self.cellGrad = -mesh.faceDiv.T
        elif getattr(self.survey.prob, 'bc_type', None) == 'Neumann':
            if self.mesh._meshType == "TREE":
                raise NotImplementedError()
            mesh.setCellGradBC("neumann")
            self.cellGrad = mesh.cellGrad
        else:
            mesh.setCellGradBC("neumann")
            self.cellGrad = mesh.cellGrad

    def startup(self):
        # self.prob = self.survey.prob
        self._MfRhoI = self.survey.prob.MfRhoI
        self._MfRho = self.survey.prob.MfRho
        self._aveF2CCV = self.survey.prob.mesh.aveF2CCV
        self._nC = self.survey.prob.mesh.nC
        self._Grad = self.survey.prob.Grad
        self._MfI = self.survey.prob.MfI
        self._Vol = self.survey.prob.Vol
        self._faceDiv = self.survey.prob.mesh.faceDiv

    def _GLoc(self, fieldType):
        if fieldType == 'phi':
            return 'CC'
        elif fieldType == 'e' or fieldType == 'j':
            return 'F'
        else:
            raise Exception('Field type must be phi, e, j')

    def _phi(self, phiSolution, srcList):
        return phiSolution

    def _phiDeriv_u(self, src, v, adjoint=False):
        return Identity()*v

    def _phiDeriv_m(self, src, v, adjoint=False):
        return Zero()

    def _j(self, phiSolution, srcList):
        """
            .. math::
                \mathbf{j} = \mathbf{M}^{f \ -1}_{\rho} \mathbf{G} \phi
        """
        return self._MfRhoI*self._Grad*phiSolution

    def _e(self, phiSolution, srcList):
        """
            .. math::
                \vec{e} = \rho \vec{j}
        """
        return self._MfI*self._MfRho * self._j(phiSolution, srcList)

    def _charge(self, phiSolution, srcList):
        """
            .. math::
                \int \nabla \codt \vec{e} =  \int \frac{\rho_v }{\epsillon_0}
        """
        return epsilon_0*self._Vol*(
            self._faceDiv*self._e(phiSolution, srcList)
        )


class Fields_N(FieldsDC):
    knownFields = {'phiSolution': 'N'}
    aliasFields = {
        'phi': ['phiSolution', 'N', '_phi'],
        'j': ['phiSolution', 'E', '_j'],
        'e': ['phiSolution', 'E', '_e'],
        'charge': ['phiSolution', 'N', '_charge'],
    }
    # primary - secondary
    # N variables

    def __init__(self, mesh, survey, **kwargs):
        FieldsDC.__init__(self, mesh, survey, **kwargs)

    def startup(self):
        self.prob = self.survey.prob

    def _GLoc(self, fieldType):
        if fieldType == 'phi':
            return 'N'
        elif fieldType == 'e' or fieldType == 'j':
            return 'E'
        else:
            raise Exception('Field type must be phi, e, j')

    def _phi(self, phiSolution, srcList):
        return phiSolution

    def _phiDeriv_u(self, src, v, adjoint=False):
        return Identity()*v

    def _phiDeriv_m(self, src, v, adjoint=False):
        return Zero()

    def _j(self, phiSolution, srcList):
        """
            In EB formulation j is not well-defined!!
            .. math::
                \mathbf{j} = - \mathbf{M}^{e}_{\sigma} \mathbf{G} \phi
        """
        return self.prob.MeI * self.prob.MeSigma * self._e(phiSolution, srcList)

    def _e(self, phiSolution, srcList):
        """
            In HJ formulation e is not well-defined!!
            .. math::
                \vec{e} = -\nabla \phi
        """
        return -self.mesh.nodalGrad * phiSolution

    def _charge(self, phiSolution, srcList):
        """
            .. math::
                \int \nabla \codt \vec{e} =  \int \frac{\rho_v }{\epsillon_0}
        """
        return - epsilon_0*(self.mesh.nodalGrad.T *
                            self.mesh.getEdgeInnerProduct() *
                            self._e(phiSolution, srcList))
