import numpy as np
from scipy.constants import epsilon_0

from ....fields import Fields
from ....utils import Identity, Zero
from ....utils.code_utils import deprecate_class


class FieldsDC(Fields):
    knownFields = {}
    dtype = float

    @property
    def survey(self):
        mini = self.simulation._mini_survey
        if mini is not None:
            return mini
        return self.simulation.survey

    def _phiDeriv(self, src, du_dm_v, v, adjoint=False):
        if (
            getattr(self, "_phiDeriv_u", None) is None
            or getattr(self, "_phiDeriv_m", None) is None
        ):
            raise NotImplementedError(
                "Getting phiDerivs from {0!s} is not "
                "implemented".format(self.knownFields.keys()[0])
            )

        if adjoint:
            return (
                self._phiDeriv_u(src, v, adjoint=adjoint),
                self._phiDeriv_m(src, v, adjoint=adjoint),
            )

        return np.array(
            self._phiDeriv_u(src, du_dm_v, adjoint) + self._phiDeriv_m(src, v, adjoint),
            dtype=float,
        )

    def _eDeriv(self, src, du_dm_v, v, adjoint=False):
        if (
            getattr(self, "_eDeriv_u", None) is None
            or getattr(self, "_eDeriv_m", None) is None
        ):
            raise NotImplementedError(
                "Getting eDerivs from {0!s} is not "
                "implemented".format(self.knownFields.keys()[0])
            )

        if adjoint:
            return (self._eDeriv_u(src, v, adjoint), self._eDeriv_m(src, v, adjoint))
        return np.array(
            self._eDeriv_u(src, du_dm_v, adjoint) + self._eDeriv_m(src, v, adjoint),
            dtype=float,
        )

    def _jDeriv(self, src, du_dm_v, v, adjoint=False):
        if (
            getattr(self, "_jDeriv_u", None) is None
            or getattr(self, "_jDeriv_m", None) is None
        ):
            raise NotImplementedError(
                "Getting jDerivs from {0!s} is not "
                "implemented".format(self.knownFields.keys()[0])
            )

        if adjoint:
            return (self._jDeriv_u(src, v, adjoint), self._jDeriv_m(src, v, adjoint))
        return np.array(
            self._jDeriv_u(src, du_dm_v, adjoint) + self._jDeriv_m(src, v, adjoint),
            dtype=float,
        )


class Fields3DCellCentered(FieldsDC):
    knownFields = {"phiSolution": "CC"}
    aliasFields = {
        "phi": ["phiSolution", "CC", "_phi"],
        "j": ["phiSolution", "F", "_j"],
        "e": ["phiSolution", "F", "_e"],
        "charge": ["phiSolution", "CC", "_charge"],
        "charge_density": ["phiSolution", "CC", "_charge_density"],
    }
    # primary - secondary
    # CC variables

    # def __init__(self, mesh, survey, **kwargs):
    #     FieldsDC.__init__(self, mesh, survey, **kwargs)

    def startup(self):
        mesh = self.simulation.mesh
        if getattr(self.simulation, "bc_type", None) == "Dirichlet":
            self.cellGrad = -mesh.faceDiv.T
        elif getattr(self.simulation, "bc_type", None) == "Neumann":
            if self.mesh._meshType == "TREE":
                raise NotImplementedError()
            mesh.setCellGradBC("neumann")
            self.cellGrad = mesh.cellGrad
        else:
            mesh.setCellGradBC("neumann")
            self.cellGrad = mesh.cellGrad

        self._MfRhoI = self.simulation.MfRhoI
        self._MfRhoIDeriv = self.simulation.MfRhoIDeriv
        self._MfRho = self.simulation.MfRho
        self._aveF2CCV = self.simulation.mesh.aveF2CCV
        self._nC = self.simulation.mesh.nC
        self._Grad = self.simulation.Grad
        self._MfI = self.simulation.MfI
        self._Vol = self.simulation.Vol
        self._faceDiv = self.simulation.mesh.faceDiv

    def _GLoc(self, fieldType):
        if fieldType == "phi":
            return "CC"
        elif fieldType == "e" or fieldType == "j":
            return "F"
        else:
            raise Exception("Field type must be phi, e, j")

    def _phi(self, phiSolution, source_list):
        return phiSolution

    def _phiDeriv_u(self, src, v, adjoint=False):
        return v

    def _phiDeriv_m(self, src, v, adjoint=False):
        return Zero()

    def _j(self, phiSolution, source_list):
        """
            .. math::

                \mathbf{j} = \mathbf{M}^{f \ -1}_{\rho} \mathbf{G} \phi
        """
        return self._MfRhoI * self._Grad * phiSolution

    def _jDeriv_u(self, src, v, adjoint=False):
        if adjoint:
            return self._Grad.T * (self._MfRhoI.T * v)
        return self._MfRhoI * (self._Grad * v)

    def _jDeriv_m(self, src, v, adjoint=False):
        if adjoint:
            return self._Grad.T * self._MfRhoIDeriv(v, adjoint=True)
        return self._MfRhoIDeriv(self._Grad * v)

    def _e(self, phiSolution, source_list):
        """
            .. math::

                \vec{e} = \rho \vec{j}
        """
        # return self._MfI * self._MfRho * self._j(phiSolution, source_list)
        return self._MfI * self._Grad * phiSolution
        # simulation._MfI * cart_mesh.faceDiv.T * p

    def _eDeriv_u(self, src, v, adjoint=False):
        if adjoint:
            return self._Grad.T * (self._MfI.T * v)
        return self._MfI * (self._Grad * v)

    def _eDeriv_m(self, src, v, adjoint=False):
        return Zero()

    def _charge(self, phiSolution, source_list):
        """
            .. math::

                \int \nabla \codt \vec{e} =  \int \frac{\rho_v }{\epsillon_0}
        """
        return (
            epsilon_0 * self._Vol * (self._faceDiv * self._e(phiSolution, source_list))
        )

    def _charge_density(self, phiSolution, source_list):
        """
            .. math::

                \frac{1}{V}\int \nabla \codt \vec{e} =
                \frac{1}{V}\int \frac{\rho_v }{\epsillon_0}
        """
        return epsilon_0 * (self._faceDiv * self._e(phiSolution, source_list))


class Fields3DNodal(FieldsDC):
    knownFields = {"phiSolution": "N"}
    aliasFields = {
        "phi": ["phiSolution", "N", "_phi"],
        "j": ["phiSolution", "E", "_j"],
        "e": ["phiSolution", "E", "_e"],
        "charge_density": ["phiSolution", "CC", "_charge_density"],
        "charge": ["phiSolution", "N", "_charge"],
    }
    # primary - secondary
    # N variables

    # def __init__(self, mesh, survey, **kwargs):
    #     FieldsDC.__init__(self, mesh, survey, **kwargs)

    def _GLoc(self, fieldType):
        if fieldType == "phi":
            return "N"
        elif fieldType == "e" or fieldType == "j":
            return "E"
        else:
            raise Exception("Field type must be phi, e, j")

    def _phi(self, phiSolution, source_list):
        return phiSolution

    def _phiDeriv_u(self, src, v, adjoint=False):
        return Identity() * v

    def _phiDeriv_m(self, src, v, adjoint=False):
        return Zero()

    def _j(self, phiSolution, source_list):
        """
            In EB formulation j is not well-defined!!
            .. math::
                \mathbf{j} = - \mathbf{M}^{e}_{\sigma} \mathbf{G} \phi
        """
        return (
            self.simulation.MeI
            * self.simulation.MeSigma
            * self._e(phiSolution, source_list)
        )

    def _e(self, phiSolution, source_list):
        """
            In HJ formulation e is not well-defined!!
            .. math::
                \vec{e} = -\nabla \phi
        """
        return -self.mesh.nodalGrad * phiSolution

    def _charge(self, phiSolution, source_list):
        """
            .. math::
                \int \nabla \codt \vec{e} =  \int \frac{\rho_v }{\epsillon_0}
        """
        return -epsilon_0 * (
            self.mesh.nodalGrad.T
            * self.mesh.getEdgeInnerProduct()
            * self._e(phiSolution, source_list)
        )

    def _charge_density(self, phiSolution, source_list):
        return (
            self.mesh.aveN2CC * self._charge(phiSolution, source_list)
        ) / self.mesh.vol[:, None]


Fields3DCellCentred = Fields3DCellCentered


############
# Deprecated
############
@deprecate_class(removal_version="0.15.0")
class Fields_CC(Fields3DCellCentered):
    pass


@deprecate_class(removal_version="0.15.0")
class Fields_N(Fields3DNodal):
    pass
