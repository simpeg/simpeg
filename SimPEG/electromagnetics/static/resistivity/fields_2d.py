import numpy as np
import properties
from scipy.constants import epsilon_0
from ....utils.code_utils import deprecate_class

from ....fields import TimeFields
from ....utils import Identity, Zero

# TODO: this should be the BaseDCSimulation2D --> but circular imports at the
# moment, so we can settle for its base at the moment
from ...base import BaseEMSimulation


class Fields2D(TimeFields):

    """

    Fancy Field Storage for a 2.5D code.

    u[:,'phi', kyInd] = phi
    print(u[src0,'phi'])

    Only one field type is stored for
    each problem, the rest are computed. The fields obejct acts like an array
    and is indexed by

    .. code-block:: python

        f = problem.fields(m)
        e = f[srcList,'e']
        j = f[srcList,'j']

    If accessing all sources for a given field, use the

    .. code-block:: python

        f = problem.fields(m)
        phi = f[:,'phi']
        e = f[:,'e']
        b = f[:,'b']

    The array returned will be size (nE or nF, nSrcs :math:`\\times`
    nFrequencies)

    """

    simulation = properties.Instance("2D DC simulation", BaseEMSimulation)

    knownFields = {}
    dtype = float

    @property
    def survey(self):
        mini = self.simulation._mini_survey
        if mini is not None:
            return mini
        return self.simulation.survey

    def _phiDeriv(self, kyInd, src, du_dm_v, v, adjoint=False):
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
                self._phiDeriv_u(kyInd, src, v, adjoint=adjoint),
                self._phiDeriv_m(kyInd, src, v, adjoint=adjoint),
            )

        return np.array(
            self._phiDeriv_u(kyInd, src, du_dm_v, adjoint)
            + self._phiDeriv_m(kyInd, src, v, adjoint),
            dtype=float,
        )

    def _eDeriv(self, kyInd, src, du_dm_v, v, adjoint=False):
        if (
            getattr(self, "_eDeriv_u", None) is None
            or getattr(self, "_eDeriv_m", None) is None
        ):
            raise NotImplementedError(
                "Getting eDerivs from {0!s} is not "
                "implemented".format(self.knownFields.keys()[0])
            )

        if adjoint:
            return (
                self._eDeriv_u(kyInd, src, v, adjoint),
                self._eDeriv_m(kyInd, src, v, adjoint),
            )
        return np.array(
            self._eDeriv_u(kyInd, src, du_dm_v, adjoint)
            + self._eDeriv_m(kyInd, src, v, adjoint),
            dtype=float,
        )

    def _jDeriv(self, kyInd, src, du_dm_v, v, adjoint=False):
        if (
            getattr(self, "_jDeriv_u", None) is None
            or getattr(self, "_jDeriv_m", None) is None
        ):
            raise NotImplementedError(
                "Getting jDerivs from {0!s} is not "
                "implemented".format(self.knownFields.keys()[0])
            )

        if adjoint:
            return (
                self._jDeriv_u(kyInd, src, v, adjoint),
                self._jDeriv_m(kyInd, src, v, adjoint),
            )
        return np.array(
            self._jDeriv_u(kyInd, src, du_dm_v, adjoint)
            + self._jDeriv_m(kyInd, src, v, adjoint),
            dtype=float,
        )

    def _phi_ky(self, phiSolution, srcList, kyInd):
        return phiSolution

    def _phi(self, phiSolution, srcList):
        return phiSolution.dot(self.simulation._quad_weights)

    def _phiDeriv_u(self, kyInd, src, v, adjoint=False):
        return Identity() * v

    def _phiDeriv_m(self, kyInd, src, v, adjoint=False):
        return Zero()

    def _j(self, phiSolution, source_list):
        raise NotImplementedError()

    def _e(self, phiSolution, source_list):
        raise NotImplementedError()


class Fields2DCellCentered(Fields2D):
    """
    Fancy Field Storage for a 2.5D cell centered code.
    """

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
    def _GLoc(self, fieldType):
        if fieldType == "phi":
            return "CC"
        elif fieldType == "e" or fieldType == "j":
            return "F"
        else:
            raise Exception("Field type must be phi, e, j")

    def _j(self, phiSolution, source_list):
        phi = self._phi(phiSolution, source_list)
        sim = self.simulation
        return sim.MfRhoI * sim.Grad * phi

    def _e(self, phiSolution, source_list):
        phi = self._phi(phiSolution, source_list)
        sim = self.simulation
        return sim.MfI * sim.Grad * phi

    def _charge(self, phiSolution, source_list):
        """
            .. math::

                \int \nabla \codt \vec{e} =  \int \frac{\rho_v }{\epsillon_0}
        """
        sim = self.simulation
        return (
            epsilon_0
            * sim.mesh.vol[:, None]
            * (sim.mesh.faceDiv * self._e(phiSolution, source_list))
        )

    def _charge_density(self, phiSolution, source_list):
        """
            .. math::

                \frac{1}{V}\int \nabla \codt \vec{e} =
                \frac{1}{V}\int \frac{\rho_v }{\epsillon_0}
        """
        sim = self.simulation
        return epsilon_0 * (sim.mesh.faceDiv * self._e(phiSolution, source_list))


class Fields2DNodal(Fields2D):
    """
    Fancy Field Storage for a 2.5D nodal code.
    """

    knownFields = {"phiSolution": "N"}
    aliasFields = {
        "phi": ["phiSolution", "N", "_phi"],
        "j": ["phiSolution", "E", "_j"],
        "e": ["phiSolution", "E", "_e"],
        "charge_density": ["phiSolution", "CC", "_charge_density"],
        "charge": ["phiSolution", "N", "_charge"],
    }
    # primary - secondary
    # CC variables

    def _GLoc(self, fieldType):
        if fieldType == "phi":
            return "N"
        elif fieldType == "e" or fieldType == "j":
            return "E"
        else:
            raise Exception("Field type must be phi, e, j")

    def _j(self, phiSolution, source_list):
        sim = self.simulation
        return sim.MeI * sim.MeSigma * self._e(phiSolution, source_list)

    def _e(self, phiSolution, source_list):
        """
            In HJ formulation e is not well-defined!!
            .. math::
                \vec{e} = -\nabla \phi
        """
        return -self.mesh.nodalGrad * self._phi(phiSolution, source_list)

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


Fields2DCellCentred = Fields2DCellCentered


############
# Deprecated
############
@deprecate_class(removal_version="0.15.0")
class Fields_ky(Fields2D):
    pass


@deprecate_class(removal_version="0.15.0")
class Fields_ky_CC(Fields2DCellCentered):
    pass


@deprecate_class(removal_version="0.15.0")
class Fields_ky_N(Fields2DNodal):
    pass
