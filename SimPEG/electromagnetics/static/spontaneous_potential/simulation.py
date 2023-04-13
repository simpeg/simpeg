import numpy as np
import scipy.sparse as sp

from .... import maps, props
from ....utils import validate_list_of_types
from ..resistivity import Simulation3DCellCentered
from ....survey import BaseSurvey
from .sources import StreamingCurrents


class Simulation3DCellCentered(Simulation3DCellCentered):
    q, qMap, qDeriv = props.Invertible("Charge density accumulation rate (C/(s m^3))")

    def __init__(
        self, mesh, survey=None, sigma=None, rho=None, q=None, qMap=None, **kwargs
    ):
        if sigma is None:
            if rho is None:
                raise ValueError("Must set either conductivity or resistivity.")
        else:
            if rho is not None:
                raise ValueError("Cannot set both conductivity and resistivity.")
        super().__init__(
            mesh=mesh,
            survey=survey,
            sigma=sigma,
            rho=rho,
            sigmaMap=None,
            rhoMap=None,
            **kwargs
        )
        self.q = q
        self.qMap = qMap

    def getRHS(self):
        return self.Vol @ self.q

    def getRHSDeriv(self, source, v, adjoint=False):
        # The q deriv is taken care of in the inherited Jvec and Jtvec ops
        # (to handle potentially non-linear maps)
        return self.Vol @ v

    def getJtJdiag(self, m, W=None, f=None):
        diag = sp.diags(np.sqrt(super().getJtJdiag(m, W=W, f=f)))
        return np.asarray((diag @ self.qDeriv).power(2).sum(axis=0)).flatten()

    def Jvec(self, m, v, f=None):
        v = self.qDeriv @ v
        return super().Jvec(m, v, f=f)

    def Jtvec(self, m, v, f=None):
        v = super().Jtvec(m, v, f=f)
        return self.qDeriv.T @ v

    @property
    def deleteTheseOnModelUpdate(self):
        if self.rhoMap is None:
            return []
        else:
            return super().deleteTheseOnModelUpdate


class CurrentDensityMap(maps.LinearMap):
    r"""Maps current density to charge density accumulation rate.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh

    Notes
    -----
    .. math::
        q = -\nabla \cdot \vec{j}_s
    """

    def __init__(self, mesh):
        A = -mesh.face_divergence @ mesh.AvgCCV2F
        super().__init__(A)


class HydraulicHeadMap(maps.LinearMap):
    """Maps hydraulic head to charge density accumulation rate.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
    L : array_like
        Cross coupling property model (units????)

    Notes
    -----
    .. math::
        q = \nabla \cdot L \nabla h
    """

    def __init__(self, mesh, L):
        div = mesh.face_divergence
        MfLiI = mesh.get_face_inner_product(L, invert_model=True, invert_matrix=True)
        A = div.T @ MfLiI @ div @ sp.diags(self.mesh.cell_volumes, format="csr")
        super().__init__(A)


class Survey(BaseSurvey):
    @property
    def source_list(self):
        return self._source_list

    @source_list.setter
    def source_list(self, new_list):
        self._source_list = validate_list_of_types(
            "source_list",
            new_list,
            StreamingCurrents,
        )
