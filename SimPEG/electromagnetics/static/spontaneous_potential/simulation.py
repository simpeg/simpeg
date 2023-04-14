import scipy.sparse as sp

from .... import maps, props
from ....utils import validate_list_of_types
from ..resistivity import Simulation3DCellCentered, Survey as BaseDCSurvey
from .sources import StreamingCurrents


class Simulation3DCellCentered(Simulation3DCellCentered):
    q, qMap, qDeriv = props.Invertible("Charge density accumulation rate (C/(s m^3))")

    def __init__(
        self, mesh, survey=None, sigma=None, rho=None, q=None, qMap=None, **kwargs
    ):
        # These below checks can be commented out, correspondingly do
        # not set sigmaMap and rhoMap to None on the super call, to enable
        # derivatives with respect to resistivity/conductivity.
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
        if adjoint:
            return self.qDeriv.T @ (self.Vol @ v)
        return self.Vol @ (self.qDeriv @ v)

    @property
    def deleteTheseOnModelUpdate(self):
        # When enabling resistivity derivatives, uncomment these lines
        # if self.rhoMap is not None:
        #     return super().deleteTheseOnModelUpdate
        return []


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
        A = -mesh.face_divergence @ mesh.average_cell_vector_to_face
        super().__init__(A)


class HydraulicHeadMap(maps.LinearMap):
    r"""Maps hydraulic head to charge density accumulation rate.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
    L : array_like
        Cross coupling property model (A/m^2)

    Notes
    -----
    .. math::
        q = \nabla \cdot L \nabla h
    """

    def __init__(self, mesh, L):
        div = mesh.face_divergence
        MfLiI = mesh.get_face_inner_product(L, invert_model=True, invert_matrix=True)
        A = div @ MfLiI @ div.T @ sp.diags(mesh.cell_volumes, format="csr")
        super().__init__(A)


class Survey(BaseDCSurvey):
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
