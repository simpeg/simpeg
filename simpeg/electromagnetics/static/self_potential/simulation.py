import scipy.sparse as sp
import numpy as np

from .... import maps, props
from ....utils import validate_list_of_types, validate_active_indices
from .. import resistivity as dc
from .sources import StreamingCurrents


class Simulation3DCellCentered(dc.Simulation3DCellCentered):
    r"""A self potential simulation.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
    survey : simpeg.electromagnetics.static.self_potential.Survey
    sigma, rho : float or array_like
        The conductivity/resistivity model of the subsurface.
    q : float, array_like, optional
        The charge density accumulation rate model (C/(s m^3)), also
        physically represents the volumetric current density (A/m^3).
    qMap : simpeg.maps.IdentityMap, optional
        The mapping used to go from the simulation model to `q`. Set this
        to invert for `q`.
    **kwargs
        arguments passed on to :class:`.resistivity.Simulation3DCellCentered`

    Notes
    -----
    The charge density accumulation rate, :math:`q`, is related to the self
    electric potential, :math:`\phi`, with the same PDE, that relates current
    sources to potential in the resistivity case.

    .. math:: - \nabla \cdot \sigma \nabla \phi = q

    This equation is solve for potential with a finite volume approach,
    discretized with :math:`\phi` and :math:`q` on cell centers, electrical
    conductivity :math`\sigma` as a cell property, and therefore current density
    lives on the faces between cells.

    By default the boundary conditions assume a Robin condition on the subsurface
    boundaries, and a zero Nuemann boundary at the top. For more details on the
    boundary conditions, check out the resistivity simulations.
    """

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
            **kwargs,
        )
        self.q = q
        self.qMap = qMap

    def getRHS(self):
        return self.Vol @ self.q

    def getRHSDeriv(self, source, v, adjoint=False):
        if adjoint:
            return self.qDeriv.T @ (self.Vol @ v)
        return self.Vol @ (self.qDeriv @ v)

    @property
    def deleteTheseOnModelUpdate(self):
        # When enabling resistivity derivatives, uncomment these lines
        # if self.rhoMap is not None:
        #     return super().deleteTheseOnModelUpdate
        if self.storeJ and self.qMap is not None and not self.qMap.is_linear:
            return ["_Jmatrix", "_gtgdiag"]
        return []


class CurrentDensityMap(maps.LinearMap):
    r"""Maps current density to charge density accumulation rate.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
    active_cells : index_array, optional
        Defaults to all cells being active. This is used to apply a
        0 Nuemann boundary condition at topographic faces.

    Notes
    -----
    This map relates current density :math:`\vec{j}_s` in `A/m^2` to the
    charge density accumulation rate :math:`q` (`C/(s m^3)`) by taking
    the negative divergence.

    .. math::

        q = -\nabla \cdot \vec{j}_s
    """

    def __init__(self, mesh, active_cells=None):
        cv_to_f = mesh.average_cell_vector_to_face
        if active_cells is not None:
            active_cells = validate_active_indices(
                "active_cells", active_cells, mesh.n_cells
            )
            active_cell_comps = np.concatenate(mesh.dim * [active_cells])
            # need to repeat above for each vector component for cells
            # Test which faces are on the boundary of the active domain
            boundary_faces = (mesh.face_divergence.T @ active_cells) != 0
            # use to set the value of j on the boundaries to zero
            # do not need to worry about the faces that are interior to
            # the inactive cells because the values on each side of a
            # face are already set to zero from the projection from active
            # to full space.
            face_weights = sp.diags(1.0 * (~boundary_faces))
            cv_to_f = face_weights @ cv_to_f[:, active_cell_comps]
        A = -mesh.face_divergence @ cv_to_f
        super().__init__(A)


class HydraulicHeadMap(maps.LinearMap):
    r"""Maps hydraulic head to charge density accumulation rate.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
    L : float or (mesh.n_cells,) array_like
        Cross coupling property model (`A/m^2`).

    Notes
    -----
    This map relates hydraulic head, :math:`h`, in `m` to the volumetric current
    density, :math:`q`, (`A/m^3`) using a cross coupling term to convert the
    negative gradient of hydrualic head into secondary current densities.
    This cross coupling parameter :math:`L` is a multiplication of the hydraulic
    conductivity (m/s) of the material and the ionic density (C/m^3)
    of the fluid.

    The mapping is:

    .. math::

        q = \nabla \cdot L \nabla h
    """

    def __init__(self, mesh, L):
        div = mesh.face_divergence
        MfLiI = mesh.get_face_inner_product(L, invert_model=True, invert_matrix=True)
        A = div @ MfLiI @ div.T @ sp.diags(mesh.cell_volumes, format="csr")
        super().__init__(A)


class Survey(dc.Survey):
    """Streaming potential survey

    Parameters
    ----------
    source_list : list of .sources.StreamingCurrents
    """

    @property
    def source_list(self):
        """List of sources.

        Returns
        -------
        list of .sources.StreamingCurrents
        """
        return self._source_list

    @source_list.setter
    def source_list(self, new_list):
        self._source_list = validate_list_of_types(
            "source_list",
            new_list,
            StreamingCurrents,
        )
