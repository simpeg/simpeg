import scipy.sparse as sp

from .... import maps, props
from ....utils import validate_list_of_types
from .. import resistivity as dc
from .sources import StreamingCurrents


class Simulation3DCellCentered(dc.Simulation3DCellCentered):
    r"""A Spontaneous potential simulation.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
    survey : spontaneous_potential.Survey
    sigma, rho : float or array_like
        The conductivity/resistivity model of the subsurface.
    q : float, array_like, optional
        The charge density accumulation rate model (C/(s m^3)), also
        physically represents the volumetric current density (A/m^3).
    qMap : SimPEG.maps.IdentityMap, optional
        The mapping used to go from the simulation model to `q`. Set this
        to invert for `q`.
    **kwargs
        arguments passed on to :class:`.resistivity.Simulation3DCellCentered`

    Notes
    -----
    The charge density accumulation rate, :math:`q`, is related to the spontaneous
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
            **kwargs
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
        # Need some way to determine if the `qMap` was linear...
        # if it was non-linear, we would need to re-compute a
        # stored jacobian matrix.
        return []


class CurrentDensityMap(maps.LinearMap):
    r"""Maps current density to charge density accumulation rate.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh

    Notes
    -----
    This map relates current density :math:`\vec{j}_s` in `A/m^2` to the
    charge density accumulation rate :math:`q` (`C/(s m^3)`) by taking
    the negative divergence.

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
    source_list : list of sources.StreamingCurrents
    """

    @property
    def source_list(self):
        """List of sources.

        Returns
        -------
        list of sources.StreamingCurrents
        """
        return self._source_list

    @source_list.setter
    def source_list(self, new_list):
        self._source_list = validate_list_of_types(
            "source_list",
            new_list,
            StreamingCurrents,
        )
