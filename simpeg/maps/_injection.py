"""
Injection and interpolation map classes.
"""

import discretize
import numpy as np
import scipy.sparse as sp
from numbers import Number

from ..utils import (
    validate_type,
    validate_ndarray_with_shape,
    validate_float,
    validate_active_indices,
)
from ._base import IdentityMap
from ..utils.code_utils import deprecate_property


class Mesh2Mesh(IdentityMap):
    """
    Takes a model on one mesh are translates it to another mesh.
    """

    def __init__(self, meshes, active_cells=None, **kwargs):
        # Sanity checks for the meshes parameter
        try:
            mesh, mesh2 = meshes
        except TypeError:
            raise TypeError("Couldn't unpack 'meshes' into two meshes.")

        # Deprecate indActive argument
        if kwargs.pop("indActive", None) is not None:
            raise TypeError(
                "'indActive' was removed in SimPEG v0.24.0, please use 'active_cells' instead.",
            )

        super().__init__(mesh=mesh, **kwargs)

        self.mesh2 = mesh2
        # Check dimensions of both meshes
        if mesh.dim != mesh2.dim:
            raise ValueError(
                f"Found meshes with dimensions '{mesh.dim}' and '{mesh2.dim}'. "
                + "Both meshes must have the same dimension."
            )

        self.active_cells = active_cells

    # reset to not accepted None for mesh
    @IdentityMap.mesh.setter
    def mesh(self, value):
        self._mesh = validate_type("mesh", value, discretize.base.BaseMesh, cast=False)

    @property
    def mesh2(self):
        """The source mesh used for the mapping.

        Returns
        -------
        discretize.base.BaseMesh
        """
        return self._mesh2

    @mesh2.setter
    def mesh2(self, value):
        self._mesh2 = validate_type(
            "mesh2", value, discretize.base.BaseMesh, cast=False
        )

    @property
    def active_cells(self):
        """Active indices on target mesh.

        Returns
        -------
        (mesh.n_cells) numpy.ndarray of bool or none
        """
        return self._active_cells

    @active_cells.setter
    def active_cells(self, value):
        if value is not None:
            value = validate_active_indices("active_cells", value, self.mesh.n_cells)
        self._active_cells = value

    indActive = deprecate_property(
        active_cells,
        "indActive",
        "active_cells",
        removal_version="0.24.0",
        error=True,
    )

    @property
    def P(self):
        if getattr(self, "_P", None) is None:
            self._P = self.mesh2.get_interpolation_matrix(
                (
                    self.mesh.cell_centers[self.active_cells, :]
                    if self.active_cells is not None
                    else self.mesh.cell_centers
                ),
                "CC",
                zeros_outside=True,
            )
        return self._P

    @property
    def shape(self):
        """Number of parameters in the model."""
        if self.active_cells is not None:
            return (self.active_cells.sum(), self.mesh2.nC)
        return (self.mesh.nC, self.mesh2.nC)

    @property
    def nP(self):
        """Number of parameters in the model."""
        return self.mesh2.nC

    def _transform(self, m):
        return self.P * m

    def deriv(self, m, v=None):
        if v is not None:
            return self.P * v
        return self.P


class InjectActiveCells(IdentityMap):
    r"""Map active cells model to all cell of a mesh.

    The ``InjectActiveCells`` class is used to define the mapping when
    the model consists of physical property values for a set of active
    mesh cells; e.g. cells below topography. For a discrete set of
    model parameters :math:`\mathbf{m}` defined on a set of active
    cells, the mapping :math:`\mathbf{u}(\mathbf{m})` is defined as:

    .. math::
        \mathbf{u}(\mathbf{m}) = \mathbf{Pm} + \mathbf{d}\, m_\perp

    where :math:`\mathbf{P}` is a (*nC* , *nP*) projection matrix from
    active cells to all mesh cells, and :math:`\mathbf{d}` is a
    (*nC* , 1) matrix that projects the inactive cell value
    :math:`m_\perp` to all inactive mesh cells.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A discretize mesh
    active_cells : numpy.ndarray
        Active cells array. Can be a boolean ``numpy.ndarray`` of length *mesh.nC*
        or a ``numpy.ndarray`` of ``int`` containing the indices of the active cells.
    value_inactive : float or numpy.ndarray
        The physical property value assigned to all inactive cells in the mesh
    """

    def __init__(
        self,
        mesh,
        active_cells=None,
        value_inactive=0.0,
        nC=None,
        **kwargs,
    ):
        self.mesh = mesh
        self.nC = nC or mesh.nC

        # Deprecate indActive argument
        if kwargs.pop("indActive", None) is not None:
            raise TypeError(
                "'indActive' was removed in SimPEG v0.24.0, please use 'active_cells' instead."
            )
        # Deprecate valInactive argument
        if kwargs.pop("valInactive", None) is not None:
            raise TypeError(
                "'valInactive' was removed in SimPEG v0.24.0, please use 'value_inactive' instead."
            )
        if kwargs:  # TODO Remove this when removing kwargs argument.
            raise TypeError("Unsupported keyword argument " + kwargs.popitem()[0])

        self.active_cells = active_cells
        self._nP = np.sum(self.active_cells)

        self.P = sp.eye(self.nC, format="csr")[:, self.active_cells]

        self.value_inactive = value_inactive

    @property
    def value_inactive(self):
        """The physical property value assigned to all inactive cells in the mesh.

        Returns
        -------
        numpy.ndarray
        """
        return self._value_inactive

    @value_inactive.setter
    def value_inactive(self, value):
        n_inactive = self.nC - self.nP
        if isinstance(value, Number):
            value = validate_float("value_inactive", value)
            value = np.full(n_inactive, value)
        value = validate_ndarray_with_shape(
            "value_inactive", value, shape=(n_inactive,)
        )
        value_inactive = np.zeros(self.nC, dtype=float)
        value_inactive[~self.active_cells] = value
        self._value_inactive = value_inactive

    valInactive = deprecate_property(
        value_inactive,
        "valInactive",
        "value_inactive",
        removal_version="0.24.0",
        error=True,
    )

    @property
    def active_cells(self):
        """A boolean array representing the active values in the map's output array.

        Returns
        -------
        numpy.ndarray of bool

        """
        return self._active_cells

    @active_cells.setter
    def active_cells(self, value):
        if value is not None:
            value = validate_active_indices("active_cells", value, self.nC)
        self._active_cells = value

    indActive = deprecate_property(
        active_cells,
        "indActive",
        "active_cells",
        removal_version="0.24.0",
        error=True,
    )

    @property
    def shape(self):
        """Dimensions of the mapping

        Returns
        -------
        tuple of int
            Where *nP* is the number of active cells and *nC* is
            number of cell in the mesh, **shape** returns a
            tuple (*nC* , *nP*).
        """
        return (self.nC, self.nP)

    @property
    def nP(self):
        """Number of parameters the model acts on.

        Returns
        -------
        int
            Number of parameters the model acts on; i.e. the number of active cells
        """
        return int(self.active_cells.sum())

    def _transform(self, m):
        if m.ndim > 1:
            return self.P * m + self.value_inactive[:, None]
        return self.P * m + self.value_inactive

    def inverse(self, u):
        r"""Recover the model parameters (active cells) from a set of physical
        property values defined on the entire mesh.

        For a discrete set of model parameters :math:`\mathbf{m}` defined
        on a set of active cells, the mapping :math:`\mathbf{u}(\mathbf{m})`
        is defined as:

        .. math::
            \mathbf{u}(\mathbf{m}) = \mathbf{Pm} + \mathbf{d} \,m_\perp

        where :math:`\mathbf{P}` is a (*nC* , *nP*) projection matrix from
        active cells to all mesh cells, and :math:`\mathbf{d}` is a
        (*nC* , 1) matrix that projects the inactive cell value
        :math:`m_\perp` to all inactive mesh cells.

        The inverse mapping is given by:

        .. math::
            \mathbf{m}(\mathbf{u}) = \mathbf{P^T u}

        Parameters
        ----------
        u : (mesh.nC) numpy.ndarray
            A vector which contains physical property values for all
            mesh cells.
        """
        return self.P.T * u

    def deriv(self, m, v=None):
        r"""Derivative of the mapping with respect to the input parameters.

        For a discrete set of model parameters :math:`\mathbf{m}` defined
        on a set of active cells, the mapping :math:`\mathbf{u}(\mathbf{m})`
        is defined as:

        .. math::
            \mathbf{u}(\mathbf{m}) = \mathbf{Pm} + \mathbf{d} \, m_\perp

        where :math:`\mathbf{P}` is a (*nC* , *nP*) projection matrix from
        active cells to all mesh cells, and :math:`\mathbf{d}` is a
        (*nC* , 1) matrix that projects the inactive cell value
        :math:`m_\perp` to all inactive mesh cells.

        the **deriv** method returns the derivative of :math:`\mathbf{u}` with respect
        to the model parameters; i.e.:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}} = \mathbf{P}

        Note that in this case, **deriv** simply returns a sparse projection matrix.

        Parameters
        ----------
        m : (nP) numpy.ndarray
            A vector representing a set of model parameters
        v : (nP) numpy.ndarray
            If not ``None``, the method returns the derivative times the vector *v*

        Returns
        -------
        scipy.sparse.csr_matrix
            Derivative of the mapping with respect to the model parameters. If the
            input argument *v* is not ``None``, the method returns the derivative times
            the vector *v*.
        """
        if v is not None:
            return self.P * v
        return self.P
