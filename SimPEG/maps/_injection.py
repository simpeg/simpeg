"""
Injection and interpolation map classes.
"""

import warnings
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

    def __init__(self, meshes, active_faceslls=None, indActive=None, **kwargs):
        # Sanity checks for the meshes parameter
        try:
            mesh, mesh2 = meshes
        except TypeError:
            raise TypeError("Couldn't unpack 'meshes' into two meshes.")

        super().__init__(mesh=mesh, **kwargs)

        self.mesh2 = mesh2
        # Check dimensions of both meshes
        if mesh.dim != mesh2.dim:
            raise ValueError(
                f"Found meshes with dimensions '{mesh.dim}' and '{mesh2.dim}'. "
                + "Both meshes must have the same dimension."
            )

        # Deprecate indActive argument
        if indActive is not None:
            if active_faceslls is not None:
                raise TypeError(
                    "Cannot pass both 'active_faceslls' and 'indActive'."
                    "'indActive' has been deprecated and will be removed in "
                    " SimPEG v0.24.0, please use 'active_faceslls' instead.",
                )
            warnings.warn(
                "'indActive' has been deprecated and will be removed in "
                " SimPEG v0.24.0, please use 'active_faceslls' instead.",
                FutureWarning,
                stacklevel=2,
            )
            active_faceslls = indActive

        self.active_faceslls = active_faceslls

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
    def active_faceslls(self):
        """Active indices on target mesh.

        Returns
        -------
        (mesh.n_cells) numpy.ndarray of bool or none
        """
        return self._active_faceslls

    @active_faceslls.setter
    def active_faceslls(self, value):
        if value is not None:
            value = validate_active_indices("active_faceslls", value, self.mesh.n_cells)
        self._active_faceslls = value

    indActive = deprecate_property(
        active_faceslls,
        "indActive",
        "active_faceslls",
        removal_version="0.24.0",
        future_warn=True,
        error=False,
    )

    @property
    def P(self):
        if getattr(self, "_P", None) is None:
            self._P = self.mesh2.get_interpolation_matrix(
                (
                    self.mesh.cell_centers[self.active_faceslls, :]
                    if self.active_faceslls is not None
                    else self.mesh.cell_centers
                ),
                "CC",
                zeros_outside=True,
            )
        return self._P

    @property
    def shape(self):
        """Number of parameters in the model."""
        if self.active_faceslls is not None:
            return (self.active_faceslls.sum(), self.mesh2.nC)
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
    active_faceslls : numpy.ndarray
        Active cells array. Can be a boolean ``numpy.ndarray`` of length *mesh.nC*
        or a ``numpy.ndarray`` of ``int`` containing the indices of the active cells.
    value_inactive : float or numpy.ndarray
        The physical property value assigned to all inactive cells in the mesh
    indActive : numpy.ndarray

        .. deprecated:: 0.23.0

           Argument ``indActive`` is deprecated in favor of ``active_faceslls`` and will
           be removed in SimPEG v0.24.0.

    valInactive : float or numpy.ndarray

        .. deprecated:: 0.23.0

           Argument ``valInactive`` is deprecated in favor of ``value_inactive`` and
           will be removed in SimPEG v0.24.0.

    """

    def __init__(
        self,
        mesh,
        active_faceslls=None,
        value_inactive=0.0,
        nC=None,
        indActive=None,
        valInactive=0.0,
    ):
        self.mesh = mesh
        self.nC = nC or mesh.nC

        # Deprecate indActive argument
        if indActive is not None:
            if active_faceslls is not None:
                raise TypeError(
                    "Cannot pass both 'active_faceslls' and 'indActive'."
                    "'indActive' has been deprecated and will be removed in "
                    " SimPEG v0.24.0, please use 'active_faceslls' instead.",
                )
            warnings.warn(
                "'indActive' has been deprecated and will be removed in "
                " SimPEG v0.24.0, please use 'active_faceslls' instead.",
                FutureWarning,
                stacklevel=2,
            )
            active_faceslls = indActive

        # Deprecate valInactive argument
        if not isinstance(valInactive, Number) or valInactive != 0.0:
            if not isinstance(value_inactive, Number) or value_inactive != 0.0:
                raise TypeError(
                    "Cannot pass both 'value_inactive' and 'valInactive'."
                    "'valInactive' has been deprecated and will be removed in "
                    " SimPEG v0.24.0, please use 'value_inactive' instead.",
                )
            warnings.warn(
                "'valInactive' has been deprecated and will be removed in "
                " SimPEG v0.24.0, please use 'value_inactive' instead.",
                FutureWarning,
                stacklevel=2,
            )
            value_inactive = valInactive

        self.active_faceslls = active_faceslls
        self._nP = np.sum(self.active_faceslls)

        self.P = sp.eye(self.nC, format="csr")[:, self.active_faceslls]

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
        value_inactive[~self.active_faceslls] = value
        self._value_inactive = value_inactive

    valInactive = deprecate_property(
        value_inactive,
        "valInactive",
        "value_inactive",
        removal_version="0.24.0",
        future_warn=True,
        error=False,
    )

    @property
    def active_faceslls(self):
        """

        Returns
        -------
        numpy.ndarray of bool

        """
        return self._active_faceslls

    @active_faceslls.setter
    def active_faceslls(self, value):
        if value is not None:
            value = validate_active_indices("active_faceslls", value, self.nC)
        self._active_faceslls = value

    indActive = deprecate_property(
        active_faceslls,
        "indActive",
        "active_faceslls",
        removal_version="0.24.0",
        future_warn=True,
        error=False,
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
        return int(self.active_faceslls.sum())

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


class InjectActiveFaces(IdentityMap):
    r"""Map active faces model to all faces of a mesh.

    The ``InjectActiveFaces`` class is used to define the mapping when
    the model consists of diagnostic property values defined on a set of active
    mesh faces; e.g. faces below topography, z-faces only. For a discrete set of
    model parameters :math:`\mathbf{m}` defined on a set of active
    faces, the mapping :math:`\mathbf{u}(\mathbf{m})` is defined as:

    .. math::
        \mathbf{u}(\mathbf{m}) = \mathbf{Pm} + \mathbf{d}\, m_\perp

    where :math:`\mathbf{P}` is a (*nF* , *nP*) projection matrix from
    active faces to all mesh faces, and :math:`\mathbf{d}` is a
    (*nF* , 1) matrix that projects the inactive faces value
    :math:`m_\perp` to all mesh faces.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A discretize mesh
    active_faces : numpy.ndarray
        Active faces array. Can be a boolean ``numpy.ndarray`` of length *mesh.nF*
        or a ``numpy.ndarray`` of ``int`` containing the indices of the active faces.
    valInactive : float or numpy.ndarray
        The physical property value assigned to all inactive faces in the mesh

    """

    def __init__(self, mesh, active_faces=None, value_inactive=0.0, nF=None):
        self.mesh = mesh
        self.nF = nF or mesh.nF

        self._active_faces = validate_active_indices("active_faces", active_faces, self.nF)
        self._nP = np.sum(self.active_faces)

        self.P = sp.eye(self.nF, format="csr")[:, self.active_faces]

        self.value_inactive = value_inactive

    @property
    def value_inactive(self):
        """The physical property value assigned to all inactive faces in the mesh.

        Returns
        -------
        numpy.ndarray
        """
        return self._value_inactive

    @value_inactive.setter
    def value_inactive(self, value):
        n_inactive = self.nF - self.nP
        try:
            value = validate_float("value_inactive", value)
            value = np.full(n_inactive, value)
        except Exception:
            pass
        value = validate_ndarray_with_shape("value_inactive", value, shape=(n_inactive,))

        self._value_inactive = np.zeros(self.nF, dtype=float)
        self._value_inactive[~self.active_faces] = value

    @property
    def active_faces(self):
        """

        Returns
        -------
        numpy.ndarray of bool

        """
        return self._active_faces

    @property
    def shape(self):
        """Dimensions of the mapping

        Returns
        -------
        tuple of int
            Where *nP* is the number of active faces and *nF* is
            number of faces in the mesh, **shape** returns a
            tuple (*nF* , *nP*).
        """
        return (self.nF, self.nP)

    @property
    def nP(self):
        """Number of parameters the model acts on.

        Returns
        -------
        int
            Number of parameters the model acts on; i.e. the number of active faces.
        """
        return int(self.active_faces.sum())

    def _transform(self, m):
        if m.ndim > 1:
            return self.P * m + self.value_inactive[:, None]
        return self.P * m + self.value_inactive

    def inverse(self, u):
        r"""Recover the model parameters (active faces) from a set of physical
        property values defined on the entire mesh.

        For a discrete set of model parameters :math:`\mathbf{m}` defined
        on a set of active faces, the mapping :math:`\mathbf{u}(\mathbf{m})`
        is defined as:

        .. math::
            \mathbf{u}(\mathbf{m}) = \mathbf{Pm} + \mathbf{d} \,m_\perp

        where :math:`\mathbf{P}` is a (*nF* , *nP*) projection matrix from
        active faces to all mesh faces, and :math:`\mathbf{d}` is a
        (*nR* , 1) matrix that projects the inactive face value
        :math:`m_\perp` to all mesh faces.

        The inverse mapping is given by:

        .. math::
            \mathbf{m}(\mathbf{u}) = \mathbf{P^T u}

        Parameters
        ----------
        u : (mesh.nF) numpy.ndarray
            A vector which contains physical property values for all
            mesh faces.
        """
        return self.P.T * u

    def deriv(self, m, v=None):
        r"""Derivative of the mapping with respect to the input parameters.

        For a discrete set of model parameters :math:`\mathbf{m}` defined
        on a set of active faces, the mapping :math:`\mathbf{u}(\mathbf{m})`
        is defined as:

        .. math::
            \mathbf{u}(\mathbf{m}) = \mathbf{Pm} + \mathbf{d} \, m_\perp

        where :math:`\mathbf{P}` is a (*nF* , *nP*) projection matrix from
        active faces to all mesh faces, and :math:`\mathbf{d}` is a
        (*nF* , 1) matrix that projects the inactive face value
        :math:`m_\perp` to all inactive mesh faces.

        the **deriv** method returns the derivative of :math:`\mathbf{u}` with respect
        to the model parameters; i.e.:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}} = \mathbf{P}

        Note that in this case, **deriv** simply returns a sparse projection matrix.

        Parameters
        ----------
        m : (nP) numpy.ndarray
            A vector representing a set of model parameters.
        v : (nP) numpy.ndarray
            If not ``None``, the method returns the derivative times the vector *v*.

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


class InjectActiveEdges(IdentityMap):
    r"""Map active edges model to all edges of a mesh.

    The ``InjectActiveEdges`` class is used to define the mapping when
    the model consists of diagnostic property values defined on a set of active
    mesh edges; e.g. edges below topography, z-edges only. For a discrete set of
    model parameters :math:`\mathbf{m}` defined on a set of active
    edges, the mapping :math:`\mathbf{u}(\mathbf{m})` is defined as:

    .. math::
        \mathbf{u}(\mathbf{m}) = \mathbf{Pm} + \mathbf{d}\, m_\perp

    where :math:`\mathbf{P}` is a (*nE* , *nP*) projection matrix from
    active edges to all mesh edges, and :math:`\mathbf{d}` is a
    (*nE* , 1) matrix that projects the inactive edges value
    :math:`m_\perp` to all mesh edges.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        A discretize mesh
    active_edges : numpy.ndarray
        Active edges array. Can be a boolean ``numpy.ndarray`` of length *mesh.nE*
        or a ``numpy.ndarray`` of ``int`` containing the indices of the active edges.
    value_inactive : float or numpy.ndarray
        The physical property value assigned to all inactive edges in the mesh.

    """

    def __init__(self, mesh, active_edges=None, value_inactive=0.0, nE=None):
        self.mesh = mesh
        self.nE = nE or mesh.nE

        self._active_edges = validate_active_indices("active_edges", active_edges, self.nE)
        self._nP = np.sum(self.active_edges)

        self.P = sp.eye(self.nE, format="csr")[:, self.active_edges]

        self.value_inactive = value_inactive

    @property
    def value_inactive(self):
        """The physical property value assigned to all inactive edges in the mesh.

        Returns
        -------
        numpy.ndarray
        """
        return self._value_inactive

    @value_inactive.setter
    def value_inactive(self, value):
        n_inactive = self.nE - self.nP
        try:
            value = validate_float("value_inactive", value)
            value = np.full(n_inactive, value)
        except Exception:
            pass
        value = validate_ndarray_with_shape("value_inactive", value, shape=(n_inactive,))

        self._value_inactive = np.zeros(self.nE, dtype=float)
        self._value_inactive[~self.active_edges] = value

    @property
    def active_edges(self):
        """

        Returns
        -------
        numpy.ndarray of bool.

        """
        return self._active_edges

    @property
    def shape(self):
        """Dimensions of the mapping

        Returns
        -------
        tuple of int
            Where *nP* is the number of active edges and *nE* is
            number of edges in the mesh, **shape** returns a
            tuple (*nE* , *nP*).
        """
        return (self.nE, self.nP)

    @property
    def nP(self):
        """Number of parameters the model acts on.

        Returns
        -------
        int
            Number of parameters the model acts on; i.e. the number of active edges.
        """
        return int(self.active_edges.sum())

    def _transform(self, m):
        if m.ndim > 1:
            return self.P * m + self.value_inactive[:, None]
        return self.P * m + self.value_inactive

    def inverse(self, u):
        r"""Recover the model parameters (active edges) from a set of physical
        property values defined on the entire mesh.

        For a discrete set of model parameters :math:`\mathbf{m}` defined
        on a set of active edges, the mapping :math:`\mathbf{u}(\mathbf{m})`
        is defined as:

        .. math::
            \mathbf{u}(\mathbf{m}) = \mathbf{Pm} + \mathbf{d} \,m_\perp

        where :math:`\mathbf{P}` is a (*nE* , *nP*) projection matrix from
        active edges to all mesh edges, and :math:`\mathbf{d}` is a
        (*nE* , 1) matrix that projects the inactive edge value
        :math:`m_\perp` to all mesh edges.

        The inverse mapping is given by:

        .. math::
            \mathbf{m}(\mathbf{u}) = \mathbf{P^T u}

        Parameters
        ----------
        u : (mesh.nE) numpy.ndarray
            A vector which contains physical property values for all
            mesh edges.
        """
        return self.P.T * u

    def deriv(self, m, v=None):
        r"""Derivative of the mapping with respect to the input parameters.

        For a discrete set of model parameters :math:`\mathbf{m}` defined
        on a set of active edges, the mapping :math:`\mathbf{u}(\mathbf{m})`
        is defined as:

        .. math::
            \mathbf{u}(\mathbf{m}) = \mathbf{Pm} + \mathbf{d} \, m_\perp

        where :math:`\mathbf{P}` is a (*nE* , *nP*) projection matrix from
        active edges to all mesh edges, and :math:`\mathbf{d}` is a
        (*nF* , 1) matrix that projects the inactive edge value
        :math:`m_\perp` to all mesh edges.

        the **deriv** method returns the derivative of :math:`\mathbf{u}` with respect
        to the model parameters; i.e.:

        .. math::
            \frac{\partial \mathbf{u}}{\partial \mathbf{m}} = \mathbf{P}

        Note that in this case, **deriv** simply returns a sparse projection matrix.

        Parameters
        ----------
        m : (nP) numpy.ndarray
            A vector representing a set of model parameters.
        v : (nP) numpy.ndarray
            If not ``None``, the method returns the derivative times the vector *v*.

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
