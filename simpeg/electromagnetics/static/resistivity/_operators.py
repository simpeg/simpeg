"""
Discrete operators for the 3D DC resistivity problem.

The classes in this module assemble the system matrix of the DC resistivity
problem, and its derivative with respect to the model, for a given mesh and
boundary condition. They hold only quantities that depend on the mesh. Anything
that depends on the physical properties (inner product matrices and their
derivatives) is requested from the object that owns those properties, which is
passed to the methods as ``physprops``. Any
:class:`~simpeg.base.BaseElectricalPDESimulation` can be used as ``physprops``.

This allows other simulations that need to solve a DC problem (e.g. to compute
the initial fields of a grounded source in the time domain) to reuse the
discretization without having to create a DC resistivity simulation.
"""

import numpy as np
import scipy.sparse as sp
from discretize.utils import make_boundary_bool

from ....base.pde_simulation import _inner_mat_mul_op
from ....utils import Zero, validate_string


def _top_boundary_faces(mesh):
    """Find the boundary faces on the top of the mesh.

    This is the default choice for the surface faces. It does not look at the
    model or at the topography: it only selects the uppermost layer of
    boundary faces of the mesh.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh

    Returns
    -------
    (n_boundary_faces, ) numpy.ndarray of bool
        ``True`` for the boundary faces on the top of the mesh.
    """
    boundary_faces = mesh.boundary_faces
    if mesh._meshType.lower() == "tree":
        top_v = np.max(mesh.nodes[:, -1])
        return boundary_faces[:, -1] == top_v
    elif mesh._meshType.lower() in ["tensor", "curv"]:
        # mesh faces are ordered, faces_x, faces_y, faces_z so...
        if mesh.dim == 2:
            is_b = make_boundary_bool(mesh.shape_faces_y)
            is_t = np.zeros(mesh.shape_faces_y, dtype=bool, order="F")
            is_t[:, -1] = True
        else:
            is_b = make_boundary_bool(mesh.shape_faces_z)
            is_t = np.zeros(mesh.shape_faces_z, dtype=bool, order="F")
            is_t[:, :, -1] = True
        is_t = is_t.reshape(-1, order="F")[is_b]
        surface_faces = np.zeros(boundary_faces.shape[0], dtype=bool)
        surface_faces[-len(is_t) :] = is_t
        return surface_faces
    raise NotImplementedError(
        f"Unable to infer surface boundaries for {type(mesh)}, please "
        f"set the `surface_faces` property."
    )


def _robin_alpha(mesh, surface_faces):
    r"""Coefficient of the Robin condition on the boundary faces of the mesh.

    The Robin boundary condition is :math:`\alpha \phi + \partial_n \phi = 0`.
    The surface faces get a homogeneous Neumann condition (:math:`\alpha = 0`).
    On every other boundary face, the potential is assumed to decay as the one
    of a point source located at the middle of the top of the mesh.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
    surface_faces : (n_boundary_faces, ) numpy.ndarray of bool
        The boundary faces that are on the surface.

    Returns
    -------
    (n_boundary_faces, ) numpy.ndarray
    """
    boundary_faces = mesh.boundary_faces
    boundary_normals = mesh.boundary_face_outward_normals

    # Top gets 0 Nuemann
    alpha = np.zeros(len(boundary_faces))

    # assume a source point at the middle of the top of the mesh
    middle = np.median(mesh.nodes, axis=0)
    top_v = np.max(mesh.nodes[:, -1])
    source_point = np.r_[middle[:-1], top_v]

    # Others: Robin: alpha * phi + d phi dn = 0
    # where alpha = 1 / r  * r_hat_dot_n
    # TODO: Implement Zhang et al. (1995)
    r_vec = boundary_faces - source_point
    r = np.linalg.norm(r_vec, axis=-1)
    r_hat = r_vec / r[:, None]
    r_dot_n = np.einsum("ij,ij->i", r_hat, boundary_normals)

    not_top = ~surface_faces
    alpha[not_top] = (r_dot_n / r)[not_top]
    return alpha


def _remove_null_space(A, symmetric=False):
    """Perturb the first row of ``A`` to remove the null space of constants.

    By default, the first row is replaced by a row of the identity. If
    ``symmetric`` is ``True``, one is added to the first diagonal entry
    instead, which keeps the matrix symmetric.
    """
    if symmetric:
        A[0, 0] = A[0, 0] + 1.0
        return A
    I, J, V = sp.find(A[0, :])
    for jj in J:
        A[0, jj] = 0.0
    A[0, 0] = 1.0
    return A


class CellCenteredDCOperator:
    r"""Cell centered discretization of the 3D DC resistivity problem.

    The electric potentials live on cell centers and the system matrix is

    .. math::
        \mathbf{A} = \mathbf{D \, M_{f\rho}^{-1} \, G}

    where :math:`\mathbf{D}` is the face divergence scaled by the cell volumes,
    :math:`\mathbf{G}` is the cell gradient with the boundary conditions
    imposed, and :math:`\mathbf{M_{f\rho}^{-1}}` is the inverse of the inner
    product matrix for resistivities projected to faces.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
    bc_type : {"Robin", "Dirichlet", "Neumann", "Mixed"}
        Type of boundary condition. ``"Mixed"`` is equivalent to ``"Robin"``.
    surface_faces : None or (n_boundary_faces, ) numpy.ndarray of bool, optional
        The boundary faces that are on the surface, used for the Robin
        condition. If ``None``, the faces on the top of the mesh are used.

    Attributes
    ----------
    Div : (n_cells, n_faces) scipy.sparse.csr_matrix
        Face divergence scaled by the cell volumes.
    Grad : (n_faces, n_cells) scipy.sparse.csr_matrix
        Cell gradient with the boundary conditions imposed.
    surface_faces : None or (n_boundary_faces, ) numpy.ndarray of bool
        The surface faces used for the Robin condition. ``None`` for the other
        boundary conditions.
    """

    def __init__(self, mesh, bc_type="Robin", surface_faces=None):
        self.mesh = mesh
        self.bc_type = validate_string(
            "bc_type", bc_type, ["Dirichlet", "Neumann", ("Robin", "Mixed")]
        )
        self.surface_faces = None

        V = sp.diags(mesh.cell_volumes)
        self.Div = V @ mesh.face_divergence
        self.Grad = self.Div.T

        if self.bc_type == "Dirichlet":
            # Homogeneous Dirichlet is the natural BC for this CC discretization.
            return
        elif self.bc_type == "Neumann":
            alpha, beta, gamma = 0, 1, 0
        else:
            if surface_faces is None:
                surface_faces = _top_boundary_faces(mesh)
            self.surface_faces = surface_faces
            alpha = _robin_alpha(mesh, surface_faces)
            beta = np.ones(len(alpha))
            gamma = 0

        B, bc = mesh.cell_gradient_weak_form_robin(alpha, beta, gamma)
        # bc should always be 0 because gamma was always 0 above
        self.Grad = self.Grad - B

    def system_matrix(self, physprops):
        """System matrix of the DC resistivity problem.

        Parameters
        ----------
        physprops : simpeg.base.BaseElectricalPDESimulation
            Object that holds the electrical properties. Its ``MfRhoI``
            property is used.

        Returns
        -------
        (n_cells, n_cells) scipy.sparse.csr_matrix
        """
        return self._assemble(physprops.MfRhoI)

    def _assemble(self, MfRhoI):
        A = self.Div @ MfRhoI @ self.Grad
        if self.bc_type == "Neumann":
            A = _remove_null_space(A)
        return A

    def system_matrix_deriv(self, physprops, phi, v, adjoint=False):
        r"""Derivative of the system matrix times the potentials, times a vector.

        For fixed potentials :math:`\boldsymbol{\phi}`, compute

        .. math::
            \frac{\partial (\mathbf{A} \boldsymbol{\phi})}{\partial \mathbf{m}}
            \, \mathbf{v}

        or the adjoint operation.

        Parameters
        ----------
        physprops : simpeg.base.BaseElectricalPDESimulation
            Object that holds the electrical properties. Its ``rhoMap`` and
            ``MfRhoIDeriv`` are used.
        phi : (n_cells, ) numpy.ndarray
            Electric potentials on cell centers.
        v : numpy.ndarray
            The vector. (n_param, ) for the standard operation, (n_cells, ) for
            the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray or simpeg.utils.Zero
            (n_cells, ) for the standard operation, (n_param, ) for the adjoint
            operation.
        """
        if physprops.rhoMap is None:
            return Zero()
        D = self.Div
        G = self.Grad
        if adjoint:
            return physprops.MfRhoIDeriv(G @ phi, D.T @ v, adjoint)
        return D * physprops.MfRhoIDeriv(G @ phi, v, adjoint)


class NodalDCOperator:
    r"""Nodal discretization of the 3D DC resistivity problem.

    The electric potentials live on nodes and the system matrix is

    .. math::
        \mathbf{A} = \mathbf{G^T \, M_{e\sigma} \, G}

    where :math:`\mathbf{G}` is the nodal gradient and
    :math:`\mathbf{M_{e\sigma}}` is the inner product matrix for conductivities
    projected to edges. A diagonal term that depends on the conductivity is
    added for the Robin boundary condition.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
    bc_type : {"Robin", "Neumann", "Mixed"}
        Type of boundary condition. ``"Mixed"`` is equivalent to ``"Robin"``.
    surface_faces : None or (n_boundary_faces, ) numpy.ndarray of bool, optional
        The boundary faces that are on the surface, used for the Robin
        condition. If ``None``, the faces on the top of the mesh are used.
    symmetric_null_space_fix : bool, optional
        How to remove the null space of constants for the Neumann condition.
        If ``False``, the first row of the system matrix is replaced by a row
        of the identity. If ``True``, one is added to its first diagonal entry
        instead, which keeps the system matrix symmetric.

    Attributes
    ----------
    Grad : (n_edges, n_nodes) scipy.sparse.csr_matrix
        Nodal gradient.
    surface_faces : None or (n_boundary_faces, ) numpy.ndarray of bool
        The surface faces used for the Robin condition. ``None`` for the
        Neumann condition.

    Notes
    -----
    For the Robin boundary condition, :meth:`system_matrix_deriv` stashes a
    matrix that depends on the model in the ``_MBC_sigma`` attribute of
    ``physprops``. The owner of the properties must delete this attribute when
    the conductivity is updated.
    """

    def __init__(
        self,
        mesh,
        bc_type="Robin",
        surface_faces=None,
        symmetric_null_space_fix=False,
    ):
        self.mesh = mesh
        self.bc_type = validate_string(
            "bc_type", bc_type, ["Neumann", ("Robin", "Mixed")]
        )
        self.symmetric_null_space_fix = symmetric_null_space_fix
        self.surface_faces = None
        self.Grad = mesh.nodal_gradient

        if self.bc_type == "Neumann":
            # Homogeneous Neumann is the natural BC for this nodal discretization.
            return

        if surface_faces is None:
            surface_faces = _top_boundary_faces(mesh)
        self.surface_faces = surface_faces
        alpha = _robin_alpha(mesh, surface_faces)

        P_bf = mesh.project_face_to_boundary_face

        AvgN2Fb = P_bf @ mesh.average_node_to_face
        AvgCC2Fb = P_bf @ mesh.average_cell_to_face

        AvgCC2Fb = sp.diags(alpha * (P_bf @ mesh.face_areas)) @ AvgCC2Fb
        self._AvgBC = AvgN2Fb.T @ AvgCC2Fb

    def system_matrix(self, physprops):
        """System matrix of the DC resistivity problem.

        Parameters
        ----------
        physprops : simpeg.base.BaseElectricalPDESimulation
            Object that holds the electrical properties. Its ``MeSigma``
            property is used, and its ``sigma`` for the Robin boundary
            condition.

        Returns
        -------
        (n_nodes, n_nodes) scipy.sparse.csr_matrix
        """
        return self._assemble(physprops.MeSigma, physprops)

    def _assemble(self, MeSigma, physprops):
        Grad = self.Grad
        A = Grad.T.tocsr() @ MeSigma @ Grad

        if self.bc_type == "Neumann":
            A = _remove_null_space(A, symmetric=self.symmetric_null_space_fix)
        else:
            # This will fail if sigma is anisotropic
            sigma = physprops.sigma
            try:
                A = A + sp.diags(self._AvgBC @ sigma, format="csr")
            except ValueError as err:
                if len(sigma) != len(self.mesh):
                    raise NotImplementedError(
                        "Anisotropic conductivity is not supported for Robin boundary "
                        "conditions, please use 'Neumann'."
                    )
                else:
                    raise err
        return A

    def system_matrix_deriv(self, physprops, phi, v, adjoint=False):
        r"""Derivative of the system matrix times the potentials, times a vector.

        For fixed potentials :math:`\boldsymbol{\phi}`, compute

        .. math::
            \frac{\partial (\mathbf{A} \boldsymbol{\phi})}{\partial \mathbf{m}}
            \, \mathbf{v}

        or the adjoint operation.

        Parameters
        ----------
        physprops : simpeg.base.BaseElectricalPDESimulation
            Object that holds the electrical properties. Its ``MeSigmaDeriv``
            is used, and its ``sigmaMap`` and ``sigmaDeriv`` for the Robin
            boundary condition.
        phi : (n_nodes, ) numpy.ndarray
            Electric potentials on nodes.
        v : numpy.ndarray
            The vector. (n_param, ) for the standard operation, (n_nodes, ) for
            the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            (n_nodes, ) for the standard operation, (n_param, ) for the adjoint
            operation.
        """
        Grad = self.Grad
        if not adjoint:
            out = Grad.T @ physprops.MeSigmaDeriv(Grad @ phi, v, adjoint)
        else:
            out = physprops.MeSigmaDeriv(Grad @ phi, Grad @ v, adjoint)
        if self.bc_type != "Neumann" and physprops.sigmaMap is not None:
            if getattr(physprops, "_MBC_sigma", None) is None:
                physprops._MBC_sigma = self._AvgBC @ physprops.sigmaDeriv
            out += _inner_mat_mul_op(physprops._MBC_sigma, phi, v, adjoint)
        return out
