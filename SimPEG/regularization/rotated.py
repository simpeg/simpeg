from .base import BaseRegularization
import numpy as np
import scipy.sparse as sp
from ..utils.code_utils import validate_ndarray_with_shape


class SmoothnessFullGradient(BaseRegularization):
    r"""Measures the gradient of a model using optionally anisotropic weighting.

    This regularizer measures the first order smoothness in a mesh ambivalent way
    by observing that the N-d smoothness operator can be represented as an
    inner product with an arbitrarily anisotropic weight.

    By default it assumes uniform weighting in each dimension, which works
    for most ``discretize`` mesh types.

    Parameters
    ----------
    mesh : discretize.BaseMesh
        The mesh object to use for regularization. The mesh should either have
        a `cell_gradient` or a `stencil_cell_gradient` defined.
    alphas : (mesh.dim,) or (mesh.n_cells, mesh.dim) array_like of float, optional.
        The weights of the regularization for each axis. This can be defined for each cell
        in the mesh. Default is uniform weights equal to the smallest edge length squared.
    reg_dirs : (mesh.dim, mesh.dim) or (mesh.n_cells, mesh.dim, mesh.dim) array_like of float
        Matrix or list of matrices whose columns represent the regularization directions.
        Each matrix should be orthonormal. Default is Identity.
    ortho_check : bool, optional
        Whether to check `reg_dirs` for orthogonality.
    **kwargs
        Keyword arguments passed to the parent class ``BaseRegularization``.

    Examples
    --------
    Construct of 2D measure with uniform smoothing in each direction.

    >>> from discretize import TensorMesh
    >>> from SimPEG.regularization import SmoothnessFullGradient
    >>> mesh = TensorMesh([32, 32])
    >>> reg = SmoothnessFullGradient(mesh)

    We can instead create a measure that smooths twice as much in the 1st dimension
    than it does in the second dimension.
    >>> reg = SmoothnessFullGradient(mesh, [2, 1])

    The `alphas` parameter can also be indepenant for each cell. Here we set all cells
    lower than 0.5 in the x2 to twice as much in the first dimension
    otherwise it is uniform smoothing.
    >>> alphas = np.ones((mesh.n_cells, mesh.dim))
    >>> alphas[mesh.cell_centers[:, 1] < 0.5] = [2, 1]
    >>> reg = SmoothnessFullGradient(mesh, alphas)

    We can also rotate the axis in which we want to preferentially smooth. Say we want to
    smooth twice as much along the +x1,+x2 diagonal as we do along the -x1,+x2 diagonal,
    effectively rotating our smoothing 45 degrees. Note and the columns of the matrix
    represent the directional vectors (not the rows).
    >>> sqrt2 = np.sqrt(2)
    >>> reg_dirs = np.array([
    ...     [sqrt2, -sqrt2],
    ...     [sqrt2, sqrt2],
    ... ])
    >>> reg = SmoothnessFullGradient(mesh, alphas, reg_dirs=reg_dirs)

    Notes
    -----
    The regularization object is the discretized form of the continuous regularization

    ..math:
       f(m) = \int_V \nabla m \cdot \mathbf{a} \nabla m \hspace{5pt} \partial V

    The tensor quantity `a` is used to represent the potential preferential directions of
    regularization. `a` must be symmetric positive semi-definite with an eigendecomposition of:

    ..math:
      \mathbf{a} = \mathbf{Q}\mathbf{L}\mathbf{Q}^{-1}

    `Q` is then the regularization directions ``reg_dirs``, and `L` is represents the weighting
    along each direction, with ``alphas`` along its diagonal. These are multiplied to form the
    anisotropic alpha used for rotated gradients.
    """

    def __init__(self, mesh, alphas=None, reg_dirs=None, ortho_check=True, **kwargs):
        if mesh.dim < 2:
            raise TypeError("Mesh must have dimension higher than 1")
        super().__init__(mesh=mesh, **kwargs)

        if alphas is None:
            edge_length = np.min(mesh.edge_lengths)
            alphas = edge_length**2 * np.ones(mesh.dim)
        alphas = validate_ndarray_with_shape(
            "alphas",
            alphas,
            shape=[(mesh.dim,), ("*", mesh.dim)],
            dtype=float,
        )
        n_active_cells = self.regularization_mesh.n_cells
        if len(alphas.shape) == 1:
            alphas = np.tile(alphas, (mesh.n_cells, 1))
        if alphas.shape[0] != mesh.n_cells:
            # check if I need to expand from active cells to all cells (needed for discretize)
            if alphas.shape[0] == n_active_cells and self.active_cells is not None:
                alpha_temp = np.zeros((mesh.n_cells, mesh.dim))
                alpha_temp[self.active_cells] = alphas
                alphas = alpha_temp
            else:
                raise IndexError(
                    f"`alphas` first dimension, {alphas.shape[0]}, must be either number "
                    f"of active cells {n_cells}, or the number of mesh cells {mesh.n_cells}. "
                )
        if np.any(alphas < 0):
            raise ValueError("`alpha` must be non-negative")
        anis_alpha = alphas

        if reg_dirs is not None:
            reg_dirs = validate_ndarray_with_shape(
                "reg_dirs",
                reg_dirs,
                shape=[(mesh.dim, mesh.dim), ("*", mesh.dim, mesh.dim)],
                dtype=float,
            )
            if reg_dirs.shape == (mesh.dim, mesh.dim):
                reg_dirs = np.tile(reg_dirs, (mesh.n_cells, 1, 1))
            if reg_dirs.shape[0] != mesh.n_cells:
                # check if I need to expand from active cells to all cells (needed for discretize)
                if (
                    reg_dirs.shape[0] == n_active_cells
                    and self.active_cells is not None
                ):
                    reg_dirs_temp = np.zeros((mesh.n_cells, mesh.dim, mesh.dim))
                    reg_dirs_temp[self.active_cells] = reg_dirs
                    reg_dirs = reg_dirs_temp
                else:
                    raise IndexError(
                        f"`reg_dirs` first dimension, {reg_dirs.shape[0]}, must be either number "
                        f"of active cells {n_cells}, or the number of mesh cells {mesh.n_cells}. "
                    )
            # check orthogonality?
            if ortho_check:
                eye = np.eye(mesh.dim)
                for i, M in enumerate(reg_dirs):
                    if not np.allclose(eye, M @ M.T):
                        raise ValueError(f"Matrix {i} is not orthonormal")
            # create a stack of matrices of dir @ alphas @ dir.T
            anis_alpha = np.einsum("ink,ik,imk->inm", reg_dirs, anis_alpha, reg_dirs)
            # Then select the upper diagonal components for input to discretize
            if mesh.dim == 2:
                anis_alpha = np.stack(
                    (
                        anis_alpha[..., 0, 0],
                        anis_alpha[..., 1, 1],
                        anis_alpha[..., 0, 1],
                    ),
                    axis=-1,
                )
            elif mesh.dim == 3:
                anis_alpha = np.stack(
                    (
                        anis_alpha[..., 0, 0],
                        anis_alpha[..., 1, 1],
                        anis_alpha[..., 2, 2],
                        anis_alpha[..., 0, 1],
                        anis_alpha[..., 0, 2],
                        anis_alpha[..., 1, 2],
                    ),
                    axis=-1,
                )
        self._anis_alpha = anis_alpha

    # overwrite the call, deriv, and deriv2...
    def __call__(self, m):
        G = self.cell_gradient
        M_f = self.W
        r = G @ (self.mapping * (self._delta_m(m)))
        return 0.5 * r @ M_f @ r

    def deriv(self, m):
        m_d = self.mapping.deriv(self._delta_m(m))
        G = self.cell_gradient
        M_f = self.W
        r = G @ (self.mapping * (self._delta_m(m)))
        return m_d.T * (G.T @ (M_f @ r))

    def deriv2(self, m, v=None):
        m_d_v = self.mapping.deriv(self._delta_m(m), v)
        G = self.cell_gradient
        M_f = self.W
        if v is None:
            return m_d_v.T @ (G.T @ M_f @ G) @ m_d_v

        return m_d_v.T @ (G.T @ (M_f @ (G @ m_d_v)))

    @property
    def cell_gradient(self):
        """The (approximate) cell gradient operator

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        if getattr(self, "_cell_gradient", None) is None:
            mesh = self.regularization_mesh.mesh
            try:
                cell_gradient = mesh.cell_gradient
            except AttributeError:
                a = mesh.face_areas
                v = mesh.average_cell_to_face @ mesh.cell_volumes
                cell_gradient = sp.diags(a / v) @ mesh.stencil_cell_gradient

            v = np.ones(mesh.n_cells)
            # Turn off cell_gradient at boundary faces
            if self.active_cells is not None:
                v[~self.active_cells] = 0

            dv = cell_gradient @ v
            P = sp.diags((np.abs(dv) <= 1e-16).astype(int))
            cell_gradient = P @ cell_gradient
            if self.active_cells is not None:
                cell_gradient = cell_gradient[:, self.active_cells]
            self._cell_gradient = cell_gradient
        return self._cell_gradient

    @property
    def W(self):
        """The inner product operator using rotated coordinates

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        if getattr(self, "_W", None) is None:
            mesh = self.regularization_mesh.mesh
            cell_weights = np.ones(len(mesh))
            for values in self._weights.values():
                cell_weights *= values
            reg_model = self._anis_alpha * cell_weights[:, None]
            # turn off measure in inactive cells
            if self.active_cells is not None:
                reg_model[~self.active_cells] = 0.0

            self._W = mesh.get_face_inner_product(reg_model)
        return self._W
