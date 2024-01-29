from typing import Literal

import numpy as np
import scipy.sparse as sp
from discretize import TensorMesh, TreeMesh
from discretize.base import BaseMesh
from scipy.interpolate import NearestNDInterpolator

from ..utils.code_utils import (
    validate_float,
    validate_ndarray_with_shape,
    validate_type,
)
from ..utils.mat_utils import coterminal
from . import BaseRegularization, RegularizationMesh, Sparse, SparseSmallness


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
    kwargs :
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

    _multiplier_pair = "alpha_x"

    def __init__(
        self,
        mesh,
        alphas=None,
        reg_dirs=None,
        ortho_check=True,
        norm=2,
        irls_scaled=True,
        irls_threshold=1e-8,
        reference_model_in_smooth=False,
        **kwargs,
    ):
        self.reference_model_in_smooth = reference_model_in_smooth

        if mesh.dim < 2:
            raise TypeError("Mesh must have dimension higher than 1")
        super().__init__(mesh=mesh, **kwargs)

        self.norm = norm
        self.irls_threshold = irls_threshold
        self.irls_scaled = irls_scaled

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
                    f"of active cells {mesh.n_cells}, or the number of mesh cells {mesh.n_cells}. "
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
                        f"of active cells {mesh.n_cells}, or the number of mesh cells {mesh.n_cells}. "
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

    @property
    def reference_model_in_smooth(self) -> bool:
        """
        whether to include reference model in gradient or not

        :return: True or False
        """
        return self._reference_model_in_smooth

    @reference_model_in_smooth.setter
    def reference_model_in_smooth(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(
                f"'reference_model_in_smooth must be of type 'bool'. Value of type {type(value)} provided."
            )
        self._reference_model_in_smooth = value

    def _delta_m(self, m):
        if self.reference_model is None or not self.reference_model_in_smooth:
            return m
        return m - self.reference_model

    def f_m(self, m):
        dfm_dl = self.cell_gradient @ (self.mapping * self._delta_m(m))

        if self.units is not None and self.units.lower() == "radian":
            return coterminal(dfm_dl * self._cell_distances) / self._cell_distances
        return dfm_dl

    def f_m_deriv(self, m):
        return self.cell_gradient @ self.mapping.deriv(self._delta_m(m))

    # overwrite the call, deriv, and deriv2...
    def __call__(self, m):
        M_f = self.W
        r = self.f_m(m)
        return 0.5 * r @ M_f @ r

    def deriv(self, m):
        m_d = self.f_m_deriv(m)
        M_f = self.W
        r = self.f_m(m)
        return m_d.T @ (M_f @ r)

    def deriv2(self, m, v=None):
        m_d = self.f_m_deriv(m)
        M_f = self.W
        if v is None:
            return m_d.T @ (M_f @ m_d)

        return m_d.T @ (M_f @ (m_d @ v))

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
                # project values to full mesh
                # dirty fix of original PR
                projection = NearestNDInterpolator(
                    mesh.cell_centers[self.active_cells], values
                )
                proj_values = projection(mesh.cell_centers)
                cell_weights *= proj_values
            reg_model = self._anis_alpha * cell_weights[:, None]
            # turn off measure in inactive cells
            if self.active_cells is not None:
                reg_model[~self.active_cells] = 0.0

            self._W = mesh.get_face_inner_product(reg_model)
        return self._W

    def update_weights(self, m):
        f_m = self.f_m(m)
        irls_weights = self.get_lp_weights(f_m)
        irls_weights = self.regularization_mesh.mesh.average_face_to_cell @ irls_weights
        self.set_weights(irls=irls_weights[self.active_cells])

    def get_lp_weights(self, f_m):
        lp_scale = np.ones_like(f_m)
        if self.irls_scaled:
            # Scale on l2-norm gradient: f_m.max()
            l2_max = np.ones_like(f_m) * np.abs(f_m).max()
            # Compute theoretical maximum gradients for p < 1
            l2_max[self.norm < 1] = self.irls_threshold / np.sqrt(
                1.0 - self.norm[self.norm < 1]
            )
            lp_values = l2_max / (l2_max**2.0 + self.irls_threshold**2.0) ** (
                1.0 - self.norm / 2.0
            )
            lp_scale[lp_values != 0] = np.abs(f_m).max() / lp_values[lp_values != 0]

        return lp_scale / (f_m**2.0 + self.irls_threshold**2.0) ** (
            1.0 - self.norm / 2.0
        )

    @property
    def irls_scaled(self) -> bool:
        """Scale IRLS weights.

        When ``True``, scaling is applied when computing IRLS weights.
        The scaling acts to preserve the balance between the data misfit and the components of
        the regularization based on the derivative of the l2-norm measure. And it assists the
        convergence by ensuring the model does not deviate
        aggressively from the global 2-norm solution during the first few IRLS iterations.
        For a comprehensive description, see the documentation for :py:meth:`get_lp_weights` .

        Returns
        -------
        bool
            Whether to scale IRLS weights.
        """
        return self._irls_scaled

    @irls_scaled.setter
    def irls_scaled(self, value: bool):
        self._irls_scaled = validate_type("irls_scaled", value, bool, cast=False)

    @property
    def irls_threshold(self):
        r"""Stability constant for computing IRLS weights.

        Returns
        -------
        float
            Stability constant for computing IRLS weights.
        """
        return self._irls_threshold

    @irls_threshold.setter
    def irls_threshold(self, value):
        self._irls_threshold = validate_float(
            "irls_threshold", value, min_val=0.0, inclusive_min=False
        )

    @property
    def norm(self):
        r"""Norm for the sparse regularization.

        Returns
        -------
        None, float, (n_cells, ) numpy.ndarray
            Norm for the sparse regularization. If ``None``, a 2-norm is used.
            A float within the interval [0,2] represents a constant norm applied for all cells.
            A ``numpy.ndarray`` object, where each entry is used to apply a different norm to each cell in the mesh.
        """
        return self._norm

    @norm.setter
    def norm(self, value: float | np.ndarray | None):
        if value is None:
            value = np.ones(self.cell_gradient.shape[0]) * 2.0
        else:
            value = np.ones(self.cell_gradient.shape[0]) * value
        if np.any(value < 0) or np.any(value > 2):
            raise ValueError(
                "Value provided for 'norm' should be in the interval [0, 2]"
            )
        self._norm = value

    @property
    def units(self) -> str | None:
        """Units for the model parameters.

        Some regularization classes behave differently depending on the units; e.g. 'radian'.

        Returns
        -------
        str
            Units for the model parameters.
        """
        return self._units

    @units.setter
    def units(self, units: str | None):
        if units is not None and not isinstance(units, str):
            raise TypeError(
                f"'units' must be None or type str. Value of type {type(units)} provided."
            )
        self._units = units

    @property
    def _cell_distances(self) -> np.ndarray:
        """
        cell size average on faces

        :return: np.ndarray
        """
        cell_distances = self.cell_gradient.max(axis=1).toarray().ravel()
        cell_distances[cell_distances == 0] = 1
        cell_distances = cell_distances ** (-1)

        return cell_distances


class RotatedSparse(Sparse):
    """
    Class that wraps the rotated gradients in a ComboObjectiveFunction similar to Sparse.
    """

    def __init__(
        self,
        mesh: TensorMesh | TreeMesh,
        reg_dirs: np.ndarray,
        alphas_rot: tuple[float, float, float],
        active_cells: np.ndarray | None = None,
        norms: list[float] = [2.0, 2.0],
        gradient_type: Literal["components", "total"] = "total",
        irls_scaled: bool = True,
        irls_threshold: float = 1e-8,
        objfcts: list[BaseRegularization] | None = None,
        **kwargs,
    ):
        """
        Class to wrap rotated gradient into a ComboObjective Function

        :param mesh: mesh
        :param reg_dirs: rotation matrix
        :param alphas_rot: alphas for rotated gradients
        :param active_cells: active cells, defaults to None
        :param norms: norms, defaults to [2, 2]
        :param gradient_type: gradient_type, defaults to "total"
        :param irls_scaled: irls_scaled, defaults to True
        :param irls_threshold: irls_threshold, defaults to 1e-8
        :param objfcts: objfcts, defaults to None
        """
        if not isinstance(mesh, RegularizationMesh):
            mesh = RegularizationMesh(mesh)

        if not isinstance(mesh, RegularizationMesh):
            TypeError(
                f"'regularization_mesh' must be of type {RegularizationMesh} or {BaseMesh}. "
                f"Value of type {type(mesh)} provided."
            )
        self._regularization_mesh = mesh
        if active_cells is not None:
            self._regularization_mesh.active_cells = active_cells

        if objfcts is None:
            objfcts = [
                SparseSmallness(mesh=self.regularization_mesh),
                SmoothnessFullGradient(
                    mesh=self.regularization_mesh.mesh,
                    active_cells=active_cells,
                    reg_dirs=reg_dirs,
                    alphas=alphas_rot,
                    norm=norms[1],
                    irls_scaled=irls_scaled,
                    irls_threshold=irls_threshold,
                ),
            ]

        super().__init__(
            self.regularization_mesh,
            objfcts=objfcts,
            active_cells=active_cells,
            gradient_type=gradient_type,
            norms=norms[:2],
            irls_scaled=irls_scaled,
            irls_threshold=irls_threshold,
            **kwargs,
        )

    @property
    def alpha_y(self):
        """Multiplier constant for first-order smoothness along y.

        Returns
        -------
        float
            Multiplier constant for first-order smoothness along y.
        """
        return self._alpha_y

    @alpha_y.setter
    def alpha_y(self, value):
        self._alpha_y = None

    @property
    def alpha_z(self):
        """Multiplier constant for first-order smoothness along z.

        Returns
        -------
        float
            Multiplier constant for first-order smoothness along z.
        """
        return self._alpha_z

    @alpha_z.setter
    def alpha_z(self, value):
        self._alpha_z = None
