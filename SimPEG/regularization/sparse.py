from __future__ import annotations

import numpy as np

from discretize.base import BaseMesh

from .base import (
    BaseRegularization,
    WeightedLeastSquares,
    RegularizationMesh,
    Smallness,
    SmoothnessFirstOrder,
)
from .. import utils
from ..utils import (
    validate_ndarray_with_shape,
    validate_float,
    validate_type,
    validate_string,
)


class BaseSparse(BaseRegularization):
    """Base class for sparse-norm regularization.

    The ``BaseSparse`` class defines properties and methods inherited by sparse-norm
    regularization classes. Sparse-norm regularization in SimPEG is implemented using
    an iteratively re-weighted least squares (IRLS) approach. The ``BaseSparse`` class
    however, is not directly used to define the regularization for the inverse problem.

    Parameters
    ----------
    mesh : SimPEG.regularization.RegularizationMesh, discretize.base.BaseMesh
        Mesh on which the regularization is discretized. This is not necessarily the same as the mesh on which the simulation is defined.
    norm : float
        The norm used in the regularization function. Must be between within the interval [0, 2].
    irls_scaled : bool
        If ``True``, scale the IRLS weights to preserve magnitude of the regularization function. If ``False``, do not scale.
    irls_threshold : float
        Constant added to IRLS weights to ensures stability in the algorithm.
    active_cells : None, numpy.ndarray of bool
        Array of bool defining the set of :py:class:`~SimPEG.regularization.RegularizationMesh` cells that are active in the inversion.
        If ``None``, all cells are active.
    mapping : None, SimPEG.maps.BaseMap
        The mapping from the model parameters to the quantity defined in the regularization. If ``None``, the mapping is the identity map.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model values used to constrain the inversion. If ``None``, the reference model is equal to the starting model for the inversion.
    units : None, str
        Units for the model parameters. Some regularization classes behave differently depending on the units; e.g. 'radian'.
    weights : None, dict
        Weight multipliers to customize the least-squares function. Each key points to a (n_cells, ) numpy.ndarray that is defined on
        the :py:class:`~SimPEG.regularization.RegularizationMesh`.
    """

    def __init__(self, mesh, norm=2.0, irls_scaled=True, irls_threshold=1e-8, **kwargs):
        super().__init__(mesh=mesh, **kwargs)
        self.norm = norm
        self.irls_scaled = irls_scaled
        self.irls_threshold = irls_threshold

    @property
    def irls_scaled(self) -> bool:
        """Scale IRLS weights.

        When ``True``, the iteratively re-weighted least-squares (IRLS) weights are scaled at
        each update to generally preserve the magnitude of the regularization term.

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
        r"""Constant added to the denominator of the IRLS weights for stability.

        At each IRLS iteration, the IRLS weights are updated.
        The weight at iteration :math:`k` corresponding the cell :math:`i` is given by:

        .. math::
            r_i (\mathbf{m}^{(k)}) = \Big ( ( m_i^{(k-1)})^2 + \epsilon^2 \Big )^{p/2 - 1}

        where the `irls_threshold` :math:`\epsilon` is a constant that stabilizes the expression.

        Returns
        -------
        float
            Constant added to the denominator of the IRLS weights for stability.
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

        Sparse-norm regularization functions within SimPEG take the form:

        .. math::
            \gamma (m) = \frac{1}{2} \int_\Omega \, w(r) \, \Big ( \mathbb{F} \big [ m(r) - m_{ref}(r) \big ] \Bigg )^p \, dv

        where :math:`m(r)` is the model, :math:`m_{ref}(r)` is a reference model, and :math:`w(r)`
        is a user-defined weighting function. :math:`\mathbb{F}` is a placeholder for a function that
        acts on the difference between :math:`m` and :math:`m_{ref}`; e.g. a differential operator.
        
        The parameter :math:`p \in [0,2]` defines the norm, where a smaller norm is used to a model
        with more sparse structures.

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
            value = np.ones(self._weights_shapes[0]) * 2.0
        expected_shapes = self._weights_shapes
        if isinstance(expected_shapes, list):
            expected_shapes = expected_shapes[0]
        value = validate_ndarray_with_shape(
            "norm", value, shape=[expected_shapes, (1,)], dtype=float
        )
        if value.shape == (1,):
            value = np.full(expected_shapes[0], value)

        if np.any(value < 0) or np.any(value > 2):
            raise ValueError(
                "Value provided for 'norm' should be in the interval [0, 2]"
            )
        self._norm = value

    def get_lp_weights(self, f_m):
        """Compute and return standard IRLS weights.

        The IRLS weights are applied to ensure the discrete approximation of the sparse
        regularization function is evaluated using the specified norm. Since the weighting
        is model dependent, it must be recomputed at every IRLS iteration.

        Parameters
        ----------
        f_m : fcn
            The least-squares regularization kernel.
        """
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


class SparseSmallness(BaseSparse, Smallness):
    r"""Sparse smallness (compactness) regularization.

    ``SparseSmallness`` is used to recover models comprised of compact structures.
    The level of compactness is controlled by the `norm` within the regularization function;
    with more compact structures being recovered when a smaller norm is used.
    Optionally, the `weights` argument can be used to supply custom weights to control the degree of compactness being enforced
    throughout different regions the model.

    See the *Notes* section below for a full mathematical description.

    Parameters
    ----------
    mesh : SimPEG.regularization.RegularizationMesh, discretize.base.BaseMesh
        Mesh on which the regularization is discretized. This is not necessarily the same as the mesh on which the simulation is defined.
    norm : float
        The norm used in the regularization function. Must be within the interval [0, 2].
    irls_scaled : bool
        If ``True``, scale the IRLS weights to preserve magnitude of the regularization function. If ``False``, do not scale.
    irls_threshold : float
        Constant added to IRLS weights to ensures stability in the algorithm.
    active_cells : None, numpy.ndarray of bool
        Array of bool defining the set of :py:class:`~SimPEG.regularization.RegularizationMesh` cells that are active in the inversion.
        If ``None``, all cells are active.
    mapping : None, SimPEG.maps.BaseMap
        The mapping from the model parameters to the quantity defined in the regularization. If ``None``, the mapping is the identity map.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model values used to constrain the inversion. If ``None``, the reference model is equal to the starting model for the inversion.
    units : None, str
        Units for the model parameters. Some regularization classes behave differently depending on the units; e.g. 'radian'.
    weights : None, dict
        Weight multipliers to customize the least-squares function. Each key points to a (n_cells, ) numpy.ndarray that is defined on
        the :py:class:`~SimPEG.regularization.RegularizationMesh`.

    Notes
    -----
    The regularization function for sparse smallness (compactness) is given by:

    .. math::
        \gamma (m) = \frac{1}{2} \int_\Omega \, w(r) \, \big [ m(r) - m_{ref}(r) \big ]^p \, dv

    where :math:`m(r)` is the model, :math:`m_{ref}(r)` is a reference model, and :math:`w(r)`
    is a user-defined weighting function. By this definition, :math:`m(r)`, :math:`m_{ref}(r)`
    and :math:`w(r)` are continuous variables as a function of location :math:`r`.
    The parameter :math:`p \in [0,2]` is defined using the `norm` input argument,
    where a smaller norm is used to recovery more compact structures.
    
    For practical implementation, the regularization function and the aforementioned variables
    are discretized onto a mesh. The discrete approximation to the regularization function is given by:

    .. math::
        \gamma (\mathbf{m}) = \frac{1}{2} \big ( \mathbf{m - m_{ref}})^T \mathbf{W^T R^T R W} (\mathbf{m} - \mathbf{m_{ref}})

    where

        - :math:`\mathbf{m}` are the discrete model parameters,
        - :math:`\mathbf{m_{ref}}` is a reference model (optional, set using `reference_model`),
        - :math:`\mathbf{W}` is the weighting matrix, and
        - :math:`\mathbf{R}` is a model-dependent re-weighting matrix that is updated at every IRLS iteration.

    The weighting matrix is given by:

    .. math::
        \mathbf{W} = \textrm{diag} \bigg [ \bigg ( \mathbf{v} \odot \prod_i \mathbf{w_i} \bigg )^{1/2} \bigg ]

    where :math:`\mathbf{w_1, \; w_2, \; w_3, \; ...}` represents a set of custom cell weights;
    optionally set using `weights`. And :math:`\mathbf{\tilde{v}}` accounts for all cell volumes and dimensions
    when the regularization function is discretized to the mesh.

    The re-weighting matrix :math:`\mathbf{R}` is responsible for evaluating the proper norm in the regularization function.
    :math:`\mathbf{R}` is a sparse diagonal matrix whose elements are recomputed at every IRLS iteration. For IRLS iteration :math:`k`, the
    :math:`i^{th}` diagonal element is computed as follows: 

    .. math::
        R_{ii}^{(k)} = \bigg [ \Big ( ( m_i^{(k-1)})^2 + \epsilon^2 \Big )^{p/2 - 1} \bigg ]^{1/2}

    where :math:`\epsilon` is a small constant added for stability of the algorithm (set using `irls_threshold`).
    """

    _multiplier_pair = "alpha_s"

    def update_weights(self, m):
        r"""Compute and update the IRLS weights.

        The re-weighting matrix :math:`\mathbf{R}` ensure the specified norm is evaluated in the regularization function.
        For a mathematically description of IRLS weights, see the *Notes* section in the :class:`SparseSmallness` class documentation.

        Parameters
        ----------
        m : numpy.ndarray
            The model.
        """
        f_m = self.f_m(m)
        self.set_weights(irls=self.get_lp_weights(f_m))


class SparseSmoothness(BaseSparse, SmoothnessFirstOrder):
    r"""Sparse smoothness (blockiness) regularization.

    ``SparseSmoothness`` is used to recover models comprised of blocky structures.
    The level of blockiness is controlled by the choice in `norm` within the regularization function;
    with more blocky structures being recovered when a smaller norm is used.
    Optionally, the `weights` argument can be used to supplied custom weighting to control
    the degree of blockiness being enforced throughout different regions the model.
    
    See the *Notes* section below for a full mathematical description.

    Parameters
    ----------
    mesh : SimPEG.regularization.RegularizationMesh, discretize.base.BaseMesh
        Mesh on which the regularization is discretized. This is not necessarily the same as the mesh on which the simulation is defined.
    orientation : str {'x','y','z'}
        The direction along which sparse smoothness is applied.
    gradient_type : str {"total", "component"}
        Gradient measure used in the IRLS re-weighting. Whether to re-weight using the total gradient or components of the gradient.
    norm : float
        The norm used in the regularization function. Must be within the interval [0, 2].
    irls_scaled : bool
        If ``True``, scale the IRLS weights to preserve magnitude of the regularization function. If ``False``, do not scale.
    irls_threshold : float
        Constant added to IRLS weights to ensures stability in the algorithm.
    active_cells : None, numpy.ndarray of bool
        Array of bool defining the set of :py:class:`~SimPEG.regularization.RegularizationMesh` cells that are active in the inversion.
        If ``None``, all cells are active.
    mapping : None, SimPEG.maps.BaseMap
        The mapping from the model parameters to the quantity defined in the regularization. If ``None``, the mapping is the identity map.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model values used to constrain the inversion. If ``None``, the reference model is equal to the starting model for the inversion.
    units : None, str
        Units for the model parameters. Some regularization classes behave differently depending on the units; e.g. 'radian'.
    weights : None, dict
        Weight multipliers to customize the least-squares function. Each key points to a (n_cells, ) numpy.ndarray that is defined on
        the :py:class:`~SimPEG.regularization.RegularizationMesh`.

    Notes
    -----
    The regularization function for sparse smoothness (blockiness) is given by:

    .. math::
        \gamma (m) = \frac{1}{2} \int_\Omega \, w(r) \, \bigg ( \frac{\partial}{\partial x} \Big [ m(r) - m_{ref}(r) \Big ] \bigg)^p \, dv

    where :math:`m(r)` is the model, :math:`m_{ref}(r)` is a reference model, and :math:`w(r)`
    is a user-defined weighting function. By this definition, :math:`m(r)`, :math:`m_{ref}(r)`
    and :math:`w(r)` are continuous variables as a function of location :math:`r`.
    The parameter :math:`p \in [0,2]` is defined using the `norm` input argument,
    where a smaller norm is used to recovery more blocky structures.
    
    For practical implementation within SimPEG, the regularization function and the aforementioned variables
    are discretized onto a mesh. The discrete approximation of the regularization function is evaluated using a
    weighted least-squares (IRLS) approach. For blockiness (sharp boundaries) along
    the x-direction, the discrete regularization function is given by:

    .. math::
        \gamma (\mathbf{m}) = \frac{1}{2} \big ( \mathbf{m - m_{ref}})^T \mathbf{G_x^T W^T R^T R W G_x} (\mathbf{m} - \mathbf{m_{ref}})

    where

        - :math:`\mathbf{m}` are the discrete model parameters,
        - :math:`\mathbf{m_{ref}}` is a reference model (optional, and set using `reference_model`),
        - :math:`\mathbf{G_x}` is partial cell gradient operator along the x-direction,
        - :math:`\mathbf{W}` is the weighting matrix, and
        - :math:`\mathbf{R}` is a model-dependent re-weighting matrix that is updated at every IRLS iteration.

    The weighting matrix is given by:

    .. math::
        \mathbf{W} = \textrm{diag} \bigg [ \bigg ( \mathbf{\tilde{v}} \odot \prod_i \mathbf{w_i} \bigg )^{1/2} \bigg ]

    where :math:`\mathbf{w_1, \; w_2, \; w_3, \; ...}` represents a set of custom cell weights;
    optionally set using `weights`. :math:`\mathbf{A_{fc}}` averages from faces to cell centers
    and :math:`\mathbf{\tilde{v}}` accounts for all cell volumes and dimensions
    when the regularization function is discretized to the mesh.

    The re-weighting matrix :math:`\mathbf{R}` is responsible for evaluating the proper norm in the regularization function.
    :math:`\mathbf{R}` is a sparse diagonal matrix whose elements are recomputed at every IRLS iteration. For IRLS iteration :math:`k`, the
    :math:`i^{th}` diagonal element is computed as follows: 

    .. math::
        R_{ii}^{(k)} = \bigg [ \Big ( ( m_i^{(k-1)})^2 + \epsilon^2 \Big )^{p/2 - 1} \bigg ]^{1/2}

    where :math:`\epsilon` is a small constant added for stability of the algorithm (set using `irls_threshold`).
    """

    def __init__(self, mesh, orientation="x", gradient_type="total", **kwargs):
        if "gradientType" in kwargs:
            self.gradientType = kwargs.pop("gradientType")
        else:
            self.gradient_type = gradient_type
        super().__init__(mesh=mesh, orientation=orientation, **kwargs)

    def update_weights(self, m):
        r"""Compute and update the IRLS weights.

        The re-weighting matrix :math:`\mathbf{R}` ensure the specified norm is evaluated in the regularization function.
        For a mathematically description of IRLS weights, see the *Notes* section in the :class:`SparseSmallness` class documentation.

        Parameters
        ----------
        m : numpy.ndarray
            The model.
        """
        if self.gradient_type == "total":
            delta_m = self.mapping * self._delta_m(m)
            f_m = np.zeros_like(delta_m)
            for ii, comp in enumerate("xyz"):
                if self.regularization_mesh.dim > ii:
                    dm = (
                        getattr(self.regularization_mesh, f"cell_gradient_{comp}")
                        * delta_m
                    )

                    if self.units is not None and self.units.lower() == "radian":
                        Ave = getattr(self.regularization_mesh, f"aveCC2F{comp}")
                        length_scales = Ave * (
                            self.regularization_mesh.Pac.T
                            * self.regularization_mesh.mesh.h_gridded[:, ii]
                        )
                        dm = (
                            utils.mat_utils.coterminal(dm * length_scales)
                            / length_scales
                        )

                    f_m += np.abs(
                        getattr(self.regularization_mesh, f"aveF{comp}2CC") * dm
                    )

            f_m = getattr(self.regularization_mesh, f"aveCC2F{self.orientation}") * f_m

        else:
            f_m = self.f_m(m)

        self.set_weights(irls=self.get_lp_weights(f_m))

    @property
    def gradient_type(self) -> str:
        """Gradient measure used in the IRLS re-weighting.

        Returns
        -------
        str in {"total", "components"}
            Whether to re-weight using the total gradient or components of the gradient.
        """
        return self._gradient_type

    @gradient_type.setter
    def gradient_type(self, value: str):
        self._gradient_type = validate_string(
            "gradient_type", value, ["total", "components"]
        )

    gradientType = utils.code_utils.deprecate_property(
        gradient_type, "gradientType", "0.19.0", error=False, future_warn=True
    )


class Sparse(WeightedLeastSquares):
    r"""Sparse norm weighted least squares regularization.

    Construct a regularization for recovering compact and/or blocky structures
    using a weighted sum of :class:`SparseSmallness` and :class:`SparseSmoothness`
    regularization functions.

    See the *Notes* section below for a full mathematical description of the regularization.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        Mesh on which the model parameters are defined.
    gradient_type : str {"total", "component"}
        Gradient measure used in the IRLS re-weighting. Whether to re-weight using the total gradient or components of the gradient.
    norms : (dim+1, ) array_like
        The respective norms used for the sparse smallness, x-smoothness, (y-smoothness and z-smoothness) regularization function.
        Must all be within the interval [0, 2].
    irls_scaled : bool
        If ``True``, scale the IRLS weights to preserve magnitude of the regularization function. If ``False``, do not scale.
    irls_threshold : float
        Constant added to IRLS weights to ensures stability in the algorithm.
    active_cells : array_like of bool or int, optional
        List of active cell indices, or a `mesh.n_cells` boolean array
        describing active cells.
    alpha_s : float, optional
        Smallness weight
    alpha_x, alpha_y, alpha_z : float or None, optional
        First order smoothness weights for the respective dimensions.
        `None` implies setting these weights using the `length_scale`
        parameters.
    length_scale_x, length_scale_y, length_scale_z : float, optional
        First order smoothness length scales for the respective dimensions.
    mapping : SimPEG.maps.IdentityMap, optional
        A mapping to apply to the model before regularization.
    reference_model : array_like, optional
        Reference model values used to constrain the inversion. If ``None``, the reference model is equal to the starting model for the inversion.
    reference_model_in_smooth : bool, optional
        Whether to include the reference model in the smoothness terms.
    weights : None, array_like, or dict of array_like, optional
        User defined weights. It is recommended to interact with weights using
        the :py:meth:`~get_weights`, :py:meth:`~set_weights` methods.

    Notes
    -----
    The model objective function :math:`\phi_m (m)` defined by the weighted sum of sparse smallness and smoothness
    regularization functions is given by:

    .. math::
        \phi_m (m) = \frac{\alpha_s}{2} \int_\Omega \, w(r) \Big [ m(r) - m_{ref}(r) \Big ]^{p_s} \, dv +
        \sum_{i=x,y,z} \frac{\alpha_i}{2} \int_\Omega \, w(r) \Bigg ( \frac{\partial}{\partial \xi_i} \Big [ m(r) - m_{ref}(r) \Big ] \Bigg )^{p_i} \, dv
        
    where :math:`m(r)` is the model, :math:`m_{ref}(r)` is a reference model that may or may not be
    included in the smoothness terms, and :math:`w(r)` is a user-defined weighting function. :math:`\xi_i` is
    the unit direction along :math:`i`. Constants :math:`\alpha_s` and :math:`\alpha_i` weight the respective
    contributions of the smallness and first-order smoothness regularization functions. By our definition,
    :math:`m(r)`, :math:`m_{ref}(r)` and :math:`w(r)` are continuous variables as a function of location :math:`r`.
    
    For practical implementation within SimPEG, the model objective function and the aforementioned variables
    are discretized onto a mesh; set upon instantiation. The discrete approximation to the model objective function is given by:

    .. math::
        \phi_m (\mathbf{m}) = \frac{\alpha_s}{2} \| \mathbf{R_s W_s} (\mathbf{m} - \mathbf{m_{ref}} ) \|^2
        + \sum_{i=x,y,z} \frac{\alpha_i}{2} \| \mathbf{R_i W_i G_i} (\mathbf{m} - \mathbf{m_{ref}} ) \|^2

    where

        - :math:`\mathbf{m}` are the set of discrete model parameters (i.e. the model),
        - :math:`\mathbf{m_{ref}}` is a reference model which may or may not be inclulded in the smoothess terms,
        - :math:`\mathbf{G_i}` are partial cell gradients operators along x, y and z,
        - :math:`\mathbf{W}` are weighting matrices, and
        - :math:`\mathbf{R}` are the IRLS re-weighting matrices

    See the documentation for :class:`SparseSmallness` and :class:`SparseSmoothness`
    for more details on how weighting matrices are constructed for each term.
    """

    def __init__(
        self,
        mesh,
        active_cells=None,
        norms=None,
        gradient_type="total",
        irls_scaled=True,
        irls_threshold=1e-8,
        **kwargs,
    ):
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

        objfcts = [
            SparseSmallness(mesh=self.regularization_mesh),
            SparseSmoothness(mesh=self.regularization_mesh, orientation="x"),
        ]

        if mesh.dim > 1:
            objfcts.append(
                SparseSmoothness(mesh=self.regularization_mesh, orientation="y")
            )

        if mesh.dim > 2:
            objfcts.append(
                SparseSmoothness(mesh=self.regularization_mesh, orientation="z")
            )

        gradientType = kwargs.pop("gradientType", None)
        super().__init__(
            self.regularization_mesh,
            objfcts=objfcts,
            **kwargs,
        )
        if norms is None:
            norms = [1] * (mesh.dim + 1)
        self.norms = norms

        if gradientType is not None:
            # Trigger deprecation warning
            self.gradientType = gradientType
        else:
            self.gradient_type = gradient_type

        self.irls_scaled = irls_scaled
        self.irls_threshold = irls_threshold

    @property
    def gradient_type(self) -> str:
        """
        Choice of gradient measure used in the irls weights
        """
        return self._gradient_type

    @gradient_type.setter
    def gradient_type(self, value: str):
        for fct in self.objfcts:
            if hasattr(fct, "gradient_type"):
                fct.gradient_type = value

        self._gradient_type = value

    gradientType = utils.code_utils.deprecate_property(
        gradient_type, "gradientType", "0.19.0", error=False, future_warn=True
    )

    @property
    def norms(self):
        """
        Value of the norm
        """
        return self._norms

    @norms.setter
    def norms(self, values: list | np.ndarray | None):
        if values is not None:
            if len(values) != len(self.objfcts):
                raise ValueError(
                    f"The number of values provided for 'norms', {len(values)}, does not "
                    f"match the number of regularization functions, {len(self.objfcts)}."
                )
        else:
            values = [None] * len(self.objfcts)
        previous_norms = getattr(self, "_norms", [None] * len(self.objfcts))
        try:
            for val, fct in zip(values, self.objfcts):
                fct.norm = val
            self._norms = values
        except Exception as err:
            # reset the norms if failed
            for val, fct in zip(previous_norms, self.objfcts):
                fct.norm = val
            raise err

    @property
    def irls_scaled(self) -> bool:
        """
        Scale irls weights.
        """
        return self._irls_scaled

    @irls_scaled.setter
    def irls_scaled(self, value: bool):
        value = validate_type("irls_scaled", value, bool, cast=False)
        for fct in self.objfcts:
            fct.irls_scaled = value
        self._irls_scaled = value

    @property
    def irls_threshold(self):
        """
        Constant added to the denominator of the IRLS weights for stability.
        """
        return self._irls_threshold

    @irls_threshold.setter
    def irls_threshold(self, value):
        value = validate_float(
            "irls_threshold", value, min_val=0.0, inclusive_min=False
        )
        for fct in self.objfcts:
            fct.irls_threshold = value
        self._irls_threshold = value

    def update_weights(self, model):
        """
        Trigger irls update on all children
        """
        for fct in self.objfcts:
            fct.update_weights(model)
