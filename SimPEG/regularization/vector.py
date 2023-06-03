from __future__ import annotations

# Regularizations for vector models.

import scipy.sparse as sp
import numpy as np
from .base import BaseRegularization, Smallness


class BaseVectorRegularization(BaseRegularization):
    """Base regularization class for models defined by vector quantities.

    The ``BaseVectorRegularization`` class defines properties and methods used
    by regularization classes for inversion to recover vector quantities.
    It is not directly used to constrain inversions.
    """

    @property
    def _weights_shapes(self) -> list[tuple[int]]:
        """Acceptable lengths for the weights

        Returns
        -------
        list of tuple
            Each tuple represents accetable shapes for the weights
        """
        mesh = self.regularization_mesh
        return [(mesh.nC,), (mesh.dim * mesh.nC,), (mesh.nC, mesh.dim)]


class CrossReferenceRegularization(Smallness, BaseVectorRegularization):
    r"""Cross reference regularization for inversion to recover vector quantities.

    This regularizer measures the magnitude of the cross product of the vector model
    with a reference vector model. This encourages the vectors in the model to point
    in the reference direction. The cross product of two vectors is minimized when they
    are parallel (or anti-parallel) to each other, and maximized when the vectors are
    perpendicular to each other.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh, .RegularizationMesh
        The mesh defining the model discretization.
    ref_dir : (mesh.dim,) array_like or (mesh.dim, n_active) array_like
        The reference direction model. This can be either a constant vector applied
        to every model cell, or different for every active model cell.
    active_cells : index_array, optional
        Boolean array or an array of active indices indicating the active cells of the
        inversion domain mesh.
    mapping : SimPEG.maps.IdentityMap, optional
        An optional linear mapping that would go from the model space to the space where
        the cross-product is enforced.
    weights : dict of [str: array_like], optional
        Any cell based weights for the regularization. Note if given a weight that is
        (n_cells, dim), meaning it is dependent on the vector component, it will compute
        the geometric mean of the component weights per cell and use that as a weight.
    **kwargs
        Arguments passed on to the parent classes: :py:class:`.Smallness` and
        :py:class:`.BaseVectorRegularization`.

    Notes
    -----
    Consider the case where the model is a vector quantity :math:`\vec{m}(r)`.
    The regularization function (objective function) for cross-reference
    regularization is given by:

    .. math::
        \phi (\vec{m}) = \frac{1}{2} \int_\Omega \, \vec{w}(r) \, \cdot \,
        \Big [ \vec{m}(r) \, \times \, \vec{m}^{(ref)}(r) \Big ]^2 \, dv

    where :math:`\vec{m}^{(ref)}(r)` is the reference model vector and :math:`\vec{w}(r)`
    is a user-defined weighting function.

    For implementation within SimPEG, the regularization function and its variables
    must be discretized onto a `mesh`. The discretized approximation for the regularization
    function (objective function) is given by:

    .. math::
        \phi (\vec{m}) \approx \frac{1}{2} \sum_i \tilde{w}_i \, \cdot \,
        \Big [ \vec{m}_i \, \times \, \vec{m}_i^{(ref)} \Big ]^2

    where :math:`\tilde{m}_i` are the model vectors at cell centers and
    :math:`\tilde{w}_i \in \mathbf{\tilde{w}}` are amalgamated weighting constants that 1) account
    for cell dimensions in the discretization and 2) apply any user-defined weighting.

    In practice, we frequently define the model :math:`\mathbf{m}` as a discrete
    vector of the form:

    .. math::
        \mathbf{m} = \begin{bmatrix} \mathbf{m_p} \\ \mathbf{m_s} \\ \mathbf{m_t} \end{bmatrix}

    where :math:`\mathbf{m_p}`, :math:`\mathbf{m_s}` and :math:`\mathbf{m_t}` represent vector
    components in the primary, secondary and tertiary directions at cell centers, respectively.
    The cross product between :math:`\mathbf{m}` and a similar reference vector
    :math:`\mathbf{m^{ref}}` is a linear operation of the form:

    .. math::
        \mathbf{m} \times \mathbf{m^{ref}} = \mathbf{X m}
        \begin{bmatrix}
        \mathbf{0} & -\boldsymbol{\Lambda_s} & \boldsymbol{\Lambda_t} \\
        \boldsymbol{\Lambda_p} & \mathbf{0} & -\boldsymbol{\Lambda_t} \\
        -\boldsymbol{\Lambda_p} & \boldsymbol{\Lambda_s} & \mathbf{0}
        \end{bmatrix}
        \begin{bmatrix} \mathbf{m_p} \\ \mathbf{m_s} \\ \mathbf{m_t} \end{bmatrix}

    where

    .. math:
        \boldsymbol{\Lambda_j} = \textrm{diag} \Big ( \mathbf{m_j^{(red)}} \Big )
        \;\;\;\; \textrm{for} \; j=p,s,t

    The discrete regularization function in linear form is given by:

    .. math::
        \phi (\mathbf{m}) = \frac{1}{2}
        \Big \| \mathbf{W X m} \, \Big \|^2

    where

        - :math:`\boldsymbol{\Lambda}` applies the cross-products, and
        - :math:`\mathbf{W}` is the weighting matrix.

    **Custom weights and the weighting matrix:**

    Let :math:`\mathbf{w_1, \; w_2, \; w_3, \; ...}` each represent an optional set of
    custom cell weights. Each individual set of cell weights acts independently on the
    the directional components of the cross-product and is a discrete vector of the form:

    .. math::
        \mathbf{w_j} = \begin{bmatrix} \mathbf{w_p} \\ \mathbf{w_s} \\ \mathbf{w_t} \end{bmatrix}

    The weighting applied within the objective function is given by:

    .. math::
        \mathbf{\tilde{w}} = \big ( \mathbf{e_3 \otimes v} ) \odot \prod_j \mathbf{w_j}

    where

        - :math:`\mathbf{e_3}` is a vector of ones of length 3,
        - :math:`\otimes` is the Kronecker product, and
        - :math:`\mathbf{v}` are the cell volumes.

    The weighting matrix used to apply the weights for smallness regularization is given by:

    .. math::
        \mathbf{W} = \textrm{diag} \Big ( \, \mathbf{\tilde{w}}^{1/2} \Big )

    Each set of custom cell weights is stored within a ``dict`` as a ``list`` of (n_cells, )
    ``numpy.ndarray``. The weights can be set all at once during instantiation
    with the `weights` keyword argument as follows:

    >>> reg = Smallness(mesh, weights={'weights_1': array_1, 'weights_2': array_2})

    or set after instantiation using the `set_weights` method:

    >>> reg.set_weights(weights_1=array_1, weights_2=array_2})

    The default weights that account for cell dimensions in the regularization are accessed via:

    >>> reg.get_weights('volume')

    """

    def __init__(
        self, mesh, ref_dir, active_cells=None, mapping=None, weights=None, **kwargs
    ):
        kwargs.pop("reference_model", None)
        super().__init__(
            mesh=mesh,
            active_cells=active_cells,
            mapping=mapping,
            weights=weights,
            **kwargs,
        )
        self.ref_dir = ref_dir
        self.reference_model = 0.0

    @property
    def _nC_residual(self):
        return np.prod(self.ref_dir.shape)

    @property
    def ref_dir(self):
        """The reference direction model.

        Returns
        -------
        (n_active, dim) numpy.ndarray
            The reference direction model.
        """
        return self._ref_dir

    @ref_dir.setter
    def ref_dir(self, value):
        mesh = self.regularization_mesh
        nC = mesh.nC
        value = np.asarray(value)
        if value.shape != (nC, mesh.dim):
            if value.shape == (mesh.dim,):
                # expand it out for each mesh cell
                value = np.tile(value, (nC, 1))
            else:
                raise ValueError(f"ref_dir must be shape {(nC, mesh.dim)}")
        self._ref_dir = value

        R0 = sp.diags(value[:, 0])
        R1 = sp.diags(value[:, 1])
        if value.shape[1] == 2:
            X = sp.bmat([[R1, -R0]])
        elif value.shape[1] == 3:
            Z = sp.csr_matrix((nC, nC))
            R2 = sp.diags(value[:, 2])
            X = sp.bmat(
                [
                    [Z, R2, -R1],
                    [-R2, Z, R0],
                    [R1, -R0, Z],
                ]
            )
        self._X = X

    def f_m(self, m):
        r"""Evaluate the regularization kernel function.

        For cross reference regularization, the regularization kernel function is given by:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{X m}

        where :math:`\mathbf{m}` are the discrete model parameters and :math:`\mathbf{X}`
        carries out the cross-product with a reference vector model.
        For a more detailed description, see the *Notes* section below.

        Parameters
        ----------
        m : numpy.ndarray
            The vector model.

        Returns
        -------
        numpy.ndarray
            The regularization kernel function evaluated for the model provided.

        Notes
        -----
        The objective function for cross reference regularization is given by:

        .. math::
            \phi_m (\mathbf{m}) = \frac{1}{2}
            \Big \| \mathbf{W X m} \, \Big \|^2

        where :math:`\mathbf{m}` are the discrete vector model parameters defined on the mesh (model),
        :math:`\mathbf{X}` carries out the cross-product with a reference vector model, and :math:`\mathbf{W}` is
        the weighting matrix. See the :class:`CrossReferenceRegularization` class documentation for more detail.

        We define the regularization kernel function :math:`\mathbf{f_m}` as:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{X m}

        such that

        .. math::
            \phi_m (\mathbf{m}) = \frac{1}{2} \Big \| \mathbf{W} \, \mathbf{f_m} \Big \|^2

        """
        return self._X @ (self.mapping * m)

    def f_m_deriv(self, m):
        r"""Derivative of the regularization kernel function.

        For ``CrossReferenceRegularization``, the derivative of the regularization kernel function
        with respect to the model is given by:

        .. math::
            \frac{\partial \mathbf{f_m}}{\partial \mathbf{m}} = \mathbf{X}

        where :math:`\mathbf{X}` is a linear operator that carries out the
        cross-product with a reference vector model.

        Parameters
        ----------
        m : numpy.ndarray
            The vector model.

        Returns
        -------
        scipy.sparse.csr_matrix
            The derivative of the regularization kernel function.

        Notes
        -----
        The objective function for cross reference regularization is given by:

        .. math::
            \phi_m (\mathbf{m}) = \frac{1}{2}
            \Big \| \mathbf{W X m} \, \Big \|^2

        where :math:`\mathbf{m}` are the discrete vector model parameters defined on the mesh (model),
        :math:`\mathbf{X}` carries out the cross-product with a reference vector model, and :math:`\mathbf{W}` is
        the weighting matrix. See the :class:`CrossReferenceRegularization` class documentation for more detail.

        We define the regularization kernel function :math:`\mathbf{f_m}` as:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{X m}

        such that

        .. math::
            \phi_m (\mathbf{m}) = \frac{1}{2} \Big \| \mathbf{W} \, \mathbf{f_m} \Big \|^2

        Thus, the derivative with respect to the model is:

        .. math::
            \frac{\partial \mathbf{f_m}}{\partial \mathbf{m}} = \mathbf{X}

        """
        return self._X @ self.mapping.deriv(m)

    @property
    def W(self):
        r"""Weighting matrix.

        Returns the weighting matrix for the objective function. To see how the
        weighting matrix is constructed, see the *Notes* section for the
        :class:`CrossReferenceRegularization` class.

        Returns
        -------
        scipy.sparse.csr_matrix
            The weighting matrix applied in the objective function.
        """
        if getattr(self, "_W", None) is None:
            mesh = self.regularization_mesh
            nC = mesh.nC

            weights = np.ones(
                nC,
            )
            for value in self._weights.values():
                if value.shape == (nC,):
                    weights *= value
                elif value.size == mesh.dim * nC:
                    weights *= np.linalg.norm(
                        value.reshape((nC, mesh.dim), order="F"), axis=1
                    )
            weights = np.sqrt(weights)
            if mesh.dim == 2:
                diag = weights
            else:
                diag = np.r_[weights, weights, weights]
            self._W = sp.diags(diag, format="csr")
        return self._W
