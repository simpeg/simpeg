import numpy as np
import scipy.sparse as sp
from ..utils import validate_ndarray_with_shape

from .. import utils
from .base import BaseSimilarityMeasure


class LinearCorrespondence(BaseSimilarityMeasure):
    r"""Linear correspondence regularization for joint inversion with two physical properties.

    ``LinearCorrespondence`` is used to recover a model where the differences between the model
    parameter values for two physical property types are minimal. ``LinearCorrespondence``
    can also be used to minimize the squared L2-norm of a linear combination of model parameters
    for two physical property types. See the *Notes* section for a comprehensive description.

    Parameters
    ----------
    mesh : simpeg.regularization.RegularizationMesh, discretize.base.BaseMesh
        Mesh on which the regularization is discretized. This is not necessarily
        the same as the mesh on which the simulation is defined.
    active_cells : None, (n_cells, ) numpy.ndarray of bool
        Boolean array defining the set of :py:class:`~.regularization.RegularizationMesh`
        cells that are active in the inversion. If ``None``, all cells are active.
    wire_map : simpeg.maps.Wires
        Wire map connecting physical properties defined on active cells of the
        :class:`RegularizationMesh`` to the entire model.
    coefficients : None, (3) numpy.ndarray of float
        Coefficients :math:`\{ \lambda_1, \lambda_2, \lambda_3 \}` for the linear relationship
        between model parameters. If ``None``, the coefficients are set to
        :math:`\{ 1, -1, 0 \}`.

    Notes
    -----
    Let :math:`\mathbf{m}` be a discrete model consisting of two physical property types such that:

    .. math::
        \mathbf{m} = \begin{bmatrix} \mathbf{m_1} \\ \mathbf{m_2} \end{bmatrix}

    Where :math:`\{ \lambda_1 , \lambda_2 , \lambda_3 \}` define scalar coefficients for a
    linear combination of vectors :math:`\mathbf{m_1}` and :math:`\mathbf{m_2}`, the regularization
    function (objective function) is given by:

    .. math::
        \phi (\mathbf{m})
        = \big \| \lambda_1 \mathbf{m_1} + \lambda_2 \mathbf{m_2} + \lambda_3 \big \|^2

    Scalar coefficients :math:`\{ \lambda_1 , \lambda_2 , \lambda_3 \}` are set using the
    `coefficients` property. For a true linear correspondence constraint, we set
    :math:`\{ \lambda_1 , \lambda_2 , \lambda_3 \}` to :math:`\{ 1, -1, 0 \}`.

    """

    def __init__(self, mesh, wire_map, coefficients=None, **kwargs):
        super().__init__(mesh, wire_map, **kwargs)
        if coefficients is None:
            coefficients = np.r_[1.0, -1.0, 0.0]
        self.coefficients = coefficients

    @property
    def coefficients(self):
        r"""Coefficients for the linear relationship between model parameters.

        For a relation vector:

        .. math::
            \mathbf{f}(\mathbf{m}) = \lambda_1 \mathbf{m_1} + \lambda_2 \mathbf{m_2} + \lambda_3

        where

        .. math::
            \mathbf{m} = \begin{bmatrix} \mathbf{m_1} \\ \mathbf{m_2} \end{bmatrix}

        This property defines the coefficients :math:`\{ \lambda_1 , \lambda_2 , \lambda_3 \}`.

        Returns
        -------
        (3, ) numpy.ndarray of float
            Coefficients for the linear relationship between model parameters.
        """
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value):
        self._coefficients = validate_ndarray_with_shape(
            "coefficients", value, shape=(3,)
        )

    def relation(self, model):
        r"""Computes the relation vector for the model provided.

        For a model consisting of two physical properties such that:

        .. math::
            \mathbf{m} = \begin{bmatrix} \mathbf{m_1} \\ \mathbf{m_2} \end{bmatrix}

        this method computer the relation vector for coefficients
        :math:`\{ \lambda_1 , \lambda_2 , \lambda_3 \}` as follows:

        .. math::
            \mathbf{f}(\mathbf{m}) = \lambda_1 \mathbf{m_1} + \lambda_2 \mathbf{m_2} + \lambda_3

        Parameters
        ----------
        model : (n_param, ) numpy.ndarray
            The model for which the relation vector is evaluated.

        Returns
        -------
        float
            The relation vector for the model provided.
        """
        m1, m2 = self.wire_map * model
        k1, k2, k3 = self.coefficients

        return k1 * m1 + k2 * m2 + k3

    def __call__(self, model):
        """Evaluate the regularization function for the model provided.

        Parameters
        ----------
        model : (n_param, ) numpy.ndarray
            The model for which the function is evaluated.

        Returns
        -------
        float
            The regularization function evaluated for the model provided.
        """

        result = self.relation(model)
        return result.T @ result

    def deriv(self, model):
        r"""Gradient of the regularization function evaluated for the model provided.

        Where :math:`\phi (\mathbf{m})` is the discrete regularization function (objective function),
        this method evaluates and returns the derivative with respect to the model parameters;
        i.e. the gradient. For a model :math:`\mathbf{m}` consisting of two physical properties
        such that:

        .. math::
            \mathbf{m} = \begin{bmatrix} \mathbf{m_1} \\ \mathbf{m_2} \end{bmatrix}

        The gradient has the form:

        .. math::
            \frac{\partial \phi}{\partial \mathbf{m}} =
            \begin{bmatrix} \dfrac{\partial \phi}{\partial \mathbf{m_1}} \\
            \dfrac{\partial \phi}{\partial \mathbf{m_2}} \end{bmatrix}

        Parameters
        ----------
        model : (n_param, ) numpy.ndarray
            The model; a vector array containing all physical properties.

        Returns
        -------
        (n_param, ) numpy.ndarray
            Gradient of the regularization function evaluated for the model provided.
        """
        k1, k2, k3 = self.coefficients
        r = self.relation(model)
        dc_dm1 = k1 * r
        dc_dm2 = k2 * r

        result = np.r_[dc_dm1, dc_dm2]

        return 2 * result

    def deriv2(self, model, v=None):
        r"""Hessian of the regularization function evaluated for the model provided.

        Where :math:`\phi (\mathbf{m})` is the discrete regularization function (objective function),
        this method evalutate and returns the second derivative (Hessian) with respect to the
        model parameters. For a model :math:`\mathbf{m}` consisting of two physical properties
        such that:

        .. math::
            \mathbf{m} = \begin{bmatrix} \mathbf{m_1} \\ \mathbf{m_2} \end{bmatrix}

        The Hessian has the form:

        .. math::
            \frac{\partial^2 \phi}{\partial \mathbf{m}^2} =
            \begin{bmatrix}
            \dfrac{\partial^2 \phi}{\partial \mathbf{m_1}^2} &
            \dfrac{\partial^2 \phi}{\partial \mathbf{m_1} \partial \mathbf{m_2}} \\
            \dfrac{\partial^2 \phi}{\partial \mathbf{m_2} \partial \mathbf{m_1}} &
            \dfrac{\partial^2 \phi}{\partial \mathbf{m_2}^2}
            \end{bmatrix}

        When a vector :math:`(\mathbf{v})` is supplied, the method returns the Hessian
        times the vector:

        .. math::
            \frac{\partial^2 \phi}{\partial \mathbf{m}^2} \, \mathbf{v}

        Parameters
        ----------
        model : (n_param, ) numpy.ndarray
            The model; a vector array containing all physical properties.
        v : None, (n_param, ) numpy.ndarray (optional)
            A numpy array to model the Hessian by.

        Returns
        -------
        (n_param, n_param) scipy.sparse.csr_matrix | (n_param, ) numpy.ndarray
            If the input argument *v* is ``None``, the Hessian
            for the models provided is returned. If *v* is not ``None``,
            the Hessian multiplied by the vector provided is returned.
        """

        k1, k2, k3 = self.coefficients
        if v is not None:
            v1, v2 = self.wire_map * v
            p1 = k1**2 * v1 + k2 * k1 * v2
            p2 = k2 * k1 * v1 + k2**2 * v2
            return 2 * np.r_[p1, p2]
        else:
            n = self.regularization_mesh.nC
            A = utils.sdiag(np.ones(n) * (k1**2))
            B = utils.sdiag(np.ones(n) * (k2**2))
            C = utils.sdiag(np.ones(n) * (k1 * k2))
            return 2 * sp.bmat([[A, C], [C, B]], format="csr")
