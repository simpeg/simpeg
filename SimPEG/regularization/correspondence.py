import numpy as np
import scipy.sparse as sp
import properties

from .. import utils
from .base import BaseSimilarityMeasure


class LinearCorrespondence(BaseSimilarityMeasure):
    """
    The petrophysical linear constraint for joint inversions.

    ..math::
        \\phi_c({\\mathbf m}_{\\mathbf1},{\\mathbf m}_{\\mathbf2})=\\lambda\\sum_{i=1}^M
        (k_1*m_1 + k_2*m_2 + k_3)

    Assuming that we are working with two models only.

    """

    coefficients = properties.Array(
        "coefficients for the linear relationship between parameters",
        shape=(3,),
        default=np.array([1.0, -1.0, 0.0]),
    )

    def relation(self, model):
        """
        Computes the values of petrophysical linear relationship between two different
        geophysical models.

        The linear relationship is defined as:

        f(m1, m2)  = k1*m1 + k2*m2 + k3

        :param numpy.ndarray model: stacked array of individual models
                                    np.c_[model1, model2,...]

        :rtype: float
        :return: linearly related petrophysical values of two different models,
                  dimension: M by 1, :M number of model parameters.

        """
        m1, m2 = self.wire_map * model
        k1, k2, k3 = self.coefficients

        return k1 * m1 + k2 * m2 + k3

    def __call__(self, model):
        """
        Computes the sum of values of petrophysical linear relationship
        between two different geophysical models.

        :param numpy.ndarray model: stacked array of individual models
                                    np.c_[model1, model2,...]

        :rtype: float
        :return: a scalar value.
        """

        result = self.relation(model)
        return 0.5 * result.T @ result

    def deriv(self, model):
        """Computes the Jacobian of the coupling term.

        :param list of numpy.ndarray ind_models: [model1, model2,...]

        :rtype: numpy.ndarray
        :return: result: gradient of the coupling term with respect to model1, model2,
                 :dimension 2M by 1, :M number of model parameters.
        """
        k1, k2, k3 = self.coefficients
        r = self.relation(model)
        dc_dm1 = k1 * r
        dc_dm2 = k2 * r

        result = np.r_[dc_dm1, dc_dm2]

        return result

    def deriv2(self, model, v=None):
        """Computes the Hessian of the linear coupling term.

        :param list of numpy.ndarray ind_models: [model1, model2, ...]
        :param numpy.ndarray v: vector to be multiplied by Hessian
        :rtype: scipy.sparse.csr_matrix if v is None
                numpy.ndarray if v is not None
        :return Hessian matrix: | h11, h21 | :dimension 2M*2M.
                                |          |
                                | h12, h22 |
        """

        k1, k2, k3 = self.coefficients
        if v is not None:
            v1, v2 = self.wire_map * v
            p1 = k1 ** 2 * v1 + k2 * k1 * v2
            p2 = k2 * k1 * v1 + k2 ** 2 * v2
            return np.r_[p1, p2]
        else:
            n = self.regularization_mesh.nC
            A = utils.sdiag(np.ones(n) * (k1 ** 2))
            B = utils.sdiag(np.ones(n) * (k2 ** 2))
            C = utils.sdiag(np.ones(n) * (k1 * k2))
            return sp.bmat([[A, C], [C, B]], format="csr")
