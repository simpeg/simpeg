"""
Map classes for petrophysics clusters.
"""

import numpy as np
from numpy.polynomial import polynomial


from ..utils import validate_ndarray_with_shape

from ._base import IdentityMap


class PolynomialPetroClusterMap(IdentityMap):
    """
    Modeling polynomial relationships between physical properties

    Parameters
    ----------
    coeffxx : array_like, optional
        Coefficients for the xx component. Default is [0, 1]
    coeffxy : array_like, optional
        Coefficients for the xy component. Default is [0]
    coeffyx : array_like, optional
        Coefficients for the yx component. Default is [0]
    coeffyy : array_like, optional
        Coefficients for the yy component. Default is [0, 1]
    """

    def __init__(
        self,
        coeffxx=None,
        coeffxy=None,
        coeffyx=None,
        coeffyy=None,
        mesh=None,
        nP=None,
        **kwargs,
    ):
        if coeffxx is None:
            coeffxx = np.r_[0.0, 1.0]
        if coeffxy is None:
            coeffxy = np.r_[0.0]
        if coeffyx is None:
            coeffyx = np.r_[0.0]
        if coeffyy is None:
            coeffyy = np.r_[0.0, 1.0]

        self._coeffxx = validate_ndarray_with_shape("coeffxx", coeffxx, shape=("*",))
        self._coeffxy = validate_ndarray_with_shape("coeffxy", coeffxy, shape=("*",))
        self._coeffyx = validate_ndarray_with_shape("coeffyx", coeffyx, shape=("*",))
        self._coeffyy = validate_ndarray_with_shape("coeffyy", coeffyy, shape=("*",))

        self._polynomialxx = polynomial.Polynomial(self.coeffxx)
        self._polynomialxy = polynomial.Polynomial(self.coeffxy)
        self._polynomialyx = polynomial.Polynomial(self.coeffyx)
        self._polynomialyy = polynomial.Polynomial(self.coeffyy)
        self._polynomialxx_deriv = self._polynomialxx.deriv(m=1)
        self._polynomialxy_deriv = self._polynomialxy.deriv(m=1)
        self._polynomialyx_deriv = self._polynomialyx.deriv(m=1)
        self._polynomialyy_deriv = self._polynomialyy.deriv(m=1)

        super().__init__(mesh=mesh, nP=nP, **kwargs)

    @property
    def coeffxx(self):
        """Coefficients for the xx component.

        Returns
        -------
        numpy.ndarray
        """
        return self._coeffxx

    @property
    def coeffxy(self):
        """Coefficients for the xy component.

        Returns
        -------
        numpy.ndarray
        """
        return self._coeffxy

    @property
    def coeffyx(self):
        """Coefficients for the yx component.

        Returns
        -------
        numpy.ndarray
        """
        return self._coeffyx

    @property
    def coeffyy(self):
        """Coefficients for the yy component.

        Returns
        -------
        numpy.ndarray
        """
        return self._coeffyy

    def _transform(self, m):
        out = m.copy()
        out[:, 0] = self._polynomialxx(m[:, 0]) + self._polynomialxy(m[:, 1])
        out[:, 1] = self._polynomialyx(m[:, 0]) + self._polynomialyy(m[:, 1])
        return out

    def inverse(self, D):
        r"""
        :param numpy.array D: physical property
        :rtype: numpy.array
        :return: model

        The *transformInverse* changes the physical property into the
        model.

        .. math::

            m = \log{\sigma}

        """
        raise NotImplementedError("Inverse is not implemented.")

    def _derivmatrix(self, m):
        return np.r_[
            [
                [
                    self._polynomialxx_deriv(m[:, 0])[0],
                    self._polynomialyx_deriv(m[:, 0])[0],
                ],
                [
                    self._polynomialxy_deriv(m[:, 1])[0],
                    self._polynomialyy_deriv(m[:, 1])[0],
                ],
            ]
        ]

    def deriv(self, m, v=None):
        """"""
        if v is None:
            out = self._derivmatrix(m.reshape(-1, 2))
            return out
        else:
            out = np.dot(self._derivmatrix(m.reshape(-1, 2)), v.reshape(2, -1))
            return out

    @property
    def is_linear(self):
        return False
