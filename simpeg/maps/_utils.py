"""
Utility functions for maps.

.. important::

    Classes and functions within this file are not meant to be public, just to be used
    internally by maps.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import LinearOperator


class ColumnStackedMatrix(LinearOperator):
    r"""
    Represent a matrix formed by a collection of dense columns.

    This is a LinearOperator that represents a dense matrix given by its columns.
    It's purpose is to avoid allocating memory for the large matrix if it'll only be
    used to compute a dot product with a vector.

    Parameters
    ----------
    columns : list of array
        List of 1D arrays. Each one of these arrays will be a column of the represented
        matrix. All arrays should have the same size.
    dtype : dtype, optional
        Data type used for the LinearOperator. Default to ``np.float64``.

    Notes
    -----
    Given a collection of vectors
    :math:`\left[\mathbf{c}_0, \dots, \mathbf{c}_M\right]`,
    this linear operator will represent the following matrix:

    .. math::

        \mathbf{M} =
        \begin{bmatrix}
            \vdots & \dots & \vdots           //
            \mathbf{c}_0 & \dots \mathbf{c}_M //
            \vdots & \dots & \vdots
        \end{bmatrix}
    """

    def __init__(self, columns: list[npt.NDArray], *, dtype=np.float64):
        # Raise error on empty list
        if not columns:
            msg = "Invalid empty columns list."
            raise ValueError(msg)

        # Check columns as 1D arrays
        columns = [np.asarray(c, dtype=dtype, copy=False) for c in columns]
        for column in columns:
            if column.ndim != 1:
                msg = f"Invalid array '{column}' with '{column.ndim}' dimensions."
                raise ValueError(msg)

        # Check sizes
        sizes = np.array([column.size for column in columns])
        if not (sizes[0] == sizes).all():
            msg = "Invalid column arrays with different sizes."
            raise ValueError(msg)

        n_rows = sizes[0]
        shape = (n_rows, len(columns))
        self._columns = columns
        super().__init__(shape=shape, dtype=dtype)

    @property
    def columns(self):
        return self._columns

    def _matvec(self, x):
        result = np.zeros(self.shape[0])
        for column, value in zip(self.columns, x, strict=True):
            result += column * value
        return result

    def _rmatvec(self, x):
        return np.array([column @ x for column in self.columns])
