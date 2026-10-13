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
        List of 1D or 2D arrays. Each one of these arrays will be one (or several)
        column of the represented matrix.
        If columns are 1D arrays, they should all have the same size.
        If any 2D array is passed, then it'll be consider as a whole block of the matrix. It should have a compatible shape.
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
            \vdots       & \dots & \vdots       \\
            \mathbf{c}_0 & \dots & \mathbf{c}_M \\
            \vdots       & \dots & \vdots
        \end{bmatrix}

    If one of the objects in the collection is a 2D matrix, it'll be consider as a full
    block of the whole matrix. For example, the following collection
    :math:`\left[\mathbf{c}_0, \dots, \mathbf{D}, \dots, \mathbf{c}_M\right]`,
    will generate a linear operator that represents the following matrix:

    .. math::

        \mathbf{M} =
        \begin{bmatrix}
            \vdots       & \dots & D_{11} & D_{12} & \dots & \vdots \\
            \mathbf{c}_0 & \dots & \vdots & \vdots & \dots & \mathbf{c}_M \\
            \vdots       & \dots & D_{N1} & D_{N2} & \dots & \vdots
        \end{bmatrix}
    """

    def __init__(self, columns: list[npt.NDArray], *, dtype=np.float64):
        # Raise error on empty list
        if not columns:
            msg = "Invalid empty columns list."
            raise ValueError(msg)

        # Check columns as arrays
        columns = [np.asarray(c, dtype=dtype, copy=False) for c in columns]
        for column in columns:
            if column.ndim not in (1, 2):
                msg = f"Invalid array '{column}' with '{column.ndim}' dimensions."
                raise ValueError(msg)

        # Check if all arrays will create compatible number of rows
        n_rows = np.array([column.shape[0] for column in columns])
        if not (n_rows[0] == n_rows).all():
            msg = "Invalid column arrays with incompatible sizes."
            raise ValueError(msg)
        n_rows = n_rows[0]

        # Get total number of columns
        n_cols = sum(1 if column.ndim == 1 else column.shape[1] for column in columns)

        shape = (n_rows, n_cols)
        self._columns = columns
        super().__init__(shape=shape, dtype=dtype)

    @property
    def columns(self):
        return self._columns

    def _matvec(self, x):
        result = np.zeros(self.shape[0], dtype=self.dtype)
        index = 0
        for column in self.columns:
            if column.ndim == 1:
                result += column * x[index]
                index += 1
            else:
                new_index = index + column.shape[1]
                result += column @ x[index:new_index]
                index = new_index
        return result

    def _rmatvec(self, x):
        result = np.zeros(self.shape[1], dtype=self.dtype)
        index = 0
        for column in self.columns:
            if column.ndim == 1:
                result[index] = column @ x
                index += 1
            else:
                new_index = index + column.shape[1]
                result[index:new_index] = column.T @ x
                index = new_index
        return result
