"""
Utility functions for maps.

.. important::

    Classes and functions within this file are not meant to be public, just to be used
    internally by maps.
"""

from collections.abc import Iterable
import numpy.typing as npt


def columns_dot_vector(columns: Iterable[npt.NDArray], vector: npt.NDArray):
    """
    Compute the dot product between a collection of columns and a vector.

    Use this function to compute the dot product between a matrix represented by
    a sequence of column vectors and a given vector. It can perform the dot product
    computation without allocating the full matrix.

    .. hint:

        We can pass a list of pre-allocated arrays, or a ``Generator`` that will produce
        each one of the arrays in-place when it gets iterated.
        The ``Generator`` approach could allow us to reduce memory consumption by
        allocating only one column array at a time.

    Parameters
    ----------
    columns : iterable of (m,) or (m, p) array
        A collection of the column vectors that form the matrix. They can either be
        1D arrays (a single column each) or 2D arrays (a column block of the matrix).
        The collection should contain a total number of *matrix columns* equal to the
        size of the expected ``result`` vector (``n``).
    vector : (m,) array
        Vector that will be used in the dot product.

    Returns
    -------
    result : (n,) vector
        Resulting vector.
    """
    empty = True  # flag used to check if the iterable is empty
    index = 0
    result = 0
    for column in columns:
        empty = False
        match column.ndim:
            case 1:
                # Add the column times the index-th value of the vector to the result.
                result += column * vector[index]
                index += 1
            case 2:
                # Add to result the dot product between the 2D column block and the
                # corresponding slice of the vector.
                new_index = index + column.shape[1]
                result += column @ vector[index:new_index]
                index = new_index
            case _:
                msg = f"Invalid column '{column}' with '{column.ndim}' dimensions."
                raise ValueError(msg)

    if empty:
        msg = "Invalid empty 'columns' argument. It must contain at last one element."
        raise ValueError(msg)

    if index != vector.size:
        msg = (
            f"The total number of columns ('{index}') passed through the `columns` "
            f"is not equal to the size of the vector ('{vector.size}'). "
        )
        raise ValueError(msg)

    return result
