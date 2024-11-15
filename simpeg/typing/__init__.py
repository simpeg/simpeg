"""
=============================
Typing (:mod:`simpeg.typing`)
=============================

This module provides additional `PEP 484 <https://peps.python.org/pep-0484/>`_
type aliases used in ``simpeg``'s codebase.

API
---

.. autosummary::
   :toctree: generated/

    IndexArray
    RandomSeed

"""

import numpy as np
import numpy.typing as npt
from typing import Union, TypeAlias

RandomSeed: TypeAlias = Union[
    int,
    npt.NDArray[np.int_],
    np.random.SeedSequence,
    np.random.BitGenerator,
    np.random.Generator,
]

IndexArray: TypeAlias = Union[npt.NDArray[np.int_], npt.NDArray[np.bool_]]


RandomSeed.__doc__ = """
A ``typing.Union`` for random seeds and Numpy's random number generators.

These type of variables can be used throughout ``simpeg`` to control random
states of functions and classes. These variables can either be an integer that
will be used as a ``seed`` to define a Numpy's :class:`numpy.random.Generator`, or
a predefined random number generator.

Examples
--------

>>> import numpy as np
>>> from simpeg.typing import RandomSeed
>>>
>>> def my_function(seed: RandomSeed = None):
...     rng = np.random.default_rng(seed=seed)
...     ...
"""

IndexArray.__doc__ = """
A ``typing.Union`` for the arrays representing active indexes into another array.

This type is commonly associated with describing active cells in a mesh.

Examples
--------
>>> import numpy as np
>>> from simpeg.typing import IndexArray
>>> array = np.linspace(0, 10, 11)
>>> def get_elements(array: np.ndarray, index_array: IndexArray) -> np.ndarray:
...     return array[index_array]

We can use a boolean array of the same length to index the array.
>>> bool_array = array < 5
>>> get_elements(array, bool_index)
array([0., 1., 2., 3., 4.])

Or we can pass an array of indices to access specific elements.
>>> index_array = [0, 2, 4, 6, 8]
>>> get_elements(array, index_array)
array([0., 2., 4., 6., 8.])
"""
