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

    RandomSeed
    MinimizeCallable

"""

import numpy as np
import numpy.typing as npt
from typing import Union, TypeAlias
from collections.abc import Callable
from scipy.sparse.linalg import LinearOperator

RandomSeed: TypeAlias = Union[
    int,
    npt.NDArray[np.int_],
    np.random.SeedSequence,
    np.random.BitGenerator,
    np.random.Generator,
]
"""
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

MinimizeCallable: TypeAlias = Callable[
    [np.ndarray, bool, bool],
    float
    | tuple[float, np.ndarray | LinearOperator]
    | tuple[float, np.ndarray, LinearOperator],
]
"""
The callable expected for the minimization operations.

The function's signature should look like::

    func(x: numpy.ndarray, return_g: bool, return_H: bool)

It should output up to three values ordered as::

    f_val : float
    gradient : numpy.ndarray
    H : LinearOperator

`f_val` is always returned, `gradient` is returned if `return_g`, and `H_func` is returned if `return_H`.
`f_val` should always be the first value returned, `gradient` will always be the second, and `H_func` will
always be the last. If `return_g == return_H == False`, then only the single argument `f_val` is
returned.
"""
