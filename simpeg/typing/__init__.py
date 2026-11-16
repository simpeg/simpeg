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
    MapLike

"""

import numpy as np
import numpy.typing as npt
from typing import Union, TypeAlias, Protocol, runtime_checkable
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


@runtime_checkable
class MapLike(Protocol):
    """Structural type describing what a physical property parametrizer needs to support.

    This documents the interface :class:`simpeg.props.PhysicalProperty` relies
    on when a physical property is parametrized by a mapping, e.g. through
    :meth:`simpeg.props.HasModel.parametrize`: a ``shape`` attribute, a
    ``deriv`` method, and support for ``*``/``@`` composition.

    Notes
    -----
    This ``Protocol`` is provided for typing and documentation purposes only.
    :meth:`simpeg.props.HasModel.parametrize` still requires an actual
    :class:`simpeg.maps.IdentityMap` instance at runtime, not merely an object
    satisfying this ``Protocol``, since :mod:`simpeg.maps`'s own internals
    (e.g. :class:`~simpeg.maps.ComboMap` construction) perform their own
    nominal ``isinstance`` checks against :class:`~simpeg.maps.IdentityMap`.
    A duck-typed object that only matches this ``Protocol``'s shape could pass
    a relaxed runtime check here, only to fail later, less legibly, deep
    inside :mod:`simpeg.maps`.
    """

    @property
    def shape(self) -> tuple: ...

    def deriv(self, m, v=None): ...

    def __mul__(self, val): ...

    def __matmul__(self, map1): ...
