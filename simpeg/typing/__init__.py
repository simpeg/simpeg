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

"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Union

# Use try and except to support Python<3.10
try:
    from typing import TypeAlias

    RandomSeed: TypeAlias = Union[
        int,
        npt.NDArray[np.int_],
        np.random.SeedSequence,
        np.random.BitGenerator,
        np.random.Generator,
    ]
except ImportError:
    RandomSeed = Union[
        int,
        npt.NDArray[np.int_],
        np.random.SeedSequence,
        np.random.BitGenerator,
        np.random.Generator,
    ]

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
