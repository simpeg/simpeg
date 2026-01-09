"""
Numba functions for gravity simulations.
"""

from ._2d_mesh import NUMBA_FUNCTIONS_2D
from ._3d_mesh import NUMBA_FUNCTIONS_3D

try:
    import choclo
except ImportError:
    choclo = None

__all__ = ["choclo", "NUMBA_FUNCTIONS_3D", "NUMBA_FUNCTIONS_2D"]
