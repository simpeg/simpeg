from .base import BaseRegularization, BaseComboRegularization
from .regularization_mesh import RegularizationMesh
from .tikhonov import (
    SimpleSmall, SimpleSmoothDeriv, Simple,
    Small, SmoothDeriv, SmoothDeriv2, Tikhonov
)
from .sparse import SparseSmall, SparseDeriv, Sparse
