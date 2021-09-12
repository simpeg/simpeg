from .base import BaseRegularization, L2Regularization, BaseComboRegularization, Small, SmoothDeriv, SmoothDeriv2
from .regularization_mesh import RegularizationMesh
from .tikhonov import (
    SimpleSmall,
    SimpleSmoothDeriv,
    Simple,
    Small,
    SmoothDeriv,
    SmoothDeriv2,
    Tikhonov,
)
from .sparse import SparseSmall, SparseDeriv, Sparse
from .pgi import (
    SimplePGIsmallness,
    PGIsmallness,
    SimplePGI,
    PGI,
    SimplePGIwithNonlinearRelationshipsSmallness,
    SimplePGIwithRelationships,
)
