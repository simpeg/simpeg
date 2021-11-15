from .base import BaseRegularization, BaseComboRegularization, BaseCoupling
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
    PGIsmallness,
    PGI,
    PGIwithNonlinearRelationshipsSmallness,
    PGIwithRelationships,
)
from .cross_gradient import CrossGradient
from .correspondence import LinearCorrespondence
