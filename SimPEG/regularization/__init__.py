from .base import (
    BaseRegularization,
    BaseComboRegularization,
    BaseSimilarityMeasure,
    Small,
    SmoothDeriv,
    SmoothDeriv2,
    L2Regularization
)
from .regularization_mesh import RegularizationMesh
from .sparse import SparseSmall, SparseDeriv, Sparse
from .pgi import (
    PGIsmallness,
    PGI,
    PGIwithNonlinearRelationshipsSmallness,
    PGIwithRelationships,
)
from .cross_gradient import CrossGradient
from .correspondence import LinearCorrespondence
from .jtv import JointTotalVariation
