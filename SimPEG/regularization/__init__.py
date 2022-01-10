from .base import (
    BaseRegularization,
    LeastSquaresRegularization,
    BaseSimilarityMeasure,
    Small,
    SmoothDeriv,
    SmoothDeriv2,
)
from .regularization_mesh import RegularizationMesh
from .sparse import SparseSmall, SparseDeriv, Sparse
from .pgi import (
    PGIsmallness,
    PGI,
    PGIwithNonlinearRelationshipsSmallness,
    PGIwithRelationships,
)
from .tikhonov import Tikhonov
from .cross_gradient import CrossGradient
from .correspondence import LinearCorrespondence
from .jtv import JointTotalVariation
