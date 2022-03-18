from .base import BaseRegularization, BaseComboRegularization, BaseSimilarityMeasure
from .regularization_mesh import RegularizationMesh, LCRegularizationMesh
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
from .laterally_constrained import (
    LaterallyConstrained,
    LaterallyConstrainedSmall,
    LaterallyConstrainedDeriv,
)
from .pgi import (
    PGIsmallness,
    PGI,
    PGIwithNonlinearRelationshipsSmallness,
    PGIwithRelationships,
)
from .cross_gradient import CrossGradient
from .correspondence import LinearCorrespondence
from .jtv import JointTotalVariation
