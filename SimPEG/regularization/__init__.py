from ..utils.code_utils import deprecate_class
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
    PGI
)
from .cross_gradient import CrossGradient
from .correspondence import LinearCorrespondence
from .jtv import JointTotalVariation


@deprecate_class(removal_version="0.x.0", future_warn=True)
class SimpleSmall(Small):
    def __init__(self, mesh=None, **kwargs):
        super().__init__(mesh=mesh, **kwargs)


@deprecate_class(removal_version="0.x.0", future_warn=True)
class SimpleSmoothDeriv(SmoothDeriv):
    def __init__(self, mesh=None, **kwargs):
        super().__init__(mesh=mesh, **kwargs)


@deprecate_class(removal_version="0.x.0", future_warn=True)
class Simple(LeastSquaresRegularization):
    def __init__(self, mesh=None, **kwargs):
        super().__init__(mesh=mesh, **kwargs)


@deprecate_class(removal_version="0.x.0", future_warn=True)
class Tikhonov(LeastSquaresRegularization):
    pass

@deprecate_class(removal_version="0.x.0", future_warn=True)
class PGIwithNonlinearRelationshipsSmallness(PGIsmallness):
    def __init__(self, gmm):
        super().__init__(gmm, non_linear_relationships=True, **kwargs)

@deprecate_class(removal_version="0.x.0", future_warn=True)
class PGIwithRelationships(PGI):
    def __init__(self, mesh, gmmref, **kwargs):
        super().__init__(mesh, gmmref, non_linear_relationships=True, **kwargs)
