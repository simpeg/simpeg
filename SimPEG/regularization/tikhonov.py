from .base import Small, SmoothDeriv, L2Regularization
from ..utils.code_utils import deprecate_class


@deprecate_class(removal_version="0.16.0", future_warn=True)
class SimpleSmall(Small):
    def __init__(self, mesh=None, **kwargs):
        if "alpha_s" not in kwargs:
            kwargs["alpha_s"] = 1
        super().__init__(mesh=mesh, **kwargs)


@deprecate_class(removal_version="0.16.0", future_warn=True)
class SimpleSmoothDeriv(SmoothDeriv):
    def __init__(self, mesh=None, **kwargs):
        super().__init__(mesh=mesh, normalized_gradients=True, **kwargs)


@deprecate_class(removal_version="0.16.0", future_warn=True)
class Simple(L2Regularization):
    def __init__(self, mesh=None, **kwargs):
        if "alpha_s" not in kwargs:
            kwargs["alpha_s"] = 1
        super().__init__(mesh=mesh, normalized_gradients=True, **kwargs)


@deprecate_class(removal_version="0.16.0", future_warn=True)
class Tikhonov(L2Regularization):
    pass
