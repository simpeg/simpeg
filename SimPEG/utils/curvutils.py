from .code_utils import deprecate_module

deprecate_module("curvutils", "curv_utils", "0.16.0", error=True)

from .curv_utils import *
