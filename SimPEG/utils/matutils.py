from .code_utils import deprecate_module

deprecate_module("matutils", "mat_utils", "0.16.0", future_warn=True)

from .mat_utils import *
