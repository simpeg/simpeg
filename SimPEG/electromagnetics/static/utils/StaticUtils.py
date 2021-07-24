from ....utils.code_utils import deprecate_module

deprecate_module("StaticUtils", "static_utils", "0.16.0", future_warn=True)

from .static_utils import *
