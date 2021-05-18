from ...utils.code_utils import deprecate_module

deprecate_module("CurrentUtils", "current_utils", "0.16.0", future_warn=True)

from .current_utils import *
