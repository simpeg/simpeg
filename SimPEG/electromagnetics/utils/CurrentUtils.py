from ...utils.code_utils import deprecate_module

deprecate_module("CurrentUtils", "current_utils", "0.16.0", error=True)

from .current_utils import *
