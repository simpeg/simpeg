from ...utils.code_utils import deprecate_module

deprecate_module("testingUtils", "testing_utils", "0.16.0", future_warn=True)

from .testing_utils import *
