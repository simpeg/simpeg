from ...utils.code_utils import deprecate_module

deprecate_module("testingUtils", "testing_utils", "0.16.0", error=True)

from .testing_utils import *
