from ....utils.code_utils import deprecate_module

deprecate_module("testUtils", "test_utils", "0.16.0", error=True)

from .test_utils import *
