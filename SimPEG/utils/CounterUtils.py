from .code_utils import deprecate_module

deprecate_module("CounterUtils", "counter_utils", "0.16.0", error=True)

from .counter_utils import *
