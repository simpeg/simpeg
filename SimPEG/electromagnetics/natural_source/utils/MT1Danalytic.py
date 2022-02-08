from ....utils.code_utils import deprecate_module

deprecate_module("MT1Danalytic", "analytic_1d", "0.16.0", error=True)

from .analytic_1d import *
