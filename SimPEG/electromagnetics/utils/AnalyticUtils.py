from ...utils.code_utils import deprecate_module

deprecate_module("AnalyticUtils", "analytic_utils", "0.16.0", future_warn=True)

from .analytic_utils import *
