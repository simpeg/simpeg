from ....utils.code_utils import deprecate_module

deprecate_module("MT1Dsolutions", "solutions_1d", "0.16.0", future_warn=True)

from .solutions_1d import *
