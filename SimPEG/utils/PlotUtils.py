from .code_utils import deprecate_module

deprecate_module("PlotUtils", "plot_utils", "0.16.0", error=True)

from .plot_utils import *
