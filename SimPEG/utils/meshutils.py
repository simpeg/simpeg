from .code_utils import deprecate_module

deprecate_module("meshutils", "mesh_utils", "0.16.0", future_warn=True)

from .mesh_utils import *
