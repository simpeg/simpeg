from .code_utils import deprecate_module

deprecate_module("ModelBuilder", "model_builder", "0.16.0", error=True)

from .model_builder import *
