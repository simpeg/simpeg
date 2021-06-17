from ...utils.code_utils import deprecate_module

deprecate_module("EMUtils", "waveform_utils", "0.16.0", future_warn=True)

from .waveform_utils import *
