"""
Backward compatibility with the ``simpeg.directives.directives`` submodule.

This file will be deleted when the ``simpeg.directives.directives`` submodule is
removed.
"""

import warnings
from ._directives import *  # noqa: F403,F401

warnings.warn(
    "The `simpeg.directives.directives` submodule has been deprecated, "
    "and will be removed in SimPEG v0.26.0."
    "Import any directive class directly from the `simpeg.directives` module. "
    "E.g.: `from simpeg.directives import BetaSchedule`",
    FutureWarning,
    stacklevel=2,
)
