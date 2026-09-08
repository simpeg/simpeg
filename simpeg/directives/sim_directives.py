"""
Backward compatibility with the ``simpeg.directives.sim_directives`` submodule.

This file will be deleted when the ``simpeg.directives.sim_directives`` submodule
is removed.
"""

import warnings
from ._sim_directives import *  # noqa: F403,F401

warnings.warn(
    "The `simpeg.directives.sim_directives` submodule has been deprecated, "
    "and will be removed in SimPEG v0.26.0."
    "Import any directive class directly from the `simpeg.directives` module. "
    "E.g.: `from simpeg.directives import PairedBetaEstimate_ByEig`",
    FutureWarning,
    stacklevel=2,
)
