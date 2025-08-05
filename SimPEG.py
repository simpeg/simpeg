import sys
import warnings

warnings.warn(
    "Importing `SimPEG` is deprecated. please import from `simpeg`.",
    FutureWarning,
    stacklevel=2,
)
import simpeg

sys.modules["SimPEG"] = simpeg
