import sys

print("Importing `SimPEG` is deprecated. please import from `simpeg`.")
import simpeg

sys.modules["SimPEG"] = simpeg
