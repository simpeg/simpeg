from .....electromagnetics.static.induced_polarization.simulation import (
    BaseIPSimulation as Sim,
)
from .....utils import Zero, mkvc
from .....data import Data
from ....utils import compute_chunk_sizes
import dask
import dask.array as da
from dask.distributed import Future
import numpy as np
import zarr
import os
import shutil
import numcodecs

numcodecs.blosc.use_threads = False

Sim.sensitivity_path = './sensitivity/'

from ..resistivity.simulation import (
    dask_fields, dask_getJtJdiag, dask_Jvec, dask_Jtvec,
    compute_J, dask_dpred, dask_getSourceTerm,
)

Sim.fields = dask_fields
Sim.getJtJdiag = dask_getJtJdiag
Sim.Jvec = dask_Jvec
Sim.Jtvec = dask_Jtvec
Sim.compute_J = compute_J
Sim.dpred = dask_dpred
Sim.getSourceTerm = dask_getSourceTerm
