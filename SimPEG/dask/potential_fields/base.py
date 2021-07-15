import numpy as np
from ...potential_fields.base import BasePFSimulation as Sim
import os
from dask import delayed, array, config
from dask.diagnostics import ProgressBar
from ..utils import compute_chunk_sizes
from dask.distributed import get_client, Future, Client

Sim._chunk_format = "equal"


@property
def chunk_format(self):
    "Apply memory chunks along rows of G, either 'equal', 'row', or 'auto'"
    return self._chunk_format


@chunk_format.setter
def chunk_format(self, other):
    if other not in ["equal", "row", "auto"]:
        raise ValueError("Chunk format must be 'equal', 'row', or 'auto'")
    self._chunk_format = other


Sim.chunk_format = chunk_format


@property
def Jmatrix(self):
    if getattr(self, "_Jmatrix", None) is None:
        client = get_client()
        self._Jmatrix = client.compute(
                delayed(self.linear_operator)(),
            workers=self.workers
        )
    elif isinstance(self._Jmatrix, Future):
        client = get_client()
        self._Jmatrix = client.gather(self._Jmatrix)

    return self._Jmatrix


Sim.Jmatrix = Jmatrix


def dask_dpred(self, m=None, f=None, compute_J=False):
    if m is not None:
        self.model = m
    if f is not None:
        return f
    return self.fields(self.model)


Sim.dpred = dask_dpred


def dask_residual(self, m, dobs, f=None):
    return self.dpred(m, f=f) - dobs


Sim.residual = dask_residual