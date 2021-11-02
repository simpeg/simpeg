import numpy as np
from ...potential_fields.base import BasePFSimulation as Sim
import os
from dask import delayed, array, config
from dask.diagnostics import ProgressBar
from ..utils import compute_chunk_sizes
from dask.distributed import get_client, Future, Client

Sim._chunk_format = "row"


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


def dask_fields(self, m):
    """
    Fields computed from a linear operation
    """
    self.model = m
    fields = self.G @ (self.modelMap @ m).astype(np.float32)

    return fields


Sim.fields = dask_fields


@property
def Jmatrix(self):
    if getattr(self, "_Jmatrix", None) is None:
        if self.store_sensitivities == "ram":
            self._Jmatrix = np.asarray(self.linear_operator())
        else:
            if self.workers is not None:
                client = get_client()
                self.Xn, self.Yn, self.Zn = client.scatter(
                    [self.Xn, self.Yn, self.Zn], workers=self.workers if isinstance(self.workers, tuple) else None
                )

                if getattr(self, "tmi_projection", None) is not None:
                    self.tmi_projection = client.scatter(
                        [self.tmi_projection], workers=self.workers if isinstance(self.workers, tuple) else None
                    )

                if getattr(self, "M", None) is not None:
                    self.M = client.scatter(
                        [self.M], workers=self.workers
                    )

                self._Jmatrix = client.compute(
                        self.linear_operator(),
                    workers=self.workers if isinstance(self.workers, tuple) else None
                )
            else:
                delayed_array = self.linear_operator()

                if "store-map" not in delayed_array.name:
                    self._Jmatrix = delayed_array
                else:
                    return delayed_array

    elif isinstance(self._Jmatrix, Future):
        self._Jmatrix.result()
        self._Jmatrix = array.from_zarr(os.path.join(self.sensitivity_path, "J.zarr"))

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