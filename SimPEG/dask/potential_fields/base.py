import shutil
import warnings
import numpy as np
from ...potential_fields.base import BasePFSimulation as Sim
import os
from dask import delayed, array, config
from dask.diagnostics import ProgressBar
from zarr.errors import ArrayNotFoundError
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


def linear_operator(self):
    self.nC = self.model_map.shape[0]
    active_components = np.hstack(
        [np.c_[values] for values in self.survey.components.values()]
    ).tolist()

    sens_name = os.path.join(self.sensitivity_path, "J.zarr")
    if os.path.exists(sens_name):
        try:
            kernel = array.from_zarr(sens_name)
            if np.all(kernel.shape == (np.sum(active_components), self.nC)):
                return kernel
        except ArrayNotFoundError:
            warnings.warn(
                f"Malformed sensitivity matrix found in {sens_name}. Re-computing",
                UserWarning,
            )
            shutil.rmtree(sens_name)
            pass

    stack = self.make_row_stack()

    # Chunking options
    if self.chunk_format == "row" or self.store_sensitivities == "forward_only":
        config.set({"array.chunk-size": f"{self.max_chunk_size}MiB"})
        # Autochunking by rows is faster and more memory efficient for
        # very large problems sensitivty and forward calculations
        stack = stack.rechunk({0: "auto", 1: -1})

    elif self.chunk_format == "equal":
        # Manual chunks for equal number of blocks along rows and columns.
        # Optimal for Jvec and Jtvec operations
        row_chunk, col_chunk = compute_chunk_sizes(*stack.shape, self.max_chunk_size)
        stack = stack.rechunk((row_chunk, col_chunk))
    else:
        # Auto chunking by columns is faster for Inversions
        config.set({"array.chunk-size": f"{self.max_chunk_size}MiB"})
        stack = stack.rechunk({0: -1, 1: "auto"})

    if self.store_sensitivities not in ["disk", "forward_only"]:
        return array.asarray(stack)

    print("Saving sensitivities to zarr: " + sens_name)
    kernel = array.to_zarr(
        stack, sens_name,
        compute=False, return_stored=True, overwrite=True
    )
    return kernel


Sim.linear_operator = linear_operator


@property
def Jmatrix(self):
    if getattr(self, "_Jmatrix", None) is None:
        if self.store_sensitivities == "ram":
            self._Jmatrix = np.asarray(self.linear_operator())
        else:
            try:
                client = get_client()
                workers = self.workers if isinstance(self.workers, tuple) else None
                self.Xn, self.Yn, self.Zn = client.scatter(
                    [self.Xn, self.Yn, self.Zn], workers=workers
                )

                # if getattr(self, "tmi_projection", None) is not None:
                #     self._tmi_projection = client.scatter(
                #         [self.tmi_projection], workers=workers
                #     )
                #
                # if getattr(self, "M", None) is not None:
                #     self._M = client.scatter(
                #         [self._M], workers=workers
                #     )

                self._Jmatrix = client.compute(
                        self.linear_operator(),
                    workers=workers
                )
            except ValueError:
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