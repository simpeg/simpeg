import numpy as np

from ...potential_fields.base import BasePFSimulation as Sim
from dask.distributed import get_client
import os
from dask import delayed, array, config
from dask.diagnostics import ProgressBar
from ..utils import compute_chunk_sizes


_chunk_format = "row"


@property
def chunk_format(self):
    "Apply memory chunks along rows of G, either 'equal', 'row', or 'auto'"
    return self._chunk_format


@chunk_format.setter
def chunk_format(self, other):
    if other not in ["equal", "row", "auto"]:
        raise ValueError("Chunk format must be 'equal', 'row', or 'auto'")
    self._chunk_format = other


def dpred(self, m=None, f=None):
    if m is not None:
        self.model = m
    if f is not None:
        return f
    return self.fields(self.model)


def residual(self, m, dobs, f=None):
    return self.dpred(m, f=f) - dobs


def block_compute(sim, rows, components):
    block = []
    for row in rows:
        block.append(sim.evaluate_integral(row, components))

    if sim.store_sensitivities == "forward_only":
        return np.hstack(block)

    return np.vstack(block)


def linear_operator(self):
    forward_only = self.store_sensitivities == "forward_only"
    n_cells = self.nC
    if getattr(self, "model_type", None) == "vector":
        n_cells *= 3

    n_components = len(self.survey.components)
    n_blocks = np.ceil(
        (n_cells * n_components * self.survey.receiver_locations.shape[0] * 8.0 * 1e-6)
        / self.max_chunk_size
    )
    block_split = np.array_split(self.survey.receiver_locations, n_blocks)

    try:
        client = get_client()
    except ValueError:
        client = None

    if client:
        sim = client.scatter(self, workers=self.worker)
    else:
        delayed_compute = delayed(block_compute)

    rows = []
    for block in block_split:
        if client:
            rows.append(
                client.submit(
                    block_compute,
                    sim,
                    block,
                    self.survey.components,
                    workers=self.worker,
                )
            )
        else:
            chunk = delayed_compute(self, block, self.survey.components)
            rows.append(
                array.from_delayed(
                    chunk,
                    dtype=self.sensitivity_dtype,
                    shape=(
                        (len(block) * n_components,)
                        if forward_only
                        else (len(block) * n_components, n_cells)
                    ),
                )
            )

    if client:
        if forward_only:
            return np.hstack(client.gather(rows))
        return np.vstack(client.gather(rows))

    if forward_only:
        stack = array.concatenate(rows)
    else:
        stack = array.vstack(rows)
        # Chunking options
        if self.chunk_format == "row":
            config.set({"array.chunk-size": f"{self.max_chunk_size}MiB"})
            # Autochunking by rows is faster and more memory efficient for
            # very large problems sensitivty and forward calculations
            stack = stack.rechunk({0: "auto", 1: -1})
        elif self.chunk_format == "equal":
            # Manual chunks for equal number of blocks along rows and columns.
            # Optimal for Jvec and Jtvec operations
            row_chunk, col_chunk = compute_chunk_sizes(
                *stack.shape, self.max_chunk_size
            )
            stack = stack.rechunk((row_chunk, col_chunk))
        else:
            # Auto chunking by columns is faster for Inversions
            config.set({"array.chunk-size": f"{self.max_chunk_size}MiB"})
            stack = stack.rechunk({0: -1, 1: "auto"})

    if self.store_sensitivities == "disk":
        sens_name = os.path.join(self.sensitivity_path, "sensitivity.zarr")
        if os.path.exists(sens_name):
            kernel = array.from_zarr(sens_name)
            if np.all(
                np.r_[
                    np.any(np.r_[kernel.chunks[0]] == stack.chunks[0]),
                    np.any(np.r_[kernel.chunks[1]] == stack.chunks[1]),
                    np.r_[kernel.shape] == np.r_[stack.shape],
                ]
            ):
                # Check that loaded kernel matches supplied data and mesh
                print("Zarr file detected with same shape and chunksize ... re-loading")
                return kernel

        print("Writing Zarr file to disk")
        with ProgressBar():
            print("Saving kernel to zarr: " + sens_name)
            kernel = array.to_zarr(
                stack, sens_name, compute=True, return_stored=True, overwrite=True
            )

    with ProgressBar():
        kernel = stack.compute()
    return kernel


def compute_J(self, _, f=None):
    return self.linear_operator()


@property
def Jmatrix(self):
    if getattr(self, "_Jmatrix", None) is None:
        self._Jmatrix = self.compute_J(self.model)
    return self._Jmatrix


@Jmatrix.setter
def Jmatrix(self, value):
    self._Jmatrix = value


Sim._delete_on_model_update = []
Sim._chunk_format = _chunk_format
Sim.chunk_format = chunk_format
Sim.dpred = dpred
Sim.residual = residual
Sim.linear_operator = linear_operator
Sim.compute_J = compute_J
Sim.Jmatrix = Jmatrix
