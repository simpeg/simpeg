import numpy as np

from ...potential_fields.base import BasePFSimulation as Sim

import os
from dask import delayed, array, config

from dask.diagnostics import ProgressBar
from ..utils import compute_chunk_sizes
import zarr


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
    if f is None:
        f = self.fields(self.model)

    if isinstance(f, array.Array):
        return np.asarray(f)
    return f


def residual(self, m, dobs, f=None):
    return self.dpred(m, f=f) - dobs


def block_compute(sim, rows, components):
    block = []
    for row in rows:
        block.append(sim.evaluate_integral(row, components))

    if sim.store_sensitivities == "forward_only":
        return np.hstack(block)

    return np.vstack(block)


def storage_formatter(
    rows: list[np.ndarray],
    device: str,
    chunk_format="rows",
    sens_name: str = "./sensitivities.zarr",
    max_chunk_size: float = 256,
    compute: bool = True,
):
    """
    Format the storage of the sensitivity matrix.

    :param rows: List of dask arrays representing blocks of the sensitivity matrix.
    :param device: Storage option, either "forward_only", "disk", or "memory".
    :param chunk_format: Chunking format for disk storage, either "row", "equal
        or "auto".
    :param sens_name: File path to store the sensitivity matrix if device is "disk".
    :param max_chunk_size: Maximum chunk size in MiB for disk storage.
    :param compute: If True, compute the dask array before returning.
    """

    if device == "forward_only":
        if compute:
            return np.hstack(rows)
        return array.concatenate(rows)
    elif device == "disk":
        stack = array.vstack(rows)
        # Chunking options
        if chunk_format == "row":
            config.set({"array.chunk-size": f"{max_chunk_size}MiB"})
            # Autochunking by rows is faster and more memory efficient for
            # very large problems sensitivty and forward calculations
            stack = stack.rechunk({0: "auto", 1: -1})
        elif chunk_format == "equal":
            # Manual chunks for equal number of blocks along rows and columns.
            # Optimal for Jvec and Jtvec operations
            row_chunk, col_chunk = compute_chunk_sizes(*stack.shape, max_chunk_size)
            stack = stack.rechunk((row_chunk, col_chunk))
        else:
            # Auto chunking by columns is faster for Inversions
            config.set({"array.chunk-size": f"{max_chunk_size}MiB"})
            stack = stack.rechunk({0: -1, 1: "auto"})

        return array.to_zarr(
            stack, sens_name, compute=False, return_stored=False, overwrite=True
        )
    else:
        if compute:
            return np.vstack(rows)

        return array.vstack(rows)


def set_orthogonal_selection(row, Jmatrix, count):
    n_rows = row.shape[0]
    Jmatrix[count : count + n_rows, :] = row
    return None


def linear_operator(self):
    forward_only = self.store_sensitivities == "forward_only"
    n_cells = self.nC
    if getattr(self, "model_type", None) == "vector":
        n_cells *= 3

    if self.store_sensitivities == "disk" and os.path.exists(self.sensitivity_path):
        kernel = array.from_zarr(self.sensitivity_path)
        return kernel

    n_components = len(self.survey.components)
    n_blocks = np.ceil(
        (n_cells * n_components * self.survey.receiver_locations.shape[0] * 8.0 * 1e-6)
        / self.max_chunk_size
    )
    block_split = np.array_split(self.survey.receiver_locations, n_blocks)

    client, worker = self._get_client_worker()

    if client:
        sim = client.scatter(self, workers=worker)
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
                    workers=worker,
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

        if self.store_sensitivities == "disk":
            Jmatrix = zarr.open(
                self.sensitivity_path,
                mode="w",
                shape=(self.survey.nD, n_cells),
                chunks=(self.max_chunk_size, n_cells),
            )

            count = 0
            future = []
            for row, block in zip(rows, block_split):
                row_chunks = block.shape[0]
                future.append(
                    client.submit(set_orthogonal_selection, row, Jmatrix, count)
                )

                count += row_chunks

        else:
            future = client.submit(
                storage_formatter,
                rows,
                device=self.store_sensitivities,
                chunk_format=self.chunk_format,
                sens_name=self.sensitivity_path,
                max_chunk_size=self.max_chunk_size,
                workers=worker,
            )

        kernel = client.gather(future)

    else:

        with ProgressBar():
            kernel = storage_formatter(
                rows,
                device=self.store_sensitivities,
                chunk_format=self.chunk_format,
                sens_name=self.sensitivity_path,
                max_chunk_size=self.max_chunk_size,
                compute=False,
            ).compute()

    if self.store_sensitivities == "disk" and os.path.exists(self.sensitivity_path):
        kernel = array.from_zarr(self.sensitivity_path)

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
