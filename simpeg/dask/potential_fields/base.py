import numpy as np

from ...potential_fields.base import BasePFSimulation as Sim

import os
from dask import delayed, array, compute

from dask.diagnostics import ProgressBar

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


def block_compute(sim, rows, components, j_matrix, count):
    block = []
    for row in rows:
        block.append(sim.evaluate_integral(row, components))

    if sim.store_sensitivities == "forward_only":
        return np.hstack(block)

    values = np.vstack(block)
    return storage_formatter(values, count, j_matrix)


def storage_formatter(
    rows: np.ndarray,
    count: int,
    j_matrix: zarr.Array | None = None,
):
    """
    Format the storage of the sensitivity matrix.

    :param rows: List of dask arrays representing blocks of the sensitivity matrix.
    :param count: Current row count offset.
    :param j_matrix: Zarr array to store the sensitivity matrix on disk, if applicable

    :return: If j_matrix is provided, returns None after storing the rows; otherwise,
    returns the stacked rows as a NumPy array.
    """

    if isinstance(j_matrix, zarr.Array):
        j_matrix.set_orthogonal_selection(
            (np.arange(count, count + rows.shape[0]), slice(None)),
            rows.astype(np.float32),
        )
        return None

    return rows


def linear_operator(self):
    forward_only = self.store_sensitivities == "forward_only"
    n_cells = self.nC
    if getattr(self, "model_type", None) == "vector":
        n_cells *= 3

    if self.store_sensitivities == "disk":

        if os.path.exists(self.sensitivity_path):
            return array.from_zarr(self.sensitivity_path)

        Jmatrix = zarr.open(
            self.sensitivity_path,
            mode="w",
            shape=(self.survey.nD, n_cells),
            chunks=(self.max_chunk_size, n_cells),
        )
    else:
        Jmatrix = None

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
    count = 0
    for block in block_split:
        if client:
            row = client.submit(
                block_compute,
                sim,
                block,
                self.survey.components,
                Jmatrix,
                count,
                workers=worker,
            )

        else:
            chunk = delayed_compute(self, block, self.survey.components, Jmatrix, count)
            row = array.from_delayed(
                chunk,
                dtype=self.sensitivity_dtype,
                shape=(
                    (len(block) * n_components,)
                    if forward_only
                    else (len(block) * n_components, n_cells)
                ),
            )
        count += block.shape[0]
        rows.append(row)

    if client:
        kernel = client.gather(rows)
    else:
        with ProgressBar():
            kernel = compute(rows)[0]

    if self.store_sensitivities == "disk" and os.path.exists(self.sensitivity_path):
        return array.from_zarr(self.sensitivity_path)

    if forward_only:
        return np.hstack(kernel)

    return np.vstack(kernel)


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
