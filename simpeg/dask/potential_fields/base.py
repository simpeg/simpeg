import numpy as np
from ...potential_fields.base import BasePFSimulation as Sim

import os
from dask import delayed, array, config

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


def linear_operator(self):
    forward_only = self.store_sensitivities == "forward_only"
    row = delayed(self.evaluate_integral, pure=True)
    n_cells = self.nC
    if getattr(self, "model_type", None) == "vector":
        n_cells *= 3

    rows = [
        array.from_delayed(
            row(receiver_location, components),
            dtype=self.sensitivity_dtype,
            shape=((len(components),) if forward_only else (len(components), n_cells)),
        )
        for receiver_location, components in self.survey._location_component_iterator()
    ]
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

        return array.to_zarr(
            stack, sens_name, compute=False, return_stored=True, overwrite=True
        )
    return stack


@property
def G(self):
    """
    Gravity forward operator
    """
    if getattr(self, "_G", None) is None:
        self._G = self.Jmatrix

    return self._G


def compute_J(self, _, f=None):
    return self.linear_operator().persist()


@property
def Jmatrix(self):
    if getattr(self, "_Jmatrix", None) is None:
        self._Jmatrix = self.compute_J(self.model)
    return self._Jmatrix


@Jmatrix.setter
def Jmatrix(self, value):
    self._Jmatrix = value


Sim.G = G
Sim._chunk_format = _chunk_format
Sim.chunk_format = chunk_format
Sim.dpred = dpred
Sim.residual = residual
Sim.linear_operator = linear_operator
Sim.compute_J = compute_J
Sim.Jmatrix = Jmatrix
