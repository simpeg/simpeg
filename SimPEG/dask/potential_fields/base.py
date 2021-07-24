import numpy as np
from ...potential_fields.base import BasePFSimulation as Sim
import os
from dask import delayed, array, config
from dask.diagnostics import ProgressBar
from ..utils import compute_chunk_sizes

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


def dask_linear_operator(self):
    self.nC = self.modelMap.shape[0]

    n_data_comp = len(self.survey.components)
    components = np.array(list(self.survey.components.keys()))
    active_components = np.hstack(
        [np.c_[values] for values in self.survey.components.values()]
    ).tolist()

    row = delayed(self.evaluate_integral, pure=True)
    rows = [
        array.from_delayed(
            row(receiver_location, components[component]),
            dtype=np.float32,
            shape=(n_data_comp, self.nC),
        )
        for receiver_location, component in zip(
            self.survey.receiver_locations.tolist(), active_components
        )
    ]
    stack = array.vstack(rows)

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

    if self.store_sensitivities == "disk":
        sens_name = self.sensitivity_path + "sensitivity.zarr"
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
        else:
            print("Writing Zarr file to disk")
            with ProgressBar():
                print("Saving kernel to zarr: " + sens_name)
                kernel = array.to_zarr(
                    stack, sens_name, compute=True, return_stored=True, overwrite=True
                )
    elif self.store_sensitivities == "forward_only":
        with ProgressBar():
            print("Forward calculation: ")
            pred = (stack @ self.model).compute()
        return pred
    else:
        print(stack.chunks)
        with ProgressBar():
            print("Computing sensitivities to local ram")
            kernel = array.asarray(stack.compute())
    return kernel


Sim.linear_operator = dask_linear_operator
