import numpy as np
from ...potential_fields.base import BasePFSimulation as Sim
import os
from dask import delayed, array, config
from dask.diagnostics import ProgressBar
from ..utils import compute_chunk_sizes


def dask_linear_operator(self):
    forward_only = self.store_sensitivities == "forward_only"
    row = delayed(self.evaluate_integral, pure=True)
    n_cells = self.nC
    if getattr(self, "model_type", None) == "vector":
        n_cells *= 3

    rows = [
        array.from_delayed(
            row(receiver_location, components),
            dtype=self.sensitivity_dtype,
            shape=(len(components),) if forward_only else (len(components), n_cells),
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
        else:
            print("Writing Zarr file to disk")
            with ProgressBar():
                print("Saving kernel to zarr: " + sens_name)
                kernel = array.to_zarr(
                    stack, sens_name, compute=True, return_stored=True, overwrite=True
                )
    elif forward_only:
        with ProgressBar():
            print("Forward calculation: ")
            kernel = stack.compute()
    else:
        with ProgressBar():
            print("Computing sensitivities to local ram")
            kernel = stack.persist()
    return kernel


Sim.linear_operator = dask_linear_operator
