import numpy as np
from simpeg.utils import mat_utils
from simpeg.directives import _directives
from simpeg.typing import RandomSeed


def compute_chunk_sizes(M, N, target_chunk_size):
    """
    Compute row and collumn chunk sizes for a matrix of shape MxN,
    such that the chunks are below a certain threshold target_chunk_size (in MB)
    """
    nChunks_col = 1
    nChunks_row = 1
    rowChunk = int(np.ceil(M / nChunks_row))
    colChunk = int(np.ceil(N / nChunks_col))
    chunk_size = rowChunk * colChunk * 8 * 1e-6

    # Add more chunks until memory falls below target
    while chunk_size >= target_chunk_size:
        if rowChunk > colChunk:
            nChunks_row += 1
        else:
            nChunks_col += 1

        rowChunk = int(np.ceil(M / nChunks_row))
        colChunk = int(np.ceil(N / nChunks_col))
        chunk_size = rowChunk * colChunk * 8 * 1e-6  # in MB
    return rowChunk, colChunk


def get_parallel_blocks(
    source_list: list, data_block_size, optimize=True, thread_count=64
) -> list:
    """
    Get the blocks of sources and receivers to be computed in parallel.

    Stored as a list of tuples for
    (source, receiver, block index) and array of indices
    for the rows of the sensitivity matrix.
    """
    row_count = 0
    row_index = 0
    block_count = 0
    blocks = [[]]

    for s_id, src in enumerate(source_list):
        for r_id, rx in enumerate(src.receiver_list):
            indices = np.arange(rx.nD).astype(int)
            chunks = np.array_split(
                indices, int(np.ceil(len(indices) / data_block_size))
            )

            for ind, chunk in enumerate(chunks):
                chunk_size = len(chunk)

                # Condition to start a new block
                if (row_count + chunk_size) > (data_block_size * thread_count):
                    row_count = 0
                    block_count += 1
                    blocks.append([])

                locs = rx.locations

                if isinstance(locs, tuple):
                    locs = locs[0]

                blocks[block_count].append(
                    (
                        (s_id, r_id, ind),
                        (
                            chunk,
                            np.arange(row_index, row_index + chunk_size).astype(int),
                            locs.shape[0],
                        ),
                    )
                )
                row_index += chunk_size
                row_count += chunk_size

    # # Re-split over cpu_count if too few blocks
    if len(blocks) < thread_count and optimize:
        flatten_blocks = []
        for block in blocks:
            flatten_blocks += block

        chunks = np.array_split(np.arange(len(flatten_blocks)), thread_count)
        return [
            [flatten_blocks[i] for i in chunk] for chunk in chunks if len(chunk) > 0
        ]

    return blocks


def dask_eigenvalue_by_power_iteration(
    combo_objfct,
    model,
    n_pw_iter=4,
    fields_list=None,
    random_seed: RandomSeed | None = None,
):
    """
    Estimate largest eigenvalue in absolute value using power iteration.

    Overloaded function used by simpeg.directives._directives.BetaEstimate_ByEig class to
    use dask for parallelization.
    """
    rng = np.random.default_rng(seed=random_seed)

    # Initial guess for eigen-vector
    x0 = rng.random(size=model.shape)
    x0 = x0 / np.linalg.norm(x0)

    # Power iteration: estimate eigenvector
    for _ in range(n_pw_iter):
        x1 = combo_objfct.deriv2(model, v=x0)
        x0 = x1 / np.linalg.norm(x1)

    eigenvalue = x0.dot(combo_objfct.deriv2(model, v=x0))

    return eigenvalue


mat_utils.eigenvalue_by_power_iteration = dask_eigenvalue_by_power_iteration
_directives.eigenvalue_by_power_iteration = dask_eigenvalue_by_power_iteration
