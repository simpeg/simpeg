import numpy as np
from dask.distributed import get_client
from multiprocessing import cpu_count


def compute_chunk_sizes(M, N, target_chunk_size):
    """
    Compute row and collumn chunk sizes for a matrix of shape MxN,
    such that the chunks are below a certain threshold target_chunk_size (in Mb)
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
        chunk_size = rowChunk * colChunk * 8 * 1e-6  # in Mb
    return rowChunk, colChunk


def compute(self, job):
    """
    Compute dask job for either dask array or client.
    """
    if isinstance(job, np.ndarray):
        return job
    try:
        client = get_client()
        return client.compute(job, workers=self.workers)
    except ValueError:
        return job.compute()


def get_parallel_blocks(source_list: list, data_block_size, optimize=True) -> list:
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
                if (row_count + chunk_size) > (data_block_size * cpu_count()):
                    row_count = 0
                    block_count += 1
                    blocks.append([])

                blocks[block_count].append(
                    (
                        (s_id, r_id, ind),
                        (
                            chunk,
                            np.arange(row_index, row_index + chunk_size).astype(int),
                            rx.locations.shape[0],
                        ),
                    )
                )
                row_index += chunk_size
                row_count += chunk_size

    # Re-split over cpu_count if too few blocks
    if len(blocks) < cpu_count() and optimize:
        flatten_blocks = []
        for block in blocks:
            flatten_blocks += block

        chunks = np.array_split(np.arange(len(flatten_blocks)), cpu_count())
        return [[flatten_blocks[i] for i in chunk] for chunk in chunks]
    return blocks
