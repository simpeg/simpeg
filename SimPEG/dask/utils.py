import numpy as np


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
