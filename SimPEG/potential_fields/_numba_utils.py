"""
Utility functions for Numba implementations

These functions are meant to be used both in the Numba-based gravity and
magnetic simulations.
"""

try:
    from numba import jit
except ImportError:
    # Define dummy jit decorator
    def jit(*args, **kwargs):
        return lambda f: f


@jit(nopython=True)
def kernels_in_nodes_to_cell(kernels, nodes_indices):
    """
    Evaluate integral on a given cell from evaluation of kernels on nodes

    Parameters
    ----------
    kernels : (n_active_nodes,) array
        Array with kernel values on each one of the nodes in the mesh.
    nodes_indices : (8,) array of int
        Indices of the nodes for the current cell in "F" order (x changes
        faster than y, and y faster than z).

    Returns
    -------
    float
    """
    result = (
        -kernels[nodes_indices[0]]
        + kernels[nodes_indices[1]]
        + kernels[nodes_indices[2]]
        - kernels[nodes_indices[3]]
        + kernels[nodes_indices[4]]
        - kernels[nodes_indices[5]]
        - kernels[nodes_indices[6]]
        + kernels[nodes_indices[7]]
    )
    return result
