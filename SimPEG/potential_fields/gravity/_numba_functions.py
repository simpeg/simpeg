"""
Numba functions for gravity simulation using Choclo.
"""
import numpy as np

try:
    import choclo
except ImportError:
    # Define dummy jit decorator
    def jit(*args, **kwargs):
        return lambda f: f

    choclo = None
else:
    from numba import jit, prange


def _forward_gravity(
    receivers,
    nodes,
    densities,
    fields,
    cell_nodes,
    kernel_func,
    constant_factor,
):
    """
    Forward model the gravity field of active cells on receivers

    This function should be used with a `numba.jit` decorator, for example:

    ..code::

        from numba import jit

        jit_forward_gravity = jit(nopython=True, parallel=True)(_forward_gravity)
        jit_forward_gravity(
            receivers, nodes, densities, fields, cell_nodes, kernel_func, const_factor
        )

    Parameters
    ----------
    receivers : (n_receivers, 3) array
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) array
        Array with the location of the mesh nodes.
    densities : (n_active_cells)
        Array with densities of each active cell in the mesh.
    fields : (n_receivers) array
        Array full of zeros where the gravity fields on each receiver will be
        stored. This could be a preallocated array or a slice of it.
    cell_nodes : (n_active_cells, 8) array
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    kernel_func : callable
        Kernel function that will be evaluated on each node of the mesh. Choose
        one of the kernel functions in ``choclo.prism``.
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        ``fields`` array.

    Notes
    -----
    The constant factor is applied here to each element of fields because
    it's more efficient than doing it afterwards: it would require to
    index the elements that corresponds to each component.
    """
    n_receivers = receivers.shape[0]
    n_nodes = nodes.shape[0]
    n_cells = cell_nodes.shape[0]
    # Evaluate kernel function on each node, for each receiver location
    for i in prange(n_receivers):
        # Allocate vector for kernels evaluated on mesh nodes
        kernels = np.empty(n_nodes)
        for j in range(n_nodes):
            dx = nodes[j, 0] - receivers[i, 0]
            dy = nodes[j, 1] - receivers[i, 1]
            dz = nodes[j, 2] - receivers[i, 2]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            kernels[j] = kernel_func(dx, dy, dz, distance)
        # Compute fields from the kernel values
        for k in range(n_cells):
            fields[i] += (
                constant_factor
                * densities[k]
                * _kernels_in_nodes_to_cell(kernels, cell_nodes[k, :])
            )


def _sensitivity_gravity(
    receivers,
    nodes,
    sensitivity_matrix,
    cell_nodes,
    kernel_func,
    constant_factor,
):
    """
    Fill the sensitivity matrix

    This function should be used with a `numba.jit` decorator, for example:

    ..code::

        from numba import jit

        jit_sensitivity = jit(nopython=True, parallel=True)(_sensitivity_gravity)
        jit_sensitivity(
            receivers, nodes, densities, fields, cell_nodes, kernel_func, const_factor
        )

    Parameters
    ----------
    receivers : (n_receivers, 3) array
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) array
        Array with the location of the mesh nodes.
    sensitivity_matrix : (n_receivers, n_active_nodes) array
        Empty 2d array where the sensitivity matrix elements will be filled.
        This could be a preallocated empty array or a slice of it.
    cell_nodes : (n_active_cells, 8) array
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    kernel_func : callable
        Kernel function that will be evaluated on each node of the mesh. Choose
        one of the kernel functions in ``choclo.prism``.
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        sensitivity matrix.

    Notes
    -----
    The constant factor is applied here to each row of the sensitivity matrix
    because it's more efficient than doing it afterwards: it would require to
    index the rows that corresponds to each component.
    """
    n_receivers = receivers.shape[0]
    n_nodes = nodes.shape[0]
    n_cells = cell_nodes.shape[0]
    # Evaluate kernel function on each node, for each receiver location
    for i in prange(n_receivers):
        # Allocate vector for kernels evaluated on mesh nodes
        kernels = np.empty(n_nodes)
        for j in range(n_nodes):
            dx = nodes[j, 0] - receivers[i, 0]
            dy = nodes[j, 1] - receivers[i, 1]
            dz = nodes[j, 2] - receivers[i, 2]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            kernels[j] = kernel_func(dx, dy, dz, distance)
        # Compute sensitivity matrix elements from the kernel values
        for k in range(n_cells):
            sensitivity_matrix[i, k] = constant_factor * _kernels_in_nodes_to_cell(
                kernels, cell_nodes[k, :]
            )


@jit(nopython=True)
def _kernels_in_nodes_to_cell(kernels, nodes_indices):
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


# Define decorated versions of these functions
_sensitivity_gravity_parallel = jit(nopython=True, parallel=True)(_sensitivity_gravity)
_sensitivity_gravity_serial = jit(nopython=True, parallel=False)(_sensitivity_gravity)
_forward_gravity_parallel = jit(nopython=True, parallel=True)(_forward_gravity)
_forward_gravity_serial = jit(nopython=True, parallel=False)(_forward_gravity)
