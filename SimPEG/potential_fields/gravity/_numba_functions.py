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

    Parameters
    ----------
    receivers : (n_receivers, 3) numpy.ndarray
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) numpy.ndarray
        Array with the location of the mesh nodes.
    densities : (n_active_cells) numpy.ndarray
        Array with densities of each active cell in the mesh.
    fields : (n_receivers) numpy.ndarray
        Array full of zeros where the gravity fields on each receiver will be
        stored. This could be a preallocated array or a slice of it.
    cell_nodes : (n_active_cells, 8) numpy.ndarray
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
            kernels[j] = _evaluate_kernel(
                receivers[i, 0],
                receivers[i, 1],
                receivers[i, 2],
                nodes[j, 0],
                nodes[j, 1],
                nodes[j, 2],
                kernel_func,
            )
        # Compute fields from the kernel values
        for k in range(n_cells):
            fields[i] += (
                constant_factor
                * densities[k]
                * _kernels_in_nodes_to_cell(
                    kernels,
                    cell_nodes[k, 0],
                    cell_nodes[k, 1],
                    cell_nodes[k, 2],
                    cell_nodes[k, 3],
                    cell_nodes[k, 4],
                    cell_nodes[k, 5],
                    cell_nodes[k, 6],
                    cell_nodes[k, 7],
                )
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

    Parameters
    ----------
    receivers : (n_receivers, 3) numpy.ndarray
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) numpy.ndarray
        Array with the location of the mesh nodes.
    sensitivity_matrix : (n_receivers, n_active_nodes) array
        Empty 2d array where the sensitivity matrix elements will be filled.
        This could be a preallocated empty array or a slice of it.
    cell_nodes : (n_active_cells, 8) numpy.ndarray
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
            kernels[j] = _evaluate_kernel(
                receivers[i, 0],
                receivers[i, 1],
                receivers[i, 2],
                nodes[j, 0],
                nodes[j, 1],
                nodes[j, 2],
                kernel_func,
            )
        # Compute sensitivity matrix elements from the kernel values
        for k in range(n_cells):
            sensitivity_matrix[i, k] = constant_factor * _kernels_in_nodes_to_cell(
                kernels,
                cell_nodes[k, 0],
                cell_nodes[k, 1],
                cell_nodes[k, 2],
                cell_nodes[k, 3],
                cell_nodes[k, 4],
                cell_nodes[k, 5],
                cell_nodes[k, 6],
                cell_nodes[k, 7],
            )


@jit(nopython=True)
def _evaluate_kernel(
    receiver_x, receiver_y, receiver_z, node_x, node_y, node_z, kernel_func
):
    """
    Evaluate a kernel function for a single node and receiver

    Parameters
    ----------
    receiver_x, receiver_y, receiver_z : floats
        Coordinates of the receiver.
    node_x, node_y, node_z : floats
        Coordinates of the node.
    kernel_func : callable
        Kernel function that should be evaluated. For example, use one of the
        kernel functions in ``choclo.prism``.

    Returns
    -------
    float
        Kernel evaluated on the given node and receiver.
    """
    dx = node_x - receiver_x
    dy = node_y - receiver_y
    dz = node_z - receiver_z
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    return kernel_func(dx, dy, dz, distance)


@jit(nopython=True)
def _kernels_in_nodes_to_cell(
    kernels,
    nodes_indices_0,
    nodes_indices_1,
    nodes_indices_2,
    nodes_indices_3,
    nodes_indices_4,
    nodes_indices_5,
    nodes_indices_6,
    nodes_indices_7,
):
    """
    Evaluate integral on a given cell from evaluation of kernels on nodes

    Parameters
    ----------
    kernels : (n_active_nodes,) numpy.ndarray
        Array with kernel values on each one of the nodes in the mesh.
    nodes_indices : ints
        Indices of the nodes for the current cell in "F" order (x changes
        faster than y, and y faster than z).

    Returns
    -------
    float
    """
    result = (
        -kernels[nodes_indices_0]
        + kernels[nodes_indices_1]
        + kernels[nodes_indices_2]
        - kernels[nodes_indices_3]
        + kernels[nodes_indices_4]
        - kernels[nodes_indices_5]
        - kernels[nodes_indices_6]
        + kernels[nodes_indices_7]
    )
    return result


# Define decorated versions of these functions
_sensitivity_gravity_parallel = jit(nopython=True, parallel=True)(_sensitivity_gravity)
_sensitivity_gravity_serial = jit(nopython=True, parallel=False)(_sensitivity_gravity)
_forward_gravity_parallel = jit(nopython=True, parallel=True)(_forward_gravity)
_forward_gravity_serial = jit(nopython=True, parallel=False)(_forward_gravity)
