"""
Numba functions for gravity simulation on 3D meshes.
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

from ..._numba_utils import kernels_in_nodes_to_cell


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

    .. code::

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
                * kernels_in_nodes_to_cell(
                    kernels,
                    cell_nodes[k, :],
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

    .. code::

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
            sensitivity_matrix[i, k] = constant_factor * kernels_in_nodes_to_cell(
                kernels,
                cell_nodes[k, :],
            )


@jit(nopython=True, parallel=False)
def _diagonal_G_T_dot_G_serial(
    receivers,
    nodes,
    cell_nodes,
    kernel_func,
    constant_factor,
    weights,
    diagonal,
):
    """
    Diagonal of ``G.T @ W.T @ W @ G`` without storing ``G``, in serial.

    Parameters
    ----------
    receivers : (n_receivers, 3) numpy.ndarray
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) numpy.ndarray
        Array with the location of the mesh nodes.
    cell_nodes : (n_active_cells, 8) numpy.ndarray
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    kernel_func : callable
        Kernel function that will be evaluated on each node of the mesh. Choose
        one of the kernel functions in ``choclo.prism``.
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        sensitivity matrix.
    weights : (n_receivers,) numpy.ndarray
        Array with data weights. It should be the diagonal of the ``W`` matrix,
        squared.
    diagonal : (n_active_cells,) numpy.ndarray
        Array where the diagonal of ``G.T @ G`` will be added to.

    Notes
    -----
    This function is meant to be run in serial. Use the
    ``_diagonal_G_T_dot_G_parallel`` one for parallelized computations.
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
        # Add diagonal components in the running result.
        for k in range(n_cells):
            diagonal[k] += (
                weights[i]
                * (
                    constant_factor
                    * kernels_in_nodes_to_cell(
                        kernels,
                        cell_nodes[k, :],
                    )
                )
                ** 2
            )


@jit(nopython=True, parallel=True)
def _diagonal_G_T_dot_G_parallel(
    receivers,
    nodes,
    cell_nodes,
    kernel_func,
    constant_factor,
    weights,
    diagonal,
):
    """
    Diagonal of ``G.T @ W.T @ W @ G`` without storing ``G``, in parallel.

    Parameters
    ----------
    receivers : (n_receivers, 3) numpy.ndarray
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) numpy.ndarray
        Array with the location of the mesh nodes.
    cell_nodes : (n_active_cells, 8) numpy.ndarray
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    kernel_func : callable
        Kernel function that will be evaluated on each node of the mesh. Choose
        one of the kernel functions in ``choclo.prism``.
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        sensitivity matrix.
    weights : (n_receivers,) numpy.ndarray
        Array with data weights. It should be the diagonal of the ``W`` matrix,
        squared.
    diagonal : (n_active_cells,) numpy.ndarray
        Array where the diagonal of ``G.T @ G`` will be added to.

    Notes
    -----
    This function is meant to be run in parallel. Use the
    ``_diagonal_G_T_dot_G_serial`` one for serialized computations.

    This implementation instructs each thread to allocate their own array for
    the current row of the sensitivity matrix. After computing the elements of
    that row, it gets added to the running ``result`` array through a reduction
    operation handled by Numba.
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
        # Allocate array for the diagonal elements for the current receiver.
        local_diagonal = np.empty(n_cells)
        for k in range(n_cells):
            local_diagonal[k] = (
                weights[i]
                * (
                    constant_factor
                    * kernels_in_nodes_to_cell(
                        kernels,
                        cell_nodes[k, :],
                    )
                )
                ** 2
            )
        # Add the result to the diagonal.
        # Apply reduction operation to add the values of the local diagonal to
        # the running diagonal array. Avoid slicing the `diagonal` array when
        # updating it to avoid racing conditions, just add the `local_diagonal`
        # to the `diagonal` variable.
        diagonal += local_diagonal


@jit(nopython=True, parallel=False)
def _sensitivity_gravity_t_dot_v_serial(
    receivers,
    nodes,
    cell_nodes,
    kernel_func,
    constant_factor,
    vector,
    result,
):
    """
    Compute ``G.T @ v`` in serial, without building G.

    Parameters
    ----------
    receivers : (n_receivers, 3) numpy.ndarray
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) numpy.ndarray
        Array with the location of the mesh nodes.
    cell_nodes : (n_active_cells, 8) numpy.ndarray
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    kernel_func : callable
        Kernel function that will be evaluated on each node of the mesh. Choose
        one of the kernel functions in ``choclo.prism``.
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        sensitivity matrix.
    vector : (n_receivers) numpy.ndarray
        Array that represents the vector used in the dot product.
    result : (n_active_cells) numpy.ndarray
        Running result array where the output of the dot product will be added to.

    Notes
    -----
    This function is meant to be run in serial. Writing to the ``result`` array
    inside a parallel loop over the receivers generates a race condition that
    leads to corrupted outputs.

    A parallel implementation of this function is available in
    ``_sensitivity_gravity_t_dot_v_parallel``.
    """
    n_receivers = receivers.shape[0]
    n_nodes = nodes.shape[0]
    n_cells = cell_nodes.shape[0]
    # Evaluate kernel function on each node, for each receiver location
    for i in range(n_receivers):
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
        # Compute the i-th row of the sensitivity matrix and multiply it by the
        # i-th element of the vector.
        for k in range(n_cells):
            result[k] += (
                constant_factor
                * vector[i]
                * kernels_in_nodes_to_cell(
                    kernels,
                    cell_nodes[k, :],
                )
            )


@jit(nopython=True, parallel=True)
def _sensitivity_gravity_t_dot_v_parallel(
    receivers,
    nodes,
    cell_nodes,
    kernel_func,
    constant_factor,
    vector,
    result,
):
    """
    Compute ``G.T @ v`` in parallel without building G.

    Parameters
    ----------
    receivers : (n_receivers, 3) numpy.ndarray
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) numpy.ndarray
        Array with the location of the mesh nodes.
    cell_nodes : (n_active_cells, 8) numpy.ndarray
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    kernel_func : callable
        Kernel function that will be evaluated on each node of the mesh. Choose
        one of the kernel functions in ``choclo.prism``.
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        sensitivity matrix.
    vector : (n_receivers) numpy.ndarray
        Array that represents the vector used in the dot product.
    result : (n_active_cells) numpy.ndarray
        Running result array where the output of the dot product will be added to.

    Notes
    -----
    This function is meant to be run in parallel.
    This implementation instructs each thread to allocate their own array for
    the current row of the sensitivity matrix. After computing the elements of
    that row, it gets added to the running ``result`` array through a reduction
    operation handled by Numba.

    A serialized implementation of this function is available in
    ``_sensitivity_gravity_t_dot_v_serial``.
    """
    n_receivers = receivers.shape[0]
    n_nodes = nodes.shape[0]
    n_cells = cell_nodes.shape[0]
    # Evaluate kernel function on each node, for each receiver location
    for i in prange(n_receivers):
        # Allocate vector for kernels evaluated on mesh nodes
        kernels = np.empty(n_nodes)
        # Allocate array for the current row of the sensitivity matrix
        local_row = np.empty(n_cells)
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
            local_row[k] = (
                constant_factor
                * vector[i]
                * kernels_in_nodes_to_cell(
                    kernels,
                    cell_nodes[k, :],
                )
            )
        # Apply reduction operation to add the values of the row to the running
        # result. Avoid slicing the `result` array when updating it to avoid
        # racing conditions, just add the `local_row` to the `results`
        # variable.
        result += local_row


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


# Define a dictionary with decorated versions of the Numba functions.
NUMBA_FUNCTIONS_3D = {
    "sensitivity": {
        parallel: jit(nopython=True, parallel=parallel)(_sensitivity_gravity)
        for parallel in (True, False)
    },
    "forward": {
        parallel: jit(nopython=True, parallel=parallel)(_forward_gravity)
        for parallel in (True, False)
    },
    "diagonal_gtg": {
        False: _diagonal_G_T_dot_G_serial,
        True: _diagonal_G_T_dot_G_parallel,
    },
    "gt_dot_v": {
        False: _sensitivity_gravity_t_dot_v_serial,
        True: _sensitivity_gravity_t_dot_v_parallel,
    },
}
