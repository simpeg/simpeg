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

from .._numba_utils import kernels_in_nodes_to_cell


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


def _forward_gravity_2d_mesh(
    receivers,
    cells_bounds,
    top,
    bottom,
    densities,
    fields,
    forward_func,
    constant_factor,
):
    """
    Forward gravity fields of 2D meshes.

    This function is designed to be used with equivalent sources, where the
    mesh is a 2D mesh (prism layer). The top and bottom boundaries of each cell
    are passed through the ``top`` and ``bottom`` arrays.

    This function should be used with a `numba.jit` decorator, for example:

    .. code::

        from numba import jit

        jit_function = jit(nopython=True, parallel=True)(_forward_gravity_2d_mesh)

    Parameters
    ----------
    receivers : (n_receivers, 3) numpy.ndarray
        Array with the locations of the receivers
    cells_bounds : (n_active_cells, 4) numpy.ndarray
        Array with the bounds of each active cell in the 2D mesh. For each row, the
        bounds should be passed in the following order: ``x_min``, ``x_max``,
        ``y_min``, ``y_max``.
    top : (n_active_cells) np.ndarray
        Array with the top boundaries of each active cell in the 2D mesh.
    bottom : (n_active_cells) np.ndarray
        Array with the bottom boundaries of each active cell in the 2D mesh.
    densities : (n_active_cells) numpy.ndarray
        Array with densities of each active cell in the mesh.
    fields : (n_receivers) numpy.ndarray
        Array full of zeros where the gravity fields on each receiver will be
        stored. This could be a preallocated array or a slice of it.
    forward_func : callable
        Forward function that will be evaluated on each node of the mesh. Choose
        one of the forward functions in ``choclo.prism``.
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
    n_cells = cells_bounds.shape[0]
    # Forward model the gravity field of each cell on each receiver location
    for i in prange(n_receivers):
        for j in range(n_cells):
            fields[i] += constant_factor * forward_func(
                receivers[i, 0],
                receivers[i, 1],
                receivers[i, 2],
                cells_bounds[j, 0],
                cells_bounds[j, 1],
                cells_bounds[j, 2],
                cells_bounds[j, 3],
                bottom[j],
                top[j],
                densities[j],
            )


def _sensitivity_gravity_2d_mesh(
    receivers,
    cells_bounds,
    top,
    bottom,
    sensitivity_matrix,
    forward_func,
    constant_factor,
):
    """
    Fill the sensitivity matrix

    This function should be used with a `numba.jit` decorator, for example:

    .. code::

        from numba import jit

        jit_function = jit(nopython=True, parallel=True)(_sensitivity_gravity_2d_mesh)

    Parameters
    ----------
    receivers : (n_receivers, 3) numpy.ndarray
        Array with the locations of the receivers
    cells_bounds : (n_active_cells, 4) numpy.ndarray
        Array with the bounds of each active cell in the 2D mesh. For each row, the
        bounds should be passed in the following order: ``x_min``, ``x_max``,
        ``y_min``, ``y_max``.
    top : (n_active_cells) np.ndarray
        Array with the top boundaries of each active cell in the 2D mesh.
    bottom : (n_active_cells) np.ndarray
        Array with the bottom boundaries of each active cell in the 2D mesh.
    sensitivity_matrix : (n_receivers, n_active_nodes) array
        Empty 2d array where the sensitivity matrix elements will be filled.
        This could be a preallocated empty array or a slice of it.
    forward_func : callable
        Forward function that will be evaluated on each node of the mesh. Choose
        one of the forward functions in ``choclo.prism``.
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
    n_cells = cells_bounds.shape[0]
    # Evaluate kernel function on each node, for each receiver location
    for i in prange(n_receivers):
        for j in range(n_cells):
            sensitivity_matrix[i, j] = constant_factor * forward_func(
                receivers[i, 0],
                receivers[i, 1],
                receivers[i, 2],
                cells_bounds[j, 0],
                cells_bounds[j, 1],
                cells_bounds[j, 2],
                cells_bounds[j, 3],
                bottom[j],
                top[j],
                1.0,  # use unitary density to get sensitivities
            )


# Define decorated versions of these functions
_sensitivity_gravity_parallel = jit(nopython=True, parallel=True)(_sensitivity_gravity)
_sensitivity_gravity_serial = jit(nopython=True, parallel=False)(_sensitivity_gravity)
_forward_gravity_parallel = jit(nopython=True, parallel=True)(_forward_gravity)
_forward_gravity_serial = jit(nopython=True, parallel=False)(_forward_gravity)
_forward_gravity_2d_mesh_serial = jit(nopython=True, parallel=False)(
    _forward_gravity_2d_mesh
)
_forward_gravity_2d_mesh_parallel = jit(nopython=True, parallel=True)(
    _forward_gravity_2d_mesh
)
_sensitivity_gravity_2d_mesh_serial = jit(nopython=True, parallel=False)(
    _sensitivity_gravity_2d_mesh
)
_sensitivity_gravity_2d_mesh_parallel = jit(nopython=True, parallel=True)(
    _sensitivity_gravity_2d_mesh
)
