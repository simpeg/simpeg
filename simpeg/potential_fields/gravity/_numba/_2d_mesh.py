"""
Numba functions for gravity simulation on 2D meshes.

These functions assumes 3D prisms formed by a 2D mesh plus top and bottom boundaries for
each prism.
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

        jit_function = jit(nopython=True, parallel=True)(_forward_gravity)

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


def _sensitivity_gravity(
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

        jit_function = jit(nopython=True, parallel=True)(_sensitivity_gravity)

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


@jit(nopython=True, parallel=False)
def _g_t_dot_v_serial(
    receivers,
    cells_bounds,
    top,
    bottom,
    forward_func,
    constant_factor,
    vector,
    result,
):
    """
    Compute ``G.T @ v`` in serial, without building G, for a 2D mesh.

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
    forward_func : callable
        Forward function that will be evaluated on each node of the mesh. Choose
        one of the forward functions in ``choclo.prism``.
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
    """
    n_receivers = receivers.shape[0]
    n_cells = cells_bounds.shape[0]
    for i in range(n_receivers):
        for j in range(n_cells):
            # Compute the i-th row of the sensitivity matrix and multiply it by the
            # i-th element of the vector.
            result[j] += constant_factor * forward_func(
                receivers[i, 0],
                receivers[i, 1],
                receivers[i, 2],
                cells_bounds[j, 0],
                cells_bounds[j, 1],
                cells_bounds[j, 2],
                cells_bounds[j, 3],
                bottom[j],
                top[j],
                vector[i],
            )


@jit(nopython=True, parallel=True)
def _g_t_dot_v_parallel(
    receivers,
    cells_bounds,
    top,
    bottom,
    forward_func,
    constant_factor,
    vector,
    result,
):
    """
    Compute ``G.T @ v`` in parallel, without building G, for a 2D mesh.

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
    forward_func : callable
        Forward function that will be evaluated on each node of the mesh. Choose
        one of the forward functions in ``choclo.prism``.
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
    """
    n_receivers = receivers.shape[0]
    n_cells = cells_bounds.shape[0]
    for i in prange(n_receivers):
        # Allocate array for the current row of the sensitivity matrix
        local_row = np.empty(n_cells)
        for j in range(n_cells):
            # Compute the i-th row of the sensitivity matrix and multiply it by the
            # i-th element of the vector.
            local_row[j] = constant_factor * forward_func(
                receivers[i, 0],
                receivers[i, 1],
                receivers[i, 2],
                cells_bounds[j, 0],
                cells_bounds[j, 1],
                cells_bounds[j, 2],
                cells_bounds[j, 3],
                bottom[j],
                top[j],
                vector[i],
            )
        # Apply reduction operation to add the values of the row to the running
        # result. Avoid slicing the `result` array when updating it to avoid
        # racing conditions, just add the `local_row` to the `results`
        # variable.
        result += local_row


@jit(nopython=True, parallel=False)
def _diagonal_G_T_dot_G_serial(
    receivers,
    cells_bounds,
    top,
    bottom,
    forward_func,
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
    cells_bounds : (n_active_cells, 4) numpy.ndarray
        Array with the bounds of each active cell in the 2D mesh. For each row, the
        bounds should be passed in the following order: ``x_min``, ``x_max``,
        ``y_min``, ``y_max``.
    top : (n_active_cells) np.ndarray
        Array with the top boundaries of each active cell in the 2D mesh.
    bottom : (n_active_cells) np.ndarray
        Array with the bottom boundaries of each active cell in the 2D mesh.
    forward_func : callable
        Forward function that will be evaluated on each node of the mesh. Choose
        one of the forward functions in ``choclo.prism``.
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
    n_cells = cells_bounds.shape[0]
    # Evaluate kernel function on each node, for each receiver location
    for i in range(n_receivers):
        for j in range(n_cells):
            g_element = constant_factor * forward_func(
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
            diagonal[j] += weights[i] * g_element**2


@jit(nopython=True, parallel=True)
def _diagonal_G_T_dot_G_parallel(
    receivers,
    cells_bounds,
    top,
    bottom,
    forward_func,
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
    cells_bounds : (n_active_cells, 4) numpy.ndarray
        Array with the bounds of each active cell in the 2D mesh. For each row, the
        bounds should be passed in the following order: ``x_min``, ``x_max``,
        ``y_min``, ``y_max``.
    top : (n_active_cells) np.ndarray
        Array with the top boundaries of each active cell in the 2D mesh.
    bottom : (n_active_cells) np.ndarray
        Array with the bottom boundaries of each active cell in the 2D mesh.
    forward_func : callable
        Forward function that will be evaluated on each node of the mesh. Choose
        one of the forward functions in ``choclo.prism``.
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
    the diagonal elements of ``G.T @ G`` that correspond to a single receiver.
    After computing them, the ``local_diagonal`` array gets added to the running
    ``diagonal`` array through a reduction operation handled by Numba.
    """
    n_receivers = receivers.shape[0]
    n_cells = cells_bounds.shape[0]
    # Evaluate kernel function on each node, for each receiver location
    for i in prange(n_receivers):
        # Allocate array for the diagonal elements for the current receiver.
        local_diagonal = np.empty(n_cells)
        for j in range(n_cells):
            g_element = constant_factor * forward_func(
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
            local_diagonal[j] = weights[i] * g_element**2
        # Add the result to the diagonal.
        # Apply reduction operation to add the values of the local diagonal to
        # the running diagonal array. Avoid slicing the `diagonal` array when
        # updating it to avoid racing conditions, just add the `local_diagonal`
        # to the `diagonal` variable.
        diagonal += local_diagonal


# Define a dictionary with decorated versions of the Numba functions.
NUMBA_FUNCTIONS_2D = {
    "sensitivity": {
        parallel: jit(nopython=True, parallel=parallel)(_sensitivity_gravity)
        for parallel in (True, False)
    },
    "forward": {
        parallel: jit(nopython=True, parallel=parallel)(_forward_gravity)
        for parallel in (True, False)
    },
    "gt_dot_v": {
        False: _g_t_dot_v_serial,
        True: _g_t_dot_v_parallel,
    },
    "diagonal_gtg": {
        False: _diagonal_G_T_dot_G_serial,
        True: _diagonal_G_T_dot_G_parallel,
    },
}
