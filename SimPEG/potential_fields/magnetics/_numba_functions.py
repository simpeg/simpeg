"""
Numba functions for magnetic simulation of rectangular prisms
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


def _sensitivity_mag_scalar(
    receivers,
    nodes,
    sensitivity_matrix,
    cell_nodes,
    regional_field,
    kernel_x,
    kernel_y,
    kernel_z,
    constant_factor,
):
    """
    Fill the sensitivity matrix for single mag component and scalar data

    This function should be used with a `numba.jit` decorator, for example:

    ..code::

        from numba import jit

        jit_sensitivity = jit(nopython=True, parallel=True)(
            _sensitivity_matrix_scalar
        )
        jit_sensitivity(
            receivers, nodes, matrix, cell_nodes, regional_field, constant_factor
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
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    kernel_x, kernel_y, kernel_z : callable
        Kernels used to compute the desired magnetic component. For example,
        for computing bx we need to use ``kernel_x=kernel_ee``,
        ``kernel_y=kernel_en``, ``kernel_z=kernel_eu``.
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        sensitivity matrix.
    """
    n_receivers = receivers.shape[0]
    n_nodes = nodes.shape[0]
    n_cells = cell_nodes.shape[0]
    fx, fy, fz = regional_field
    regional_field_amplitude = np.sqrt(fx**2 + fy**2 + fz**2)
    fx /= regional_field_amplitude
    fy /= regional_field_amplitude
    fz /= regional_field_amplitude
    # Evaluate kernel function on each node, for each receiver location
    for i in prange(n_receivers):
        # Allocate vectors for kernels evaluated on mesh nodes
        kx, ky, kz = np.empty(n_nodes), np.empty(n_nodes), np.empty(n_nodes)
        # Allocate small vector for the nodes indices for a given cell
        nodes_indices = np.empty(8, dtype=cell_nodes.dtype)
        for j in range(n_nodes):
            dx = nodes[j, 0] - receivers[i, 0]
            dy = nodes[j, 1] - receivers[i, 1]
            dz = nodes[j, 2] - receivers[i, 2]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            kx[j] = kernel_x(dx, dy, dz, distance)
            ky[j] = kernel_y(dx, dy, dz, distance)
            kz[j] = kernel_z(dx, dy, dz, distance)
        # Compute sensitivity matrix elements from the kernel values
        for k in range(n_cells):
            nodes_indices = cell_nodes[k, :]
            ux = _kernels_in_nodes_to_cell(kx, nodes_indices)
            uy = _kernels_in_nodes_to_cell(ky, nodes_indices)
            uz = _kernels_in_nodes_to_cell(kz, nodes_indices)
            sensitivity_matrix[i, k] = (
                constant_factor
                * regional_field_amplitude
                * (ux * fx + uy * fy + uz * fz)
            )


def _sensitivity_tmi_scalar(
    receivers,
    nodes,
    sensitivity_matrix,
    cell_nodes,
    regional_field,
    constant_factor,
):
    """
    Fill the sensitivity matrix for TMI and scalar data (susceptibility only)

    This function should be used with a `numba.jit` decorator, for example:

    ..code::

        from numba import jit

        jit_sensitivity = jit(nopython=True, parallel=True)(
            _sensitivity_matrix_tmi_scalar
        )
        jit_sensitivity(
            receivers, nodes, matrix, cell_nodes, regional_field, constant_factor
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
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        sensitivity matrix.
    """
    n_receivers = receivers.shape[0]
    n_nodes = nodes.shape[0]
    n_cells = cell_nodes.shape[0]
    fx, fy, fz = regional_field
    regional_field_amplitude = np.sqrt(fx**2 + fy**2 + fz**2)
    fx /= regional_field_amplitude
    fy /= regional_field_amplitude
    fz /= regional_field_amplitude
    # Evaluate kernel function on each node, for each receiver location
    for i in prange(n_receivers):
        # Allocate vectors for kernels evaluated on mesh nodes
        kxx, kyy, kzz = np.empty(n_nodes), np.empty(n_nodes), np.empty(n_nodes)
        kxy, kxz, kyz = np.empty(n_nodes), np.empty(n_nodes), np.empty(n_nodes)
        # Allocate small vector for the nodes indices for a given cell
        nodes_indices = np.empty(8, dtype=cell_nodes.dtype)
        for j in range(n_nodes):
            dx = nodes[j, 0] - receivers[i, 0]
            dy = nodes[j, 1] - receivers[i, 1]
            dz = nodes[j, 2] - receivers[i, 2]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            kxx[j] = choclo.prism.kernel_ee(dx, dy, dz, distance)
            kyy[j] = choclo.prism.kernel_nn(dx, dy, dz, distance)
            kzz[j] = choclo.prism.kernel_uu(dx, dy, dz, distance)
            kxy[j] = choclo.prism.kernel_en(dx, dy, dz, distance)
            kxz[j] = choclo.prism.kernel_eu(dx, dy, dz, distance)
            kyz[j] = choclo.prism.kernel_nu(dx, dy, dz, distance)
        # Compute sensitivity matrix elements from the kernel values
        for k in range(n_cells):
            nodes_indices = cell_nodes[k, :]
            uxx = _kernels_in_nodes_to_cell(kxx, nodes_indices)
            uyy = _kernels_in_nodes_to_cell(kyy, nodes_indices)
            uzz = _kernels_in_nodes_to_cell(kzz, nodes_indices)
            uxy = _kernels_in_nodes_to_cell(kxy, nodes_indices)
            uxz = _kernels_in_nodes_to_cell(kxz, nodes_indices)
            uyz = _kernels_in_nodes_to_cell(kyz, nodes_indices)
            bx = uxx * fx + uxy * fy + uxz * fz
            by = uxy * fx + uyy * fy + uyz * fz
            bz = uxz * fx + uyz * fy + uzz * fz
            sensitivity_matrix[i, k] = (
                constant_factor
                * regional_field_amplitude
                * (bx * fx + by * fy + bz * fz)
            )


def _fill_sensitivity_tmi_vector(
    receivers,
    nodes,
    sensitivity_matrix,
    cell_nodes,
    regional_field,
    constant_factor,
):
    """
    Fill the sensitivity matrix for TMI and vector data (effective susceptibility)

    This function should be used with a `numba.jit` decorator, for example:

    ..code::

        from numba import jit

        jit_sensitivity = jit(nopython=True, parallel=True)(
            _fill_sensitivity_matrix_tmi_vector
        )
        jit_sensitivity(
            receivers, nodes, matrix, cell_nodes, regional_field, constant_factor
        )

    Parameters
    ----------
    receivers : (n_receivers, 3) array
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) array
        Array with the location of the mesh nodes.
    sensitivity_matrix : (n_receivers, 3 * n_active_nodes) array
        Empty 2d array where the sensitivity matrix elements will be filled.
        This could be a preallocated empty array or a slice of it.
        The number of columns is three times the number of active nodes since
        the vector model has ``3 * n_active_nodes`` elements: three components
        for each active cell.
    cell_nodes : (n_active_cells, 8) array
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        sensitivity matrix.
    """
    n_receivers = receivers.shape[0]
    n_nodes = nodes.shape[0]
    n_cells = cell_nodes.shape[0]
    fx, fy, fz = regional_field
    # Evaluate kernel function on each node, for each receiver location
    for i in prange(n_receivers):
        # Allocate vectors for kernels evaluated on mesh nodes
        kxx, kyy, kzz = np.empty(n_nodes), np.empty(n_nodes), np.empty(n_nodes)
        kxy, kxz, kyz = np.empty(n_nodes), np.empty(n_nodes), np.empty(n_nodes)
        # Allocate small vector for the nodes indices for a given cell
        nodes_indices = np.empty(8, dtype=cell_nodes.dtype)
        for j in range(n_nodes):
            dx = nodes[j, 0] - receivers[i, 0]
            dy = nodes[j, 1] - receivers[i, 1]
            dz = nodes[j, 2] - receivers[i, 2]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            kxx[j] = choclo.prism.kernel_ee(dx, dy, dz, distance)
            kyy[j] = choclo.prism.kernel_nn(dx, dy, dz, distance)
            kzz[j] = choclo.prism.kernel_uu(dx, dy, dz, distance)
            kxy[j] = choclo.prism.kernel_en(dx, dy, dz, distance)
            kxz[j] = choclo.prism.kernel_eu(dx, dy, dz, distance)
            kyz[j] = choclo.prism.kernel_nu(dx, dy, dz, distance)
        # Compute sensitivity matrix elements from the kernel values
        for k in range(n_cells):
            nodes_indices = cell_nodes[k, :]
            uxx = _kernels_in_nodes_to_cell(kxx, nodes_indices)
            uyy = _kernels_in_nodes_to_cell(kyy, nodes_indices)
            uzz = _kernels_in_nodes_to_cell(kzz, nodes_indices)
            uxy = _kernels_in_nodes_to_cell(kxy, nodes_indices)
            uxz = _kernels_in_nodes_to_cell(kxz, nodes_indices)
            uyz = _kernels_in_nodes_to_cell(kyz, nodes_indices)
            bx = uxx * fx + uxy * fy + uxz * fz
            by = uxy * fx + uyy * fy + uyz * fz
            bz = uxz * fx + uyz * fy + uzz * fz
            # Fill the sensitivity matrix elements that correspond to the
            # current active cell
            sensitivity_matrix[i, k] = constant_factor * bx
            sensitivity_matrix[i, k + n_cells] = constant_factor * by
            sensitivity_matrix[i, k + 2 * n_cells] = constant_factor * bz


def _forward_tmi_scalar(
    receivers,
    nodes,
    susceptibilities,
    fields,
    cell_nodes,
    regional_field,
    constant_factor,
):
    """
    Forward model the TMI with scalar data (susceptibility only)

    This function should be used with a `numba.jit` decorator, for example:

    ..code::

        from numba import jit

        jit_forward = jit(nopython=True, parallel=True)(_forward_tmi_scalar)
        jit_forward(
            receivers, nodes, mag_sus, fields, cell_nodes, regional_field, const_factor
        )

    Parameters
    ----------
    receivers : (n_receivers, 3) array
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) array
        Array with the location of the mesh nodes.
    susceptibilities : (n_active_cells)
        Array with the susceptibility of each active cell in the mesh.
    fields : (n_receivers) array
        Array full of zeros where the TMI on each receiver will be stored. This
        could be a preallocated array or a slice of it.
    cell_nodes : (n_active_cells, 8) array
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        sensitivity matrix.
    """
    n_receivers = receivers.shape[0]
    n_nodes = nodes.shape[0]
    n_cells = cell_nodes.shape[0]
    fx, fy, fz = regional_field
    regional_field_amplitude = np.sqrt(fx**2 + fy**2 + fz**2)
    fx /= regional_field_amplitude
    fy /= regional_field_amplitude
    fz /= regional_field_amplitude
    # Evaluate kernel function on each node, for each receiver location
    for i in prange(n_receivers):
        # Allocate vectors for kernels evaluated on mesh nodes
        kxx, kyy, kzz = np.empty(n_nodes), np.empty(n_nodes), np.empty(n_nodes)
        kxy, kxz, kyz = np.empty(n_nodes), np.empty(n_nodes), np.empty(n_nodes)
        # Allocate small vector for the nodes indices for a given cell
        nodes_indices = np.empty(8, dtype=cell_nodes.dtype)
        for j in range(n_nodes):
            dx = nodes[j, 0] - receivers[i, 0]
            dy = nodes[j, 1] - receivers[i, 1]
            dz = nodes[j, 2] - receivers[i, 2]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            kxx[j] = choclo.prism.kernel_ee(dx, dy, dz, distance)
            kyy[j] = choclo.prism.kernel_nn(dx, dy, dz, distance)
            kzz[j] = choclo.prism.kernel_uu(dx, dy, dz, distance)
            kxy[j] = choclo.prism.kernel_en(dx, dy, dz, distance)
            kxz[j] = choclo.prism.kernel_eu(dx, dy, dz, distance)
            kyz[j] = choclo.prism.kernel_nu(dx, dy, dz, distance)
        # Compute sensitivity matrix elements from the kernel values
        for k in range(n_cells):
            nodes_indices = cell_nodes[k, :]
            uxx = _kernels_in_nodes_to_cell(kxx, nodes_indices)
            uyy = _kernels_in_nodes_to_cell(kyy, nodes_indices)
            uzz = _kernels_in_nodes_to_cell(kzz, nodes_indices)
            uxy = _kernels_in_nodes_to_cell(kxy, nodes_indices)
            uxz = _kernels_in_nodes_to_cell(kxz, nodes_indices)
            uyz = _kernels_in_nodes_to_cell(kyz, nodes_indices)
            bx = uxx * fx + uxy * fy + uxz * fz
            by = uxy * fx + uyy * fy + uyz * fz
            bz = uxz * fx + uyz * fy + uzz * fz
            fields[i] += (
                constant_factor
                * susceptibilities[k]
                * regional_field_amplitude
                * (bx * fx + by * fy + bz * fz)
            )


def _forward_tmi_vector(
    receivers,
    nodes,
    effective_susceptibilities,
    fields,
    cell_nodes,
    regional_field,
    constant_factor,
):
    """
    Forward model the TMI with vector data (effective susceptibility)

    This function should be used with a `numba.jit` decorator, for example:

    ..code::

        from numba import jit

        jit_forward = jit(nopython=True, parallel=True)(_forward_tmi_vector)
        jit_forward(
            receivers, nodes, effective_sus, fields, cell_nodes, regional_field, const_factor
        )

    Parameters
    ----------
    receivers : (n_receivers, 3) array
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) array
        Array with the location of the mesh nodes.
    effective_susceptibilities : (3 * n_active_cells)
        Array with the effective susceptibility vector components of each
        active cell in the mesh.
        The order of the components should be the following: all x components
        for every cell, then all y components for every cell and then all
        z components for every cell.
    fields : (n_receivers) array
        Array full of zeros where the TMI on each receiver will be stored. This
        could be a preallocated array or a slice of it.
    cell_nodes : (n_active_cells, 8) array
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        sensitivity matrix.
    """
    n_receivers = receivers.shape[0]
    n_nodes = nodes.shape[0]
    n_cells = cell_nodes.shape[0]
    fx, fy, fz = regional_field
    # Evaluate kernel function on each node, for each receiver location
    for i in prange(n_receivers):
        # Allocate vectors for kernels evaluated on mesh nodes
        kxx, kyy, kzz = np.empty(n_nodes), np.empty(n_nodes), np.empty(n_nodes)
        kxy, kxz, kyz = np.empty(n_nodes), np.empty(n_nodes), np.empty(n_nodes)
        # Allocate small vector for the nodes indices for a given cell
        nodes_indices = np.empty(8, dtype=cell_nodes.dtype)
        for j in range(n_nodes):
            dx = nodes[j, 0] - receivers[i, 0]
            dy = nodes[j, 1] - receivers[i, 1]
            dz = nodes[j, 2] - receivers[i, 2]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            kxx[j] = choclo.prism.kernel_ee(dx, dy, dz, distance)
            kyy[j] = choclo.prism.kernel_nn(dx, dy, dz, distance)
            kzz[j] = choclo.prism.kernel_uu(dx, dy, dz, distance)
            kxy[j] = choclo.prism.kernel_en(dx, dy, dz, distance)
            kxz[j] = choclo.prism.kernel_eu(dx, dy, dz, distance)
            kyz[j] = choclo.prism.kernel_nu(dx, dy, dz, distance)
        # Compute sensitivity matrix elements from the kernel values
        for k in range(n_cells):
            nodes_indices = cell_nodes[k, :]
            uxx = _kernels_in_nodes_to_cell(kxx, nodes_indices)
            uyy = _kernels_in_nodes_to_cell(kyy, nodes_indices)
            uzz = _kernels_in_nodes_to_cell(kzz, nodes_indices)
            uxy = _kernels_in_nodes_to_cell(kxy, nodes_indices)
            uxz = _kernels_in_nodes_to_cell(kxz, nodes_indices)
            uyz = _kernels_in_nodes_to_cell(kyz, nodes_indices)
            bx = uxx * fx + uxy * fy + uxz * fz
            by = uxy * fx + uyy * fy + uyz * fz
            bz = uxz * fx + uyz * fy + uzz * fz
            fields[i] += constant_factor * (
                bx * effective_susceptibilities[k]
                + by * effective_susceptibilities[k + n_cells]
                + bz * effective_susceptibilities[k + 2 * n_cells]
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


_sensitivity_tmi_scalar_serial = jit(nopython=True, parallel=False)(
    _sensitivity_tmi_scalar
)
_sensitivity_tmi_scalar_parallel = jit(nopython=True, parallel=True)(
    _sensitivity_tmi_scalar
)
_sensitivity_mag_scalar_serial = jit(nopython=True, parallel=False)(
    _sensitivity_mag_scalar
)
_sensitivity_mag_scalar_parallel = jit(nopython=True, parallel=True)(
    _sensitivity_mag_scalar
)
_sensitivity_tmi_vector_serial = jit(nopython=True, parallel=False)(
    _fill_sensitivity_tmi_vector
)
_sensitivity_tmi_vector_parallel = jit(nopython=True, parallel=True)(
    _fill_sensitivity_tmi_vector
)
_forward_tmi_scalar_serial = jit(nopython=True, parallel=False)(_forward_tmi_scalar)
_forward_tmi_scalar_parallel = jit(nopython=True, parallel=True)(_forward_tmi_scalar)
_forward_tmi_vector_serial = jit(nopython=True, parallel=False)(_forward_tmi_vector)
_forward_tmi_vector_parallel = jit(nopython=True, parallel=True)(_forward_tmi_vector)
