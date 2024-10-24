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

from .._numba_utils import kernels_in_nodes_to_cell


def _sensitivity_mag(
    receivers,
    nodes,
    sensitivity_matrix,
    cell_nodes,
    regional_field,
    kernel_x,
    kernel_y,
    kernel_z,
    constant_factor,
    scalar_model,
):
    r"""
    Fill the sensitivity matrix for single mag component

    This function should be used with a `numba.jit` decorator, for example:

    .. code::

        from numba import jit

        jit_sensitivity_mag = jit(nopython=True, parallel=True)(_sensitivity_mag)

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
    scalar_model : bool
        If True, the sensitivity matrix is built to work with scalar models
        (susceptibilities).
        If False, the sensitivity matrix is built to work with vector models
        (effective susceptibilities).

    Notes
    -----

    About the kernel functions
    ^^^^^^^^^^^^^^^^^^^^^^^^^^

    For computing the ``bx`` component of the magnetic field we need to use the
    following kernels:

    .. code::

        kernel_x=kernel_ee, kernel_y=kernel_en, kernel_z=kernel_eu


    For computing the ``by`` component of the magnetic field we need to use the
    following kernels:

    .. code::

        kernel_x=kernel_en, kernel_y=kernel_nn, kernel_z=kernel_nu

    For computing the ``bz`` component of the magnetic field we need to use the
    following kernels:

    .. code::

        kernel_x=kernel_eu, kernel_y=kernel_nu, kernel_z=kernel_uu

    About the model array
    ^^^^^^^^^^^^^^^^^^^^^

    The ``model`` must always be a 1d array:

    * If ``scalar_model`` is ``True``, then ``model`` should be a 1d array with
      the same number of elements as active cells in the mesh. It should store
      the magnetic susceptibilities of each active cell in SI units.
    * If ``scalar_model`` is ``False``, then ``model`` should be a 1d array
      with a number of elements equal to three times the active cells in the
      mesh. It should store the components of the magnetization vector of each
      active cell in :math:`Am^{-1}`. The order in which the components should
      be passed are:
          * every _easting_ component of each active cell,
          * then every _northing_ component of each active cell,
          * and finally every _upward_ component of each active cell.

    About the sensitivity matrix
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Each row of the sensitivity matrix corresponds to a single receiver
    location.

    If ``scalar_model`` is True, then each element of the row will
    correspond to the partial derivative of the selected magnetic component
    with respect to the susceptibility of each cell in the mesh.

    If ``scalar_model`` is False, then each row can be split in three sections
    containing:

    * the partial derivatives of the selected magnetic component with respect
      to the _x_ component of the effective susceptibility of each cell; then
    * the partial derivatives of the selected magnetic component with respect
      to the _y_ component of the effective susceptibility of each cell; and then
    * the partial derivatives of the selected magnetic component with respect
      to the _z_ component of the effective susceptibility of each cell.

    So, if we call :math:`B_j` the magnetic field component on the receiver
    :math:`j`, and :math:`\bar{\chi}^{(i)} = (\chi_x^{(i)}, \chi_y^{(i)},
    \chi_z^{(i)})` the effective susceptibility of the active cell :math:`i`,
    then each row of the sensitivity matrix will be:

    .. math::

        \left[
            \frac{\partial B_j}{\partial \chi_x^{(1)}},
            \dots,
            \frac{\partial B_j}{\partial \chi_x^{(N)}},
            \frac{\partial B_j}{\partial \chi_y^{(1)}},
            \dots,
            \frac{\partial B_j}{\partial \chi_y^{(N)}},
            \frac{\partial B_j}{\partial \chi_z^{(1)}},
            \dots,
            \frac{\partial B_j}{\partial \chi_z^{(N)}}
        \right]

    where :math:`N` is the total number of active cells.
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
            ux = kernels_in_nodes_to_cell(kx, nodes_indices)
            uy = kernels_in_nodes_to_cell(ky, nodes_indices)
            uz = kernels_in_nodes_to_cell(kz, nodes_indices)
            if scalar_model:
                sensitivity_matrix[i, k] = (
                    constant_factor
                    * regional_field_amplitude
                    * (ux * fx + uy * fy + uz * fz)
                )
            else:
                sensitivity_matrix[i, k] = (
                    constant_factor * regional_field_amplitude * ux
                )
                sensitivity_matrix[i, k + n_cells] = (
                    constant_factor * regional_field_amplitude * uy
                )
                sensitivity_matrix[i, k + 2 * n_cells] = (
                    constant_factor * regional_field_amplitude * uz
                )


def _sensitivity_tmi(
    receivers,
    nodes,
    sensitivity_matrix,
    cell_nodes,
    regional_field,
    constant_factor,
    scalar_model,
):
    r"""
    Fill the sensitivity matrix for TMI

    This function should be used with a `numba.jit` decorator, for example:

    .. code::

        from numba import jit

        jit_sensitivity_tmi = jit(nopython=True, parallel=True)(_sensitivity_tmi)

    Parameters
    ----------
    receivers : (n_receivers, 3) array
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) array
        Array with the location of the mesh nodes.
    sensitivity_matrix : array
        Empty 2d array where the sensitivity matrix elements will be filled.
        This could be a preallocated empty array or a slice of it.
        The array should have a shape of ``(n_receivers, n_active_nodes)``
        if ``scalar_model`` is True.
        The array should have a shape of ``(n_receivers, 3 * n_active_nodes)``
        if ``scalar_model`` is False.
    cell_nodes : (n_active_cells, 8) array
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        sensitivity matrix.
    scalar_model : bool
        If True, the sensitivity matrix is build to work with scalar models
        (susceptibilities).
        If False, the sensitivity matrix is build to work with vector models
        (effective susceptibilities).

    Notes
    -----

    About the model array
    ^^^^^^^^^^^^^^^^^^^^^

    The ``model`` must always be a 1d array:

    * If ``scalar_model`` is ``True``, then ``model`` should be a 1d array with
      the same number of elements as active cells in the mesh. It should store
      the magnetic susceptibilities of each active cell in SI units.
    * If ``scalar_model`` is ``False``, then ``model`` should be a 1d array
      with a number of elements equal to three times the active cells in the
      mesh. It should store the components of the magnetization vector of each
      active cell in :math:`Am^{-1}`. The order in which the components should
      be passed are:
          * every _easting_ component of each active cell,
          * then every _northing_ component of each active cell,
          * and finally every _upward_ component of each active cell.

    About the sensitivity matrix
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Each row of the sensitivity matrix corresponds to a single receiver
    location.

    If ``scalar_model`` is True, then each element of the row will
    correspond to the partial derivative of the tmi
    with respect to the susceptibility of each cell in the mesh.

    If ``scalar_model`` is False, then each row can be split in three sections
    containing:

    * the partial derivatives of the tmi with respect
      to the _x_ component of the effective susceptibility of each cell; then
    * the partial derivatives of the tmi with respect
      to the _y_ component of the effective susceptibility of each cell; and then
    * the partial derivatives of the tmi with respect
      to the _z_ component of the effective susceptibility of each cell.

    So, if we call :math:`T_j` the tmi on the receiver
    :math:`j`, and :math:`\bar{\chi}^{(i)} = (\chi_x^{(i)}, \chi_y^{(i)},
    \chi_z^{(i)})` the effective susceptibility of the active cell :math:`i`,
    then each row of the sensitivity matrix will be:

    .. math::

        \left[
            \frac{\partial T_j}{\partial \chi_x^{(1)}},
            \dots,
            \frac{\partial T_j}{\partial \chi_x^{(N)}},
            \frac{\partial T_j}{\partial \chi_y^{(1)}},
            \dots,
            \frac{\partial T_j}{\partial \chi_y^{(N)}},
            \frac{\partial T_j}{\partial \chi_z^{(1)}},
            \dots,
            \frac{\partial T_j}{\partial \chi_z^{(N)}}
        \right]

    where :math:`N` is the total number of active cells.
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
            uxx = kernels_in_nodes_to_cell(kxx, nodes_indices)
            uyy = kernels_in_nodes_to_cell(kyy, nodes_indices)
            uzz = kernels_in_nodes_to_cell(kzz, nodes_indices)
            uxy = kernels_in_nodes_to_cell(kxy, nodes_indices)
            uxz = kernels_in_nodes_to_cell(kxz, nodes_indices)
            uyz = kernels_in_nodes_to_cell(kyz, nodes_indices)
            bx = uxx * fx + uxy * fy + uxz * fz
            by = uxy * fx + uyy * fy + uyz * fz
            bz = uxz * fx + uyz * fy + uzz * fz
            # Fill the sensitivity matrix element(s) that correspond to the
            # current active cell
            if scalar_model:
                sensitivity_matrix[i, k] = (
                    constant_factor
                    * regional_field_amplitude
                    * (bx * fx + by * fy + bz * fz)
                )
            else:
                sensitivity_matrix[i, k] = (
                    constant_factor * regional_field_amplitude * bx
                )
                sensitivity_matrix[i, k + n_cells] = (
                    constant_factor * regional_field_amplitude * by
                )
                sensitivity_matrix[i, k + 2 * n_cells] = (
                    constant_factor * regional_field_amplitude * bz
                )


def _forward_mag(
    receivers,
    nodes,
    model,
    fields,
    cell_nodes,
    regional_field,
    kernel_x,
    kernel_y,
    kernel_z,
    constant_factor,
    scalar_model,
):
    """
    Forward model single magnetic component

    This function should be used with a `numba.jit` decorator, for example:

    .. code::

        from numba import jit

        jit_forward = jit(nopython=True, parallel=True)(_forward_mag)

    Parameters
    ----------
    receivers : (n_receivers, 3) array
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) array
        Array with the location of the mesh nodes.
    model : (n_active_cells) or (3 * n_active_cells) array
        Array containing the susceptibilities (scalar) or effective
        susceptibilities (vector) of the active cells in the mesh, in SI
        units.
        Susceptibilities are expected if ``scalar_model`` is True,
        and the array should have ``n_active_cells`` elements.
        Effective susceptibilities are expected if ``scalar_model`` is False,
        and the array should have ``3 * n_active_cells`` elements.
    fields : (n_receivers) array
        Array full of zeros where the magnetic component on each receiver will
        be stored. This could be a preallocated array or a slice of it.
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
    scalar_model : bool
        If True, the forward will be computing assuming that the ``model`` has
        susceptibilities (scalar model) for each active cell.
        If False, the forward will be computing assuming that the ``model`` has
        effective susceptibilities (vector model) for each active cell.

    Notes
    -----

    About the kernel functions
    ^^^^^^^^^^^^^^^^^^^^^^^^^^

    For computing the ``bx`` component of the magnetic field we need to use the
    following kernels:

    .. code::

        kernel_x=kernel_ee, kernel_y=kernel_en, kernel_z=kernel_eu


    For computing the ``by`` component of the magnetic field we need to use the
    following kernels:

    .. code::

        kernel_x=kernel_en, kernel_y=kernel_nn, kernel_z=kernel_nu

    For computing the ``bz`` component of the magnetic field we need to use the
    following kernels:

    .. code::

        kernel_x=kernel_eu, kernel_y=kernel_nu, kernel_z=kernel_uu


    About the model array
    ^^^^^^^^^^^^^^^^^^^^^

    The ``model`` must always be a 1d array:

    * If ``scalar_model`` is ``True``, then ``model`` should be a 1d array with
      the same number of elements as active cells in the mesh. It should store
      the magnetic susceptibilities of each active cell in SI units.
    * If ``scalar_model`` is ``False``, then ``model`` should be a 1d array
      with a number of elements equal to three times the active cells in the
      mesh. It should store the components of the magnetization vector of each
      active cell in :math:`Am^{-1}`. The order in which the components should
      be passed are:
          * every _easting_ component of each active cell,
          * then every _northing_ component of each active cell,
          * and finally every _upward_ component of each active cell.

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
            ux = kernels_in_nodes_to_cell(kx, nodes_indices)
            uy = kernels_in_nodes_to_cell(ky, nodes_indices)
            uz = kernels_in_nodes_to_cell(kz, nodes_indices)
            if scalar_model:
                fields[i] += (
                    constant_factor
                    * model[k]
                    * regional_field_amplitude
                    * (ux * fx + uy * fy + uz * fz)
                )
            else:
                fields[i] += (
                    constant_factor
                    * regional_field_amplitude
                    * (
                        ux * model[k]
                        + uy * model[k + n_cells]
                        + uz * model[k + 2 * n_cells]
                    )
                )


def _forward_tmi(
    receivers,
    nodes,
    model,
    fields,
    cell_nodes,
    regional_field,
    constant_factor,
    scalar_model,
):
    """
    Forward model the TMI

    This function should be used with a `numba.jit` decorator, for example:

    .. code::

        from numba import jit

        jit_forward = jit(nopython=True, parallel=True)(_forward_tmi)

    Parameters
    ----------
    receivers : (n_receivers, 3) array
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) array
        Array with the location of the mesh nodes.
    model : (n_active_cells) or (3 * n_active_cells)
        Array with the susceptibility (scalar model) or the effective
        susceptibility (vector model) of each active cell in the mesh.
        If the model is scalar, the ``model`` array should have
        ``n_active_cells`` elements and ``scalar_model`` should be True.
        If the model is vector, the ``model`` array should have
        ``3 * n_active_cells`` elements and ``scalar_model`` should be False.
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
    scalar_model : bool
        If True, the sensitivity matrix is build to work with scalar models
        (susceptibilities).
        If False, the sensitivity matrix is build to work with vector models
        (effective susceptibilities).

    Notes
    -----

    The ``model`` must always be a 1d array:

    * If ``scalar_model`` is ``True``, then ``model`` should be a 1d array with
      the same number of elements as active cells in the mesh. It should store
      the magnetic susceptibilities of each active cell in SI units.
    * If ``scalar_model`` is ``False``, then ``model`` should be a 1d array
      with a number of elements equal to three times the active cells in the
      mesh. It should store the components of the magnetization vector of each
      active cell in :math:`Am^{-1}`. The order in which the components should
      be passed are:
          * every _easting_ component of each active cell,
          * then every _northing_ component of each active cell,
          * and finally every _upward_ component of each active cell.

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
            uxx = kernels_in_nodes_to_cell(kxx, nodes_indices)
            uyy = kernels_in_nodes_to_cell(kyy, nodes_indices)
            uzz = kernels_in_nodes_to_cell(kzz, nodes_indices)
            uxy = kernels_in_nodes_to_cell(kxy, nodes_indices)
            uxz = kernels_in_nodes_to_cell(kxz, nodes_indices)
            uyz = kernels_in_nodes_to_cell(kyz, nodes_indices)
            bx = uxx * fx + uxy * fy + uxz * fz
            by = uxy * fx + uyy * fy + uyz * fz
            bz = uxz * fx + uyz * fy + uzz * fz
            if scalar_model:
                fields[i] += (
                    constant_factor
                    * model[k]
                    * regional_field_amplitude
                    * (bx * fx + by * fy + bz * fz)
                )
            else:
                fields[i] += (
                    constant_factor
                    * regional_field_amplitude
                    * (
                        bx * model[k]
                        + by * model[k + n_cells]
                        + bz * model[k + 2 * n_cells]
                    )
                )


_sensitivity_tmi_serial = jit(nopython=True, parallel=False)(_sensitivity_tmi)
_sensitivity_tmi_parallel = jit(nopython=True, parallel=True)(_sensitivity_tmi)
_forward_tmi_serial = jit(nopython=True, parallel=False)(_forward_tmi)
_forward_tmi_parallel = jit(nopython=True, parallel=True)(_forward_tmi)
_forward_mag_serial = jit(nopython=True, parallel=False)(_forward_mag)
_forward_mag_parallel = jit(nopython=True, parallel=True)(_forward_mag)
_sensitivity_mag_serial = jit(nopython=True, parallel=False)(_sensitivity_mag)
_sensitivity_mag_parallel = jit(nopython=True, parallel=True)(_sensitivity_mag)
