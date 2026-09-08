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

from ..._numba_utils import kernels_in_nodes_to_cell


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
    sensitivity_matrix : array
        Empty 2d array where the sensitivity matrix elements will be filled.
        This could be a preallocated empty array or a slice of it.
        The array should have a shape of ``(n_receivers, n_active_cells)``
        if ``scalar_model`` is True.
        The array should have a shape of ``(n_receivers, 3 * n_active_cells)``
        if ``scalar_model`` is False.
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
        The array should have a shape of ``(n_receivers, n_active_cells)``
        if ``scalar_model`` is True.
        The array should have a shape of ``(n_receivers, 3 * n_active_cells)``
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


def _sensitivity_tmi_derivative(
    receivers,
    nodes,
    sensitivity_matrix,
    cell_nodes,
    regional_field,
    kernel_xx,
    kernel_yy,
    kernel_zz,
    kernel_xy,
    kernel_xz,
    kernel_yz,
    constant_factor,
    scalar_model,
):
    r"""
    Fill the sensitivity matrix for a TMI derivative.

    This function should be used with a `numba.jit` decorator, for example:

    .. code::

        from numba import jit

        jit_sens = jit(nopython=True, parallel=True)(_sensitivity_tmi_derivative)

    Parameters
    ----------
    receivers : (n_receivers, 3) array
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) array
        Array with the location of the mesh nodes.
    sensitivity_matrix : array
        Empty 2d array where the sensitivity matrix elements will be filled.
        This could be a preallocated empty array or a slice of it.
        The array should have a shape of ``(n_receivers, n_active_cells)``
        if ``scalar_model`` is True.
        The array should have a shape of ``(n_receivers, 3 * n_active_cells)``
        if ``scalar_model`` is False.
    cell_nodes : (n_active_cells, 8) array
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz : callables
        Kernel functions used for computing the desired TMI derivative.
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

    About the kernel functions
    ^^^^^^^^^^^^^^^^^^^^^^^^^^

    To compute the :math:`\alpha` derivative of the TMI :math:`\Delta T` (with
    :math:`\alpha \in \{x, y, z\}` we need to evaluate third order kernels
    functions for the prism. The kernels we need to evaluate can be obtained by
    fixing one of the subindices to the direction of the derivative
    (:math:`\alpha`) and cycle through combinations of the other two.

    For ``tmi_x`` we need to pass:

    .. code::

        kernel_xx=kernel_eee, kernel_yy=kernel_enn, kernel_zz=kernel_euu,
        kernel_xy=kernel_een, kernel_xz=kernel_eeu, kernel_yz=kernel_enu

    For ``tmi_y`` we need to pass:

    .. code::

        kernel_xx=kernel_een, kernel_yy=kernel_nnn, kernel_zz=kernel_nuu,
        kernel_xy=kernel_enn, kernel_xz=kernel_enu, kernel_yz=kernel_nnu

    For ``tmi_z`` we need to pass:

    .. code::

        kernel_xx=kernel_eeu, kernel_yy=kernel_nnu, kernel_zz=kernel_uuu,
        kernel_xy=kernel_enu, kernel_xz=kernel_euu, kernel_yz=kernel_nuu


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
    correspond to the partial derivative of the tmi derivative (spatial)
    with respect to the susceptibility of each cell in the mesh.

    If ``scalar_model`` is False, then each row can be split in three sections
    containing:

    * the partial derivatives of the tmi derivative with respect
      to the _x_ component of the effective susceptibility of each cell; then
    * the partial derivatives of the tmi derivative with respect
      to the _y_ component of the effective susceptibility of each cell; and then
    * the partial derivatives of the tmi derivative with respect
      to the _z_ component of the effective susceptibility of each cell.
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
            kxx[j] = kernel_xx(dx, dy, dz, distance)
            kyy[j] = kernel_yy(dx, dy, dz, distance)
            kzz[j] = kernel_zz(dx, dy, dz, distance)
            kxy[j] = kernel_xy(dx, dy, dz, distance)
            kxz[j] = kernel_xz(dx, dy, dz, distance)
            kyz[j] = kernel_yz(dx, dy, dz, distance)
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


@jit(nopython=True, parallel=False)
def _mag_sensitivity_t_dot_v_serial(
    receivers,
    nodes,
    cell_nodes,
    regional_field,
    kernel_x,
    kernel_y,
    kernel_z,
    constant_factor,
    scalar_model,
    vector,
    result,
):
    r"""
    Compute ``G.T @ v`` in serial, without building G, for a single magnetic component.

    Parameters
    ----------
    receivers : (n_receivers, 3) array
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) array
        Array with the location of the mesh nodes.
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
        If True, the result will be computed assuming that the ``model`` has
        susceptibilities (scalar model) for each active cell.
        If False, the result will be computed assuming that the ``model`` has
        effective susceptibilities (vector model) for each active cell.
    vector : (n_receivers) numpy.ndarray
        Array that represents the vector used in the dot product.
    result : (n_active_cells) or (3 * n_active_cells) numpy.ndarray
        Running result array where the output of the dot product will be added to.
        The array should have ``n_active_cells`` elements if ``scalar_model``
        is True, or ``3 * n_active_cells`` otherwise.

    Notes
    -----
    This function is meant to be run in serial. Writing to the ``result`` array
    inside a parallel loop over the receivers generates a race condition that
    leads to corrupted outputs.

    A parallel implementation of this function is available in
    ``_mag_sensitivity_t_dot_v_parallel``.

    See also
    --------
    _sensitivity_mag
        Compute the sensitivity matrix for a single magnetic component by
        allocating it in memory.
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
                result[k] += (
                    constant_factor
                    * vector[i]
                    * regional_field_amplitude
                    * (ux * fx + uy * fy + uz * fz)
                )
            else:
                result[k] += constant_factor * vector[i] * regional_field_amplitude * ux
                result[k + n_cells] += (
                    constant_factor * vector[i] * regional_field_amplitude * uy
                )
                result[k + 2 * n_cells] += (
                    constant_factor * vector[i] * regional_field_amplitude * uz
                )


@jit(nopython=True, parallel=True)
def _mag_sensitivity_t_dot_v_parallel(
    receivers,
    nodes,
    cell_nodes,
    regional_field,
    kernel_x,
    kernel_y,
    kernel_z,
    constant_factor,
    scalar_model,
    vector,
    result,
):
    r"""
    Compute ``G.T @ v`` in parallel, without building G, for a single magnetic component

    Parameters
    ----------
    receivers : (n_receivers, 3) array
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) array
        Array with the location of the mesh nodes.
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
        If True, the result will be computed assuming that the ``model`` has
        susceptibilities (scalar model) for each active cell.
        If False, the result will be computed assuming that the ``model`` has
        effective susceptibilities (vector model) for each active cell.
    vector : (n_receivers) numpy.ndarray
        Array that represents the vector used in the dot product.
    result : (n_active_cells) or (3 * n_active_cells) numpy.ndarray
        Running result array where the output of the dot product will be added to.
        The array should have ``n_active_cells`` elements if ``scalar_model``
        is True, or ``3 * n_active_cells`` otherwise.

    Notes
    -----
    This function is meant to be run in parallel.
    This implementation instructs each thread to allocate their own array for
    the current row of the sensitivity matrix. After computing the elements of
    that row, it gets added to the running ``result`` array through a reduction
    operation handled by Numba.

    A serialized implementation of this function is available in
    ``_mag_sensitivity_t_dot_v_serial``.

    See also
    --------
    _sensitivity_mag
        Compute the sensitivity matrix for a single magnetic component by
        allocating it in memory.
    """
    n_receivers = receivers.shape[0]
    n_nodes = nodes.shape[0]
    n_cells = cell_nodes.shape[0]
    fx, fy, fz = regional_field
    regional_field_amplitude = np.sqrt(fx**2 + fy**2 + fz**2)
    fx /= regional_field_amplitude
    fy /= regional_field_amplitude
    fz /= regional_field_amplitude
    result_size = result.size
    # Evaluate kernel function on each node, for each receiver location
    for i in prange(n_receivers):
        # Allocate vectors for kernels evaluated on mesh nodes
        kx, ky, kz = np.empty(n_nodes), np.empty(n_nodes), np.empty(n_nodes)
        # Allocate array for the current row of the sensitivity matrix
        local_row = np.empty(result_size)
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
                local_row[k] = (
                    constant_factor
                    * vector[i]
                    * regional_field_amplitude
                    * (ux * fx + uy * fy + uz * fz)
                )
            else:
                local_row[k] = (
                    constant_factor * vector[i] * regional_field_amplitude * ux
                )
                local_row[k + n_cells] = (
                    constant_factor * vector[i] * regional_field_amplitude * uy
                )
                local_row[k + 2 * n_cells] = (
                    constant_factor * vector[i] * regional_field_amplitude * uz
                )
        # Apply reduction operation to add the values of the row to the running
        # result. Avoid slicing the `result` array when updating it to avoid
        # racing conditions, just add the `local_row` to the `results`
        # variable.
        result += local_row


@jit(nopython=True, parallel=False)
def _tmi_sensitivity_t_dot_v_serial(
    receivers,
    nodes,
    cell_nodes,
    regional_field,
    constant_factor,
    scalar_model,
    vector,
    result,
):
    r"""
    Compute ``G.T @ v`` in serial, without building G, for TMI.

    Parameters
    ----------
    receivers : (n_receivers, 3) array
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) array
        Array with the location of the mesh nodes.
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
        If True, the result will be computed assuming that the ``model`` has
        susceptibilities (scalar model) for each active cell.
        If False, the result will be computed assuming that the ``model`` has
        effective susceptibilities (vector model) for each active cell.
    vector : (n_receivers) numpy.ndarray
        Array that represents the vector used in the dot product.
    result : (n_active_cells) or (3 * n_active_cells) numpy.ndarray
        Running result array where the output of the dot product will be added to.
        The array should have ``n_active_cells`` elements if ``scalar_model``
        is True, or ``3 * n_active_cells`` otherwise.

    Notes
    -----
    This function is meant to be run in serial. Writing to the ``result`` array
    inside a parallel loop over the receivers generates a race condition that
    leads to corrupted outputs.

    A parallel implementation of this function is available in
    ``_tmi_sensitivity_t_dot_v_parallel``.

    See also
    --------
    _sensitivity_tmi
        Compute the sensitivity matrix for TMI by allocating it in memory.
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
                result[k] += (
                    constant_factor
                    * vector[i]
                    * regional_field_amplitude
                    * (bx * fx + by * fy + bz * fz)
                )
            else:
                result[k] += constant_factor * vector[i] * regional_field_amplitude * bx
                result[k + n_cells] += (
                    constant_factor * vector[i] * regional_field_amplitude * by
                )
                result[k + 2 * n_cells] += (
                    constant_factor * vector[i] * regional_field_amplitude * bz
                )


@jit(nopython=True, parallel=True)
def _tmi_sensitivity_t_dot_v_parallel(
    receivers,
    nodes,
    cell_nodes,
    regional_field,
    constant_factor,
    scalar_model,
    vector,
    result,
):
    r"""
    Compute ``G.T @ v`` in parallel, without building G, for TMI.

    Parameters
    ----------
    receivers : (n_receivers, 3) array
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) array
        Array with the location of the mesh nodes.
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
        If True, the result will be computed assuming that the ``model`` has
        susceptibilities (scalar model) for each active cell.
        If False, the result will be computed assuming that the ``model`` has
        effective susceptibilities (vector model) for each active cell.
    vector : (n_receivers) numpy.ndarray
        Array that represents the vector used in the dot product.
    result : (n_active_cells) or (3 * n_active_cells) numpy.ndarray
        Running result array where the output of the dot product will be added to.
        The array should have ``n_active_cells`` elements if ``scalar_model``
        is True, or ``3 * n_active_cells`` otherwise.

    Notes
    -----
    This function is meant to be run in parallel.
    This implementation instructs each thread to allocate their own array for
    the current row of the sensitivity matrix. After computing the elements of
    that row, it gets added to the running ``result`` array through a reduction
    operation handled by Numba.

    A serialized implementation of this function is available in
    ``_tmi_sensitivity_t_dot_v_serial``.

    See also
    --------
    _sensitivity_tmi
        Compute the sensitivity matrix for TMI by allocating it in memory.
    """
    n_receivers = receivers.shape[0]
    n_nodes = nodes.shape[0]
    n_cells = cell_nodes.shape[0]
    fx, fy, fz = regional_field
    regional_field_amplitude = np.sqrt(fx**2 + fy**2 + fz**2)
    fx /= regional_field_amplitude
    fy /= regional_field_amplitude
    fz /= regional_field_amplitude
    result_size = result.size
    # Evaluate kernel function on each node, for each receiver location
    for i in prange(n_receivers):
        # Allocate vectors for kernels evaluated on mesh nodes
        kxx, kyy, kzz = np.empty(n_nodes), np.empty(n_nodes), np.empty(n_nodes)
        kxy, kxz, kyz = np.empty(n_nodes), np.empty(n_nodes), np.empty(n_nodes)
        # Allocate array for the current row of the sensitivity matrix
        local_row = np.empty(result_size)
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
                local_row[k] = (
                    constant_factor
                    * vector[i]
                    * regional_field_amplitude
                    * (bx * fx + by * fy + bz * fz)
                )
            else:
                local_row[k] = (
                    constant_factor * vector[i] * regional_field_amplitude * bx
                )
                local_row[k + n_cells] = (
                    constant_factor * vector[i] * regional_field_amplitude * by
                )
                local_row[k + 2 * n_cells] = (
                    constant_factor * vector[i] * regional_field_amplitude * bz
                )
        # Apply reduction operation to add the values of the row to the running
        # result. Avoid slicing the `result` array when updating it to avoid
        # racing conditions, just add the `local_row` to the `results`
        # variable.
        result += local_row


@jit(nopython=True, parallel=False)
def _tmi_derivative_sensitivity_t_dot_v_serial(
    receivers,
    nodes,
    cell_nodes,
    regional_field,
    kernel_xx,
    kernel_yy,
    kernel_zz,
    kernel_xy,
    kernel_xz,
    kernel_yz,
    constant_factor,
    scalar_model,
    vector,
    result,
):
    r"""
    Compute ``G.T @ v`` in serial, without building G, for a spatial TMI derivative.

    Parameters
    ----------
    receivers : (n_receivers, 3) array
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) array
        Array with the location of the mesh nodes.
    cell_nodes : (n_active_cells, 8) array
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz : callables
        Kernel functions used for computing the desired TMI derivative.
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        sensitivity matrix.
    scalar_model : bool
        If True, the result will be computed assuming that the ``model`` has
        susceptibilities (scalar model) for each active cell.
        If False, the result will be computed assuming that the ``model`` has
        effective susceptibilities (vector model) for each active cell.
    vector : (n_receivers) numpy.ndarray
        Array that represents the vector used in the dot product.
    result : (n_active_cells) or (3 * n_active_cells) numpy.ndarray
        Running result array where the output of the dot product will be added to.
        The array should have ``n_active_cells`` elements if ``scalar_model``
        is True, or ``3 * n_active_cells`` otherwise.

    Notes
    -----
    This function is meant to be run in serial. Writing to the ``result`` array
    inside a parallel loop over the receivers generates a race condition that
    leads to corrupted outputs.

    A parallel implementation of this function is available in
    ``_tmi_derivative_sensitivity_t_dot_v_parallel``.

    See also
    --------
    _sensitivity_tmi_derivative
        Compute the sensitivity matrix for a TMI derivative by allocating it in memory.
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
            kxx[j] = kernel_xx(dx, dy, dz, distance)
            kyy[j] = kernel_yy(dx, dy, dz, distance)
            kzz[j] = kernel_zz(dx, dy, dz, distance)
            kxy[j] = kernel_xy(dx, dy, dz, distance)
            kxz[j] = kernel_xz(dx, dy, dz, distance)
            kyz[j] = kernel_yz(dx, dy, dz, distance)
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
                result[k] += (
                    constant_factor
                    * vector[i]
                    * regional_field_amplitude
                    * (bx * fx + by * fy + bz * fz)
                )
            else:
                result[k] += constant_factor * vector[i] * regional_field_amplitude * bx
                result[k + n_cells] += (
                    constant_factor * vector[i] * regional_field_amplitude * by
                )
                result[k + 2 * n_cells] += (
                    constant_factor * vector[i] * regional_field_amplitude * bz
                )


@jit(nopython=True, parallel=True)
def _tmi_derivative_sensitivity_t_dot_v_parallel(
    receivers,
    nodes,
    cell_nodes,
    regional_field,
    kernel_xx,
    kernel_yy,
    kernel_zz,
    kernel_xy,
    kernel_xz,
    kernel_yz,
    constant_factor,
    scalar_model,
    vector,
    result,
):
    r"""
    Compute ``G.T @ v`` in parallel, without building G, for a spatial TMI derivative.

    Parameters
    ----------
    receivers : (n_receivers, 3) array
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) array
        Array with the location of the mesh nodes.
    cell_nodes : (n_active_cells, 8) array
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz : callables
        Kernel functions used for computing the desired TMI derivative.
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        sensitivity matrix.
    scalar_model : bool
        If True, the result will be computed assuming that the ``model`` has
        susceptibilities (scalar model) for each active cell.
        If False, the result will be computed assuming that the ``model`` has
        effective susceptibilities (vector model) for each active cell.
    vector : (n_receivers) numpy.ndarray
        Array that represents the vector used in the dot product.
    result : (n_active_cells) or (3 * n_active_cells) numpy.ndarray
        Running result array where the output of the dot product will be added to.
        The array should have ``n_active_cells`` elements if ``scalar_model``
        is True, or ``3 * n_active_cells`` otherwise.

    Notes
    -----
    This function is meant to be run in parallel.
    This implementation instructs each thread to allocate their own array for
    the current row of the sensitivity matrix. After computing the elements of
    that row, it gets added to the running ``result`` array through a reduction
    operation handled by Numba.

    A serialized implementation of this function is available in
    ``_tmi_derivative_sensitivity_t_dot_v_serial``.

    See also
    --------
    _sensitivity_tmi_derivative
        Compute the sensitivity matrix for a TMI derivative by allocating it in memory.
    """
    n_receivers = receivers.shape[0]
    n_nodes = nodes.shape[0]
    n_cells = cell_nodes.shape[0]
    fx, fy, fz = regional_field
    regional_field_amplitude = np.sqrt(fx**2 + fy**2 + fz**2)
    fx /= regional_field_amplitude
    fy /= regional_field_amplitude
    fz /= regional_field_amplitude
    result_size = result.size
    # Evaluate kernel function on each node, for each receiver location
    for i in prange(n_receivers):
        # Allocate vectors for kernels evaluated on mesh nodes
        kxx, kyy, kzz = np.empty(n_nodes), np.empty(n_nodes), np.empty(n_nodes)
        kxy, kxz, kyz = np.empty(n_nodes), np.empty(n_nodes), np.empty(n_nodes)
        # Allocate array for the current row of the sensitivity matrix
        local_row = np.empty(result_size)
        # Allocate small vector for the nodes indices for a given cell
        nodes_indices = np.empty(8, dtype=cell_nodes.dtype)
        for j in range(n_nodes):
            dx = nodes[j, 0] - receivers[i, 0]
            dy = nodes[j, 1] - receivers[i, 1]
            dz = nodes[j, 2] - receivers[i, 2]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            kxx[j] = kernel_xx(dx, dy, dz, distance)
            kyy[j] = kernel_yy(dx, dy, dz, distance)
            kzz[j] = kernel_zz(dx, dy, dz, distance)
            kxy[j] = kernel_xy(dx, dy, dz, distance)
            kxz[j] = kernel_xz(dx, dy, dz, distance)
            kyz[j] = kernel_yz(dx, dy, dz, distance)
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
                local_row[k] = (
                    constant_factor
                    * vector[i]
                    * regional_field_amplitude
                    * (bx * fx + by * fy + bz * fz)
                )
            else:
                local_row[k] = (
                    constant_factor * vector[i] * regional_field_amplitude * bx
                )
                local_row[k + n_cells] = (
                    constant_factor * vector[i] * regional_field_amplitude * by
                )
                local_row[k + 2 * n_cells] = (
                    constant_factor * vector[i] * regional_field_amplitude * bz
                )
        # Apply reduction operation to add the values of the row to the running
        # result. Avoid slicing the `result` array when updating it to avoid
        # racing conditions, just add the `local_row` to the `results`
        # variable.
        result += local_row


@jit(nopython=True, parallel=False)
def _diagonal_G_T_dot_G_mag_serial(
    receivers,
    nodes,
    cell_nodes,
    regional_field,
    kernel_x,
    kernel_y,
    kernel_z,
    constant_factor,
    scalar_model,
    weights,
    diagonal,
):
    """
    Diagonal of ``G.T @ W.T @ W @ G`` for single magnetic component, in serial.

    This function doesn't store the full ``G`` matrix in memory.

    Parameters
    ----------
    receivers : (n_receivers, 3) numpy.ndarray
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) numpy.ndarray
        Array with the location of the mesh nodes.
    cell_nodes : (n_active_cells, 8) numpy.ndarray
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
        If True, the result will be computed assuming that the ``model`` has
        susceptibilities (scalar model) for each active cell.
        If False, the result will be computed assuming that the ``model`` has
        effective susceptibilities (vector model) for each active cell.
    weights : (n_receivers,) numpy.ndarray
        Array with data weights. It should be the diagonal of the ``W`` matrix,
        squared.
    diagonal : (n_active_cells) or (3 * n_active_cells) numpy.ndarray
        Array where the diagonal of ``G.T @ G`` will be added to.

    Notes
    -----
    This function is meant to be run in serial. Use the
    ``_diagonal_G_T_dot_G_mag_parallel`` one for parallelized computations.
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
                g_element = (
                    constant_factor
                    * regional_field_amplitude
                    * (ux * fx + uy * fy + uz * fz)
                )
                diagonal[k] += weights[i] * g_element**2
            else:
                const = constant_factor * regional_field_amplitude
                diagonal[k] += weights[i] * (const * ux) ** 2
                diagonal[k + n_cells] += weights[i] * (const * uy) ** 2
                diagonal[k + 2 * n_cells] += weights[i] * (const * uz) ** 2


@jit(nopython=True, parallel=True)
def _diagonal_G_T_dot_G_mag_parallel(
    receivers,
    nodes,
    cell_nodes,
    regional_field,
    kernel_x,
    kernel_y,
    kernel_z,
    constant_factor,
    scalar_model,
    weights,
    diagonal,
):
    """
    Diagonal of ``G.T @ W.T @ W @ G`` for single magnetic component, in parallel.

    This function doesn't store the full ``G`` matrix in memory.

    Parameters
    ----------
    receivers : (n_receivers, 3) numpy.ndarray
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) numpy.ndarray
        Array with the location of the mesh nodes.
    cell_nodes : (n_active_cells, 8) numpy.ndarray
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
        If True, the result will be computed assuming that the ``model`` has
        susceptibilities (scalar model) for each active cell.
        If False, the result will be computed assuming that the ``model`` has
        effective susceptibilities (vector model) for each active cell.
    weights : (n_receivers,) numpy.ndarray
        Array with data weights. It should be the diagonal of the ``W`` matrix,
        squared.
    diagonal : (n_active_cells) or (3 * n_active_cells) numpy.ndarray
        Array where the diagonal of ``G.T @ G`` will be added to.

    Notes
    -----
    This function is meant to be run in parallel. Use the
    ``_diagonal_G_T_dot_G_mag_serial`` one for serialized computations.
    """
    n_receivers = receivers.shape[0]
    n_nodes = nodes.shape[0]
    n_cells = cell_nodes.shape[0]
    diagonal_size = diagonal.size
    fx, fy, fz = regional_field
    regional_field_amplitude = np.sqrt(fx**2 + fy**2 + fz**2)
    fx /= regional_field_amplitude
    fy /= regional_field_amplitude
    fz /= regional_field_amplitude
    # Evaluate kernel function on each node, for each receiver location
    for i in prange(n_receivers):
        # Allocate vectors for kernels evaluated on mesh nodes
        kx, ky, kz = np.empty(n_nodes), np.empty(n_nodes), np.empty(n_nodes)
        # Allocate array for the diagonal elements for the current receiver.
        local_diagonal = np.empty(diagonal_size)
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
                g_element = (
                    constant_factor
                    * regional_field_amplitude
                    * (ux * fx + uy * fy + uz * fz)
                )
                local_diagonal[k] = weights[i] * g_element**2
            else:
                const = constant_factor * regional_field_amplitude
                local_diagonal[k] = weights[i] * (const * ux) ** 2
                local_diagonal[k + n_cells] = weights[i] * (const * uy) ** 2
                local_diagonal[k + 2 * n_cells] = weights[i] * (const * uz) ** 2
        # Add the result to the diagonal.
        # Apply reduction operation to add the values of the local diagonal to
        # the running diagonal array. Avoid slicing the `diagonal` array when
        # updating it to avoid racing conditions, just add the `local_diagonal`
        # to the `diagonal` variable.
        diagonal += local_diagonal


@jit(nopython=True, parallel=False)
def _diagonal_G_T_dot_G_tmi_serial(
    receivers,
    nodes,
    cell_nodes,
    regional_field,
    constant_factor,
    scalar_model,
    weights,
    diagonal,
):
    """
    Diagonal of ``G.T @ W.T @ W @ G`` for TMI, in serial.

    This function doesn't store the full ``G`` matrix in memory.

    Parameters
    ----------
    receivers : (n_receivers, 3) numpy.ndarray
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) numpy.ndarray
        Array with the location of the mesh nodes.
    cell_nodes : (n_active_cells, 8) numpy.ndarray
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        sensitivity matrix.
    scalar_model : bool
        If True, the result will be computed assuming that the ``model`` has
        susceptibilities (scalar model) for each active cell.
        If False, the result will be computed assuming that the ``model`` has
        effective susceptibilities (vector model) for each active cell.
    weights : (n_receivers,) numpy.ndarray
        Array with data weights. It should be the diagonal of the ``W`` matrix,
        squared.
    diagonal : (n_active_cells) or (3 * n_active_cells) numpy.ndarray
        Array where the diagonal of ``G.T @ G`` will be added to.

    Notes
    -----
    This function is meant to be run in serial. Use the
    ``_diagonal_G_T_dot_G_tmi_parallel`` one for parallelized computations.
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
                g_element = (
                    constant_factor
                    * regional_field_amplitude
                    * (bx * fx + by * fy + bz * fz)
                )
                diagonal[k] += weights[i] * g_element**2
            else:
                const = constant_factor * regional_field_amplitude
                diagonal[k] += weights[i] * (const * bx) ** 2
                diagonal[k + n_cells] += weights[i] * (const * by) ** 2
                diagonal[k + 2 * n_cells] += weights[i] * (const * bz) ** 2


@jit(nopython=True, parallel=True)
def _diagonal_G_T_dot_G_tmi_parallel(
    receivers,
    nodes,
    cell_nodes,
    regional_field,
    constant_factor,
    scalar_model,
    weights,
    diagonal,
):
    """
    Diagonal of ``G.T @ W.T @ W @ G`` for TMI, in parallel.

    This function doesn't store the full ``G`` matrix in memory.

    Parameters
    ----------
    receivers : (n_receivers, 3) numpy.ndarray
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) numpy.ndarray
        Array with the location of the mesh nodes.
    cell_nodes : (n_active_cells, 8) numpy.ndarray
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        sensitivity matrix.
    scalar_model : bool
        If True, the result will be computed assuming that the ``model`` has
        susceptibilities (scalar model) for each active cell.
        If False, the result will be computed assuming that the ``model`` has
        effective susceptibilities (vector model) for each active cell.
    weights : (n_receivers,) numpy.ndarray
        Array with data weights. It should be the diagonal of the ``W`` matrix,
        squared.
    diagonal : (n_active_cells) or (3 * n_active_cells) numpy.ndarray
        Array where the diagonal of ``G.T @ G`` will be added to.

    Notes
    -----
    This function is meant to be run in parallel. Use the
    ``_diagonal_G_T_dot_G_tmi_serial`` one for serialized computations.
    """
    n_receivers = receivers.shape[0]
    n_nodes = nodes.shape[0]
    n_cells = cell_nodes.shape[0]
    diagonal_size = diagonal.size
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
        # Allocate array for the diagonal elements for the current receiver.
        local_diagonal = np.empty(diagonal_size)
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
                g_element = (
                    constant_factor
                    * regional_field_amplitude
                    * (bx * fx + by * fy + bz * fz)
                )
                local_diagonal[k] = weights[i] * g_element**2
            else:
                const = constant_factor * regional_field_amplitude
                local_diagonal[k] = weights[i] * (const * bx) ** 2
                local_diagonal[k + n_cells] = weights[i] * (const * by) ** 2
                local_diagonal[k + 2 * n_cells] = weights[i] * (const * bz) ** 2

        # Add the result to the diagonal.
        # Apply reduction operation to add the values of the local diagonal to
        # the running diagonal array. Avoid slicing the `diagonal` array when
        # updating it to avoid racing conditions, just add the `local_diagonal`
        # to the `diagonal` variable.
        diagonal += local_diagonal


@jit(nopython=True, parallel=False)
def _diagonal_G_T_dot_G_tmi_deriv_serial(
    receivers,
    nodes,
    cell_nodes,
    regional_field,
    kernel_xx,
    kernel_yy,
    kernel_zz,
    kernel_xy,
    kernel_xz,
    kernel_yz,
    constant_factor,
    scalar_model,
    weights,
    diagonal,
):
    """
    Diagonal of ``G.T @ W.T @ W @ G`` for TMI derivatives, in serial.

    This function doesn't store the full ``G`` matrix in memory.

    Parameters
    ----------
    receivers : (n_receivers, 3) numpy.ndarray
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) numpy.ndarray
        Array with the location of the mesh nodes.
    cell_nodes : (n_active_cells, 8) numpy.ndarray
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz : callables
        Kernel functions used for computing the desired TMI derivative.
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        sensitivity matrix.
    scalar_model : bool
        If True, the result will be computed assuming that the ``model`` has
        susceptibilities (scalar model) for each active cell.
        If False, the result will be computed assuming that the ``model`` has
        effective susceptibilities (vector model) for each active cell.
    weights : (n_receivers,) numpy.ndarray
        Array with data weights. It should be the diagonal of the ``W`` matrix,
        squared.
    diagonal : (n_active_cells) or (3 * n_active_cells) numpy.ndarray
        Array where the diagonal of ``G.T @ G`` will be added to.

    Notes
    -----
    This function is meant to be run in serial. Use the
    ``_diagonal_G_T_dot_G_tmi_deriv_parallel`` one for parallelized computations.
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
            kxx[j] = kernel_xx(dx, dy, dz, distance)
            kyy[j] = kernel_yy(dx, dy, dz, distance)
            kzz[j] = kernel_zz(dx, dy, dz, distance)
            kxy[j] = kernel_xy(dx, dy, dz, distance)
            kxz[j] = kernel_xz(dx, dy, dz, distance)
            kyz[j] = kernel_yz(dx, dy, dz, distance)
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
                g_element = (
                    constant_factor
                    * regional_field_amplitude
                    * (bx * fx + by * fy + bz * fz)
                )
                diagonal[k] += weights[i] * g_element**2
            else:
                const = constant_factor * regional_field_amplitude
                diagonal[k] += weights[i] * (const * bx) ** 2
                diagonal[k + n_cells] += weights[i] * (const * by) ** 2
                diagonal[k + 2 * n_cells] += weights[i] * (const * bz) ** 2


@jit(nopython=True, parallel=True)
def _diagonal_G_T_dot_G_tmi_deriv_parallel(
    receivers,
    nodes,
    cell_nodes,
    regional_field,
    kernel_xx,
    kernel_yy,
    kernel_zz,
    kernel_xy,
    kernel_xz,
    kernel_yz,
    constant_factor,
    scalar_model,
    weights,
    diagonal,
):
    """
    Diagonal of ``G.T @ W.T @ W @ G`` for TMI derivatives, in parallel.

    This function doesn't store the full ``G`` matrix in memory.

    Parameters
    ----------
    receivers : (n_receivers, 3) numpy.ndarray
        Array with the locations of the receivers
    nodes : (n_active_nodes, 3) numpy.ndarray
        Array with the location of the mesh nodes.
    cell_nodes : (n_active_cells, 8) numpy.ndarray
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz : callables
        Kernel functions used for computing the desired TMI derivative.
    constant_factor : float
        Constant factor that will be used to multiply each element of the
        sensitivity matrix.
    scalar_model : bool
        If True, the result will be computed assuming that the ``model`` has
        susceptibilities (scalar model) for each active cell.
        If False, the result will be computed assuming that the ``model`` has
        effective susceptibilities (vector model) for each active cell.
    weights : (n_receivers,) numpy.ndarray
        Array with data weights. It should be the diagonal of the ``W`` matrix,
        squared.
    diagonal : (n_active_cells) or (3 * n_active_cells) numpy.ndarray
        Array where the diagonal of ``G.T @ G`` will be added to.

    Notes
    -----
    This function is meant to be run in parallel. Use the
    ``_diagonal_G_T_dot_G_tmi_serial`` one for serialized computations.
    """
    n_receivers = receivers.shape[0]
    n_nodes = nodes.shape[0]
    n_cells = cell_nodes.shape[0]
    diagonal_size = diagonal.size
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
        # Allocate array for the diagonal elements for the current receiver.
        local_diagonal = np.empty(diagonal_size)
        # Allocate small vector for the nodes indices for a given cell
        nodes_indices = np.empty(8, dtype=cell_nodes.dtype)
        for j in range(n_nodes):
            dx = nodes[j, 0] - receivers[i, 0]
            dy = nodes[j, 1] - receivers[i, 1]
            dz = nodes[j, 2] - receivers[i, 2]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            kxx[j] = kernel_xx(dx, dy, dz, distance)
            kyy[j] = kernel_yy(dx, dy, dz, distance)
            kzz[j] = kernel_zz(dx, dy, dz, distance)
            kxy[j] = kernel_xy(dx, dy, dz, distance)
            kxz[j] = kernel_xz(dx, dy, dz, distance)
            kyz[j] = kernel_yz(dx, dy, dz, distance)
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
                g_element = (
                    constant_factor
                    * regional_field_amplitude
                    * (bx * fx + by * fy + bz * fz)
                )
                local_diagonal[k] = weights[i] * g_element**2
            else:
                const = constant_factor * regional_field_amplitude
                local_diagonal[k] = weights[i] * (const * bx) ** 2
                local_diagonal[k + n_cells] = weights[i] * (const * by) ** 2
                local_diagonal[k + 2 * n_cells] = weights[i] * (const * bz) ** 2

        # Add the result to the diagonal.
        # Apply reduction operation to add the values of the local diagonal to
        # the running diagonal array. Avoid slicing the `diagonal` array when
        # updating it to avoid racing conditions, just add the `local_diagonal`
        # to the `diagonal` variable.
        diagonal += local_diagonal


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
        If True, the forward will be computed assuming that the ``model`` has
        susceptibilities (scalar model) for each active cell.
        If False, the forward will be computed assuming that the ``model`` has
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


def _forward_tmi_derivative(
    receivers,
    nodes,
    model,
    fields,
    cell_nodes,
    regional_field,
    kernel_xx,
    kernel_yy,
    kernel_zz,
    kernel_xy,
    kernel_xz,
    kernel_yz,
    constant_factor,
    scalar_model,
):
    r"""
    Forward model a TMI derivative.

    This function should be used with a `numba.jit` decorator, for example:

    .. code::

        from numba import jit

        jit_forward = jit(nopython=True, parallel=True)(_forward_tmi_derivative)

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
        Array full of zeros where the TMI derivative on each receiver will be
        stored. This could be a preallocated array or a slice of it.
    cell_nodes : (n_active_cells, 8) array
        Array of integers, where each row contains the indices of the nodes for
        each active cell in the mesh.
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz : callables
        Kernel functions used for computing the desired TMI derivative.
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

    About the kernel functions
    ^^^^^^^^^^^^^^^^^^^^^^^^^^

    To compute the :math:`\alpha` derivative of the TMI :math:`\Delta T` (with
    :math:`\alpha \in \{x, y, z\}` we need to evaluate third order kernels
    functions for the prism. The kernels we need to evaluate can be obtained by
    fixing one of the subindices to the direction of the derivative
    (:math:`\alpha`) and cycle through combinations of the other two.

    For ``tmi_x`` we need to pass:

    .. code::

        kernel_xx=kernel_eee, kernel_yy=kernel_enn, kernel_zz=kernel_euu,
        kernel_xy=kernel_een, kernel_xz=kernel_eeu, kernel_yz=kernel_enu

    For ``tmi_y`` we need to pass:

    .. code::

        kernel_xx=kernel_een, kernel_yy=kernel_nnn, kernel_zz=kernel_nuu,
        kernel_xy=kernel_enn, kernel_xz=kernel_enu, kernel_yz=kernel_nnu

    For ``tmi_z`` we need to pass:

    .. code::

        kernel_xx=kernel_eeu, kernel_yy=kernel_nnu, kernel_zz=kernel_uuu,
        kernel_xy=kernel_enu, kernel_xz=kernel_euu, kernel_yz=kernel_nuu


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
        kxx, kyy, kzz = np.empty(n_nodes), np.empty(n_nodes), np.empty(n_nodes)
        kxy, kxz, kyz = np.empty(n_nodes), np.empty(n_nodes), np.empty(n_nodes)
        # Allocate small vector for the nodes indices for a given cell
        nodes_indices = np.empty(8, dtype=cell_nodes.dtype)
        for j in range(n_nodes):
            dx = nodes[j, 0] - receivers[i, 0]
            dy = nodes[j, 1] - receivers[i, 1]
            dz = nodes[j, 2] - receivers[i, 2]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            kxx[j] = kernel_xx(dx, dy, dz, distance)
            kyy[j] = kernel_yy(dx, dy, dz, distance)
            kzz[j] = kernel_zz(dx, dy, dz, distance)
            kxy[j] = kernel_xy(dx, dy, dz, distance)
            kxz[j] = kernel_xz(dx, dy, dz, distance)
            kyz[j] = kernel_yz(dx, dy, dz, distance)
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


NUMBA_FUNCTIONS_3D = {
    "forward": {
        "tmi": {
            parallel: jit(nopython=True, parallel=parallel)(_forward_tmi)
            for parallel in (True, False)
        },
        "magnetic_component": {
            parallel: jit(nopython=True, parallel=parallel)(_forward_mag)
            for parallel in (True, False)
        },
        "tmi_derivative": {
            parallel: jit(nopython=True, parallel=parallel)(_forward_tmi_derivative)
            for parallel in (True, False)
        },
    },
    "sensitivity": {
        "tmi": {
            parallel: jit(nopython=True, parallel=parallel)(_sensitivity_tmi)
            for parallel in (True, False)
        },
        "magnetic_component": {
            parallel: jit(nopython=True, parallel=parallel)(_sensitivity_mag)
            for parallel in (True, False)
        },
        "tmi_derivative": {
            parallel: jit(nopython=True, parallel=parallel)(_sensitivity_tmi_derivative)
            for parallel in (True, False)
        },
    },
    "gt_dot_v": {
        "tmi": {
            False: _tmi_sensitivity_t_dot_v_serial,
            True: _tmi_sensitivity_t_dot_v_parallel,
        },
        "magnetic_component": {
            False: _mag_sensitivity_t_dot_v_serial,
            True: _mag_sensitivity_t_dot_v_parallel,
        },
        "tmi_derivative": {
            False: _tmi_derivative_sensitivity_t_dot_v_serial,
            True: _tmi_derivative_sensitivity_t_dot_v_parallel,
        },
    },
    "diagonal_gtg": {
        "tmi": {
            False: _diagonal_G_T_dot_G_tmi_serial,
            True: _diagonal_G_T_dot_G_tmi_parallel,
        },
        "magnetic_component": {
            False: _diagonal_G_T_dot_G_mag_serial,
            True: _diagonal_G_T_dot_G_mag_parallel,
        },
        "tmi_derivative": {
            False: _diagonal_G_T_dot_G_tmi_deriv_serial,
            True: _diagonal_G_T_dot_G_tmi_deriv_serial,
        },
    },
}
