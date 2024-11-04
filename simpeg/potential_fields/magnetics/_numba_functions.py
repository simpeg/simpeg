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

from .._numba_utils import kernels_in_nodes_to_cell, evaluate_kernels_on_cell


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


def _forward_mag_2d_mesh(
    receivers,
    cells_bounds,
    top,
    bottom,
    model,
    fields,
    regional_field,
    forward_func,
    scalar_model,
):
    """
    Forward model single magnetic component for 2D meshes.

    This function is designed to be used with equivalent sources, where the
    mesh is a 2D mesh (prism layer). The top and bottom boundaries of each cell
    are passed through the ``top`` and ``bottom`` arrays.

    This function should be used with a `numba.jit` decorator, for example:

    .. code::

        from numba import jit

        jit_forward = jit(nopython=True, parallel=True)(_forward_mag_2d_mesh)

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
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    forward_func : callable
        Forward function that will be evaluated on each node of the mesh. Choose
        one of the forward functions in ``choclo.prism``.
    scalar_model : bool
        If True, the forward will be computing assuming that the ``model`` has
        susceptibilities (scalar model) for each active cell.
        If False, the forward will be computing assuming that the ``model`` has
        effective susceptibilities (vector model) for each active cell.

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

    """
    n_receivers = receivers.shape[0]
    n_cells = cells_bounds.shape[0]
    fx, fy, fz = regional_field
    regional_field_amplitude = np.sqrt(fx**2 + fy**2 + fz**2)
    fx /= regional_field_amplitude
    fy /= regional_field_amplitude
    fz /= regional_field_amplitude
    # Forward model the magnetic component of each cell on each receiver location
    for i in prange(n_receivers):
        for j in range(n_cells):
            # Define magnetization vector of the cell
            # (we we'll divide by mu_0 when adding the forward modelled field)
            if scalar_model:
                # model is susceptibility, so the vector is parallel to the
                # regional field
                magnetization_x = model[j] * fx
                magnetization_y = model[j] * fy
                magnetization_z = model[j] * fz
            else:
                # model is effective susceptibility (vector)
                magnetization_x = model[j]
                magnetization_y = model[j + n_cells]
                magnetization_z = model[j + 2 * n_cells]
            # Forward the magnetic component
            fields[i] += (
                regional_field_amplitude
                * forward_func(
                    receivers[i, 0],
                    receivers[i, 1],
                    receivers[i, 2],
                    cells_bounds[j, 0],
                    cells_bounds[j, 1],
                    cells_bounds[j, 2],
                    cells_bounds[j, 3],
                    bottom[j],
                    top[j],
                    magnetization_x,
                    magnetization_y,
                    magnetization_z,
                )
                / choclo.constants.VACUUM_MAGNETIC_PERMEABILITY
            )


def _forward_tmi_2d_mesh(
    receivers,
    cells_bounds,
    top,
    bottom,
    model,
    fields,
    regional_field,
    scalar_model,
):
    """
    Forward model the TMI for 2D meshes.

    This function is designed to be used with equivalent sources, where the
    mesh is a 2D mesh (prism layer). The top and bottom boundaries of each cell
    are passed through the ``top`` and ``bottom`` arrays.

    This function should be used with a `numba.jit` decorator, for example:

    .. code::

        from numba import jit

        jit_forward = jit(nopython=True, parallel=True)(_forward_tmi_2d_mesh)

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
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
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
    n_cells = cells_bounds.shape[0]
    fx, fy, fz = regional_field
    regional_field_amplitude = np.sqrt(fx**2 + fy**2 + fz**2)
    fx /= regional_field_amplitude
    fy /= regional_field_amplitude
    fz /= regional_field_amplitude
    # Forward model the magnetic component of each cell on each receiver location
    for i in prange(n_receivers):
        for j in range(n_cells):
            # Define magnetization vector of the cell
            # (we we'll divide by mu_0 when adding the forward modelled field)
            if scalar_model:
                # model is susceptibility, so the vector is parallel to the
                # regional field
                magnetization_x = model[j] * fx
                magnetization_y = model[j] * fy
                magnetization_z = model[j] * fz
            else:
                # model is effective susceptibility (vector)
                magnetization_x = model[j]
                magnetization_y = model[j + n_cells]
                magnetization_z = model[j + 2 * n_cells]
            # Forward the magnetic field vector and compute tmi
            bx, by, bz = choclo.prism.magnetic_field(
                receivers[i, 0],
                receivers[i, 1],
                receivers[i, 2],
                cells_bounds[j, 0],
                cells_bounds[j, 1],
                cells_bounds[j, 2],
                cells_bounds[j, 3],
                bottom[j],
                top[j],
                magnetization_x,
                magnetization_y,
                magnetization_z,
            )
            fields[i] += (
                regional_field_amplitude
                * (bx * fx + by * fy + bz * fz)
                / choclo.constants.VACUUM_MAGNETIC_PERMEABILITY
            )


def _forward_tmi_derivative_2d_mesh(
    receivers,
    cells_bounds,
    top,
    bottom,
    model,
    fields,
    regional_field,
    kernel_xx,
    kernel_yy,
    kernel_zz,
    kernel_xy,
    kernel_xz,
    kernel_yz,
    scalar_model,
):
    r"""
    Forward model a TMI derivative for 2D meshes.

    This function is designed to be used with equivalent sources, where the
    mesh is a 2D mesh (prism layer). The top and bottom boundaries of each cell
    are passed through the ``top`` and ``bottom`` arrays.

    This function should be used with a `numba.jit` decorator, for example:

    .. code::

        from numba import jit

        jit_forward = jit(nopython=True, parallel=True)(_forward_tmi_2d_mesh)

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
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz : callables
        Kernel functions used for computing the desired TMI derivative.
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
    n_cells = cells_bounds.shape[0]
    fx, fy, fz = regional_field
    regional_field_amplitude = np.sqrt(fx**2 + fy**2 + fz**2)
    fx /= regional_field_amplitude
    fy /= regional_field_amplitude
    fz /= regional_field_amplitude
    # Forward model the magnetic component of each cell on each receiver location
    for i in prange(n_receivers):
        for j in range(n_cells):
            uxx, uyy, uzz = evaluate_kernels_on_cell(
                receivers[i, 0],
                receivers[i, 1],
                receivers[i, 2],
                cells_bounds[j, 0],
                cells_bounds[j, 1],
                cells_bounds[j, 2],
                cells_bounds[j, 3],
                bottom[j],
                top[j],
                kernel_xx,
                kernel_yy,
                kernel_zz,
            )
            uxy, uxz, uyz = evaluate_kernels_on_cell(
                receivers[i, 0],
                receivers[i, 1],
                receivers[i, 2],
                cells_bounds[j, 0],
                cells_bounds[j, 1],
                cells_bounds[j, 2],
                cells_bounds[j, 3],
                bottom[j],
                top[j],
                kernel_xy,
                kernel_xz,
                kernel_yz,
            )
            if scalar_model:
                bx = uxx * fx + uxy * fy + uxz * fz
                by = uxy * fx + uyy * fy + uyz * fz
                bz = uxz * fx + uyz * fy + uzz * fz
                fields[i] += (
                    model[j]
                    * regional_field_amplitude
                    * (fx * bx + fy * by + fz * bz)
                    / (4 * np.pi)
                )
            else:
                model_x = model[j]
                model_y = model[j + n_cells]
                model_z = model[j + 2 * n_cells]
                bx = uxx * model_x + uxy * model_y + uxz * model_z
                by = uxy * model_x + uyy * model_y + uyz * model_z
                bz = uxz * model_x + uyz * model_y + uzz * model_z
                fields[i] += (
                    regional_field_amplitude * (bx * fx + by * fy + bz * fz) / 4 / np.pi
                )


def _sensitivity_mag_2d_mesh(
    receivers,
    cells_bounds,
    top,
    bottom,
    sensitivity_matrix,
    regional_field,
    kernel_x,
    kernel_y,
    kernel_z,
    scalar_model,
):
    r"""
    Fill the sensitivity matrix for single mag component for 2d meshes.

    This function is designed to be used with equivalent sources, where the
    mesh is a 2D mesh (prism layer). The top and bottom boundaries of each cell
    are passed through the ``top`` and ``bottom`` arrays.

    This function should be used with a `numba.jit` decorator, for example:

    .. code::

        from numba import jit

        jit_sensitivity = jit(nopython=True, parallel=True)(_sensitivity_mag_2d_mesh)

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
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    kernel_x, kernel_y, kernel_z : callable
        Kernels used to compute the desired magnetic component. For example,
        for computing bx we need to use ``kernel_x=kernel_ee``,
        ``kernel_y=kernel_en``, ``kernel_z=kernel_eu``.
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
    fx, fy, fz = regional_field
    regional_field_amplitude = np.sqrt(fx**2 + fy**2 + fz**2)
    fx /= regional_field_amplitude
    fy /= regional_field_amplitude
    fz /= regional_field_amplitude

    constant_factor = 1 / 4 / np.pi

    # Fill the sensitivity matrix
    n_receivers = receivers.shape[0]
    n_cells = cells_bounds.shape[0]
    for i in prange(n_receivers):
        for j in range(n_cells):
            # Evaluate kernels for the current cell and receiver
            ux, uy, uz = evaluate_kernels_on_cell(
                receivers[i, 0],
                receivers[i, 1],
                receivers[i, 2],
                cells_bounds[j, 0],
                cells_bounds[j, 1],
                cells_bounds[j, 2],
                cells_bounds[j, 3],
                bottom[j],
                top[j],
                kernel_x,
                kernel_y,
                kernel_z,
            )
            if scalar_model:
                sensitivity_matrix[i, j] = (
                    constant_factor
                    * regional_field_amplitude
                    * (ux * fx + uy * fy + uz * fz)
                )
            else:
                sensitivity_matrix[i, j] = (
                    constant_factor * regional_field_amplitude * ux
                )
                sensitivity_matrix[i, j + n_cells] = (
                    constant_factor * regional_field_amplitude * uy
                )
                sensitivity_matrix[i, j + 2 * n_cells] = (
                    constant_factor * regional_field_amplitude * uz
                )


def _sensitivity_tmi_2d_mesh(
    receivers,
    cells_bounds,
    top,
    bottom,
    sensitivity_matrix,
    regional_field,
    scalar_model,
):
    r"""
    Fill the sensitivity matrix TMI for 2d meshes.

    This function is designed to be used with equivalent sources, where the
    mesh is a 2D mesh (prism layer). The top and bottom boundaries of each cell
    are passed through the ``top`` and ``bottom`` arrays.

    This function should be used with a `numba.jit` decorator, for example:

    .. code::

        from numba import jit

        jit_tmi = jit(nopython=True, parallel=True)(_sensitivity_tmi_2d_mesh)

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
    sensitivity_matrix : array
        Empty 2d array where the sensitivity matrix elements will be filled.
        This could be a preallocated empty array or a slice of it.
        The array should have a shape of ``(n_receivers, n_active_nodes)``
        if ``scalar_model`` is True.
        The array should have a shape of ``(n_receivers, 3 * n_active_nodes)``
        if ``scalar_model`` is False.
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
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
    fx, fy, fz = regional_field
    regional_field_amplitude = np.sqrt(fx**2 + fy**2 + fz**2)
    fx /= regional_field_amplitude
    fy /= regional_field_amplitude
    fz /= regional_field_amplitude

    constant_factor = 1 / 4 / np.pi

    # Fill the sensitivity matrix
    n_receivers = receivers.shape[0]
    n_cells = cells_bounds.shape[0]
    for i in prange(n_receivers):
        for j in range(n_cells):
            # Evaluate kernels for the current cell and receiver
            uxx, uyy, uzz = evaluate_kernels_on_cell(
                receivers[i, 0],
                receivers[i, 1],
                receivers[i, 2],
                cells_bounds[j, 0],
                cells_bounds[j, 1],
                cells_bounds[j, 2],
                cells_bounds[j, 3],
                bottom[j],
                top[j],
                choclo.prism.kernel_ee,
                choclo.prism.kernel_nn,
                choclo.prism.kernel_uu,
            )
            uxy, uxz, uyz = evaluate_kernels_on_cell(
                receivers[i, 0],
                receivers[i, 1],
                receivers[i, 2],
                cells_bounds[j, 0],
                cells_bounds[j, 1],
                cells_bounds[j, 2],
                cells_bounds[j, 3],
                bottom[j],
                top[j],
                choclo.prism.kernel_en,
                choclo.prism.kernel_eu,
                choclo.prism.kernel_nu,
            )
            bx = uxx * fx + uxy * fy + uxz * fz
            by = uxy * fx + uyy * fy + uyz * fz
            bz = uxz * fx + uyz * fy + uzz * fz
            if scalar_model:
                sensitivity_matrix[i, j] = (
                    constant_factor
                    * regional_field_amplitude
                    * (bx * fx + by * fy + bz * fz)
                )
            else:
                sensitivity_matrix[i, j] = (
                    constant_factor * regional_field_amplitude * bx
                )
                sensitivity_matrix[i, j + n_cells] = (
                    constant_factor * regional_field_amplitude * by
                )
                sensitivity_matrix[i, j + 2 * n_cells] = (
                    constant_factor * regional_field_amplitude * bz
                )


def _sensitivity_tmi_derivative_2d_mesh(
    receivers,
    cells_bounds,
    top,
    bottom,
    sensitivity_matrix,
    regional_field,
    kernel_xx,
    kernel_yy,
    kernel_zz,
    kernel_xy,
    kernel_xz,
    kernel_yz,
    scalar_model,
):
    r"""
    Fill the sensitivity matrix TMI for 2d meshes.

    This function is designed to be used with equivalent sources, where the
    mesh is a 2D mesh (prism layer). The top and bottom boundaries of each cell
    are passed through the ``top`` and ``bottom`` arrays.

    This function should be used with a `numba.jit` decorator, for example:

    .. code::

        from numba import jit

        jit_tmi = jit(nopython=True, parallel=True)(_sensitivity_tmi_2d_mesh)

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
    sensitivity_matrix : array
        Empty 2d array where the sensitivity matrix elements will be filled.
        This could be a preallocated empty array or a slice of it.
        The array should have a shape of ``(n_receivers, n_active_nodes)``
        if ``scalar_model`` is True.
        The array should have a shape of ``(n_receivers, 3 * n_active_nodes)``
        if ``scalar_model`` is False.
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz : callables
        Kernel functions used for computing the desired TMI derivative.
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
    """
    fx, fy, fz = regional_field
    regional_field_amplitude = np.sqrt(fx**2 + fy**2 + fz**2)
    fx /= regional_field_amplitude
    fy /= regional_field_amplitude
    fz /= regional_field_amplitude

    constant_factor = 1 / 4 / np.pi

    # Fill the sensitivity matrix
    n_receivers = receivers.shape[0]
    n_cells = cells_bounds.shape[0]
    for i in prange(n_receivers):
        for j in range(n_cells):
            # Evaluate kernels for the current cell and receiver
            uxx, uyy, uzz = evaluate_kernels_on_cell(
                receivers[i, 0],
                receivers[i, 1],
                receivers[i, 2],
                cells_bounds[j, 0],
                cells_bounds[j, 1],
                cells_bounds[j, 2],
                cells_bounds[j, 3],
                bottom[j],
                top[j],
                kernel_xx,
                kernel_yy,
                kernel_zz,
            )
            uxy, uxz, uyz = evaluate_kernels_on_cell(
                receivers[i, 0],
                receivers[i, 1],
                receivers[i, 2],
                cells_bounds[j, 0],
                cells_bounds[j, 1],
                cells_bounds[j, 2],
                cells_bounds[j, 3],
                bottom[j],
                top[j],
                kernel_xy,
                kernel_xz,
                kernel_yz,
            )
            bx = uxx * fx + uxy * fy + uxz * fz
            by = uxy * fx + uyy * fy + uyz * fz
            bz = uxz * fx + uyz * fy + uzz * fz
            if scalar_model:
                sensitivity_matrix[i, j] = (
                    constant_factor
                    * regional_field_amplitude
                    * (bx * fx + by * fy + bz * fz)
                )
            else:
                sensitivity_matrix[i, j] = (
                    constant_factor * regional_field_amplitude * bx
                )
                sensitivity_matrix[i, j + n_cells] = (
                    constant_factor * regional_field_amplitude * by
                )
                sensitivity_matrix[i, j + 2 * n_cells] = (
                    constant_factor * regional_field_amplitude * bz
                )


_sensitivity_tmi_serial = jit(nopython=True, parallel=False)(_sensitivity_tmi)
_sensitivity_tmi_parallel = jit(nopython=True, parallel=True)(_sensitivity_tmi)
_forward_tmi_serial = jit(nopython=True, parallel=False)(_forward_tmi)
_forward_tmi_parallel = jit(nopython=True, parallel=True)(_forward_tmi)
_forward_mag_serial = jit(nopython=True, parallel=False)(_forward_mag)
_forward_mag_parallel = jit(nopython=True, parallel=True)(_forward_mag)
_sensitivity_mag_serial = jit(nopython=True, parallel=False)(_sensitivity_mag)
_sensitivity_mag_parallel = jit(nopython=True, parallel=True)(_sensitivity_mag)
_forward_tmi_derivative_parallel = jit(nopython=True, parallel=True)(
    _forward_tmi_derivative
)
_forward_tmi_derivative_serial = jit(nopython=True, parallel=False)(
    _forward_tmi_derivative
)
_sensitivity_tmi_derivative_parallel = jit(nopython=True, parallel=True)(
    _sensitivity_tmi_derivative
)
_sensitivity_tmi_derivative_serial = jit(nopython=True, parallel=False)(
    _sensitivity_tmi_derivative
)
_forward_tmi_2d_mesh_serial = jit(nopython=True, parallel=False)(_forward_tmi_2d_mesh)
_forward_tmi_2d_mesh_parallel = jit(nopython=True, parallel=True)(_forward_tmi_2d_mesh)
_forward_mag_2d_mesh_serial = jit(nopython=True, parallel=False)(_forward_mag_2d_mesh)
_forward_mag_2d_mesh_parallel = jit(nopython=True, parallel=True)(_forward_mag_2d_mesh)
_forward_tmi_derivative_2d_mesh_serial = jit(nopython=True, parallel=False)(
    _forward_tmi_derivative_2d_mesh
)
_forward_tmi_derivative_2d_mesh_parallel = jit(nopython=True, parallel=True)(
    _forward_tmi_derivative_2d_mesh
)
_sensitivity_mag_2d_mesh_serial = jit(nopython=True, parallel=False)(
    _sensitivity_mag_2d_mesh
)
_sensitivity_mag_2d_mesh_parallel = jit(nopython=True, parallel=True)(
    _sensitivity_mag_2d_mesh
)
_sensitivity_tmi_2d_mesh_serial = jit(nopython=True, parallel=False)(
    _sensitivity_tmi_2d_mesh
)
_sensitivity_tmi_2d_mesh_parallel = jit(nopython=True, parallel=True)(
    _sensitivity_tmi_2d_mesh
)
_sensitivity_tmi_derivative_2d_mesh_serial = jit(nopython=True, parallel=False)(
    _sensitivity_tmi_derivative_2d_mesh
)
_sensitivity_tmi_derivative_2d_mesh_parallel = jit(nopython=True, parallel=True)(
    _sensitivity_tmi_derivative_2d_mesh
)
