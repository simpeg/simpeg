"""
Numba functions for magnetic simulation of rectangular prisms on 2D meshes.

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

from ..._numba_utils import evaluate_kernels_on_cell


def _forward_mag(
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

        jit_forward = jit(nopython=True, parallel=True)(_forward_mag)

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


def _forward_tmi(
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

        jit_forward = jit(nopython=True, parallel=True)(_forward_tmi)

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


def _forward_tmi_derivative(
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

        jit_forward = jit(nopython=True, parallel=True)(_forward_tmi_derivative)

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


def _sensitivity_mag(
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

        jit_sensitivity = jit(nopython=True, parallel=True)(_sensitivity_mag)

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
        The array should have a shape of ``(n_receivers, n_active_cells)``
        if ``scalar_model`` is True.
        The array should have a shape of ``(n_receivers, 3 * n_active_cells)``
        if ``scalar_model`` is False.
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


def _sensitivity_tmi(
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

        jit_tmi = jit(nopython=True, parallel=True)(_sensitivity_tmi)

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
        The array should have a shape of ``(n_receivers, n_active_cells)``
        if ``scalar_model`` is True.
        The array should have a shape of ``(n_receivers, 3 * n_active_cells)``
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


def _sensitivity_tmi_derivative(
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

        jit_tmi = jit(nopython=True, parallel=True)(_sensitivity_tmi_derivative)

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
        The array should have a shape of ``(n_receivers, n_active_cells)``
        if ``scalar_model`` is True.
        The array should have a shape of ``(n_receivers, 3 * n_active_cells)``
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


@jit(nopython=True, parallel=False)
def _tmi_sensitivity_t_dot_v_serial(
    receivers,
    cells_bounds,
    top,
    bottom,
    regional_field,
    scalar_model,
    vector,
    result,
):
    r"""
    Compute ``G.T @ v`` in serial, without building G, for TMI on 2d meshes.

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
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    scalar_model : bool
        If True, the sensitivity matrix is build to work with scalar models
        (susceptibilities).
        If False, the sensitivity matrix is build to work with vector models
        (effective susceptibilities).
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
    """
    fx, fy, fz = regional_field
    regional_field_amplitude = np.sqrt(fx**2 + fy**2 + fz**2)
    fx /= regional_field_amplitude
    fy /= regional_field_amplitude
    fz /= regional_field_amplitude

    constant_factor = 1 / 4 / np.pi

    n_receivers = receivers.shape[0]
    n_cells = cells_bounds.shape[0]
    # Evaluate kernel function on each node, for each receiver location
    for i in range(n_receivers):
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
                result[j] += (
                    constant_factor
                    * vector[i]
                    * regional_field_amplitude
                    * (bx * fx + by * fy + bz * fz)
                )
            else:
                result[j] += constant_factor * vector[i] * regional_field_amplitude * bx
                result[j + n_cells] += (
                    constant_factor * vector[i] * regional_field_amplitude * by
                )
                result[j + 2 * n_cells] += (
                    constant_factor * vector[i] * regional_field_amplitude * bz
                )


@jit(nopython=True, parallel=True)
def _tmi_sensitivity_t_dot_v_parallel(
    receivers,
    cells_bounds,
    top,
    bottom,
    regional_field,
    scalar_model,
    vector,
    result,
):
    r"""
    Compute ``G.T @ v`` in parallel, without building G, for TMI on 2d meshes.

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
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    scalar_model : bool
        If True, the sensitivity matrix is build to work with scalar models
        (susceptibilities).
        If False, the sensitivity matrix is build to work with vector models
        (effective susceptibilities).
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

    A parallel implementation of this function is available in
    ``_tmi_sensitivity_t_dot_v_serial``.
    """
    fx, fy, fz = regional_field
    regional_field_amplitude = np.sqrt(fx**2 + fy**2 + fz**2)
    fx /= regional_field_amplitude
    fy /= regional_field_amplitude
    fz /= regional_field_amplitude

    constant_factor = 1 / 4 / np.pi

    n_receivers = receivers.shape[0]
    n_cells = cells_bounds.shape[0]
    # Evaluate kernel function on each node, for each receiver location
    for i in prange(n_receivers):
        # Allocate array for the current row of the sensitivity matrix
        local_row = np.empty(n_cells)
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
                local_row[j] = (
                    constant_factor
                    * vector[i]
                    * regional_field_amplitude
                    * (bx * fx + by * fy + bz * fz)
                )
            else:
                local_row[j] = (
                    constant_factor * vector[i] * regional_field_amplitude * bx
                )
                local_row[j + n_cells] = (
                    constant_factor * vector[i] * regional_field_amplitude * by
                )
                local_row[j + 2 * n_cells] = (
                    constant_factor * vector[i] * regional_field_amplitude * bz
                )
        # Apply reduction operation to add the values of the row to the running
        # result. Avoid slicing the `result` array when updating it to avoid
        # racing conditions, just add the `local_row` to the `results`
        # variable.
        result += local_row


@jit(nopython=True, parallel=False)
def _mag_sensitivity_t_dot_v_serial(
    receivers,
    cells_bounds,
    top,
    bottom,
    regional_field,
    kernel_x,
    kernel_y,
    kernel_z,
    scalar_model,
    vector,
    result,
):
    r"""
    Compute ``G.T @ v`` in serial, without building G, for mag components on 2d meshes.

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
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    kernel_x, kernel_y, kernel_z : callable
        Kernels used to compute the desired magnetic component. For example,
        for computing bx we need to use ``kernel_x=kernel_ee``,
        ``kernel_y=kernel_en``, ``kernel_z=kernel_eu``.
    scalar_model : bool
        If True, the sensitivity matrix is build to work with scalar models
        (susceptibilities).
        If False, the sensitivity matrix is build to work with vector models
        (effective susceptibilities).
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
    """
    fx, fy, fz = regional_field
    regional_field_amplitude = np.sqrt(fx**2 + fy**2 + fz**2)
    fx /= regional_field_amplitude
    fy /= regional_field_amplitude
    fz /= regional_field_amplitude

    constant_factor = 1 / 4 / np.pi

    n_receivers = receivers.shape[0]
    n_cells = cells_bounds.shape[0]
    # Evaluate kernel function on each node, for each receiver location
    for i in range(n_receivers):
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
                result[j] += (
                    constant_factor
                    * vector[i]
                    * regional_field_amplitude
                    * (ux * fx + uy * fy + uz * fz)
                )
            else:
                result[j] += constant_factor * vector[i] * regional_field_amplitude * ux
                result[j + n_cells] += (
                    constant_factor * vector[i] * regional_field_amplitude * uy
                )
                result[j + 2 * n_cells] += (
                    constant_factor * vector[i] * regional_field_amplitude * uz
                )


@jit(nopython=True, parallel=True)
def _mag_sensitivity_t_dot_v_parallel(
    receivers,
    cells_bounds,
    top,
    bottom,
    regional_field,
    kernel_x,
    kernel_y,
    kernel_z,
    scalar_model,
    vector,
    result,
):
    r"""
    Compute ``G.T @ v`` in parallel, without building G, for components on 2d meshes.

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
    regional_field : (3,) array
        Array containing the x, y and z components of the regional magnetic
        field (uniform background field).
    kernel_x, kernel_y, kernel_z : callable
        Kernels used to compute the desired magnetic component. For example,
        for computing bx we need to use ``kernel_x=kernel_ee``,
        ``kernel_y=kernel_en``, ``kernel_z=kernel_eu``.
    scalar_model : bool
        If True, the sensitivity matrix is build to work with scalar models
        (susceptibilities).
        If False, the sensitivity matrix is build to work with vector models
        (effective susceptibilities).
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
    """
    fx, fy, fz = regional_field
    regional_field_amplitude = np.sqrt(fx**2 + fy**2 + fz**2)
    fx /= regional_field_amplitude
    fy /= regional_field_amplitude
    fz /= regional_field_amplitude

    constant_factor = 1 / 4 / np.pi

    n_receivers = receivers.shape[0]
    n_cells = cells_bounds.shape[0]
    # Evaluate kernel function on each node, for each receiver location
    for i in range(n_receivers):
        # Allocate array for the current row of the sensitivity matrix
        local_row = np.empty(n_cells)
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
                local_row[j] = (
                    constant_factor
                    * vector[i]
                    * regional_field_amplitude
                    * (ux * fx + uy * fy + uz * fz)
                )
            else:
                local_row[j] = (
                    constant_factor * vector[i] * regional_field_amplitude * ux
                )
                local_row[j + n_cells] = (
                    constant_factor * vector[i] * regional_field_amplitude * uy
                )
                local_row[j + 2 * n_cells] = (
                    constant_factor * vector[i] * regional_field_amplitude * uz
                )
        # Apply reduction operation to add the values of the row to the running
        # result. Avoid slicing the `result` array when updating it to avoid
        # racing conditions, just add the `local_row` to the `results`
        # variable.
        result += local_row


NUMBA_FUNCTIONS_2D = {
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
            False: None,
            True: None,
        },
    },
    "diagonal_gtg": {
        "tmi": {
            False: None,
            True: None,
        },
        "magnetic_component": {
            False: None,
            True: None,
        },
        "tmi_derivative": {
            False: None,
            True: None,
        },
    },
}
