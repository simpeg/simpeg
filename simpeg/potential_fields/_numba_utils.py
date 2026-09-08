"""
Utility functions for Numba implementations

These functions are meant to be used both in the Numba-based gravity and
magnetic simulations.
"""

import numpy as np

try:
    from numba import jit
except ImportError:
    # Define dummy jit decorator
    def jit(*args, **kwargs):
        return lambda f: f


@jit(nopython=True)
def kernels_in_nodes_to_cell(kernels, nodes_indices):
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


@jit(nopython=True)
def evaluate_kernels_on_cell(
    easting,
    northing,
    upward,
    prism_west,
    prism_east,
    prism_south,
    prism_north,
    prism_bottom,
    prism_top,
    kernel_x,
    kernel_y,
    kernel_z,
):
    r"""
    Evaluate three kernel functions on every shifted vertex of a prism.

    .. note::

        This function was inspired in the ``_evaluate_kernel`` function in
        Choclo (released under BSD 3-clause Licence):
        https://www.fatiando.org/choclo

    Parameters
    ----------
    easting, northing, upward : float
        Easting, northing and upward coordinates of the observation point. Must
        be in meters.
    prism_west, prism_east : floats
        The West and East boundaries of the prism. Must be in meters.
    prism_south, prism_north : floats
        The South and North boundaries of the prism. Must be in meters.
    prism_bottom, prism_top : floats
        The bottom and top boundaries of the prism. Must be in meters.
    kernel_x, kernel_y, kernel_z : callable
        Kernel functions that will be evaluated on each one of the shifted
        vertices of the prism.

    Returns
    -------
    result_x, result_y, result_z : floats
        Evaluation of the kernel functions on each one of the vertices of the
        prism.

    Notes
    -----
    This function evaluates each numerical kernel :math:`k(x, y, z)` on each one
    of the vertices of the prism:

    .. math::

        v(\mathbf{p}) =
            \Bigg\lvert \Bigg\lvert \Bigg\lvert
            k(x, y, z)
            \Bigg\rvert_{X_1}^{X_2}
            \Bigg\rvert_{Y_1}^{Y_2}
            \Bigg\rvert_{Z_1}^{Z_2}

    where :math:`X_1`, :math:`X_2`, :math:`Y_1`, :math:`Y_2`, :math:`Z_1` and
    :math:`Z_2` are boundaries of the rectangular prism in the *shifted
    coordinates* defined by the Cartesian coordinate system with its origin
    located on the observation point :math:`\mathbf{p}`.
    """
    # Initialize result floats to zero
    result_x, result_y, result_z = 0.0, 0.0, 0.0
    # Iterate over the vertices of the prism
    for i in range(2):
        # Compute shifted easting coordinate
        if i == 0:
            shift_east = prism_east - easting
        else:
            shift_east = prism_west - easting
        shift_east_sq = shift_east**2
        for j in range(2):
            # Compute shifted northing coordinate
            if j == 0:
                shift_north = prism_north - northing
            else:
                shift_north = prism_south - northing
            shift_north_sq = shift_north**2
            for k in range(2):
                # Compute shifted upward coordinate
                if k == 0:
                    shift_upward = prism_top - upward
                else:
                    shift_upward = prism_bottom - upward
                shift_upward_sq = shift_upward**2
                # Compute the radius
                radius = np.sqrt(shift_east_sq + shift_north_sq + shift_upward_sq)
                # If i, j or k is 1, the corresponding shifted
                # coordinate will refer to the lower boundary,
                # meaning the corresponding term should have a minus
                # sign.
                result_x += (-1) ** (i + j + k) * kernel_x(
                    shift_east, shift_north, shift_upward, radius
                )
                result_y += (-1) ** (i + j + k) * kernel_y(
                    shift_east, shift_north, shift_upward, radius
                )
                result_z += (-1) ** (i + j + k) * kernel_z(
                    shift_east, shift_north, shift_upward, radius
                )
    return result_x, result_y, result_z


@jit(nopython=True)
def evaluate_six_kernels_on_cell(
    easting,
    northing,
    upward,
    prism_west,
    prism_east,
    prism_south,
    prism_north,
    prism_bottom,
    prism_top,
    kernel_xx,
    kernel_yy,
    kernel_zz,
    kernel_xy,
    kernel_xz,
    kernel_yz,
):
    r"""
    Evaluate six kernel functions on every shifted vertex of a prism.

    Similar to ``evaluate_kernels_on_cell``, but designed to evaluate six kernels
    instead of three. This function comes useful for magnetic forwards, when six kernels
    are needed to be evaluated.

    .. note::

        This function was inspired in the ``_evaluate_kernel`` function in
        Choclo (released under BSD 3-clause Licence):
        https://www.fatiando.org/choclo

    Parameters
    ----------
    easting, northing, upward : float
        Easting, northing and upward coordinates of the observation point. Must
        be in meters.
    prism_west, prism_east : floats
        The West and East boundaries of the prism. Must be in meters.
    prism_south, prism_north : floats
        The South and North boundaries of the prism. Must be in meters.
    prism_bottom, prism_top : floats
        The bottom and top boundaries of the prism. Must be in meters.
    kernel_xx, kernel_yy, kernel_zz, kernel_xy, kernel_xz, kernel_yz : callable
        Kernel functions that will be evaluated on each one of the shifted
        vertices of the prism.

    Returns
    -------
    result_xx, result_yy, result_zz, result_xy, result_xz, result_yz : float
        Evaluation of the kernel functions on each one of the vertices of the
        prism.
    """
    # Initialize result floats to zero
    result_xx, result_yy, result_zz = 0.0, 0.0, 0.0
    result_xy, result_xz, result_yz = 0.0, 0.0, 0.0
    # Iterate over the vertices of the prism
    for i in range(2):
        # Compute shifted easting coordinate
        if i == 0:
            shift_east = prism_east - easting
        else:
            shift_east = prism_west - easting
        shift_east_sq = shift_east**2
        for j in range(2):
            # Compute shifted northing coordinate
            if j == 0:
                shift_north = prism_north - northing
            else:
                shift_north = prism_south - northing
            shift_north_sq = shift_north**2
            for k in range(2):
                # Compute shifted upward coordinate
                if k == 0:
                    shift_upward = prism_top - upward
                else:
                    shift_upward = prism_bottom - upward
                shift_upward_sq = shift_upward**2
                # Compute the radius
                radius = np.sqrt(shift_east_sq + shift_north_sq + shift_upward_sq)
                # If i, j or k is 1, the corresponding shifted
                # coordinate will refer to the lower boundary,
                # meaning the corresponding term should have a minus
                # sign.
                result_xx += (-1) ** (i + j + k) * kernel_xx(
                    shift_east, shift_north, shift_upward, radius
                )
                result_yy += (-1) ** (i + j + k) * kernel_yy(
                    shift_east, shift_north, shift_upward, radius
                )
                result_zz += (-1) ** (i + j + k) * kernel_zz(
                    shift_east, shift_north, shift_upward, radius
                )
                result_xy += (-1) ** (i + j + k) * kernel_xy(
                    shift_east, shift_north, shift_upward, radius
                )
                result_xz += (-1) ** (i + j + k) * kernel_xz(
                    shift_east, shift_north, shift_upward, radius
                )
                result_yz += (-1) ** (i + j + k) * kernel_yz(
                    shift_east, shift_north, shift_upward, radius
                )
    return result_xx, result_yy, result_zz, result_xy, result_xz, result_yz
