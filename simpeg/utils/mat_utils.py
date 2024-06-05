from __future__ import annotations  # needed to use type operands in Python 3.8
import numpy as np
from .code_utils import deprecate_function
from ..typing import RandomSeed
from discretize.utils import (  # noqa: F401
    Zero,
    Identity,
    mkvc,
    sdiag,
    sdinv,
    speye,
    kron3,
    spzeros,
    ddx,
    av,
    av_extrap,
    ndgrid,
    ind2sub,
    sub2ind,
    get_subarray,
    inverse_3x3_block_diagonal,
    inverse_2x2_block_diagonal,
    TensorType,
    make_property_tensor,
    inverse_property_tensor,
)


def estimate_diagonal(matrix_arg, n, k=None, approach="Probing"):
    r"""Estimate the diagonal of a matrix.

    This function estimates the diagonal of a matrix using one of the following
    iterative methods:

        - **Probing:** cyclic permutations of vectors with 1's and 0's (default)
        - **Ones:** random +/- 1 entries
        - **Random:** random vectors with entries in the range [-1, 1]

    The user can estimate the diagonal of the matrix by providing the matrix,
    or by providing a function hangle which computes the dot product of the matrix
    and a vector.

    For background information on this method, see
    `Bekas (et al., 2005) <https://www-users.cs.umn.edu/~saad/PDF/umsi-2005-082.pdf>`__
    and `Selig (et al., 2012) <https://www.cita.utoronto.ca/~niels/diagonal.pdf>`__

    Parameters
    ----------
    matrix_arg : numpy.ndarray or function
        The matrix as a ``numpy.ndarray``, or a function handle which computes the dot product
        between the matrix and a vector.
    n : int
        The length of the random vectors used to compute the diagonal; equals number of columns
    k : int
        Number of vectors to be used to estimate the diagonal; i.e. number of iterations in estimation
    approach : str
        Method used for approximating diagonal. Must be one of {'probing', 'ones', 'random'}

    Returns
    -------
    numpy.ndarray
        Estimate of the diagonal elements of the input matrix

    """

    if isinstance(matrix_arg, np.ndarray):
        A = matrix_arg

        def matrix_arg(v):
            return A.dot(v)

    if k is None:
        k = np.floor(n / 10.0)

    if approach.upper() == "ONES":

        def getv(n, i=None):
            v = np.random.randn(n)
            v[v < 0] = -1.0
            v[v >= 0] = 1.0
            return v

    elif approach.upper() == "RANDOM":

        def getv(n, i=None):
            return np.random.randn(n)

    else:  # if approach == 'Probing':

        def getv(n, i):
            v = np.zeros(n)
            v[i:n:k] = 1.0
            return v

    Mv = np.zeros(n)
    vv = np.zeros(n)

    for i in range(0, k):
        vk = getv(n, i)
        Mv += matrix_arg(vk) * vk
        vv += vk * vk

    d = Mv / vv

    return d


def unique_rows(M):
    """Return unique rows, row indices and inverse indices.

    Parameters
    ----------
    M : array_like
        The input array

    Returns
    -------
    unqM : numpy.ndarray
        Array consisting of the unique rows of the input array
    unqInd : numpy.ndarray of int
        Indices to project from input array to output array
    unqInd : numpy.ndarray of int
        Indices to project from output array to input array

    """
    b = np.ascontiguousarray(M).view(np.dtype((np.void, M.dtype.itemsize * M.shape[1])))
    _, unqInd = np.unique(b, return_index=True)
    _, invInd = np.unique(b, return_inverse=True)
    unqM = M[unqInd]
    return unqM, unqInd, invInd


def eigenvalue_by_power_iteration(
    combo_objfct,
    model,
    n_pw_iter=4,
    fields_list=None,
    seed: RandomSeed | None = None,
):
    r"""Estimate largest eigenvalue in absolute value using power iteration.

    Uses the power iteration approach to estimate the largest eigenvalue in absolute
    value for a single :class:`simpeg.BaseObjectiveFunction` or a combination of
    objective functions stored in a :class:`simpeg.ComboObjectiveFunction`.

    Parameters
    ----------
    combo_objfct : simpeg.BaseObjectiveFunction
        Objective function or a combo objective function
    model : numpy.ndarray
        Current model
    n_pw_iter : int
        Number of power iterations used to estimate the highest eigenvalue
    fields_list : list (optional)
        ``list`` of fields objects for each data misfit term in combo_objfct. If none given,
        they will be evaluated within the function. If combo_objfct mixs data misfit and regularization
        terms, the list should contains simpeg.fields for the data misfit terms and None for the
        regularization term.
    seed : None or :class:`~simpeg.typing.RandomSeed`, optional
        Random seed for the initial random guess of eigenvector. It can either
        be an int, a predefined Numpy random number generator, or any valid
        input to ``numpy.random.default_rng``.

    Returns
    -------
    float
        Estimated value of the highest eigenvalue in absolute value

    Notes
    -----
    After *k* power iterations, the largest eigenvalue in absolute value is
    approximated by the Rayleigh quotient:

    .. math::
        \lambda_k = \frac{\mathbf{x_k^T A x_k}}{\mathbf{x_k^T x_k}}

    where :math:`\mathfb{A}` is our matrix and :math:`\mathfb{x_k}` is computed
    recursively according to:

    .. math::
        \mathbf{x_{k+1}} = \frac{\mathbf{A x_k}}{\| \mathbf{Ax_k} \|}

    The elements of the initial vector :math:`\mathbf{x_0}` are randomly
    selected from a uniform distribution.

    """
    rng = np.random.default_rng(seed=seed)

    # Initial guess for eigen-vector
    x0 = rng.random(size=model.shape)
    x0 = x0 / np.linalg.norm(x0)

    # transform to ComboObjectiveFunction if required
    if getattr(combo_objfct, "objfcts", None) is None:
        combo_objfct = 1.0 * combo_objfct

    # create Field for data misfit if necessary and not provided
    if fields_list is None:
        fields_list = []
        for obj in combo_objfct.objfcts:
            if hasattr(obj, "simulation"):
                fields_list += [obj.simulation.fields(model)]
            else:
                # required to put None to conserve it in the list
                # The idea is that the function can have a mixed of dmis and reg terms
                # (see test)
                fields_list += [None]
    elif not isinstance(fields_list, (list, tuple, np.ndarray)):
        fields_list = [fields_list]

    # Power iteration: estimate eigenvector
    for _ in range(n_pw_iter):
        x1 = 0.0
        for j, (mult, obj) in enumerate(
            zip(combo_objfct.multipliers, combo_objfct.objfcts)
        ):
            if hasattr(obj, "simulation"):  # if data misfit term
                aux = obj.deriv2(model, v=x0, f=fields_list[j])
                if not isinstance(aux, Zero):
                    x1 += mult * aux
            else:
                aux = obj.deriv2(model, v=x0)
                if not isinstance(aux, Zero):
                    x1 += mult * aux
        x0 = x1 / np.linalg.norm(x1)

    # Compute highest eigenvalue from estimated eigenvector
    eigenvalue = 0.0
    for j, (mult, obj) in enumerate(
        zip(combo_objfct.multipliers, combo_objfct.objfcts)
    ):
        if hasattr(obj, "simulation"):  # if data misfit term
            eigenvalue += mult * x0.dot(obj.deriv2(model, v=x0, f=fields_list[j]))
        else:
            eigenvalue += mult * x0.dot(
                obj.deriv2(
                    model,
                    v=x0,
                )
            )

    return eigenvalue


def cartesian2spherical(m):
    r"""
    Converts a set of 3D vectors from Cartesian to spherical coordinates.

    Parameters
    ----------
    m : (n, 3) array_like
        An array whose columns represent the x, y and z components of
        a set of vectors.

    Returns
    -------
    (n, 3) numpy.ndarray
        An array whose columns represent the *a*, *t* and *p* components
        of a set of vectors in spherical coordinates.

    Notes
    -----

    In Cartesian space, the components of each vector are defined as

    .. math::

        \mathbf{v} = (v_x, v_y, v_z)

    In spherical coordinates, vectors are is defined as:

    .. math::

        \mathbf{v^\prime} = (a, t, p)

    where

        - :math:`a` is the amplitude of the vector
        - :math:`t` is the azimuthal angle defined positive from vertical
        - :math:`p` is the radial angle defined positive CCW from Easting

    """

    # nC = int(len(m)/3)

    x = m[:, 0]
    y = m[:, 1]
    z = m[:, 2]

    a = (x**2.0 + y**2.0 + z**2.0) ** 0.5

    t = np.zeros_like(x)
    t[a > 0] = np.arcsin(z[a > 0] / a[a > 0])

    p = np.zeros_like(x)
    p[a > 0] = np.arctan2(y[a > 0], x[a > 0])

    m_atp = np.r_[a, t, p]

    return m_atp


def spherical2cartesian(m):
    r"""
    Converts a set of 3D vectors from spherical to Catesian coordinates.

    Parameters
    ----------
    m : (n, 3) array_like
        An array whose columns represent the *a*, *t* and *p* components of
        a set of vectors in spherical coordinates.

    Returns
    -------
    (n, 3) numpy.ndarray
        An array whose columns represent the *x*, *y* and *z* components
        of the set of vectors in Cartesian.

    Notes
    -----

    In Cartesian space, the components of each vector are defined as

    .. math::

        \mathbf{v} = (v_x, v_y, v_z)

    In spherical coordinates, vectors are is defined as:

    .. math::

        \mathbf{v^\prime} = (a, t, p)

    where

        - :math:`a` is the amplitude of the vector
        - :math:`t` is the azimuthal angle defined positive from vertical
        - :math:`p` is the radial angle defined positive CCW from Easting

    """

    a = m[:, 0] + 1e-8
    t = m[:, 1]
    p = m[:, 2]

    m_xyz = np.r_[a * np.cos(t) * np.cos(p), a * np.cos(t) * np.sin(p), a * np.sin(t)]

    return m_xyz


def dip_azimuth2cartesian(dip, azm):
    """Convert vectors from dip-azimuth to Cartesian

    This function takes the dip and azimuthal angles for a set of vectors and
    converts to Cartesian coordinates. The output is a numpy.ndarray whose
    columns represent the vectors' x, y and z components.

    Parameters
    ----------
    dip : float or 1D numpy.ndarray
        Dip angle in degrees. Values in range [0, 90]
    azm : float or 1D numpy.ndarray
        Asimuthal angle (strike) in degrees. Defined clockwise from Northing. Values is range [0, 360]

    Returns
    -------
    (n, 3) numpy.ndarray
        Numpy array whose columns represent the x, y and z components of the
        vector(s) in Cartesian coordinates

    """

    azm = np.asarray(azm)
    dip = np.asarray(dip)

    # Number of elements
    nC = azm.size

    M = np.zeros((nC, 3))

    # Modify azimuth from North to cartesian-X
    azm_X = (450.0 - np.asarray(azm)) % 360.0
    inc = -np.deg2rad(np.asarray(dip))
    dec = np.deg2rad(azm_X)

    M[:, 0] = np.cos(inc) * np.cos(dec)
    M[:, 1] = np.cos(inc) * np.sin(dec)
    M[:, 2] = np.sin(inc)

    return M


def coterminal(theta):
    r"""
    Compute coterminal angle

    For a set of angles defined in radians, this function outputs their coterminal angles.
    That is, for an angle :math:`\theta` where:

    .. math::

        \theta = 2\pi N + \gamma

    and *N* is an integer, the function returns the value of :math:`\gamma`.
    The coterminal angle :math:`\gamma` is within the range :math:`[-\pi , \pi)`.

    Parameters
    ----------
    theta : array_like
        Input angles

    Returns
    -------
    numpy.ndarray
        Coterminal angles

    """
    coterminal = (theta + np.pi) % (2 * np.pi) - np.pi
    return coterminal


def define_plane_from_points(xyz1, xyz2, xyz3):
    """Compute constants defining a plane from a set of points.

    The equation defining a plane has the form :math:`ax+by+cz+d=0`.
    This utility returns the constants a, b, c and d defining the plane.

    Parameters
    ----------
    xyz1 : (3) numpy.ndarray
        First point needed to define the plane (x1, y1, z1)
    xyz2 : (3) numpy.ndarray
        Second point needed to define the plane (x2, y2, z2)
    xyz3 : (3) numpy.ndarray
        Third point needed to define the plane (x3, y3, z3)

    Returns
    -------
    a : float
    b : float
    c : float
    d : float

    """
    v12 = (xyz2 - xyz1) / np.sqrt(np.sum((xyz2 - xyz1) ** 2, axis=0))
    v13 = (xyz3 - xyz1) / np.sqrt(np.sum((xyz3 - xyz1) ** 2, axis=0))

    a, b, c = np.cross(v12, v13)
    d = -(a * xyz1[0] + b * xyz1[1] + c * xyz1[2])

    return a, b, c, d


################################################
#             DEPRECATED FUNCTIONS
################################################


diagEst = deprecate_function(
    estimate_diagonal, "diagEst", removal_version="0.19.0", error=True
)
uniqueRows = deprecate_function(
    unique_rows, "uniqueRows", removal_version="0.19.0", error=True
)
sdInv = deprecate_function(sdinv, "sdInv", removal_version="0.19.0", error=True)
getSubArray = deprecate_function(
    get_subarray, "getSubArray", removal_version="0.19.0", error=True
)
inv3X3BlockDiagonal = deprecate_function(
    inverse_3x3_block_diagonal,
    "inv3X3BlockDiagonal",
    removal_version="0.19.0",
    error=True,
)
inv2X2BlockDiagonal = deprecate_function(
    inverse_2x2_block_diagonal,
    "inv2X2BlockDiagonal",
    removal_version="0.19.0",
    error=True,
)
makePropertyTensor = deprecate_function(
    make_property_tensor,
    "makePropertyTensor",
    removal_version="0.19.0",
    error=True,
)
invPropertyTensor = deprecate_function(
    inverse_property_tensor,
    "invPropertyTensor",
    removal_version="0.19.0",
    error=True,
)
