import warnings
from typing import Literal, Optional

import discretize
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .mat_utils import mkvc

try:
    import numba
    from numba import njit, prange
except ImportError:
    numba = None

    # Define dummy njit decorator
    def njit(*args, **kwargs):
        return lambda f: f

    # Define dummy prange function
    prange = range


def surface_layer_index(mesh, topo, index=0):
    """Find ith layer of cells below topo for a tensor mesh.

    Parameters
    ----------
    mesh : discretize.TensorMesh
        Input mesh
    topo : (n, 3) numpy.ndarray
        Topography data as a numpy array with columns [x,y,z]; can use [x,z] for 2D meshes.
        Topography data can be unstructured.
    index : int
        How many layers below the surface you want to find

    Returns
    -------
    numpy.ndarray of int
        Index vector for layer of cells
    """

    actv = np.zeros(mesh.nC, dtype="bool")
    # Get cdkTree to find top layer
    tree = cKDTree(mesh.gridCC)

    def ismember(a, b):
        bind = {}
        for i, elt in enumerate(b):
            if elt not in bind:
                bind[elt] = i
        return np.vstack([bind.get(itm, None) for itm in a])

    grid_x, grid_y = np.meshgrid(mesh.cell_centers_x, mesh.cell_centers_y)
    zInterp = mkvc(
        griddata(topo[:, :2], topo[:, 2], (grid_x, grid_y), method="nearest")
    )

    # Get nearest cells
    r, inds = tree.query(np.c_[mkvc(grid_x), mkvc(grid_y), zInterp])
    inds = np.unique(inds)

    # Extract vertical neighbors from Gradz operator
    Dz = mesh.stencil_cell_gradient_z
    Iz, Jz, _ = sp.find(Dz)
    jz = np.sort(Jz[np.argsort(Iz)].reshape((int(Iz.shape[0] / 2), 2)), axis=1)
    for _ in range(index):
        members = ismember(inds, jz[:, 1])
        inds = np.squeeze(jz[members, 0])

    actv[inds] = True

    return actv


def depth_weighting(
    mesh, reference_locs, active_cells=None, exponent=2.0, threshold=None, **kwargs
):
    r"""
    Construct diagonal elements of a depth weighting matrix

    Builds the model weights following the depth weighting strategy, a method
    to generate weights based on the vertical distance between mesh cell
    centers and some reference location(s).
    Use these weights in regularizations to counteract the natural decay of
    potential field data with depth.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        Discretized model space.
    reference_locs : float or (n, ndim) numpy.ndarray
        Reference location for the depth weighting.
        It can be a ``float``, which value is the vertical component for
        the reference location.
        Or it can be a 2d array, with multiple reference locations, where each
        row should contain the coordinates of a single location point in the
        following order: _x_, _y_, _z_ (for 3D meshes) or _x_, _z_ (for 2D
        meshes).
        The vertical coordinate of the reference location for each cell in the
        mesh will be obtained by the closest point in ``reference_locs`` using
        only their horizontal coordinates.
    active_cells : (mesh.n_cells) numpy.ndarray of bool, optional
        Index vector for the active cells on the mesh.
        If ``None``, every cell will be assumed to be active.
    exponent : float, optional
        Exponent parameter for depth weighting.
        The exponent should match the natural decay power of the potential
        field. For example, for gravity acceleration, set it to 2; for magnetic
        fields, to 3.
    threshold : float or None, optional
        Threshold parameters used in the depth weighting.
        If ``None``, it will be set to half of the smallest cell width.

    Returns
    -------
    (n_active) numpy.ndarray
        Normalized depth weights for the mesh at every active cell as
        a 1d-array.

    Notes
    -----
    Each diagonal term of the matrix is defined as:

    .. math::

        w(z) = \frac{1}{(|z - z_0| + \epsilon) ^ {\nu / 2}}

    where :math:`z` is the vertical coordinate of the mesh cell centers,
    :math:`z_0` is the vertical coordinate of the reference location,
    :math:`\nu` is the _exponent_,
    and :math:`\epsilon` is a given _threshold_.

    The depth weights array is finally normalized by dividing for its maximum
    value.
    """

    if (key := "indActive") in kwargs:
        raise TypeError(
            f"'{key}' argument has been removed. " "Please use 'active_cells' instead."
        )

    # Default threshold value
    if threshold is None:
        threshold = 0.5 * mesh.h_gridded.min()

    reference_locs = np.asarray(reference_locs)

    # Calculate depth from receiver locations, delta_z
    # reference_locs is a scalar
    if reference_locs.ndim < 2:
        delta_z = np.abs(mesh.cell_centers[:, -1] - reference_locs)

    # reference_locs is a 2d array
    elif reference_locs.ndim == 2:
        tree = cKDTree(reference_locs[:, :-1])
        _, ind = tree.query(mesh.cell_centers[:, :-1])
        delta_z = np.abs(mesh.cell_centers[:, -1] - reference_locs[ind, -1])

    else:
        raise ValueError("reference_locs must be either a scalar or 2d array!")

    wz = (delta_z + threshold) ** (-0.5 * exponent)

    if active_cells is not None:
        wz = wz[active_cells]

    return wz / np.nanmax(wz)


@njit(parallel=True)
def _distance_weighting_numba(
    cell_centers: np.ndarray,
    reference_locs: np.ndarray,
    threshold: float,
    exponent: float = 2.0,
) -> np.ndarray:
    r"""
    distance weighting kernel in numba.

    If numba is not installed, this will work as a regular for loop.

    Parameters
    ----------
    cell_centers : np.ndarray
        cell centers of the mesh.
    reference_locs : (n, ndim) numpy.ndarray
        The coordinate of the reference location, usually the receiver locations,
        for the distance weighting.
        A 2d array, with multiple reference locations, where each row should
        contain the coordinates of a single location point in the following
        order: _x_, _y_, _z_ (for 3D meshes) or _x_, _z_ (for 2D meshes).
    threshold : float
        Threshold parameters used in the distance weighting.
    exponent : float, optional
        Exponent parameter for distance weighting.
        The exponent should match the natural decay power of the potential
        field. For example, for gravity acceleration, set it to 2; for magnetic
        fields, to 3.

    Returns
    -------
    (n_active) numpy.ndarray
        Normalized distance weights for the mesh at every active cell as
        a 1d-array.
    """
    n_active_cells = cell_centers.shape[0]
    n_reference_locs = len(reference_locs)

    distance_weights = np.zeros(n_active_cells)
    for j in prange(n_active_cells):
        cell_center = cell_centers[j]
        for i in range(n_reference_locs):
            reference_loc = reference_locs[i]
            distance = np.sqrt(((cell_center - reference_loc) ** 2).sum())
            distance_weights[j] += (distance + threshold) ** (-2 * exponent)

    distance_weights = np.sqrt(distance_weights)
    distance_weights /= np.nanmax(distance_weights)
    return distance_weights


def distance_weighting(
    mesh: discretize.base.BaseMesh,
    reference_locs: np.ndarray,
    active_cells: Optional[np.ndarray] = None,
    exponent: Optional[float] = 2.0,
    threshold: Optional[float] = None,
    engine: Literal["numba", "scipy"] = "numba",
    cdist_opts: Optional[dict] = None,
):
    r"""
    Construct diagonal elements of a distance weighting matrix

    Builds the model weights following the distance weighting strategy, a method
    to generate weights based on the distance between mesh cell centers and some
    reference location(s).
    Use these weights in regularizations to counteract the natural decay of
    potential field data with distance.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        Discretized model space.
    reference_locs : (n, ndim) numpy.ndarray
        The coordinate of the reference location, usually the receiver locations,
        for the distance weighting.
        A 2d array, with multiple reference locations, where each row should
        contain the coordinates of a single location point in the following
        order: _x_, _y_, _z_ (for 3D meshes) or _x_, _z_ (for 2D meshes).
    active_cells : (mesh.n_cells) numpy.ndarray of bool, optional
        Index vector for the active cells on the mesh.
        If ``None``, every cell will be assumed to be active.
    exponent : float or None, optional
        Exponent parameter for distance weighting.
        The exponent should match the natural decay power of the potential
        field. For example, for gravity acceleration, set it to 2; for magnetic
        fields, to 3.
    threshold : float or None, optional
        Threshold parameters used in the distance weighting.
        If ``None``, it will be set to half of the smallest cell width.
    engine: str, 'numba' or 'scipy'
        Pick between a ``scipy.spatial.distance.cdist`` computation (memory
        intensive) or `for` loop implementation, parallelized with numba if
        available. Default to ``"numba"``.
    cdist_opts: dict, optional
        Only valid with ``engine=="scipy"``. Options to pass to
        ``scipy.spatial.distance.cdist``. Default to None.

    Returns
    -------
    (n_active) numpy.ndarray
        Normalized distance weights for the mesh at every active cell as
        a 1d-array.
    """

    active_cells = (
        np.ones(mesh.n_cells, dtype=bool) if active_cells is None else active_cells
    )

    # Default threshold value
    if threshold is None:
        threshold = 0.5 * mesh.h_gridded.min()

    reference_locs = np.atleast_2d(reference_locs)
    cell_centers = mesh.cell_centers[active_cells]

    # address 1D case
    if mesh.dim == 1:
        cell_centers = cell_centers.reshape(-1, 1)
        reference_locs = reference_locs.reshape(-1, 1)

    if reference_locs.shape[1] != mesh.dim:
        raise ValueError(
            f"Invalid 'reference_locs' with shape '{reference_locs.shape}'. "
            "The number of columns of the reference_locs array should match "
            f"the dimensions of the mesh ({mesh.dim})."
        )

    if engine == "numba" and cdist_opts is not None:
        raise TypeError(
            "The `cdist_opts` is valid only when engine is 'scipy'."
            "The current engine is 'numba'."
        )

    if engine == "numba":
        if numba is None:
            warnings.warn(
                "Numba is not installed. Distance computations will be slower.",
                stacklevel=2,
            )
        distance_weights = _distance_weighting_numba(
            cell_centers,
            reference_locs,
            exponent=exponent,
            threshold=threshold,
        )

    elif engine == "scipy":
        warnings.warn(
            "``scipy.spatial.distance.cdist`` computations can be memory intensive. "
            "Consider switching to `engine='numba'` "
            "if you run into memory overflow issues.",
            stacklevel=2,
        )
        cdist_opts = cdist_opts or dict()
        distance = cdist(cell_centers, reference_locs, **cdist_opts)

        distance_weights = (((distance + threshold) ** exponent) ** -2).sum(axis=1)

        distance_weights = distance_weights**0.5
        distance_weights /= np.nanmax(distance_weights)

    else:
        raise ValueError(
            f"Invalid engine '{engine}'. Engine should be either 'scipy' or 'numba'."
        )

    return distance_weights
