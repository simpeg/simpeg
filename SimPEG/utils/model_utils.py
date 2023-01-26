from .mat_utils import mkvc
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import scipy.sparse as sp
from discretize.utils import active_from_xyz
import warnings


def surface2ind_topo(mesh, topo, gridLoc="CC", method="nearest", fill_value=np.nan):
    """Get indices of active cells from topography.

    For a mesh and surface topography, this function returns the indices of cells
    lying below the discretized surface topography.

    Parameters
    ----------
    mesh : discretize.TensorMesh or discretize.TreeMesh
        Mesh on which you want to identify active cells
    topo : (n, 3) numpy.ndarray
        Topography data as a ``numpyndarray`` with columns [x,y,z]; can use [x,z] for 2D meshes.
        Topography data can be unstructured.
    gridLoc : str {'CC', 'N'}
        If 'CC', all cells whose centers are below the topography are active cells.
        If 'N', then cells must lie entirely below the topography in order to be active cells.
    method : str {'nearest','linear'}
        Interpolation method for approximating topography at cell's horizontal position.
        Default is 'nearest'.
    fill_value : float
        Defines the elevation for cells outside the horizontal extent of the topography data.
        Default is :py:class:`numpy.nan`.

    Returns
    -------
    (n_cells) numpy.ndarray of bool
        1D mask array of *bool* for the active cells below xyz.
    """
    warnings.warn(
        "The surface2ind_topo function has been deprecated, please use active_from_xyz. "
        "This will be removed in SimPEG 0.19.0",
        FutureWarning,
    )

    return active_from_xyz(mesh, topo, gridLoc, method)


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
    """A simple depth weighting function

    This function is a simple form of depth weighting based off of the vertical distance
    of mesh cell centers from the reference location(s).

    This is commonly used to counteract the natural decay of potential field data at
    depth.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        discretize model space.
    reference_locs : float or (n, dim) numpy.ndarray
        the reference values for top of the points
    active_cells : (mesh.n_cells) numpy.ndarray of bool, optional
        index vector for the active cells on the mesh.
        A value of ``None`` implies every cell is active.
    exponent : float, optional
        exponent parameter for depth weighting.
    threshold : float, optional
        The default value is half of the smallest cell width.

    Returns
    -------
    (n_active) numpy.ndarray
        Normalized depth weights for the mesh, at every active cell.

    Notes
    -----
    When *reference_locs* is a single value the function is defined as,

    >>> wz = (np.abs(mesh.cell_centers[:, -1] - reference_locs) + threshold) ** (-0.5 * exponent)

    When *reference_locs* is an array of values, the difference is between the
    nearest point (of first two dimensions) in *reference_locs*.
    'exponent' and 'threshold' are two adjustable parameters.
    """

    if "indActive" in kwargs:
        warnings.warn(
            "The indActive keyword argument has been deprecated, please use grid_loc. "
            "This will be removed in SimPEG 0.19.0",
            FutureWarning,
        )
        active_cells = kwargs["indActive"]

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
