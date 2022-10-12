from __future__ import print_function
import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
from .mat_utils import mkvc
from scipy.spatial import Delaunay
from .code_utils import deprecate_function
from discretize.base import BaseMesh


def add_block(cell_centers, model, p0, p1, prop_value):
    """Add a homogeneous block to an existing cell centered model

    Parameters
    ----------
    cell_centers : (n_cells, dim) numpy.ndarray or discretize.base.BaseMesh
        A mesh or its gridded cell center locations
    model : (n_cells) numpy.ndarray
        Cell-centered model. Currently, this is only implemented for isotropic properties.
    p0 : (dim) numpy.ndarray
        Bottom southwest corner of the block
    p1 : (dim) numpy.ndarray
        Top northeast corner of the block
    prop_value : float
        Physical property value assigned to the block

    Returns
    -------
    (n_cells) numpy.ndarray
        The updated cell-centered model which includes the block
    """
    if isinstance(cell_centers, BaseMesh):
        cell_centers = cell_centers.cell_centers

    ind = get_indices_block(p0, p1, cell_centers)
    new_model = model.copy()
    new_model[ind] = prop_value
    return new_model


def get_indices_block(p0, p1, cell_centers):
    """Get indices for cells whose centers lie inside specified block

    Parameters
    ----------
    p0 : (dim) numpy.ndarray
        Bottom southwest corner of the block
    p1 : (dim) numpy.ndarray
        Top northeast corner of the block
    cell_centers : (n_cells, dim) numpy.ndarray or discretize.base.BaseMesh
        A mesh or its gridded cell center locations

    Returns
    -------
    tuple of int
        Indices of the cells whose center lie within the specified block

    """

    if isinstance(cell_centers, BaseMesh):
        cell_centers = cell_centers.cell_centers

    # Validation: p0 and p1 live in the same dimensional space
    assert len(p0) == len(p1), "Dimension mismatch. len(p0) != len(p1)"

    # Validation: mesh and points live in the same dimensional space
    dimMesh = np.size(cell_centers[0, :])
    assert len(p0) == dimMesh, "Dimension mismatch. len(p0) != dimMesh"

    for ii in range(len(p0)):
        p0[ii], p1[ii] = np.min([p0[ii], p1[ii]]), np.max([p0[ii], p1[ii]])

    if dimMesh == 1:
        # Define the reference points
        x1 = p0[0]
        x2 = p1[0]

        indX = (x1 <= cell_centers[:, 0]) & (cell_centers[:, 0] <= x2)
        ind = np.where(indX)

    elif dimMesh == 2:
        # Define the reference points
        x1 = p0[0]
        y1 = p0[1]

        x2 = p1[0]
        y2 = p1[1]

        indX = (x1 <= cell_centers[:, 0]) & (cell_centers[:, 0] <= x2)
        indY = (y1 <= cell_centers[:, 1]) & (cell_centers[:, 1] <= y2)

        ind = np.where(indX & indY)

    elif dimMesh == 3:
        # Define the points
        x1 = p0[0]
        y1 = p0[1]
        z1 = p0[2]

        x2 = p1[0]
        y2 = p1[1]
        z2 = p1[2]

        indX = (x1 <= cell_centers[:, 0]) & (cell_centers[:, 0] <= x2)
        indY = (y1 <= cell_centers[:, 1]) & (cell_centers[:, 1] <= y2)
        indZ = (z1 <= cell_centers[:, 2]) & (cell_centers[:, 2] <= z2)

        ind = np.where(indX & indY & indZ)

    # Return a tuple
    return ind


def create_block_in_wholespace(
    cell_centers, p0, p1, background_value=0.0, block_value=1.0
):
    """Construct cell-centered model comprised of a block in a wholespace.

    Parameters
    ----------
    cell_centers : (n_cells, dim) numpy.ndarray or discretize.base.BaseMesh
        A mesh or its gridded cell center locations
    p0 : (dim) numpy.ndarray
        Bottom southwest corner of the block
    p1 : (dim) numpy.ndarray
        Top northeast corner of the block
    background_value : float, optional
        Background physical property value.
    block_value : float, optional
        Block physical property value

    Returns
    -------
    (n_cells) numpy.ndarray
        Physical property model defined at the cell centers
    """

    if isinstance(cell_centers, BaseMesh):
        cell_centers = cell_centers.cell_centers

    # Used to use a single input 'vals' for background and block
    try:
        background_value, block_value = background_value
    except TypeError:
        pass

    sigma = np.zeros(cell_centers.shape[0]) + background_value
    ind = getIndicesBlock(p0, p1, cell_centers)

    sigma[ind] = block_value

    return mkvc(sigma)


def create_ellipse_in_wholespace(
    cell_centers, center=None, anisotropy=None, slope=10.0, theta=0.0
):
    """Construct cell-centered model comprised of an ellipsoid in a wholespace.

    Parameters
    ----------
    cell_centers : (n_cells, dim) numpy.ndarray or discretize.base.BaseMesh
        A mesh or its gridded cell center locations
    center : (dim) numpy.ndarray, optional
        Center of the ellipsoid, default is [0, 0, 0]
    anisotropy : (dim) numpy.ndarray, optional
        Anisotropy, default is an isotropic elipse ([1, 1, 1]).
    slope : float, optional
        Slope
    theta : float, optional
        Angle

    Returns
    -------
    (n_cells) numpy.ndarray
        Physical property model defined at the cell centers
    """

    if isinstance(cell_centers, BaseMesh):
        cell_centers = cell_centers.cell_centers

    if center is None:
        center = [0, 0, 0]
    if anisotropy is None:
        anisotropy = [1, 1, 1]
    G = cell_centers.copy()
    dim = cell_centers.shape[1]
    for i in range(dim):
        G[:, i] = G[:, i] - center[i]

    theta = -theta * np.pi / 180
    M = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1.0],
        ]
    )
    M = M[:dim, :dim]
    G = M.dot(G.T).T

    for i in range(dim):
        G[:, i] = G[:, i] / anisotropy[i] * 2.0

    D = np.sqrt(np.sum(G ** 2, axis=1))
    return -np.arctan((D - 1) * slope) * (2.0 / np.pi) / 2.0 + 0.5


def get_indices_sphere(center, radius, cell_centers):
    """Get indices for cells whose centers lie inside a sphere

    Parameters
    ----------
    center : (dim) numpy.ndarray
        Location of the center of the sphere
    radius : float
        Radius of the sphere
    cell_centers : (n_cells, dim) numpy.ndarray or discretize.base.BaseMesh
        A mesh or its gridded cell center locations

    Returns
    -------
    tuple of int
        Indices of the cells whose center lie within the specified sphere

    """

    if isinstance(cell_centers, BaseMesh):
        cell_centers = cell_centers.cell_centers

    # Validation: mesh and point (p0) live in the same dimensional space
    dimMesh = np.size(cell_centers[0, :])
    assert len(center) == dimMesh, "Dimension mismatch. len(p0) != dimMesh"

    if dimMesh == 1:
        # Define the reference points

        ind = np.abs(center[0] - cell_centers[:, 0]) < radius

    elif dimMesh == 2:
        # Define the reference points

        ind = (
            np.sqrt(
                (center[0] - cell_centers[:, 0]) ** 2
                + (center[1] - cell_centers[:, 1]) ** 2
            )
            < radius
        )

    elif dimMesh == 3:
        # Define the points
        ind = (
            np.sqrt(
                (center[0] - cell_centers[:, 0]) ** 2
                + (center[1] - cell_centers[:, 1]) ** 2
                + (center[2] - cell_centers[:, 2]) ** 2
            )
            < radius
        )

    # Return a tuple
    return ind


def create_2_layer_model(cell_centers, depth, top_value=1.0, bottom_value=0.0):
    """Create a basic two layered model

    This function creates a physical property model consisting of 2 layers.

    Parameters
    ----------
    cell_centers : (n_cells, dim) numpy.ndarray or discretize.base.BaseMesh
        A mesh or its gridded cell center locations
    depth : float
        Depth defining the interface between layer 1 and layer 2
    top_value : float, optional
        Physical property value for the top layer
    bottom_value : float, optional
        Physical property value for the bottom layer

    Returns
    -------
    (n_cells) numpy.ndarray
        Cell-centered physical property model

    """

    if isinstance(cell_centers, BaseMesh):
        cell_centers = cell_centers.cell_centers

    try:
        top_value, bottom_value = top_value
    except TypeError:
        pass

    sigma = np.zeros(cell_centers.shape[0]) + bottom_value

    dim = np.size(cell_centers[0, :])

    p0 = np.zeros(dim)
    p1 = np.zeros(dim)

    # Identify 1st cell centered reference point
    p0[0] = cell_centers[0, 0]
    if dim > 1:
        p0[1] = cell_centers[0, 1]
    if dim > 2:
        p0[2] = cell_centers[0, 2]

    # Identify the last cell-centered reference point
    p1[0] = cell_centers[-1, 0]
    if dim > 1:
        p1[1] = cell_centers[-1, 1]
    if dim > 2:
        p1[2] = cell_centers[-1, 2]

    # The depth is always defined on the last one.
    p1[len(p1) - 1] -= depth

    ind = getIndicesBlock(p0, p1, cell_centers)

    sigma[ind] = top_value

    return mkvc(sigma)


def create_from_function(cell_centers, fun_handle):
    """Define physical property model from scalar analytic function.

    For a function handle that defines a scalar physical property as function of
    location, **create_from_function** outputs the discrete representation
    of the physical property distribution on the mesh provided.

    Parameters
    ----------
    cell_centers : (n_cells, dim) numpy.ndarray or discretize.base.BaseMesh
        A mesh or its gridded cell center locations.
    fun_handle : function
        A function which defines a scalar physical property value as a function
        of location. The input argument for the location must be an (*n*, *dim*)
        ``numpy.ndarray``.

    Returns
    -------
    (n_cells) numpy.ndarray
        Cell-centered physical property model for the mesh.
    """

    if isinstance(cell_centers, BaseMesh):
        cell_centers = cell_centers.cell_centers

    dim = np.size(cell_centers[0, :])
    CC = [cell_centers[:, 0]]
    if dim > 1:
        CC.append(cell_centers[:, 1])
    if dim > 2:
        CC.append(cell_centers[:, 2])

    sigma = fun_handle(*CC)

    return mkvc(sigma)


def create_layers_model(cell_centers, layer_tops, layer_values):
    """Create physical property model consisting of a set of infinite horizontal layers.

    Parameters
    ----------
    cell_centers : (n_cells, dim) numpy.ndarray or discretize.base.BaseMesh
        A mesh or its gridded cell center locations
    layer_tops : (n_layer) numpy.ndarray
        Elevation values (z +ve up) for the top of each layer. Layers are defined from top to bottom.
        The first value can be very large if the top layer (e.g. air) extends to infinity.
    layer_values : (n_layer) numpy.ndarray
        Physical property value for each layer from top to bottom.

    Returns
    -------
    (n_cells) numpy.ndarray
        Cell-centered physical property model for the mesh.
    """

    if isinstance(cell_centers, BaseMesh):
        cell_centers = cell_centers.cell_centers

    descending = np.linalg.norm(sorted(layer_tops, reverse=True) - layer_tops) < 1e-20

    # TODO: put an error check to make sure that there is an ordering... needs to work with inf elts
    # assert ascending or descending, "Layers must be listed in either ascending or descending order"

    # start from bottom up
    if not descending:
        zprop = np.hstack([mkvc(layer_tops, 2), mkvc(layer_values, 2)])
        zprop.sort(axis=0)
        layer_tops, layer_values = zprop[::-1, 0], zprop[::-1, 1]

    # put in vector form
    layer_tops, layer_values = mkvc(layer_tops), mkvc(layer_values)

    # initialize with bottom layer
    dim = cell_centers.shape[1]
    if dim == 3:
        z = cell_centers[:, 2]
    elif dim == 2:
        z = cell_centers[:, 1]
    elif dim == 1:
        z = cell_centers[:, 0]

    model = np.zeros(cell_centers.shape[0])

    for i, top in enumerate(layer_tops):
        zind = z <= top
        model[zind] = layer_values[i]

    return model


def create_random_model(shape, seed=1000, anisotropy=None, its=100, bounds=None):
    """Create random model by convolving a kernel with a uniformly distributed random model.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the model. Can define a vector of size (n_cells) or define the dimensions of a tensor
    seed : int, optional
        If not None, sets the seed for the random uniform model that is convolved with the kernel.
    anisotropy : numpy.ndarray
        this is the (*3*, *n*) blurring kernel that is used.
    its : int
        Number of smoothing iterations after convolutions
    bounds : list of float
        Lower and upper bound for the model values

    Returns
    -------
    numpy.ndarray
        Physical property model


    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> from SimPEG.utils.model_builder import create_random_model
    >>> m = create_random_model((50,50), bounds=[-4,0])
    >>> plt.colorbar(plt.imshow(m))
    >>> plt.title('A very cool, yet completely random model.')
    >>> plt.show()

    """
    if bounds is None:
        bounds = [0, 1]

    if seed is not None:
        np.random.seed(seed)
        print("Using a seed of: ", seed)

    if isinstance(shape, (int, float)):
        shape = (shape,)  # make it a tuple for consistency

    mr = np.random.rand(*shape)
    if anisotropy is None:
        if len(shape) == 1:
            smth = np.array([1, 10.0, 1], dtype=float)
        elif len(shape) == 2:
            smth = np.array([[1, 7, 1], [2, 10, 2], [1, 7, 1]], dtype=float)
        elif len(shape) == 3:
            kernal = np.array([1, 4, 1], dtype=float).reshape((1, 3))
            smth = np.array(
                sp.kron(sp.kron(kernal, kernal.T).todense()[:], kernal).todense()
            ).reshape((3, 3, 3))
    else:
        assert len(anisotropy.shape) is len(shape), "Anisotropy must be the same shape."
        smth = np.array(anisotropy, dtype=float)

    smth = smth / smth.sum()  # normalize
    mi = mr
    for i in range(its):
        mi = ndi.convolve(mi, smth)

    # scale the model to live between the bounds.
    mi = (mi - mi.min()) / (mi.max() - mi.min())  # scaled between 0 and 1
    mi = mi * (bounds[1] - bounds[0]) + bounds[0]

    return mi


def get_indices_polygon(mesh, pts):
    """Get indices for cells whose centers lie within the convex hull of a set of points.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        Discretize mesh
    pts : (n, dim) numpy.ndarray
        Set of points defining the convex hull

    Returns
    -------
    tuple of int
        Indices of the cells whose center lies within convex hull

    """
    if mesh.dim == 1:
        assert "Only works for a mesh greater than 1-dimension"
    elif mesh.dim == 2:
        assert ~(pts.shape[1] != 2), "Please input (*,2) array"
    elif mesh.dim == 3:
        assert ~(pts.shape[1] != 3), "Please input (*,3) array"
    hull = Delaunay(pts)
    inds = hull.find_simplex(mesh.cell_centers) >= 0
    return inds


################################################
#             DEPRECATED FUNCTIONS
################################################


addBlock = deprecate_function(add_block, "addBlock", removal_version="0.16.0")

getIndicesBlock = deprecate_function(
    get_indices_block, "getIndicesBlock", removal_version="0.16.0"
)

defineBlock = deprecate_function(
    create_block_in_wholespace, "defineBlock", removal_version="0.16.0"
)

defineEllipse = deprecate_function(
    create_ellipse_in_wholespace, "defineEllipse", removal_version="0.16.0"
)

getIndicesSphere = deprecate_function(
    get_indices_sphere, "getIndicesSphere", removal_version="0.16.0"
)

defineTwoLayers = deprecate_function(
    create_2_layer_model, "defineTwoLayers", removal_version="0.16.0"
)

layeredModel = deprecate_function(
    create_layers_model, "layeredModel", removal_version="0.16.0"
)

randomModel = deprecate_function(
    create_random_model, "randomModel", removal_version="0.16.0"
)

polygonInd = deprecate_function(
    get_indices_polygon, "polygonInd", removal_version="0.16.0"
)

scalarConductivity = deprecate_function(
    create_from_function, "scalarConductivity", removal_version="0.16.0"
)
