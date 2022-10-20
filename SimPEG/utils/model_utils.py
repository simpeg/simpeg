import discretize

from .mat_utils import mkvc, ndgrid, uniqueRows
import numpy as np
from scipy.interpolate import griddata, interp1d
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, interp1d
from scipy.spatial import cKDTree
import scipy.sparse as sp


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
    method : str {'nearest','linear','cubic'}
        Interpolation method for approximating topography at cell's horizontal position.
        Default is 'nearest'.
    fill_value : float
        Defines the elevation for cells outside the horizontal extent of the topography data.
        Default is :py:class:`numpy.nan`.

    Returns
    -------
    numpy.ndarray of int
        Index vector for cells lying below the topography
    """
    if mesh._meshType == "TENSOR":

        if mesh.dim == 3:
            # Check if Topo points are inside of the mesh
            xmin, xmax = mesh.nodes_x.min(), mesh.nodes_x.max()
            xminTopo, xmaxTopo = topo[:, 0].min(), topo[:, 0].max()
            ymin, ymax = mesh.nodes_y.min(), mesh.nodes_y.max()
            yminTopo, ymaxTopo = topo[:, 1].min(), topo[:, 1].max()
            if (
                (xminTopo > xmin)
                or (xmaxTopo < xmax)
                or (yminTopo > ymin)
                or (ymaxTopo < ymax)
            ):
                # If not, use nearest neihbor to extrapolate them
                Ftopo = NearestNDInterpolator(topo[:, :2], topo[:, 2])
                xinds = np.logical_or(xminTopo < mesh.nodes_x, xmaxTopo > mesh.nodes_x)
                yinds = np.logical_or(yminTopo < mesh.nodes_y, ymaxTopo > mesh.nodes_y)
                XYOut = ndgrid(mesh.nodes_x[xinds], mesh.nodes_y[yinds])
                topoOut = Ftopo(XYOut)
                topo = np.vstack((topo, np.c_[XYOut, topoOut]))

            if gridLoc == "CC":
                XY = ndgrid(mesh.vectorCCx, mesh.vectorCCy)
                Zcc = mesh.gridCC[:, 2].reshape(
                    (np.prod(mesh.vnC[:2]), mesh.shape_cells[2]), order="F"
                )
                gridTopo = griddata(
                    topo[:, :2], topo[:, 2], XY, method=method, fill_value=fill_value
                )
                actind = [gridTopo >= Zcc[:, ixy] for ixy in range(mesh.vnC[2])]
                actind = np.hstack(actind)

            elif gridLoc == "N":

                XY = ndgrid(mesh.nodes_x, mesh.nodes_y)
                gridTopo = griddata(
                    topo[:, :2], topo[:, 2], XY, method=method, fill_value=fill_value
                )
                gridTopo = gridTopo.reshape(mesh.vnN[:2], order="F")

                if mesh._meshType not in ["TENSOR", "CYL", "BASETENSOR"]:
                    raise NotImplementedError(
                        "Nodal surface2ind_topo not implemented for {0!s} mesh".format(
                            mesh._meshType
                        )
                    )

                # TODO: this will only work for tensor meshes
                Nz = mesh.nodes_z[1:]
                actind = np.array([False] * mesh.nC).reshape(mesh.vnC, order="F")

                for ii in range(mesh.shape_cells[0]):
                    for jj in range(mesh.shape_cells[1]):
                        actind[ii, jj, :] = [
                            np.all(gridTopo[ii : ii + 2, jj : jj + 2] >= Nz[kk])
                            for kk in range(len(Nz))
                        ]

        elif mesh.dim == 2:
            # Check if Topo points are inside of the mesh
            xmin, xmax = mesh.nodes_x.min(), mesh.nodes_x.max()
            xminTopo, xmaxTopo = topo[:, 0].min(), topo[:, 0].max()
            if (xminTopo > xmin) or (xmaxTopo < xmax):
                fill_value = "extrapolate"

            Ftopo = interp1d(topo[:, 0], topo[:, 1], fill_value=fill_value, kind=method)

            if gridLoc == "CC":
                gridTopo = Ftopo(mesh.gridCC[:, 0])
                actind = mesh.gridCC[:, 1] <= gridTopo

            elif gridLoc == "N":

                gridTopo = Ftopo(mesh.nodes_x)
                if mesh._meshType not in ["TENSOR", "CYL", "BASETENSOR"]:
                    raise NotImplementedError(
                        "Nodal surface2ind_topo not implemented for {0!s} mesh".format(
                            mesh._meshType
                        )
                    )

                # TODO: this will only work for tensor meshes
                Ny = mesh.nodes_y[1:]
                actind = np.array([False] * mesh.nC).reshape(mesh.vnC, order="F")

                for ii in range(mesh.shape_cells[0]):
                    actind[ii, :] = [
                        np.all(gridTopo[ii : ii + 2] > Ny[kk]) for kk in range(len(Ny))
                    ]

        else:
            raise NotImplementedError("surface2ind_topo not implemented for 1D mesh")

    elif mesh._meshType == "TREE":
        if mesh.dim == 3:
            if gridLoc == "CC":
                # Compute unique XY location
                uniqXY = uniqueRows(mesh.gridCC[:, :2])

                if method == "nearest":
                    Ftopo = NearestNDInterpolator(topo[:, :2], topo[:, 2])
                elif method == "linear":
                    # Check if Topo points are inside of the mesh
                    xmin, xmax = mesh.x0[0], mesh.hx.sum() + mesh.x0[0]
                    xminTopo, xmaxTopo = topo[:, 0].min(), topo[:, 0].max()
                    ymin, ymax = mesh.x0[1], mesh.hy.sum() + mesh.x0[1]
                    yminTopo, ymaxTopo = topo[:, 1].min(), topo[:, 1].max()
                    if (
                        (xminTopo > xmin)
                        or (xmaxTopo < xmax)
                        or (yminTopo > ymin)
                        or (ymaxTopo < ymax)
                    ):
                        # If not, use nearest neihbor to extrapolate them
                        Ftopo = NearestNDInterpolator(topo[:, :2], topo[:, 2])
                        xinds = np.logical_or(
                            xminTopo < uniqXY[0][:, 0], xmaxTopo > uniqXY[0][:, 0]
                        )
                        yinds = np.logical_or(
                            yminTopo < uniqXY[0][:, 1], ymaxTopo > uniqXY[0][:, 1]
                        )
                        inds = np.logical_or(xinds, yinds)
                        XYOut = uniqXY[0][inds, :]
                        topoOut = Ftopo(XYOut)
                        topo = np.vstack((topo, np.c_[XYOut, topoOut]))
                    Ftopo = LinearNDInterpolator(topo[:, :2], topo[:, 2])
                else:
                    raise NotImplementedError(
                        "Only nearest and linear method are available for TREE mesh"
                    )
                actind = np.zeros(mesh.nC, dtype="bool")
                npts = uniqXY[0].shape[0]
                for i in range(npts):
                    z = Ftopo(uniqXY[0][i, :])
                    inds = uniqXY[2] == i
                    actind[inds] = mesh.gridCC[inds, 2] < z[0]
            # Need to implement
            elif gridLoc == "N":
                raise NotImplementedError("gridLoc=N is not implemented for TREE mesh")
            else:
                raise Exception("gridLoc must be either CC or N")

        elif mesh.dim == 2:

            if gridLoc == "CC":
                # Compute unique X location
                uniqX = np.unique(
                    mesh.gridCC[:, 0], return_index=True, return_inverse=True
                )

                if method == "nearest":
                    Ftopo = interp1d(topo[:, 0], topo[:, -1], kind="nearest")
                elif method == "linear":
                    # Check if Topo points are inside of the mesh
                    xmin, xmax = mesh.x0[0], mesh.hx.sum() + mesh.x0[0]
                    xminTopo, xmaxTopo = topo[:, 0].min(), topo[:, 0].max()
                    if (xminTopo > xmin) or (xmaxTopo < xmax):
                        # If not, use nearest neihbor to extrapolate them
                        Ftopo = interp1d(topo[:, 0], topo[:, -1], kind="nearest")
                        xinds = np.logical_or(
                            xminTopo < uniqX[0][:, 0], xmaxTopo > uniqX[0][:, 0]
                        )
                        XOut = uniqX[0][xinds, :]
                        topoOut = Ftopo(XOut)
                        topo = np.vstack((topo, np.c_[XOut, topoOut]))
                    Ftopo = interp1d(topo[:, 0], topo[:, -1], kind="nearest")
                else:
                    raise NotImplementedError(
                        "Only nearest and linear method are available for TREE mesh"
                    )
                actind = np.zeros(mesh.nC, dtype="bool")
                npts = uniqX[0].shape[0]
                for i in range(npts):
                    z = Ftopo(uniqX[0][i])
                    inds = uniqX[2] == i
                    actind[inds] = mesh.gridCC[inds, 1] < z
            # Need to implement
            elif gridLoc == "N":
                raise NotImplementedError("gridLoc=N is not implemented for TREE mesh")
            else:
                raise Exception("gridLoc must be either CC or N")

        else:
            raise NotImplementedError("surface2ind_topo not implemented for 1D mesh")
    else:
        raise NotImplementedError(f"{type(mesh)} is not supported.")
    return mkvc(actind)


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

    grid_x, grid_y = np.meshgrid(mesh.vectorCCx, mesh.vectorCCy)
    zInterp = mkvc(
        griddata(topo[:, :2], topo[:, 2], (grid_x, grid_y), method="nearest")
    )

    # Get nearest cells
    r, inds = tree.query(np.c_[mkvc(grid_x), mkvc(grid_y), zInterp])
    inds = np.unique(inds)

    # Extract vertical neighbors from Gradz operator
    Dz = mesh._cellGradzStencil
    Iz, Jz, _ = sp.find(Dz)
    jz = np.sort(Jz[np.argsort(Iz)].reshape((int(Iz.shape[0] / 2), 2)), axis=1)
    for ii in range(index):

        members = ismember(inds, jz[:, 1])
        inds = np.squeeze(jz[members, 0])

    actv[inds] = True

    return actv


def depth_weighting(mesh, reference_locs, indActive=None, exponent=2.0, threshold=None):
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
    indActive : (mesh.n_cells) numpy.ndarray of bool, optional
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

    if indActive is not None:
        wz = wz[indActive]

    return wz / np.nanmax(wz)
