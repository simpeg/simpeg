from .mat_utils import mkvc, ndgrid, uniqueRows
import numpy as np
from scipy.interpolate import griddata, interp1d
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, interp1d
from scipy.spatial import cKDTree
import scipy.sparse as sp


def surface2ind_topo(mesh, topo, gridLoc="CC", method="nearest", fill_value=np.nan):
    """
    Get active indices from topography

    Parameters
    ----------

    :param TensorMesh mesh: TensorMesh object on which to discretize the topography
    :param numpy.ndarray topo: [X,Y,Z] topographic data
    :param str gridLoc: 'CC' or 'N'. Default is 'CC'.
                        Discretize the topography
                        on cells-center 'CC' or nodes 'N'
    :param str method: 'nearest' or 'linear' or 'cubic'. Default is 'nearest'.
                       Interpolation method for the topographic data
    :param float fill_value: default is np.nan. Filling value for extrapolation

    Returns
    -------

    :param numpy.ndarray actind: index vector for the active cells on the mesh
                               below the topography
    """
    if mesh._meshType == "TENSOR":

        if mesh.dim == 3:
            # Check if Topo points are inside of the mesh
            xmin, xmax = mesh.vectorNx.min(), mesh.vectorNx.max()
            xminTopo, xmaxTopo = topo[:, 0].min(), topo[:, 0].max()
            ymin, ymax = mesh.vectorNy.min(), mesh.vectorNy.max()
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
                    xminTopo < mesh.vectorNx, xmaxTopo > mesh.vectorNx
                )
                yinds = np.logical_or(
                    yminTopo < mesh.vectorNy, ymaxTopo > mesh.vectorNy
                )
                XYOut = ndgrid(mesh.vectorNx[xinds], mesh.vectorNy[yinds])
                topoOut = Ftopo(XYOut)
                topo = np.vstack((topo, np.c_[XYOut, topoOut]))

            if gridLoc == "CC":
                XY = ndgrid(mesh.vectorCCx, mesh.vectorCCy)
                Zcc = mesh.gridCC[:, 2].reshape(
                    (np.prod(mesh.vnC[:2]), mesh.nCz), order="F"
                )
                gridTopo = griddata(
                    topo[:, :2], topo[:, 2], XY, method=method, fill_value=fill_value
                )
                actind = [
                    gridTopo >= Zcc[:, ixy] for ixy in range(np.prod(mesh.vnC[2]))
                ]
                actind = np.hstack(actind)

            elif gridLoc == "N":

                XY = ndgrid(mesh.vectorNx, mesh.vectorNy)
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
                Nz = mesh.vectorNz[1:]
                actind = np.array([False] * mesh.nC).reshape(mesh.vnC, order="F")

                for ii in range(mesh.nCx):
                    for jj in range(mesh.nCy):
                        actind[ii, jj, :] = [
                            np.all(gridTopo[ii : ii + 2, jj : jj + 2] >= Nz[kk])
                            for kk in range(len(Nz))
                        ]

        elif mesh.dim == 2:
            # Check if Topo points are inside of the mesh
            xmin, xmax = mesh.vectorNx.min(), mesh.vectorNx.max()
            xminTopo, xmaxTopo = topo[:, 0].min(), topo[:, 0].max()
            if (xminTopo > xmin) or (xmaxTopo < xmax):
                fill_value = "extrapolate"

            Ftopo = interp1d(topo[:, 0], topo[:, 1], fill_value=fill_value, kind=method)

            if gridLoc == "CC":
                gridTopo = Ftopo(mesh.gridCC[:, 0])
                actind = mesh.gridCC[:, 1] <= gridTopo

            elif gridLoc == "N":

                gridTopo = Ftopo(mesh.vectorNx)
                if mesh._meshType not in ["TENSOR", "CYL", "BASETENSOR"]:
                    raise NotImplementedError(
                        "Nodal surface2ind_topo not implemented for {0!s} mesh".format(
                            mesh._meshType
                        )
                    )

                # TODO: this will only work for tensor meshes
                Ny = mesh.vectorNy[1:]
                actind = np.array([False] * mesh.nC).reshape(mesh.vnC, order="F")

                for ii in range(mesh.nCx):
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

    return mkvc(actind)


def surface_layer_index(mesh, topo, index=0):
    """
        Find the ith layer below topo
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


def tile_locations(
        locations, n_tiles, minimize=True, method="kmeans",
        bounding_box=False, count=False, unique_id=False
):
    """
        Function to tile an survey points into smaller square subsets of points

        :param numpy.ndarray locations: n x 2 array of locations [x,y]
        :param integer n_tiles: number of tiles (for 'cluster'), or number of
            refinement steps ('other')
        :param Bool minimize: shrink tile sizes to minimum
        :param string method: set to 'kmeans' to use better quality clustering, or anything
            else to use more memory efficient method for large problems
        :param bounding_box: bool [False]
            Return the SW and NE corners of each tile.
        :param count: bool [False]
            Return the number of locations in each tile.
        :param unique_id: bool [False]
            Return the unique identifiers of all tiles.

        RETURNS:
        :param list: Return a list of arrays with the for the SW and NE
                            limits of each tiles
        :param integer binCount: Number of points in each tile
        :param list labels: Cluster index of each point n=0:(nTargetTiles-1)
        :param numpy.array tile_numbers: Vector of tile numbers for each count in binCount

        NOTE: All X Y and xy products are legacy now values, and are only used
        for plotting functions. They are not used in any calculations and could
        be dropped from the return calls in future versions.


    """

    if method == "kmeans":
        # Best for smaller problems
        from sklearn.cluster import AgglomerativeClustering

        # Cluster
        cluster = AgglomerativeClustering(
            n_clusters=n_tiles, affinity="euclidean", linkage="ward"
        )
        cluster.fit_predict(locations[:, :2])

        # nData in each tile
        binCount = np.zeros(int(n_tiles))

        # x and y limits on each tile
        X1 = np.zeros_like(binCount)
        X2 = np.zeros_like(binCount)
        Y1 = np.zeros_like(binCount)
        Y2 = np.zeros_like(binCount)

        for ii in range(int(n_tiles)):

            mask = cluster.labels_ == ii
            X1[ii] = locations[mask, 0].min()
            X2[ii] = locations[mask, 0].max()
            Y1[ii] = locations[mask, 1].min()
            Y2[ii] = locations[mask, 1].max()
            binCount[ii] = mask.sum()

        xy1 = np.c_[X1[binCount > 0], Y1[binCount > 0]]
        xy2 = np.c_[X2[binCount > 0], Y2[binCount > 0]]

        # Get the tile numbers that exist, for compatibility with the next method
        tile_id = np.unique(cluster.labels_)
        labels = cluster.labels_

    else:
        # Works on larger problems
        # Initialize variables
        # Test each refinement level for maximum space coverage
        nTx = 1
        nTy = 1
        for ii in range(int(n_tiles + 1)):

            nTx += 1
            nTy += 1

            testx = np.percentile(locations[:, 0], np.arange(0, 100, 100 / nTx))
            testy = np.percentile(locations[:, 1], np.arange(0, 100, 100 / nTy))

            # if ii > 0:
            dx = testx[:-1] - testx[1:]
            dy = testy[:-1] - testy[1:]

            if np.mean(dx) > np.mean(dy):
                nTx -= 1
            else:
                nTy -= 1

            print(nTx, nTy)
        tilex = np.percentile(locations[:, 0], np.arange(0, 100, 100 / nTx))
        tiley = np.percentile(locations[:, 1], np.arange(0, 100, 100 / nTy))

        X1, Y1 = np.meshgrid(tilex, tiley)
        X2, Y2 = np.meshgrid(
            np.r_[tilex[1:], locations[:, 0].max()], np.r_[tiley[1:], locations[:, 1].max()]
        )

        # Plot data and tiles
        X1, Y1, X2, Y2 = mkvc(X1), mkvc(Y1), mkvc(X2), mkvc(Y2)
        binCount = np.zeros_like(X1)
        labels = np.zeros_like(locations[:, 0])
        for ii in range(X1.shape[0]):

            mask = (
                (locations[:, 0] >= X1[ii])
                * (locations[:, 0] <= X2[ii])
                * (locations[:, 1] >= Y1[ii])
                * (locations[:, 1] <= Y2[ii])
            ) == 1

            # Re-adjust the window size for tight fit
            if minimize:

                if mask.sum():
                    X1[ii], X2[ii] = locations[:, 0][mask].min(), locations[:, 0][mask].max()
                    Y1[ii], Y2[ii] = locations[:, 1][mask].min(), locations[:, 1][mask].max()

            labels[mask] = ii
            binCount[ii] = mask.sum()

        xy1 = np.c_[X1[binCount > 0], Y1[binCount > 0]]
        xy2 = np.c_[X2[binCount > 0], Y2[binCount > 0]]

        # Get the tile numbers that exist
        # Since some tiles may have 0 data locations, and are removed by
        # [binCount > 0], the tile numbers are no longer contiguous 0:nTiles
        tile_id = np.unique(labels)

    tiles = []
    for id in tile_id.tolist():
        tiles += [np.where(labels==id)[0]]

    out = [tiles]

    if bounding_box:
        out.append([xy1, xy2])

    if count:
        out.append(binCount[binCount > 0])

    if unique_id:
        out.append(tile_id)

    if len(out) == 1:
        return out[0]
    return tuple(out)