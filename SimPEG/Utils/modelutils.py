from .matutils import mkvc, ndgrid
import numpy as np
from scipy.interpolate import griddata, interp1d, interp2d, NearestNDInterpolator
import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay
from scipy.sparse.linalg import bicgstab
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator
import discretize as Mesh
from discretize.utils import closestPoints, kron3, speye


def surface2ind_topo(mesh, topo, gridLoc='N', method='linear',
                     fill_value=np.nan):
    """
    Get active indices from topography

    Parameters
    ----------

    :param TensorMesh mesh: TensorMesh object on which to discretize topography
    :param numpy.ndarray topo: [X,Y,Z] topographic data
    :param str gridLoc: 'CC' or 'N'. Default is 'CC'.
                        Discretize the topography
                        on cells-center 'CC' or nodes 'N'
    :param str method: 'nearest' or 'linear' or 'cubic'. Default is 'nearest'.
                       Interpolation method for the topographic data
    :param float fill_value: default is np.nan. Filling value for extrapolation

    Returns
    -------

    :param numpy.array actind: index vector for the active cells on the mesh
                               below the topography
    """

    if mesh.dim == 3:

        if mesh._meshType in ['TREE']:

                if method == 'nearest':
                    F = NearestNDInterpolator(topo[:, :2], topo[:, 2])
                    zTopo = F(mesh.gridCC[:, :2])
                else:
                    tri2D = Delaunay(topo[:, :2])
                    F = LinearNDInterpolator(tri2D, topo[:, 2])
                    zTopo = F(mesh.gridCC[:, :2])

                    if any(np.isnan(zTopo)):
                        F = NearestNDInterpolator(topo[:, :2], topo[:, 2])
                        zTopo[np.isnan(zTopo)] = F(mesh.gridCC[np.isnan(zTopo), :2])

        if gridLoc == 'CC':

            if mesh._meshType in ['TREE']:

                # Fetch elevation at cell centers
                actind = mesh.gridCC[:, 2] < zTopo

            else:
                XY = ndgrid(mesh.vectorCCx, mesh.vectorCCy)
                Zcc = mesh.gridCC[:, 2].reshape((np.prod(mesh.vnC[:2]),
                                                 mesh.nCz),
                                                order='F')

                gridTopo = griddata(topo[:, :2], topo[:, 2], XY,
                                    method=method,
                                    fill_value=fill_value)

                actind = [gridTopo >= Zcc[:, ixy]
                          for ixy in range(np.prod(mesh.vnC[2]))]

                actind = np.hstack(actind)

        elif gridLoc == 'N':

            if mesh._meshType in ['TENSOR', 'CYL', 'BASETENSOR']:

                XY = ndgrid(mesh.vectorNx, mesh.vectorNy)
                gridTopo = griddata(topo[:, :2], topo[:, 2], XY,
                                    method=method,
                                    fill_value=fill_value)

                gridTopo = gridTopo.reshape(mesh.vnN[:2], order='F')

                # TODO: this will only work for tensor meshes
                Nz = mesh.vectorNz[1:]
                actind = np.array([False]*mesh.nC).reshape(mesh.vnC, order='F')

                for ii in range(mesh.nCx):
                    for jj in range(mesh.nCy):
                        actind[ii, jj, :] = [np.all(gridTopo[ii:ii+2, jj:jj+2] >=
                                                    Nz[kk])
                                             for kk in range(len(Nz))]
            else:

                # Fetch elevation at cell centers
                actind = (mesh.gridCC[:, 2] + mesh.h_gridded[:, 2]/2.) < zTopo

    elif mesh.dim == 2:

        Ftopo = interp1d(topo[:, 0], topo[:, 1], fill_value=fill_value,
                         kind=method)

        if gridLoc == 'CC':
            gridTopo = Ftopo(mesh.gridCC[:, 0])
            actind = mesh.gridCC[:, 1] <= gridTopo

        elif gridLoc == 'N':

            if mesh._meshType in ['TENSOR', 'CYL', 'BASETENSOR']:
                gridTopo = Ftopo(mesh.vectorNx)
                    # raise NotImplementedError('Nodal surface2ind_topo not' +
                    #                           'implemented for {0!s} ' +
                    #                           'mesh'.format(mesh._meshType))

                # TODO: this will only work for tensor meshes
                Ny = mesh.vectorNy[1:]
                actind = np.array([False]*mesh.nC).reshape(mesh.vnC, order='F')

                for ii in range(mesh.nCx):
                    actind[ii, :] = [np.all(gridTopo[ii: ii+2] > Ny[kk])
                                     for kk in range(len(Ny))]

            else:
                zTopo = Ftopo(mesh.gridCC[:, 0])
                actind = (mesh.gridCC[:, 1] + mesh.h_gridded[:, 1]/2.) < zTopo

    else:
        raise NotImplementedError('surface2ind_topo not implemented' +
                                  ' for 1D mesh')

    return mkvc(actind)


def tileSurveyPoints(locs, nRefine, minimize=True, method='cluster'):
    """
        Function to tile an survey points into smaller square subsets of points

        :param numpy.ndarray locs: n x 2 array of locations [x,y]
        :param integer nRefine: number of tiles (for 'cluster'), or number of
            refinement steps ('other')
        :param Bool minimize: shrink tile sizes to minimum
        :param string method: set to 'cluster' to use better quality clustering, or anything
            else to use more memory efficient method for large problems

        RETURNS:
        :param numpy.ndarray: Return a list of arrays with the for the SW and NE
                            limits of each tiles
        :param integer binCount: Number of points in each tile
        :param numpy.array labels: Cluster index of each point n=0:(nTargetTiles-1)
        :param numpy.array tile_numbers: Vector of tile numbers for each count in binCount

        NOTE: All X Y and xy products are legacy now values, and are only used
        for plotting functions. They are not used in any calculations and could
        be dropped from the return calls in future versions.


    """

    if method is 'cluster':
        # Best for smaller problems
        from sklearn.cluster import AgglomerativeClustering

        # Cluster
        cluster = AgglomerativeClustering(n_clusters=nRefine, affinity='euclidean', linkage='ward')
        cluster.fit_predict(locs[:,:2])

        # nData in each tile
        binCount = np.zeros(int(nRefine))

        # x and y limits on each tile
        X1 = np.zeros_like(binCount)
        X2 = np.zeros_like(binCount)
        Y1 = np.zeros_like(binCount)
        Y2 = np.zeros_like(binCount)

        for ii in range(int(nRefine)):

            mask = cluster.labels_ == ii
            X1[ii] = locs[mask, 0].min()
            X2[ii] = locs[mask, 0].max()
            Y1[ii] = locs[mask, 1].min()
            Y2[ii] = locs[mask, 1].max()
            binCount[ii] = mask.sum()

        xy1 = np.c_[X1[binCount > 0], Y1[binCount > 0]]
        xy2 = np.c_[X2[binCount > 0], Y2[binCount > 0]]

        # Get the tile numbers that exist, for compatibility with the next method
        tile_numbers = np.unique(cluster.labels_)
        return [xy1, xy2], binCount[binCount > 0], cluster.labels_, tile_numbers

    else:
        # Works on larger problems
        # Initialize variables
        # Test each refinement level for maximum space coverage
        nTx = 1
        nTy = 1
        for ii in range(int(nRefine+1)):

            nTx += 1
            nTy += 1

            testx = np.percentile(locs[:, 0], np.arange(0, 100, 100/nTx))
            testy = np.percentile(locs[:, 1], np.arange(0, 100, 100/nTy))

            # if ii > 0:
            dx = testx[:-1] - testx[1:]
            dy = testy[:-1] - testy[1:]

            if np.mean(dx) > np.mean(dy):
                nTx -= 1
            else:
                nTy -= 1

            print(nTx, nTy)
        tilex = np.percentile(locs[:, 0], np.arange(0, 100, 100/nTx))
        tiley = np.percentile(locs[:, 1], np.arange(0, 100, 100/nTy))

        X1, Y1 = np.meshgrid(tilex, tiley)
        X2, Y2 = np.meshgrid(
                np.r_[tilex[1:], locs[:, 0].max()],
                np.r_[tiley[1:], locs[:, 1].max()]
        )

        # Plot data and tiles
        X1, Y1, X2, Y2 = mkvc(X1), mkvc(Y1), mkvc(X2), mkvc(Y2)
        binCount = np.zeros_like(X1)
        tile_labels = np.zeros_like(locs[:, 0])
        for ii in range(X1.shape[0]):

            mask = (
                (locs[:, 0] >= X1[ii]) * (locs[:, 0] <= X2[ii]) *
                (locs[:, 1] >= Y1[ii]) * (locs[:, 1] <= Y2[ii])
            ) == 1

            # Re-adjust the window size for tight fit
            if minimize:

                if mask.sum():
                    X1[ii], X2[ii] = locs[:, 0][mask].min(), locs[:, 0][mask].max()
                    Y1[ii], Y2[ii] = locs[:, 1][mask].min(), locs[:, 1][mask].max()

            tile_labels[mask] = ii
            binCount[ii] = mask.sum()

        xy1 = np.c_[X1[binCount > 0], Y1[binCount > 0]]
        xy2 = np.c_[X2[binCount > 0], Y2[binCount > 0]]

        # Get the tile numbers that exist
        # Since some tiles may have 0 data locations, and are removed by
        # [binCount > 0], the tile numbers are no longer contiguous 0:nTiles
        tile_numbers = np.unique(tile_labels)
        return [xy1, xy2], binCount[binCount > 0], tile_labels, tile_numbers


def minCurvatureInterp(
    locs, data, mesh=None,
    vectorX=None, vectorY=None, vectorZ=None, gridSize=10,
    tol=1e-5, iterMax=200, method='spline'
):
    """
    Interpolate properties with a minimum curvature interpolation
    :param locs:  numpy.array of size n-by-3 of point locations
    :param data: numpy.array of size n-by-m of values to be interpolated
    :param vectorX: numpy.ndarray Gridded locations along x-axis [Default:None]
    :param vectorY: numpy.ndarray Gridded locations along y-axis [Default:None]
    :param vectorZ: numpy.ndarray Gridded locations along z-axis [Default:None]
    :param gridSize: numpy float Grid point seperation in meters [DEFAULT:10]
    :param method: 'relaxation' || 'spline' [Default]
    :param tol: float tol=1e-5 [Default] Convergence criteria
    :param iterMax: int iterMax=None [Default] Maximum number of iterations

    :return: numpy.array of size nC-by-m of interpolated values

    """

    def av_extrap(n):
        """Define 1D averaging operator from cell-centers to nodes."""
        Av = (
            sp.spdiags(
                (0.5 * np.ones((n, 1)) * [1, 1]).T,
                [-1, 0],
                n + 1, n,
                format="csr"
            )
        )
        Av[0, 1], Av[-1, -2] = 0.5, 0.5
        return Av

    def aveCC2F(grid):
        "Construct the averaging operator on cell cell centers to faces."
        if grid.ndim == 1:
            aveCC2F = av_extrap(grid.shape[0])
        elif grid.ndim == 2:
            aveCC2F = sp.vstack((
                sp.kron(speye(grid.shape[1]), av_extrap(grid.shape[0])),
                sp.kron(av_extrap(grid.shape[1]), speye(grid.shape[0]))
            ), format="csr")
        elif grid.ndim == 3:
            aveCC2F = sp.vstack((
                kron3(
                    speye(grid.shape[2]), speye(grid.shape[1]), av_extrap(grid.shape[0])
                ),
                kron3(
                    speye(grid.shape[2]), av_extrap(grid.shape[1]), speye(grid.shape[0])
                ),
                kron3(
                    av_extrap(grid.shape[2]), speye(grid.shape[1]), speye(grid.shape[0])
                )
            ), format="csr")
        return aveCC2F

    assert locs.shape[0] == data.shape[0], ("Number of interpolated locs " +
                                            "must match number of data")

    if vectorY is not None:
        assert locs.shape[1] >= 2, (
                "Found vectorY as an input." +
                " Point locations must contain X and Y coordinates."
            )

    if vectorZ is not None:
        assert locs.shape[1] == 3, (
                "Found vectorZ as an input." +
                " Point locations must contain X, Y and Z coordinates."
            )

    ndim = locs.shape[1]

    # Define a new grid based on data extent
    if mesh is None:
        if vectorX is None:
            xmin, xmax = locs[:, 0].min(), locs[:, 0].max()
            nCx = int((xmax-xmin)/gridSize)
            vectorX = xmin+np.cumsum(np.ones(nCx) * gridSize)

        if vectorY is None and ndim >= 2:
            ymin, ymax = locs[:, 1].min(), locs[:, 1].max()
            nCy = int((ymax-ymin)/gridSize)
            vectorY = ymin+np.cumsum(np.ones(nCy) * gridSize)

        if vectorZ is None and ndim == 3:
            zmin, zmax = locs[:, 2].min(), locs[:, 2].max()
            nCz = int((zmax-zmin)/gridSize)
            vectorZ = zmin+np.cumsum(np.ones(nCz) * gridSize)

        if ndim == 3:

            mesh = Mesh.TensorMesh([
                np.ones(nCx) * gridSize,
                np.ones(nCy) * gridSize,
                np.ones(nCz) * gridSize
                ])
            mesh.x0 = [xmin, ymin, zmin]

            # gridCy, gridCx, gridCz = np.meshgrid(vectorY, vectorX, vectorZ)
            # gridCC = np.c_[mkvc(gridCx), mkvc(gridCy), mkvc(gridCz)]
        elif ndim == 2:

            mesh = Mesh.TensorMesh([
                np.ones(nCx) * gridSize,
                np.ones(nCy) * gridSize
                ])
            mesh.x0 = [xmin, ymin]
            # gridCy, gridCx = np.meshgrid(vectorY, vectorX)
            # gridCC = np.c_[mkvc(gridCx), mkvc(gridCy)]
        else:
            mesh = Mesh.TensorMesh([
                np.ones(nCx) * gridSize
                ])
            mesh.x0 = xmin
    else:

        gridSize = mesh.h_gridded.min()

    gridCC = mesh.gridCC
    # Build the cKDTree for distance lookup
    tree = cKDTree(locs)
    # Get the grid location
    d, ind = tree.query(gridCC, k=1)

    if data.ndim == 1:
        data = np.c_[data]

    if method == 'relaxation':
        # Ave = aveCC2F(gridCC)
        Ave = mesh.aveCC2F

        count = 0
        residual = 1.

        m = np.zeros((gridCC.shape[0], data.shape[1]))
        # Begin with neighrest primers
        for ii in range(m.shape[1]):
            # F = NearestNDInterpolator(mesh.gridCC[ijk], data[:, ii])
            m[:, ii] = data[ind, ii]

        while np.all([count < iterMax, residual > tol]):
            for ii in range(m.shape[1]):
                # Reset the closest cell grid to the contraints
                m[d < gridSize, ii] = data[ind[d < gridSize], ii]
            mtemp = m
            m = Ave.T * (Ave * m)
            residual = np.linalg.norm(m-mtemp)/np.linalg.norm(mtemp)
            count += 1

    elif method == 'spline':

        ndat = locs.shape[0]
        # nC = int(nCx*nCy)

        A = np.zeros((ndat, ndat))
        for i in range(ndat):

            r = (locs[i, 0] - locs[:, 0])**2. + (locs[i, 1] - locs[:, 1])**2.
            A[i, :] = r.T * (np.log((r.T + 1e-8)**0.5) - 1.)


        # Compute new solution
        nC = gridCC.shape[0]
        m = np.zeros((nC, data.shape[1]))

        # Solve system for the weights
        for dd in range(data.shape[1]):
            w = bicgstab(A, data[:, dd], tol=tol)

            # We can parallelize this part later
            for i in range(nC):

                r = (gridCC[i, 0] - locs[:, 0])**2. + (gridCC[i, 1] - locs[:, 1])**2.
                m[i, dd] = np.sum(w[0] * r.T * (np.log((r.T + 1e-8)**0.5) - 1.))

    return mesh, m


def activeTopoLayer(mesh, topo, index=0):
    """
        Find the ith layer below topo
    """

    actv = np.zeros(mesh.nC, dtype='bool')
    # Get cdkTree to find top layer
    tree = cKDTree(mesh.gridCC)

    def ismember(a, b):
        bind = {}
        for i, elt in enumerate(b):
            if elt not in bind:
                bind[elt] = i
        return np.vstack([bind.get(itm, None) for itm in a])

    grid_x, grid_y = np.meshgrid(mesh.vectorCCx, mesh.vectorCCy)
    zInterp = mkvc(griddata(topo[:, :2], topo[:, 2], (grid_x, grid_y), method='nearest'))
    r, inds = tree.query(np.c_[mkvc(grid_x), mkvc(grid_y), zInterp])
    inds = np.unique(inds)

    # Extract neighbors from operators
    # Dz = mesh._cellGradzStencil
    # Iz, Jz, _ = sp.find(Dz)
    # jz = np.sort(Jz[np.argsort(Iz)].reshape((int(Iz.shape[0]/2), 2)), axis=1)
    # for ii in range(index):

    #     members = ismember(inds, jz[:, 1])
    #     inds = np.squeeze(jz[members, 0])

    # actv[inds] = True

    max_level = mesh.max_level

    for layer in range(index):

        nn = []
        for ind in inds.tolist():

            skip = 2 ** (max_level - mesh[int(ind)]._level)
            if (np.floor(layer/skip) % 2) > 0:
                nn += [mesh[int(ind)].neighbors[4]]
            else:
                nn += [int(ind)]

        inds = np.hstack(nn)

    actv[np.hstack(nn)] = True

    return actv


def transfer_to_mesh(
    mesh_in, mesh_out, values,
    return_indices=False
):
    """
        transfer_to_mesh(mesh_input, mesh_output, values, return_indices=False)

        Function to transfer values from an input mesh to another.

        Parameters
        ----------
        mesh_in: discetize.mesh
            Mesh object with input values

        mesh_out: discetize.mesh
            Mesh object to be transfered onto

        values: ndarray of floats (mesh_in.nC, nD)
            Cell center values to be transfered

        return_indices: bool optional
            Option to return an array of int (mesh_out.nC, ) for the indices
            of `mesh_in` such that:
            `a = values[indices]` are values tranfered from `mesh_in` inside
            `mesh_out`

        Returns
        -------

        values_out: ndarray or float (mesh_out.nC, nD)
            Values transfered from mesh_in to mesh_out

        incides: ndarray of int (mesh_out.nC, )
            Integer values from mesh_in to mesh_out

    """

    tree = cKDTree(mesh_in.gridCC)

    _, cell_indices = tree.query(mesh_out.gridCC)

    if values.ndim > 1:
        values_out = values[cell_indices, :]
    else:
        values_out = values[cell_indices]

    if return_indices:
        return (values_out, cell_indices)

    return values_out
