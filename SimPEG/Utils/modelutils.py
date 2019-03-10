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

            gridTopo = Ftopo(mesh.vectorNx)
            if mesh._meshType not in ['TENSOR', 'CYL', 'BASETENSOR']:
                raise NotImplementedError('Nodal surface2ind_topo not' +
                                          'implemented for {0!s} ' +
                                          'mesh'.format(mesh._meshType))

            # TODO: this will only work for tensor meshes
            Ny = mesh.vectorNy[1:]
            actind = np.array([False]*mesh.nC).reshape(mesh.vnC, order='F')

            for ii in range(mesh.nCx):
                actind[ii, :] = [np.all(gridTopo[ii: ii+2] > Ny[kk])
                                 for kk in range(len(Ny))]

    else:
        raise NotImplementedError('surface2ind_topo not implemented' +
                                  ' for 1D mesh')

    return mkvc(actind)


def tileSurveyPoints(locs, nRefine, minimize=True):
    """
        Function to tile an survey points into smaller square subsets of points

        :param numpy.ndarray locs: n x 2 array of locations [x,y]
        :param integer nRefine: maximum number of points in each tile

        RETURNS:
        :param numpy.ndarray: Return a list of arrays  for the SW and NE
                            limits of each tiles

    """

    # Initialize variables
    # Test each refinement level for maximum space coverage
    nTx = 0
    nTy = 0
    for ii in range(int(nRefine+1)):

        nTx += 1
        nTy += 1

        testx = np.percentile(locs[:, 0], np.arange(0, 100, 100/nTx))
        testy = np.percentile(locs[:, 1], np.arange(0, 100, 100/nTy))

        if ii > 0:
            dx = testx[:-1] - testx[1:]
            dy = testy[:-1] - testy[1:]

            if np.mean(dx) > np.mean(dy):
                nTx -= 1
            else:
                nTy -= 1

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
    tile = []
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

        binCount[ii] = mask.sum()

    xy1 = np.c_[X1[binCount > 0], Y1[binCount > 0]]
    xy2 = np.c_[X2[binCount > 0], Y2[binCount > 0]]

    return [xy1, xy2], binCount


def meshBuilder(xyz, h, padDist, meshGlobal=None,
                expFact=1.3,
                meshType='TENSOR',
                verticalAlignment='top'):
    """
        Function to quickly generate a Tensor mesh
        given a cloud of xyz points, finest core cell size
        and padding distance.
        If a meshGlobal is provided, the core cells will be centered
        on the underlaying mesh to reduce interpolation errors.

        :param numpy.ndarray xyz: n x 3 array of locations [x, y, z]
        :param numpy.ndarray h: 1 x 3 cell size for the core mesh
        :param numpy.ndarray padDist: 2 x 3 padding distances [W,E,S,N,Down,Up]
        [OPTIONAL]
        :param numpy.ndarray padCore: Number of core cells around the xyz locs
        :object SimPEG.Mesh: Base mesh used to shift the new mesh for overlap
        :param float expFact: Expension factor for padding cells [1.3]
        :param string meshType: Specify output mesh type: "TensorMesh"

        RETURNS:
        :object SimPEG.Mesh: Mesh object

    """

    assert meshType in ['TENSOR', 'TREE'], ('Revise meshType. Only ' +
                                            ' TENSOR | TREE mesh ' +
                                            'are implemented')

    # Get extent of points
    limx = np.r_[xyz[:, 0].max(), xyz[:, 0].min()]
    limy = np.r_[xyz[:, 1].max(), xyz[:, 1].min()]
    limz = np.r_[xyz[:, 2].max(), xyz[:, 2].min()]

    # Get center of the mesh
    midX = np.mean(limx)
    midY = np.mean(limy)

    if verticalAlignment == 'center':
        midZ = np.mean(limz)
    else:
        midZ = limz[0]

    nCx = int(limx[0]-limx[1]) / h[0]
    nCy = int(limy[0]-limy[1]) / h[1]
    nCz = int(np.max([
            limz[0]-limz[1],
            int(np.min(np.r_[nCx*h[0], nCy*h[1]])/3)
            ]) / h[2])

    if meshType == 'TENSOR':
        # Make sure the core has odd number of cells for centereing
        # on global mesh
        if meshGlobal is not None:
            nCx += 1 - int(nCx % 2)
            nCy += 1 - int(nCy % 2)
            nCz += 1 - int(nCz % 2)

        # Figure out paddings
        def expand(dx, pad):
            L = 0
            nC = 0
            while L < pad:
                nC += 1
                L = np.sum(dx * expFact**(np.asarray(range(nC))+1))

            return nC

        # Figure number of padding cells required to fill the space
        npadEast = expand(h[0], padDist[0, 0])
        npadWest = expand(h[0], padDist[0, 1])
        npadSouth = expand(h[1], padDist[1, 0])
        npadNorth = expand(h[1], padDist[1, 1])
        npadDown = expand(h[2], padDist[2, 0])
        npadUp = expand(h[2], padDist[2, 1])

        # Create discretization
        hx = [(h[0], npadWest, -expFact),
              (h[0], nCx),
              (h[0], npadEast, expFact)]
        hy = [(h[1], npadSouth, -expFact),
              (h[1], nCy), (h[1],
              npadNorth, expFact)]
        hz = [(h[2], npadDown, -expFact),
              (h[2], nCz),
              (h[2], npadUp, expFact)]

        # Create mesh
        mesh = Mesh.TensorMesh([hx, hy, hz], 'CC0')

        # Re-set the mesh at the center of input locations
        # Set origin
        if verticalAlignment == 'center':
            mesh.x0 = [midX-np.sum(mesh.hx)/2., midY-np.sum(mesh.hy)/2., midZ - np.sum(mesh.hz)/2.]
        elif verticalAlignment == 'top':
            mesh.x0 = [midX-np.sum(mesh.hx)/2., midY-np.sum(mesh.hy)/2., midZ - np.sum(mesh.hz)]
        else:
            assert NotImplementedError("verticalAlignment must be 'center' | 'top'")

    elif meshType == 'TREE':

        # Figure out full extent required from input
        extent = np.max(np.r_[nCx * h[0] + padDist[0, :].sum(),
                              nCy * h[1] + padDist[1, :].sum(),
                              nCz * h[2] + padDist[2, :].sum()])

        maxLevel = int(np.log2(extent/h[0]))+1

        # Number of cells at the small octree level
        # equal in 3D
        nCx, nCy, nCz = 2**(maxLevel), 2**(maxLevel), 2**(maxLevel)

        # nCy = 2**(int(np.log2(extent/h[1]))+1)
        # nCz = 2**(int(np.log2(extent/h[2]))+1)

        # Define the mesh and origin
        mesh = Mesh.TreeMesh([np.ones(nCx)*h[0],
                              np.ones(nCx)*h[1],
                              np.ones(nCx)*h[2]])

        # Shift mesh if global mesh is used
        center = np.r_[midX, midY, midZ]
        if meshGlobal is not None:

            tree = cKDTree(meshGlobal.gridCC)
            _, ind = tree.query(center, k=1)
            center = meshGlobal.gridCC[ind, :]

        # Set origin
        if verticalAlignment == 'center':
            mesh.x0 = np.r_[center[0] - (nCx-1)*h[0]/2., center[1] - (nCy-1)*h[1]/2., center[2] - (nCz-1)*h[2]/2.]
        elif verticalAlignment == 'top':
            mesh.x0 = np.r_[center[0] - (nCx-1)*h[0]/2., center[1] - (nCy-1)*h[1]/2., center[2] - (nCz-1)*h[2]]
        else:
            assert NotImplementedError("verticalAlignment must be 'center' | 'top'")

    return mesh


def refineTree(
            mesh, xyz,
            finalize=False, dtype="radial",
            octreeLevels=[1, 1, 1],
            octreeLevels_XY=None,
            maxDist=np.inf
):

    if octreeLevels_XY is not None:

        assert len(octreeLevels_XY) == len(octreeLevels), "Arguments 'octreeLevels' and 'octreeLevels_XY' must be the same length"

    else:

        octreeLevels_XY = np.zeros_like(octreeLevels)

    maxLevel = int(np.log2(mesh.hx.shape[0]))

    tree = cKDTree(xyz)

    if dtype == "point":

        mesh.insert_cells(xyz, np.ones(xyz.shape[0])*maxLevel, finalize=False)

        stencil = np.r_[
                np.ones(octreeLevels[0]),
                np.ones(octreeLevels[1])*2,
                np.ones(octreeLevels[2])*3
            ]

        # Reflect in the opposite direction
        vec = np.r_[stencil[::-1], 1, stencil]
        vecX, vecY, vecZ = np.meshgrid(vec, vec, vec)
        gridLevel = np.maximum(np.maximum(np.abs(vecX),
                               np.abs(vecY)), np.abs(vecZ))
        gridLevel = np.kron(np.ones(xyz.shape[0]), mkvc(gridLevel))

        # Grid the coordinates
        vec = np.r_[-np.cumsum(stencil)[::-1], 0, np.cumsum(stencil)]
        vecX, vecY, vecZ = np.meshgrid(vec, vec, vec)
        offset = np.c_[
            mkvc(np.sign(vecX)*np.abs(vecX) * mesh.hx.min()),
            mkvc(np.sign(vecY)*np.abs(vecY) * mesh.hy.min()),
            mkvc(np.sign(vecZ)*np.abs(vecZ) * mesh.hz.min())
        ]

        # Replicate the point locations in each offseted grid points
        newLoc = (
            np.kron(xyz, np.ones((offset.shape[0], 1))) +
            np.kron(np.ones((xyz.shape[0], 1)), offset)
        )

        # Apply max distance
        r, ind = tree.query(newLoc)

        mesh.insert_cells(
            newLoc[r < maxDist], maxLevel-mkvc(gridLevel)[r < maxDist]+1, finalize=False
        )

        if finalize:
            mesh.finalize()

    elif dtype == "radial":

        # Compute the outer limits of each octree level
        rMax = np.cumsum(
            mesh.hx.min() *
            np.asarray(octreeLevels) *
            2**np.arange(len(octreeLevels))
        )

        def inBall(cell):
            xyz = cell.center
            r, ind = tree.query(xyz)

            for ii, nC in enumerate(octreeLevels):

                if r < rMax[ii]:

                    return maxLevel-ii

            return 0

        mesh.refine(inBall, finalize=finalize)

    elif dtype == 'surface':

        # Compute centroid and
        centroid = np.mean(xyz, axis=0)

        # Largest outer point distance
        rOut = np.linalg.norm(np.r_[
            np.abs(centroid[0]-xyz[:, 0]).max(),
            np.abs(centroid[1]-xyz[:, 1]).max()
            ]
        )

        # Compute maximum depth of refinement
        zmax = np.cumsum(
            mesh.hz.min() *
            np.asarray(octreeLevels) *
            2**np.arange(len(octreeLevels))
        )
        padWidth = np.cumsum(
            mesh.hx.min() *
            np.asarray(octreeLevels_XY) *
            2**np.arange(len(octreeLevels_XY))
        )

        depth = zmax[-1]

        # Increment the vertical offset
        zOffset = 0
        xyPad = -1
        # Cycle through the Tree levels backward
        for ii in range(len(octreeLevels)-1, -1, -1):

            dx = mesh.hx.min() * 2**ii
            dy = mesh.hy.min() * 2**ii
            dz = mesh.hz.min() * 2**ii

            # Increase the horizontal extent of the surface
            if xyPad != padWidth[ii]:
                xyPad = padWidth[ii]

                # Calculate expansion for padding XY cells
                expFactor = (rOut + xyPad) / rOut
                xLoc = (xyz - centroid)*expFactor + centroid

                # Create a new triangulated surface
                tri2D = Delaunay(xLoc[:, :2])
                F = LinearNDInterpolator(tri2D, xLoc[:, 2])

            limx = np.r_[xLoc[:, 0].max(), xLoc[:, 0].min()]
            limy = np.r_[xLoc[:, 1].max(), xLoc[:, 1].min()]

            nCx = int(np.ceil((limx[0]-limx[1]) / dx))
            nCy = int(np.ceil((limy[0]-limy[1]) / dy))

            # Create a grid at the octree level in xy
            CCx, CCy = np.meshgrid(
                np.linspace(
                    limx[1], limx[0], nCx
                    ),
                np.linspace(
                    limy[1], limy[0], nCy
                    )
            )

            xy = np.c_[CCx.reshape(-1), CCy.reshape(-1)]

            # Only keep points within triangulation
            indexTri = tri2D.find_simplex(xy)

            z = F(xy[indexTri != -1])

            newLoc = np.c_[xy[indexTri != -1], z]

            # Only keep points within maxDist
            # Apply max distance
            r, ind = tree.query(newLoc)

            # Apply vertical padding for current octree level
            zOffset = 0
            while zOffset < depth:
                indIn = r < (maxDist + padWidth[ii])
                nnz = int(np.sum(indIn))
                if nnz > 0:
                    mesh.insert_cells(
                        np.c_[
                            newLoc[indIn, :2],
                            newLoc[indIn, 2]-zOffset],
                        np.ones(nnz)*maxLevel-ii,
                        finalize=False
                    )

                zOffset += dz

            depth -= dz * octreeLevels[ii]

        if finalize:
            mesh.finalize()

    elif dtype == 'box':

        # Define the data extend [bottom SW, top NE]
        bsw = np.min(xyz, axis=0)
        tne = np.max(xyz, axis=0)

        hx = mesh.hx.min()

        if mesh.dim == 2:
            hz = mesh.hy.min()
        else:
            hz = mesh.hz.min()

        # Pre-calculate max depth of each level
        zmax = np.cumsum(
            hz * np.asarray(octreeLevels) *
            2**np.arange(len(octreeLevels))
        )

        if mesh.dim == 2:
            # Pre-calculate outer extent of each level
            padWidth = np.cumsum(
                    mesh.hx.min() *
                    np.asarray(octreeLevels_XY) *
                    2**np.arange(len(octreeLevels_XY))
                )

            # Make a list of outer limits
            BSW = [
                bsw - np.r_[padWidth[ii], zmax[ii]]
                for ii, (octZ, octXY) in enumerate(
                        zip(octreeLevels, octreeLevels_XY)
                )
            ]

            TNE = [
                tne + np.r_[padWidth[ii], zmax[ii]]
                for ii, (octZ, octXY) in enumerate(
                    zip(octreeLevels, octreeLevels_XY)
                )
            ]

        else:
            hy = mesh.hy.min()

            # Pre-calculate outer X extent of each level
            padWidth_x = np.cumsum(
                    hx * np.asarray(octreeLevels_XY) *
                    2**np.arange(len(octreeLevels_XY))
                )

            # Pre-calculate outer Y extent of each level
            padWidth_y = np.cumsum(
                    hy * np.asarray(octreeLevels_XY) *
                    2**np.arange(len(octreeLevels_XY))
                )

            # Make a list of outer limits
            BSW = [
                bsw - np.r_[padWidth_x[ii], padWidth_y[ii], zmax[ii]]
                for ii, (octZ, octXY) in enumerate(
                        zip(octreeLevels, octreeLevels_XY)
                )
            ]

            TNE = [
                tne + np.r_[padWidth_x[ii], padWidth_y[ii], zmax[ii]]
                for ii, (octZ, octXY) in enumerate(
                    zip(octreeLevels, octreeLevels_XY)
                )
            ]

        def inBox(cell):

            xyz = cell.center

            for ii, (nC, bsw, tne) in enumerate(zip(octreeLevels, BSW, TNE)):

                if np.all([xyz > bsw, xyz < tne]):
                    return maxLevel-ii

            return cell._level

        mesh.refine(inBox, finalize=finalize)

    else:
        NotImplementedError(
            "Only dtype= 'surface' | 'points' "
            " | 'radial' | 'box' has been implemented"
        )

    return mesh


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
    Dz = mesh._cellGradzStencil
    Iz, Jz, _ = sp.find(Dz)
    jz = np.sort(Jz[np.argsort(Iz)].reshape((int(Iz.shape[0]/2), 2)), axis=1)
    for ii in range(index):

        members = ismember(inds, jz[:, 1])
        inds = np.squeeze(jz[members, 0])

    actv[inds] = True

    return actv
