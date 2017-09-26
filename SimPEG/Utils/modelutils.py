from .matutils import mkvc, ndgrid
import numpy as np
from scipy.interpolate import griddata, interp1d
import numpy as np
import discretize as Mesh
from discretize.utils import closestPoints


def surface2ind_topo(mesh, topo, gridLoc='CC', method='nearest',
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

        if gridLoc == 'CC':
            XY = ndgrid(mesh.vectorCCx, mesh.vectorCCy)
            Zcc = mesh.gridCC[:, 2].reshape((np.prod(mesh.vnC[:2]), mesh.nCz),
                                            order='F')

            gridTopo = griddata(topo[:, :2], topo[:, 2], XY, method=method,
                                fill_value=fill_value)
            actind = [gridTopo >= Zcc[:, ixy]
                      for ixy in range(np.prod(mesh.vnC[2]))]

            actind = np.hstack(actind)

        elif gridLoc == 'N':

            XY = ndgrid(mesh.vectorNx, mesh.vectorNy)
            gridTopo = griddata(topo[:, :2], topo[:, 2], XY, method=method,
                                fill_value=fill_value)
            gridTopo = gridTopo.reshape(mesh.vnN[:2], order='F')

            if mesh._meshType not in ['TENSOR', 'CYL', 'BASETENSOR']:
                raise NotImplementedError('Nodal surface2ind_topo not ' +
                                          'implemented for' +
                                          '{0!s} mesh'.format(mesh._meshType))

            # TODO: this will only work for tensor meshes
            Nz = mesh.vectorNz[1:]
            actind = np.array([False]*mesh.nC).reshape(mesh.vnC, order='F')

            for ii in range(mesh.nCx):
                for jj in range(mesh.nCy):
                    actind[ii, jj, :] = [np.all(gridTopo[ii:ii+2, jj:jj+2] >=
                                                Nz[kk])
                                         for kk in range(len(Nz))]

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


def tileSurveyPoints(locs, maxNpoints):
    """
        Function to tile an survey points into smaller square subsets of points

        :param numpy.ndarray locs: n x 2 array of locations [x,y]
        :param integer maxNpoints: maximum number of points in each tile

        RETURNS:
        :param numpy.ndarray: Return a list of arrays  for the SW and NE
                            limits of each tiles

    """

    # Initialize variables
    nNx = 2
    nNy = 1
    nObs = 1e+8
    countx = 0
    county = 0
    xlim = [locs[:, 0].min(), locs[:, 0].max()]
    ylim = [locs[:, 1].min(), locs[:, 1].max()]

    # Refine the brake recursively
    while nObs > maxNpoints:

        nObs = 0

        if countx > county:
            nNx += 1
        else:
            nNy += 1

        countx = 0
        county = 0
        xtiles = np.linspace(xlim[0], xlim[1], nNx)
        ytiles = np.linspace(ylim[0], ylim[1], nNy)

        # Remove tiles without points in
        filt = np.ones((nNx-1)*(nNy-1), dtype='bool')

        for ii in range(xtiles.shape[0]-1):
            for jj in range(ytiles.shape[0]-1):
                # Mask along x axis
                maskx = np.all([locs[:, 0] >= xtiles[ii],
                               locs[:, 0] <= xtiles[int(ii+1)]], axis=0)

                # Mask along y axis
                masky = np.all([locs[:, 1] >= ytiles[jj],
                               locs[:, 1] <= ytiles[int(jj+1)]], axis=0)

                # Remember which axis has the most points (for next split)
                countx = np.max([np.sum(maskx), countx])
                county = np.max([np.sum(masky), county])

                # Full mask
                mask = np.all([maskx, masky], axis=0)
                nObs = np.max([nObs, np.sum(mask)])

                # Check if at least one point is picked
                if np.sum(mask) == 0:
                    filt[jj + ii*(nNy-1)] = False

    x1, x2 = xtiles[:-1], xtiles[1:]
    y1, y2 = ytiles[:-1], ytiles[1:]

    X1, Y1 = np.meshgrid(x1, y1)
    xy1 = np.c_[mkvc(X1)[filt], mkvc(Y1)[filt]]
    X2, Y2 = np.meshgrid(x2, y2)
    xy2 = np.c_[mkvc(X2)[filt], mkvc(Y2)[filt]]

    return [xy1, xy2]


def meshBuilder(xyz, h, padDist, nCmin=3, meshGlobal=None, expFact=1.3):
    """
        Function to quickly generate a Tensor mesh
        given a cloud of xyz points, finest core cell size
        and padding distance.
        If a meshGlobal is provided, the core cells will be centered
        on the underlaying mesh to reduce interpolation errors.
    """

    # Get center of the mesh
    midX = np.mean([xyz[:, 0].max(), xyz[:, 0].min()])
    midY = np.mean([xyz[:, 1].max(), xyz[:, 1].min()])
    midZ = np.mean([xyz[:, 2].max(), xyz[:, 2].min()])

    nCx = int((xyz[:, 0].max() - xyz[:, 0].min()) / h[0]) + nCmin*2
    nCy = int((xyz[:, 1].max() - xyz[:, 1].min()) / h[1]) + nCmin*2
    nCz = int((xyz[:, 2].max() - xyz[:, 2].min()) / h[2]) + nCmin*2

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

    # Add paddings
    hx = [(h[0], npadWest, -expFact), (h[0], nCx), (h[0], npadEast, expFact)]
    hy = [(h[1], npadSouth, -expFact), (h[1], nCy), (h[1], npadNorth, expFact)]
    hz = [(h[2], npadDown, -expFact), (h[2], nCz), (h[2], npadUp, expFact)]

    # Create mesh
    mesh = Mesh.TensorMesh([hx, hy, hz], 'CC0')

    mesh._x0 = np.r_[mesh.x0[0] + midX,
                     mesh.x0[1] + midY,
                     mesh.x0[2] - mesh.hz.sum() + midZ + h[2]*nCz/2.]

    if meshGlobal is not None:
        # Shift tile center to closest cell in base grid
        ind = closestPoints(meshGlobal, np.r_[midX, midY, midZ], gridLoc='CC')
        shift = np.squeeze(meshGlobal.gridCC[ind, :]) - np.r_[midX, midY, midZ]
        mesh._x0 = mesh.x0 + shift

    return mesh
