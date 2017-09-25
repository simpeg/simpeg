from .matutils import mkvc, ndgrid
import numpy as np
from scipy.interpolate import griddata, interp1d


def surface2ind_topo(mesh, topo, gridLoc='CC', method='nearest', fill_value=np.nan):
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
                                          'implemented for {0!s} mesh'.format(mesh._meshType))

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
                raise NotImplementedError('Nodal surface2ind_topo not implemented for {0!s} mesh'.format(mesh._meshType))

            # TODO: this will only work for tensor meshes
            Ny = mesh.vectorNy[1:]
            actind = np.array([False]*mesh.nC).reshape(mesh.vnC, order='F')

            for ii in range(mesh.nCx):
                actind[ii, :] = [np.all(gridTopo[ii: ii+2] > Ny[kk])
                                 for kk in range(len(Ny))]

    else:
        raise NotImplementedError('surface2ind_topo not implemented for 1D mesh')

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

    print(filt.shape)
    x1, x2 = xtiles[:-1], xtiles[1:]
    y1, y2 = ytiles[:-1], ytiles[1:]

    X1, Y1 = np.meshgrid(x1, y1)
    xy1 = np.c_[mkvc(X1)[filt], mkvc(Y1)[filt]]
    X2, Y2 = np.meshgrid(x2, y2)
    xy2 = np.c_[mkvc(X2)[filt], mkvc(Y2)[filt]]

    return [xy1, xy2]
