from .matutils import mkvc, ndgrid, uniqueRows
import numpy as np
from scipy.interpolate import griddata, interp1d
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from SimPEG import Mesh


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
    if mesh._meshType == "TENSOR":

        if mesh.dim == 3:
            # Check if Topo points are inside of the mesh
            xmin, xmax = mesh.vectorNx.min(), mesh.vectorNx.max()
            xminTopo, xmaxTopo = topo[:, 0].min(), topo[:, 1].min()
            ymin, ymax = mesh.vectorNy.min(), mesh.vectorNy.max()
            yminTopo, ymaxTopo = topo[:, 1].min(), topo[:, 1].max()
            if (xminTopo > xmin) or (xmaxTopo < xmax) or (yminTopo > ymin) or (ymaxTopo < ymax):
                # If not, use nearest neihbor to extrapolate them
                Ftopo = NearestNDInterpolator(topo[:, :2], topo[:, 2])
                xinds =  np.logical_or(
                    xminTopo < mesh.vectorNx, xmaxTopo > mesh.vectorNx
                    )
                yinds =  np.logical_or(
                    yminTopo < mesh.vectorNy, ymaxTopo > mesh.vectorNy
                    )
                XYOut = ndgrid(mesh.vectorNx[xinds], mesh.vectorNy[yinds])
                topoOut = Ftopo(XYOut)
                topo = np.vstack((topo, np.c_[XYOut, topoOut]))

            if gridLoc == 'CC':
                XY = ndgrid(mesh.vectorCCx, mesh.vectorCCy)
                Zcc = mesh.gridCC[:, 2].reshape((np.prod(mesh.vnC[:2]), mesh.nCz), order='F')
                gridTopo = griddata(topo[:, :2], topo[:, 2], XY, method=method, fill_value=fill_value)
                actind = [gridTopo >= Zcc[:, ixy] for ixy in range(np.prod(mesh.vnC[2]))]
                actind = np.hstack(actind)

            elif gridLoc == 'N':

                XY = ndgrid(mesh.vectorNx, mesh.vectorNy)
                gridTopo = griddata(topo[:, :2], topo[:, 2], XY, method=method, fill_value=fill_value)
                gridTopo = gridTopo.reshape(mesh.vnN[:2], order='F')

                if mesh._meshType not in ['TENSOR', 'CYL', 'BASETENSOR']:
                    raise NotImplementedError('Nodal surface2ind_topo not implemented for {0!s} mesh'.format(mesh._meshType))

                # TODO: this will only work for tensor meshes
                Nz = mesh.vectorNz[1:]
                actind = np.array([False]*mesh.nC).reshape(mesh.vnC, order='F')

                for ii in range(mesh.nCx):
                    for jj in range(mesh.nCy):
                         actind[ii, jj, :] = [np.all(gridTopo[ii:ii+2, jj:jj+2] >= Nz[kk]) for kk in range(len(Nz))]

        elif mesh.dim == 2:
            # Check if Topo points are inside of the mesh
            xmin, xmax = mesh.vectorNx.min(), mesh.vectorNx.max()
            xminTopo, xmaxTopo = topo[:, 0].min(), topo[:, 1].min()
            if (xminTopo > xmin) or (xmaxTopo < xmax):
                fill_value = "extrapolate"

            Ftopo = interp1d(topo[:, 0], topo[:, 1], fill_value=fill_value, kind=method)

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
                    actind[ii, :] = [np.all(gridTopo[ii: ii+2] > Ny[kk]) for kk in range(len(Ny))]

        else:
            raise NotImplementedError('surface2ind_topo not implemented for 1D mesh')

    elif mesh._meshType == "TREE":
        if mesh.dim == 3:
            if gridLoc == "CC":
                # Compute unique XY location
                uniqXY = uniqueRows(mesh.gridCC[:, :2])

                if method == "nearest":
                    Ftopo = NearestNDInterpolator(topo[:, :2], topo[:, 2])
                elif method == "linear":
                    # Check if Topo points are inside of the mesh
                    xmin, xmax = mesh.x0[0], mesh.hx.sum()+mesh.x0[0]
                    xminTopo, xmaxTopo = topo[:, 0].min(), topo[:, 1].min()
                    ymin, ymax = mesh.x0[1], mesh.hy.sum()+mesh.x0[1]
                    yminTopo, ymaxTopo = topo[:, 1].min(), topo[:, 1].max()
                    if (xminTopo > xmin) or (xmaxTopo < xmax) or (yminTopo > ymin) or (ymaxTopo < ymax):
                        # If not, use nearest neihbor to extrapolate them
                        Ftopo = NearestNDInterpolator(topo[:, :2], topo[:, 2])
                        xinds =  np.logical_or(
                            xminTopo < uniqXY[0][:, 0], xmaxTopo > uniqXY[0][:, 0]
                            )
                        yinds =  np.logical_or(
                            yminTopo < uniqXY[0][:, 1], ymaxTopo > uniqXY[0][:, 1]
                            )
                        inds = np.logical_or(xinds, yinds)
                        XYOut = uniqXY[0][inds, :]
                        topoOut = Ftopo(XYOut)
                        topo = np.vstack((topo, np.c_[XYOut, topoOut]))
                    Ftopo = LinearNDInterpolator(topo[:, :2], topo[:, 2])
                else:
                    raise NotImplementedError('Only nearest and linear method are available for TREE mesh')
                actind = np.zeros(mesh.nC, dtype='bool')
                npts = uniqXY[0].shape[0]
                for i in range(npts):
                    z = Ftopo(uniqXY[0][i, :])
                    inds = uniqXY[2] == i
                    actind[inds] = mesh.gridCC[inds, 2] < z[0]
            # Need to implement
            elif gridLoc == "N":
                raise NotImplementedError('gridLoc=N is not implemented for TREE mesh')
            else:
                raise Exception("gridLoc must be either CC or N")
        elif mesh.dim == 2:
            sorter = np.argsort(topo[:, 0])
            topo = topo[sorter]
            Ftopo = interp1d(topo[:, 0], topo[:, 1], bounds_error=False,
                             fill_value=(topo[0, 1], topo[-1, 1]), kind=method)
            if gridLoc == 'CC':
                grid = mesh.gridCC
            elif gridLoc == 'N':
                grid = mesh.gridN
            gridTopo = Ftopo(grid[:, 0])
            actind = grid[:, 1] <= gridTopo

    return mkvc(actind)


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
    midZ = np.mean(limz)

    nCx = int(limx[0]-limx[1]) / h[0]
    nCy = int(limy[0]-limy[1]) / h[1]
    nCz = int(limz[0]-limz[1]+int(np.min(np.r_[nCx, nCy])/3)) / h[2]

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
            mesh.x0 = [midX-np.sum(mesh.hx)/2., midY-np.sum(mesh.hy)/2., midZ-np.sum(mesh.hz)/2.]
        elif verticalAlignment == 'top':
            mesh.x0 = [midX-np.sum(mesh.hx)/2., midY-np.sum(mesh.hy)/2., limz[0]-np.sum(mesh.hz)]
        else:
            assert NotImplementedError("verticalAlignment must be 'center' | 'top'")

    elif meshType == 'TREE':

        # Figure out full extent required from input
        extent = np.max(np.r_[nCx * h[0] + padDist[0, :].sum(),
                              nCy * h[1] + padDist[1, :].sum(),
                              nCz * h[2] + padDist[2, :].sum()])

        maxLevel = int(np.log2(extent/h[0]))+1

        # Number of cells at the small octree level
        # For now equal in 3D

        nCx, nCy, nCz = 2**(maxLevel), 2**(maxLevel), 2**(maxLevel)
        # nCy = 2**(int(np.log2(extent/h[1]))+1)
        # nCz = 2**(int(np.log2(extent/h[2]))+1)

        # Define the mesh and origin
        # For now cubic cells
        mesh = Mesh.TreeMesh([np.ones(nCx)*h[0],
                              np.ones(nCx)*h[1],
                              np.ones(nCx)*h[2]])

        # Set origin
        if verticalAlignment == 'center':
            mesh.x0 = np.r_[-nCx*h[0]/2.+midX, -nCy*h[1]/2.+midY, -nCz*h[2]/2.+midZ]
        elif verticalAlignment == 'top':
            mesh.x0 = np.r_[-nCx*h[0]/2.+midX, -nCy*h[1]/2.+midY, -(nCz-1)*h[2] + limz.max()]
        else:
            assert NotImplementedError("verticalAlignment must be 'center' | 'top'")

    return mesh


def refineTree(mesh, xyz, finalize=False, dtype="point", nCpad=[1, 1, 1]):

    maxLevel = int(np.log2(mesh.hx.shape[0]))

    if dtype == "point":

        mesh.insert_cells(xyz, np.ones(xyz.shape[0])*maxLevel, finalize=False)

        stencil = np.r_[
                np.ones(nCpad[0]),
                np.ones(nCpad[1])*2,
                np.ones(nCpad[2])*3
            ]

        # Reflect in the opposite direction
        vec = np.r_[stencil[::-1], 1, stencil]
        vecX, vecY, vecZ = np.meshgrid(vec, vec, vec)
        gridLevel = np.maximum(np.maximum(vecX,
                               vecY), vecZ)
        gridLevel = np.kron(np.ones((xyz.shape[0], 1)), gridLevel)

        # Grid the coordinates
        vec = np.r_[-stencil[::-1], 0, stencil]
        vecX, vecY, vecZ = np.meshgrid(vec, vec, vec)
        offset = np.c_[
            mkvc(np.sign(vecX)*2**np.abs(vecX) * mesh.hx.min()),
            mkvc(np.sign(vecY)*2**np.abs(vecY) * mesh.hx.min()),
            mkvc(np.sign(vecZ)*2**np.abs(vecZ) * mesh.hx.min())
        ]

        # Replicate the point locations in each offseted grid points
        newLoc = (
            np.kron(xyz, np.ones((offset.shape[0], 1))) +
            np.kron(np.ones((xyz.shape[0], 1)), offset)
        )

        mesh.insert_cells(
            newLoc, maxLevel-mkvc(gridLevel)+1, finalize=finalize
        )

    elif dtype == 'surface':

        # Get extent of points
        limx = np.r_[xyz[:, 0].max(), xyz[:, 0].min()]
        limy = np.r_[xyz[:, 1].max(), xyz[:, 1].min()]

        F = NearestNDInterpolator(xyz[:, :2], xyz[:, 2])
        zOffset = 0
        # Cycle through the first 3 octree levels
        for ii in range(3):

            dx = mesh.hx.min()*2**ii

            nCx = int((limx[0]-limx[1]) / dx)
            nCy = int((limy[0]-limy[1]) / dx)

            # Create a grid at the octree level in xy
            CCx, CCy = np.meshgrid(
                np.linspace(limx[1], limx[0], nCx),
                np.linspace(limy[1], limy[0], nCy)
            )

            z = F(mkvc(CCx), mkvc(CCy))

            for level in range(int(nCpad[ii])):

                mesh.insert_cells(
                    np.c_[mkvc(CCx), mkvc(CCy), z-zOffset], np.ones_like(z)*maxLevel-ii,
                    finalize=False
                )

                zOffset += dx

        if finalize:
            mesh.finalize()

    else:
        NotImplementedError("Only dtype='points' has been implemented")

    return mesh
