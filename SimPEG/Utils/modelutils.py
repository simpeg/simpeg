from .matutils import mkvc, ndgrid
import numpy as np
from scipy.interpolate import griddata, interp1d

def surface2ind_topo(mesh, topo, gridLoc='CC', method='cubic', fill_value=np.nan):
    """
    Get active indices from topography
    """

    if mesh.dim == 3:

        if gridLoc == 'CC':
            XY = ndgrid(mesh.vectorCCx, mesh.vectorCCy)
            Zcc = mesh.gridCC[:, 2].reshape((np.prod(mesh.vnC[:2]), mesh.nCz), order='F')
            #gridTopo = Ftopo(XY)
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

    return mkvc(actind)
