from .matutils import mkvc, ndgrid
import numpy as np

def surface2ind_topo(mesh, topo, gridLoc='CC'):
# def genActiveindfromTopo(mesh, topo):
    """
    Get active indices from topography
    """


    if mesh.dim == 3:
        from scipy.interpolate import NearestNDInterpolator
        Ftopo = NearestNDInterpolator(topo[:,:2], topo[:,2])

        if gridLoc == 'CC':
            XY = ndgrid(mesh.vectorCCx, mesh.vectorCCy)
            Zcc = mesh.gridCC[:,2].reshape((np.prod(mesh.vnC[:2]), mesh.nCz), order='F')

            gridTopo = Ftopo(XY)
            actind = [gridTopo[ixy] <= Zcc[ixy,:] for ixy in range(np.prod(mesh.vnC[0]))]
            actind = np.hstack(actind)

        elif gridLoc == 'N':

            XY = ndgrid(mesh.vectorNx, mesh.vectorNy)
            gridTopo = Ftopo(XY).reshape(mesh.vnN[:2], order='F')

            if mesh._meshType not in ['TENSOR', 'CYL', 'BASETENSOR']:
                raise NotImplementedError('Nodal surface2ind_topo not implemented for {0!s} mesh'.format(mesh._meshType))

            Nz = mesh.vectorNz[1:] # TODO: this will only work for tensor meshes
            actind = np.array([False]*mesh.nC).reshape(mesh.vnC, order='F')

            for ii in range(mesh.nCx):
                for jj in range(mesh.nCy):
                     actind[ii,jj,:] = [np.all(gridTopo[ii:ii+2, jj:jj+2] >= Nz[kk]) for kk in range(len(Nz)) ]

    elif mesh.dim == 2:
        from scipy.interpolate import interp1d
        Ftopo = interp1d(topo[:,0], topo[:,1])

        if gridLoc == 'CC':
            gridTopo = Ftopo(mesh.gridCC[:,0])
            actind = mesh.gridCC[:,1] <= gridTopo

        elif gridLoc == 'N':

            gridTopo = Ftopo(mesh.vectorNx)
            if mesh._meshType not in ['TENSOR', 'CYL', 'BASETENSOR']:
                raise NotImplementedError('Nodal surface2ind_topo not implemented for {0!s} mesh'.format(mesh._meshType))

            Ny = mesh.vectorNy[1:] # TODO: this will only work for tensor meshes
            actind = np.array([False]*mesh.nC).reshape(mesh.vnC, order='F')

            for ii in range(mesh.nCx):
                actind[ii,:] = [np.all(gridTopo[ii:ii+2] > Ny[kk]) for kk in range(len(Ny)) ]

    else:
        raise NotImplementedError('surface2ind_topo not implemented for 1D mesh')

    return mkvc(actind)


