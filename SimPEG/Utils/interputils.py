import numpy as np
import scipy.sparse as sp
from sputils import spzeros
from matutils import mkvc, sub2ind

def _interp_point_1D(x, xr_i):
    """
        given a point, xr_i, this will find which two integers it lies between.

        :param numpy.ndarray x: Tensor vector of 1st dimension of grid.
        :param float xr_i: Location of a point
        :rtype: int,int,float,float
        :return: index1, index2, portion1, portion2
    """
    # TODO: This fails if the point is on the outside of the mesh. We may want to replace this by extrapolation?
    im = np.argmin(abs(x-xr_i))
    if xr_i - x[im] >= 0:  # Point on the left
        ind_x1 = im
        ind_x2 = im+1
    elif xr_i - x[im] < 0:  # Point on the right
        ind_x1 = im-1
        ind_x2 = im
    ind_x1 = max(min(ind_x1, x.size-1), 0)
    ind_x2 = max(min(ind_x2, x.size-1), 0)
    dx1 = xr_i - x[ind_x1]
    dx2 = x[ind_x2] - xr_i
    return ind_x1, ind_x2, dx1, dx2


def interpmat(locs, x, y=None, z=None):
    """
        Local interpolation computed for each receiver point in turn

        :param numpy.ndarray loc: Location of points to interpolate to
        :param numpy.ndarray x: Tensor vector of 1st dimension of grid.
        :param numpy.ndarray y: Tensor vector of 2nd dimension of grid. None by default.
        :param numpy.ndarray z: Tensor vector of 3rd dimension of grid. None by default.
        :rtype: scipy.sparse.csr.csr_matrix
        :return: Interpolation matrix

        .. plot::

            import SimPEG
            import numpy as np
            import matplotlib.pyplot as plt
            locs = np.random.rand(50)*0.8+0.1
            x = np.linspace(0,1,7)
            dense = np.linspace(0,1,200)
            fun = lambda x: np.cos(2*np.pi*x)
            Q = SimPEG.Utils.interpmat(locs, x)
            plt.plot(x, fun(x), 'bs-')
            plt.plot(dense, fun(dense), 'y:')
            plt.plot(locs, Q*fun(x), 'mo')
            plt.plot(locs, fun(locs), 'rx')
            plt.show()

    """
    if y is None and z is None:
        return _interpmat1D(locs, x)
    elif z is None:
        return _interpmat2D(locs, x, y)
    else:
        return _interpmat3D(locs, x, y, z)


def _interpmat1D(locs, x):
    """Use interpmat with only x component provided."""
    nx = x.size
    locs = mkvc(locs)
    npts = locs.shape[0]

    Q = sp.lil_matrix((npts, nx))

    for i in range(npts):
        ind_x1, ind_x2, dx1, dx2 = _interp_point_1D(x, locs[i])
        dv = (x[ind_x2] - x[ind_x1])
        Dx =  x[ind_x2] - x[ind_x1]
        # Get the row in the matrix
        inds = [ind_x1, ind_x2]
        vals = [(1-dx1/Dx),(1-dx2/Dx)]
        Q[i, inds] = vals
    return Q



def _interpmat2D(locs, x, y):
    """Use interpmat with only x and y components provided."""
    nx = x.size
    ny = y.size
    npts = locs.shape[0]

    Q = sp.lil_matrix((npts, nx*ny))


    for i in range(npts):
        ind_x1, ind_x2, dx1, dx2 = _interp_point_1D(x, locs[i, 0])
        ind_y1, ind_y2, dy1, dy2 = _interp_point_1D(y, locs[i, 1])

        dv = (x[ind_x2] - x[ind_x1]) * (y[ind_y2] - y[ind_y1])

        Dx =  x[ind_x2] - x[ind_x1]
        Dy =  y[ind_y2] - y[ind_y1]

        # Get the row in the matrix

        inds = sub2ind((nx,ny),[
            ( ind_x1,  ind_y2),
            ( ind_x1,  ind_y1),
            ( ind_x2,  ind_y1),
            ( ind_x2,  ind_y2)])

        vals = [(1-dx1/Dx)*(1-dy2/Dy),
                (1-dx1/Dx)*(1-dy1/Dy),
                (1-dx2/Dx)*(1-dy1/Dy),
                (1-dx2/Dx)*(1-dy2/Dy)]

        Q[i, mkvc(inds)] = vals

    return Q



def _interpmat3D(locs, x, y, z):
    """Use interpmat."""
    nx = x.size
    ny = y.size
    nz = z.size
    npts = locs.shape[0]

    Q = sp.lil_matrix((npts, nx*ny*nz))


    for i in range(npts):
        ind_x1, ind_x2, dx1, dx2 = _interp_point_1D(x, locs[i, 0])
        ind_y1, ind_y2, dy1, dy2 = _interp_point_1D(y, locs[i, 1])
        ind_z1, ind_z2, dz1, dz2 = _interp_point_1D(z, locs[i, 2])

        dv = (x[ind_x2] - x[ind_x1]) * (y[ind_y2] - y[ind_y1]) *(z[ind_z2] - z[ind_z1])

        Dx =  x[ind_x2] - x[ind_x1]
        Dy =  y[ind_y2] - y[ind_y1]
        Dz =  z[ind_z2] - z[ind_z1]

        # Get the row in the matrix

        inds = sub2ind((nx,ny,nz),[
            ( ind_x1,  ind_y2,  ind_z1),
            ( ind_x1,  ind_y1,  ind_z1),
            ( ind_x2,  ind_y1,  ind_z1),
            ( ind_x2,  ind_y2,  ind_z1),
            ( ind_x1,  ind_y1,  ind_z2),
            ( ind_x1,  ind_y2,  ind_z2),
            ( ind_x2,  ind_y1,  ind_z2),
            ( ind_x2,  ind_y2,  ind_z2)])

        vals = [(1-dx1/Dx)*(1-dy2/Dy)*(1-dz1/Dz),
                (1-dx1/Dx)*(1-dy1/Dy)*(1-dz1/Dz),
                (1-dx2/Dx)*(1-dy1/Dy)*(1-dz1/Dz),
                (1-dx2/Dx)*(1-dy2/Dy)*(1-dz1/Dz),
                (1-dx1/Dx)*(1-dy1/Dy)*(1-dz2/Dz),
                (1-dx1/Dx)*(1-dy2/Dy)*(1-dz2/Dz),
                (1-dx2/Dx)*(1-dy1/Dy)*(1-dz2/Dz),
                (1-dx2/Dx)*(1-dy2/Dy)*(1-dz2/Dz)]

        Q[i, mkvc(inds)] = vals

    return Q


if __name__ == '__main__':
    import SimPEG
    import numpy as np
    import matplotlib.pyplot as plt
    locs = np.random.rand(50)*0.8+0.1
    x = np.linspace(0,1,7)
    dense = np.linspace(0,1,200)
    fun = lambda x: np.cos(2*np.pi*x)
    Q = SimPEG.Utils.interpmat(locs, x)
    plt.plot(x, fun(x), 'bs-')
    plt.plot(dense, fun(dense), 'y:')
    plt.plot(locs, Q*fun(x), 'mo')
    plt.plot(locs, fun(locs), 'rx')
    plt.show()
