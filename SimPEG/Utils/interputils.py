from __future__ import print_function
import numpy as np
import scipy.sparse as sp
from .matutils import mkvc, sub2ind, spzeros

try:
    from . import interputils_cython as pyx
    _interp_point_1D = pyx._interp_point_1D
    _interpmat1D = pyx._interpmat1D
    _interpmat2D = pyx._interpmat2D
    _interpmat3D = pyx._interpmat3D
    _interpCython = True
except ImportError:
    print("""Efficiency Warning: Interpolation will be slow, use setup.py!

            python setup.py build_ext --inplace
    """)
    _interpCython = False


def interpmat(locs, x, y=None, z=None):
    """
        Local interpolation computed for each receiver point in turn

        :param numpy.ndarray loc: Location of points to interpolate to
        :param numpy.ndarray x: Tensor vector of 1st dimension of grid.
        :param numpy.ndarray y: Tensor vector of 2nd dimension of grid. None by default.
        :param numpy.ndarray z: Tensor vector of 3rd dimension of grid. None by default.
        :rtype: scipy.sparse.csr_matrix
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

    npts = locs.shape[0]
    locs = locs.astype(float)
    x = x.astype(float)
    if y is None and z is None:
        shape = [x.size,]
        inds, vals = _interpmat1D(mkvc(locs), x)
    elif z is None:
        y = y.astype(float)
        shape = [x.size, y.size]
        inds, vals = _interpmat2D(locs, x, y)
    else:
        y = y.astype(float)
        z = z.astype(float)
        shape = [x.size, y.size, z.size]
        inds, vals = _interpmat3D(locs, x, y, z)

    I = np.repeat(range(npts),2**len(shape))
    J = sub2ind(shape,inds)
    Q = sp.csr_matrix((vals,(I, J)),
                      shape=(npts, np.prod(shape)))
    return Q

if not _interpCython:
    def _interp_point_1D(x, xr_i):
        """
            given a point, xr_i, this will find which two integers it lies between.

            :param numpy.ndarray x: Tensor vector of 1st dimension of grid.
            :param float xr_i: Location of a point
            :rtype: int,int,float,float
            :return: index1, index2, portion1, portion2
        """
        im = np.argmin(abs(x-xr_i))
        if xr_i - x[im] >= 0:  # Point on the left
            ind_x1 = im
            ind_x2 = im+1
        elif xr_i - x[im] < 0:  # Point on the right
            ind_x1 = im-1
            ind_x2 = im
        ind_x1 = max(min(ind_x1, x.size-1), 0)
        ind_x2 = max(min(ind_x2, x.size-1), 0)

        if ind_x1 == ind_x2:
            return ind_x1, ind_x1, 0.5, 0.5

        hx =  x[ind_x2] - x[ind_x1]
        wx1 = 1 - (xr_i - x[ind_x1])/hx
        wx2 = 1 - (x[ind_x2] - xr_i)/hx

        return ind_x1, ind_x2, wx1, wx2

    def _interpmat1D(locs, x):
        """Use interpmat with only x component provided."""
        nx = x.size
        npts = locs.shape[0]

        inds, vals = [], []

        for i in range(npts):
            ind_x1, ind_x2, wx1, wx2 = _interp_point_1D(x, locs[i])
            inds += [ind_x1, ind_x2]
            vals += [wx1,wx2]

        return inds, vals


    def _interpmat2D(locs, x, y):
        """Use interpmat with only x and y components provided."""
        nx = x.size
        ny = y.size
        npts = locs.shape[0]

        inds, vals = [], []

        for i in range(npts):
            ind_x1, ind_x2, wx1, wx2 = _interp_point_1D(x, locs[i, 0])
            ind_y1, ind_y2, wy1, wy2 = _interp_point_1D(y, locs[i, 1])

            inds += [( ind_x1,  ind_y1),
                     ( ind_x1,  ind_y2),
                     ( ind_x2,  ind_y1),
                     ( ind_x2,  ind_y2)]

            vals += [wx1*wy1,
                     wx1*wy2,
                     wx2*wy1,
                     wx2*wy2]

        return inds, vals



    def _interpmat3D(locs, x, y, z):
        """Use interpmat."""
        nx = x.size
        ny = y.size
        nz = z.size
        npts = locs.shape[0]

        inds, vals = [], []

        for i in range(npts):
            ind_x1, ind_x2, wx1, wx2 = _interp_point_1D(x, locs[i, 0])
            ind_y1, ind_y2, wy1, wy2 = _interp_point_1D(y, locs[i, 1])
            ind_z1, ind_z2, wz1, wz2 = _interp_point_1D(z, locs[i, 2])

            inds += [( ind_x1,  ind_y1,  ind_z1),
                     ( ind_x1,  ind_y2,  ind_z1),
                     ( ind_x2,  ind_y1,  ind_z1),
                     ( ind_x2,  ind_y2,  ind_z1),
                     ( ind_x1,  ind_y1,  ind_z2),
                     ( ind_x1,  ind_y2,  ind_z2),
                     ( ind_x2,  ind_y1,  ind_z2),
                     ( ind_x2,  ind_y2,  ind_z2)]

            vals += [wx1*wy1*wz1,
                     wx1*wy2*wz1,
                     wx2*wy1*wz1,
                     wx2*wy2*wz1,
                     wx1*wy1*wz2,
                     wx1*wy2*wz2,
                     wx2*wy1*wz2,
                     wx2*wy2*wz2]

        return inds, vals


if __name__ == '__main__':
    from SimPEG import *
    import matplotlib.pyplot as plt
    locs = np.random.rand(50)*0.8+0.1
    x = np.linspace(0,1,7)
    dense = np.linspace(0,1,200)
    fun = lambda x: np.cos(2*np.pi*x)
    Q = Utils.interpmat(locs, x)
    plt.plot(x, fun(x), 'bs-')
    plt.plot(dense, fun(dense), 'y:')
    plt.plot(locs, Q*fun(x), 'mo')
    plt.plot(locs, fun(locs), 'rx')
    plt.show()
