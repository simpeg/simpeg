# from __future__ import division
import numpy as np
cimport numpy as np
# from libcpp.vector cimport vector

def _interp_point_1D(np.ndarray[np.float64_t, ndim=1] x, float xr_i):
    """
        given a point, xr_i, this will find which two integers it lies between.

        :param numpy.ndarray x: Tensor vector of 1st dimension of grid.
        :param float xr_i: Location of a point
        :rtype: int,int,float,float
        :return: index1, index2, portion1, portion2
    """
    # TODO: This fails if the point is on the outside of the mesh.
    #       We may want to replace this by extrapolation?
    cdef int im = np.argmin(abs(x-xr_i))
    cdef int ind_x1 = 0
    cdef int ind_x2 = 0
    cdef int xSize = x.shape[0]-1
    cdef float wx1 = 0.0
    cdef float wx2 = 0.0
    cdef float hx  = 0.0

    if xr_i - x[im] >= 0:  # Point on the left
        ind_x1 = im
        ind_x2 = im+1
    elif xr_i - x[im] < 0:  # Point on the right
        ind_x1 = im-1
        ind_x2 = im
    ind_x1 = max(min(ind_x1, xSize), 0)
    ind_x2 = max(min(ind_x2, xSize), 0)

    if ind_x1 == ind_x2:
        return ind_x1, ind_x1, 0.5, 0.5

    hx =  x[ind_x2] - x[ind_x1]
    wx1 = 1 - (xr_i - x[ind_x1])/hx
    wx2 = 1 - (x[ind_x2] - xr_i)/hx

    return ind_x1, ind_x2, wx1, wx2


def _interpmat1D(np.ndarray[np.float64_t, ndim=1] locs,
                 np.ndarray[np.float64_t, ndim=1] x):
    """Use interpmat with only x component provided."""
    cdef int nx = x.size
    cdef int npts = locs.shape[0]

    inds, vals = [], []

    for i in range(npts):
        ind_x1, ind_x2, wx1, wx2 = _interp_point_1D(x, locs[i])
        inds += [ind_x1, ind_x2]
        vals += [wx1,wx2]

    return inds, vals


def _interpmat2D(np.ndarray[np.float64_t, ndim=2] locs,
                 np.ndarray[np.float64_t, ndim=1] x,
                 np.ndarray[np.float64_t, ndim=1] y):
    """Use interpmat with only x and y components provided."""
    cdef int nx = x.size
    cdef int ny = y.size
    cdef int npts = locs.shape[0]

    inds, vals = [], []

    for i in range(npts):
        ind_x1, ind_x2, wx1, wx2 = _interp_point_1D(x, locs[i, 0])
        ind_y1, ind_y2, wy1, wy2 = _interp_point_1D(y, locs[i, 1])

        inds += [( ind_x1,  ind_y1),
                 ( ind_x1,  ind_y2),
                 ( ind_x2,  ind_y1),
                 ( ind_x2,  ind_y2)]

        vals += [wx1*wy1, wx1*wy2, wx2*wy1, wx2*wy2]

    return inds, vals


def _interpmat3D(np.ndarray[np.float64_t, ndim=2] locs,
                 np.ndarray[np.float64_t, ndim=1] x,
                 np.ndarray[np.float64_t, ndim=1] y,
                 np.ndarray[np.float64_t, ndim=1] z):
    """Use interpmat."""
    cdef int nx = x.size
    cdef int ny = y.size
    cdef int nz = z.size
    cdef int npts = locs.shape[0]

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
