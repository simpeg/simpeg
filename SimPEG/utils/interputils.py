import numpy as np
import scipy.sparse as sp
from sputils import spzeros
from matutils import mkvc, sub2ind

def interpmat(x,y,z,xr,yr,zr):

    """ Local interpolation computed for each receiver point in turn """

    nx = x.size
    ny = y.size
    nz = z.size
    npts = xr.shape[0]

    Q = sp.lil_matrix((npts, nx*ny*nz))


    def inter1D(x, xr_i):
        im = np.argmin(abs(x-xr_i))
        if  xr_i - x[im] >= 0:  # Point on the left
            ind_x1 = im
            ind_x2 = im+1
        elif  xr_i - x[im] < 0:  # Point on the right
            ind_x1 = im-1
            ind_x2 = im
        dx1 = xr_i - x[ind_x1]
        dx2 = x[ind_x2] - xr_i
        return ind_x1, ind_x2, dx1, dx2

    for i in range(npts):
        # in x-direction
        ind_x1, ind_x2, dx1, dx2 = inter1D(x, xr[i])
        ind_y1, ind_y2, dy1, dy2 = inter1D(y, yr[i])
        ind_z1, ind_z2, dz1, dz2 = inter1D(z, zr[i])

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
    Q = Q.tocsr()
    return Q
