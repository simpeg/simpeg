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

    for i in range(npts):
        # in x-direction
        im = np.argmin(abs(x-xr[i]))
        if  xr[i] - x[im] >= 0:  # Point on the left
            ind_x1 = im
            ind_x2 = im+1
        elif  xr[i] - x[im] < 0:  # Point on the right
            ind_x1 = im-1
            ind_x2 = im
        dx1 = xr[i] - x[ind_x1]
        dx2 = x[ind_x2] - xr[i]
        # in y-direction
        im = np.argmin(abs(y-yr[i]))
        if  yr[i] - y[im] >= 0:  # Point on the left
            ind_y1 = im
            ind_y2 = im+1
        elif  yr[i] - y[im] < 0:  # Point on the right
            ind_y1 = im-1
            ind_y2 = im
        dy1 = yr[i] - y[ind_y1]
        dy2 = y[ind_y2] - yr[i]
        # in z-direction
        im = np.argmin(abs(z-zr[i]))
        if  zr[i] - z[im] >= 0:  # Point on the left
            ind_z1 = im
            ind_z2 = im+1
        elif  zr[i] - z[im] < 0:  # Point on the right
            ind_z1 = im-1
            ind_z2 = im
        dz1 = zr[i] - z[ind_z1]
        dz2 = z[ind_z2] - zr[i]
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
