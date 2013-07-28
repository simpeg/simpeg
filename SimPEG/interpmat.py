from scipy import sparse as sp
from sputils import sdiag
from utils import sub2ind, ndgrid, mkvc
import numpy as np

def interpmat(x,y,z,xr,yr,zr):
#
# This function does local linear interpolation
# computed for each receiver point in turn
#
# [Q] = linint(x,y,z,xr,yr,zr)
# Interpolation matrix 
#

    nx = size(x)
    ny = size(y)
    nz = size(z)

    np = size(xr)

    #Q = spalloc(np,nx*ny*nz,8*np);
    Q = sparse.coo_matrix((0.0,(0,0)),shape=(nx*ny*nz,8*np))
    
    for i in range(0, np): 
        im = amin(abs(xr[i]-x))
        if  xr[i] - x[im] >= 0:  # Point on the left 
                 ind_x[0] = im;   ind_x[1] = im+1
        else:                    # Point on the right
                 ind_x[0] = im-1; ind_x[1] = im
       
       
        dx[0] = xr[i] - x[ind_x[0]]
        dx[1] = x[ind_x[1]] - xr[i]

        im = amin(abs(yr[i] - y)) 
        if  yr[i] - y[im] >= 0:     # Point on the left
            ind_y[0] = im; ind_y[1] = im+1
        else:                       # Point on the right
            ind_y[0] = im-1; ind_y[1] = im
            

        dy[0] = yr[i] - y[ind_y[0]]
        dy[1] = y[ind_y[1]] - yr[i];

        im = amin(abs(zr[i] - z));
        if  zr(i) -z(im) >= 0:  # Point on the left
            ind_z[0] = im;  ind_z[1] = im+1
        else:                    # Point on the right
            ind_z[0] = im-1; ind_z[1] = im;

        dz[0] = zr[i] - z[ind_z[0]]; dz[1] = z[ind_z[1]] - zr[i]      

        Dx =  x[ind_x[1]] - x[ind_x[0]]
        Dy =  y[ind_y[1]] - y[ind_y[0]]
        Dz =  z[ind_z[1]] - z[ind_z[0]]  
        dv = Dx*Dy*Dz

      # Get the row in the matrix
        v = zeros([nx, ny,nz]);

        v[ ind_x[0],  ind_y[0],  ind_z[0]] = (1-dx[0]/Dx)*(1-dy[0]/Dy)*(1-dz[0]/Dz)
        v[ ind_x[0],  ind_y[1],  ind_z[0]] = (1-dx[0]/Dx)*(1-dy[1]/Dy)*(1-dz[0]/Dz);
        v[ ind_x[1],  ind_y[0],  ind_z[0]] = (1-dx[1]/Dx)*(1-dy[0]/Dy)*(1-dz[0]/Dz);
        v[ ind_x[1],  ind_y[1],  ind_z[0]] = (1-dx[1]/Dx)*(1-dy[1]/Dy)*(1-dz[0]/Dz);
        v[ ind_x[0],  ind_y[0],  ind_z[1]] = (1-dx[0]/Dx)*(1-dy[0]/Dy)*(1-dz[1]/Dz);
        v[ ind_x[0],  ind_y[1],  ind_z[1]] = (1-dx[0]/Dx)*(1-dy[1]/Dy)*(1-dz[1]/Dz);
        v[ ind_x[1],  ind_y[0],  ind_z[1]] = (1-dx[1]/Dx)*(1-dy[0]/Dy)*(1-dz[1]/Dz);
        v[ ind_x[1],  ind_y[1],  ind_z[1]] = (1-dx[1]/Dx)*(1-dy[1]/Dy)*(1-dz[1]/Dz);

     
        Q[i,:] = v.flatten('F')
        
    return Q
    
    
if __name__ == '__main__':
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 2, 3, 4, 5])
    z = np.array([0, 1, 4, 6])
  
    xr = np.array([2.5,3.2])
    yr = np.array([2.4,3.6])
    zr = np.array([2.5,3.9])
      
    A = interpmat(x,y,z,xr,yr,zr)
   