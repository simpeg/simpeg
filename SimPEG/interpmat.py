from scipy import sparse as sp
import numpy as np

def interpmat(x,y,z,xr,yr,zr):
#
# This function does local linear interpolation
# computed for each receiver point in turn
#
# [Q] = linint(x,y,z,xr,yr,zr)
# Interpolation matrix 
#

    nx = np.size(x)
    ny = np.size(y)
    nz = np.size(z)

    nps = np.size(xr)

    #Q = spalloc(np,nx*ny*nz,8*np);
    Q = sp.lil_matrix((nps,nx*ny*nz))
    ind_x = np.array([0,0])
    ind_y = np.array([0,0])
    ind_z = np.array([0,0])
    dx, dy, dz = np.zeros(2), np.zeros(2), np.zeros(2)
    for i in range(0, nps): 
        im = np.amin(abs(xr[i]-x))
        if  xr[i] - x[im] >= 0:  # Point on the left 
                 ind_x[0] = im;   ind_x[1] = im+1
        else:                    # Point on the right
                 ind_x[0] = im-1; ind_x[1] = im
       
       
        dx[0] = xr[i] - x[ind_x[0]]
        dx[1] = x[ind_x[1]] - xr[i]

        im = np.amin(abs(yr[i] - y)) 
        if  yr[i] - y[im] >= 0:     # Point on the left
            ind_y[0] = im; ind_y[1] = im+1
        else:                       # Point on the right
            ind_y[0] = im-1; ind_y[1] = im
            

        dy[0] = yr[i] - y[ind_y[0]]
        dy[1] = y[ind_y[1]] - yr[i];

        im = np.amin(abs(zr[i] - z));
        if  zr[i] -z[im] >= 0:  # Point on the left
            ind_z[0] = im;  ind_z[1] = im+1
        else:                    # Point on the right
            ind_z[0] = im-1; ind_z[1] = im;

        dz[0] = zr[i] - z[ind_z[0]]; dz[1] = z[ind_z[1]] - zr[i]      

        Dx =  x[ind_x[1]] - x[ind_x[0]]
        Dy =  y[ind_y[1]] - y[ind_y[0]]
        Dz =  z[ind_z[1]] - z[ind_z[0]]  
        #dv = Dx*Dy*Dz

      # Get the row in the matrix
        v = np.zeros([nx, ny,nz])

        v[ ind_x[0],  ind_y[0],  ind_z[0]] = (1-dx[0]/Dx)*(1-dy[0]/Dy)*(1-dz[0]/Dz)
        v[ ind_x[0],  ind_y[1],  ind_z[0]] = (1-dx[0]/Dx)*(1-dy[1]/Dy)*(1-dz[0]/Dz)
        v[ ind_x[1],  ind_y[0],  ind_z[0]] = (1-dx[1]/Dx)*(1-dy[0]/Dy)*(1-dz[0]/Dz)
        v[ ind_x[1],  ind_y[1],  ind_z[0]] = (1-dx[1]/Dx)*(1-dy[1]/Dy)*(1-dz[0]/Dz)
        v[ ind_x[0],  ind_y[0],  ind_z[1]] = (1-dx[0]/Dx)*(1-dy[0]/Dy)*(1-dz[1]/Dz)
        v[ ind_x[0],  ind_y[1],  ind_z[1]] = (1-dx[0]/Dx)*(1-dy[1]/Dy)*(1-dz[1]/Dz)
        v[ ind_x[1],  ind_y[0],  ind_z[1]] = (1-dx[1]/Dx)*(1-dy[0]/Dy)*(1-dz[1]/Dz)
        v[ ind_x[1],  ind_y[1],  ind_z[1]] = (1-dx[1]/Dx)*(1-dy[1]/Dy)*(1-dz[1]/Dz)

     
        print(np.shape(v.flatten('F')))
        print(np.shape(Q))
        
        Q[i,:] = v.flatten('F')
        
     
    return Q.tocsr()
    
    
if __name__ == '__main__':
      
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 2, 3, 4, 5])
    z = np.array([0, 1, 4, 6])
  
    xr = np.array([2.5,3.2])
    yr = np.array([2.4,3.6])
    zr = np.array([2.5,3.9])
      
    A = interpmat(x,y,z,xr,yr,zr)
   