import numpy as np
import scipy.sparse as sp
from SimPEG import utils, TensorMesh
from SimPEG.utils import spzeros, mkvc

def interpmat(x,y,z,xr,yr,zr):

    """ Local nterpolation computed for each receiver point in turn """

    nx = max(x.shape)
    ny = max(y.shape)
    nz = max(z.shape)
    npts = max(xr.shape)

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

        inds = utils.sub2ind((nx,ny,nz),[ 
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

def getInterpmat(mesh, rxLoc, dataType):
    """ """
    xr = rxLoc[:,0]
    yr = rxLoc[:,1]
    zr = rxLoc[:,2]
    nrx = rxLoc.shape[0]
    if dataType == 'fx':
        Qx = interpmat(np.unique(mesh.gridFx[:,0]),
                       np.unique(mesh.gridFx[:,1]),
                       np.unique(mesh.gridFx[:,2]),
                       xr,yr,zr)
        Q = sp.hstack([Qx,spzeros(nrx,mesh.nF[1]),spzeros(nrx,mesh.nF[2])])        
    elif dataType == 'fy':
        Qy = interpmat(np.unique(mesh.gridFy[:,0]),
                       np.unique(mesh.gridFy[:,1]),
                       np.unique(mesh.gridFy[:,2]),
                       xr,yr,zr)
        Q = sp.hstack([spzeros(nrx,mesh.nF[0]),Qy,spzeros(nrx,mesh.nF[2])])        
    elif dataType == 'fz':
        Qz = interpmat(np.unique(mesh.gridFz[:,0]),
                       np.unique(mesh.gridFz[:,1]),
                       np.unique(mesh.gridFz[:,2]),
                       xr,yr,zr)
        Q = sp.hstack([spzeros(nrx,mesh.nF[0]),spzeros(nrx,mesh.nF[1]),Qz])
    elif dataType == 'ex':
        Qx = interpmat(np.unique(mesh.gridEx[:,0]),
                       np.unique(mesh.gridEx[:,1]),
                       np.unique(mesh.gridEx[:,2]),
                       xr, yr, zr)
        Q = sp.hstack([Qx,spzeros(nrx,mesh.nE[1]),spzeros(nrx,mesh.nE[2])])
    elif dataType == 'ey':
        Qy = interpmat(np.unique(mesh.gridEy[:,0]),
                       np.unique(mesh.gridEy[:,1]),
                       np.unique(mesh.gridEy[:,2]),
                       xr, yr, zr)
        Q = sp.hstack([spzeros(nrx,mesh.nE[0]),Qy,spzeros(nrx,mesh.nE[2])])       
    elif dataType == 'ez':
        Qz = interpmat(np.unique(mesh.gridEz[:,0]),
                       np.unique(mesh.gridEz[:,1]),
                       np.unique(mesh.gridEz[:,2]),
                       xr,yr,zr)
        Q = sp.hstack([spzeros(nrx,mesh.nE[0]),spzeros(nrx,mesh.nE[1]),Qz])
    else:
        assert(True), "Input either face (fx, fy, fz) or edge (ex, ey, ez) option"
    return Q

if __name__ == '__main__':
    pad = 1
    padfactor = 1.5
    cs = 100
    xpad = cs*(np.ones(pad)*padfactor)**np.arange(pad)
    ypad = cs*(np.ones(pad)*padfactor)**np.arange(pad)
    zpad = cs*(np.ones(pad)*padfactor)**np.arange(pad)

    core = 10
    xcore = cs*np.ones(core)
    ycore = cs*np.ones(core)
    zcore = cs*np.ones(core)

    hx = np.r_[xpad[::-1],xcore, cs, xcore,xpad]
    hy = np.r_[ypad[::-1],ycore, cs, ycore, ypad]
    hz = np.r_[zpad[::-1],zcore,zcore, zpad]
    x0 = np.array([-np.sum(hx)/2, -np.sum(hy)/2, -np.sum(hz)/2], )
    mesh = TensorMesh([hx, hy, hz],x0)

    xr1 = np.linspace(-500,500,5)
    yr1 = np.linspace(-500,500,5)
    zr1 = 0
    xr, yr = np.meshgrid(xr1, yr1, indexing='ij')
    zr = np.ones((xr.shape[0],xr.shape[1]))*zr1
    xr = mkvc(xr)
    yr = mkvc(yr)
    zr = mkvc(zr)
    rxLoc = np.c_[xr, yr, zr]
    Q = getInterpmat(mesh, rxLoc, 'ex')

    print Q

