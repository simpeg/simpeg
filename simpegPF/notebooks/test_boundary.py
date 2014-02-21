
from SimPEG.Utils.sputils import kron3, speye, sdiag
from SimPEG import *
import numpy as np
import scipy.sparse as sp
def ddxFaceDivBC(n, bc):

    ij   = (np.array([0, n-1]),np.array([0, 1]))
    vals = np.zeros(2)

    # Set the first side
    if(bc[0] == 'dirichlet'):
        vals[0] = 0
    elif(bc[0] == 'neumann'):
        vals[0] = -1
    # Set the second side
    if(bc[1] == 'dirichlet'):
        vals[1] = 0
    elif(bc[1] == 'neumann'):
        vals[1] = 1
    D = sp.csr_matrix((vals, ij), shape=(n,2))
    return D


def faceDivBC(mesh, BC, ind):
    """
    The facd divergence boundary condtion matrix

    .. math::



    """
    # The number of cell centers in each direction
    n = mesh.nCv
    # Compute faceDivergence operator on faces
    if(mesh.dim == 1):
        D = ddxFaceDivBC(n[0], BC[0])
    elif(mesh.dim == 2):
        D1 = sp.kron(speye(n[1]), ddxFaceDivBC(n[0]), BC[0])
        D2 = sp.kron(ddxFaceDivBC(n[1], BC[1]), speye(n[0]))
        D = sp.hstack((D1, D2), format="csr")
    elif(mesh.dim == 3):
        D1 = kron3(speye(n[2]), speye(n[1]), ddxFaceDivBC(n[0], BC[0]))
        D2 = kron3(speye(n[2]), ddxFaceDivBC(n[1], BC[1]), speye(n[0]))
        D3 = kron3(ddxFaceDivBC(n[2], BC[2]), speye(n[1]), speye(n[0]))
    D = sp.hstack((D1, D2, D3), format="csr")
    # Compute areas of cell faces & volumes
    S = mesh.area[ind]
    V = mesh.vol
    mesh._faceDiv = sdiag(1/V)*D*sdiag(S)

    return mesh._faceDiv


def faceBCind(mesh):
    """
    Find indices of boundary faces in each direction

    """
    if(mesh.dim==1):
        indxd = (mesh.gridFx[:,0]==min(mesh.gridFx[:,0]))
        indxu = (mesh.gridFx[:,0]==max(mesh.gridFx[:,0]))
        return indxd, indxu
    elif(mesh.dim==1):
        indxd = (mesh.gridFx[:,0]==min(mesh.gridFx[:,0]))
        indxu = (mesh.gridFx[:,0]==max(mesh.gridFx[:,0]))
        indyd = (mesh.gridFy[:,1]==min(mesh.gridFy[:,1]))
        indyu = (mesh.gridFy[:,1]==max(mesh.gridFy[:,1]))
        return indxd, indxu, indyd, indyu
    elif(mesh.dim==3):
        indxd = (mesh.gridFx[:,0]==min(mesh.gridFx[:,0]))
        indxu = (mesh.gridFx[:,0]==max(mesh.gridFx[:,0]))
        indyd = (mesh.gridFy[:,1]==min(mesh.gridFy[:,1]))
        indyu = (mesh.gridFy[:,1]==max(mesh.gridFy[:,1]))
        indzd = (mesh.gridFz[:,2]==min(mesh.gridFz[:,2]))
        indzu = (mesh.gridFz[:,2]==max(mesh.gridFz[:,2]))
        return indxd, indxu, indyd, indyu, indzd, indzu


def spheremodel(mesh, x0, y0, z0, r):
    """
        Generate model indicies for sphere
        - (x0, y0, z0 ): is the center location of sphere
        - r: is the radius of the sphere
        - it returns logical indicies of cell-center model
    """
    ind = np.sqrt((mesh.gridCC[:,0]-x0)**2+(mesh.gridCC[:,1]-y0)**2+(mesh.gridCC[:,2]-z0)**2 ) < r
    return ind
  


def MagSphereAnalFun(x, y, z, R, x0, y0, z0, mu1, mu2, H0, flag):
    """ 
        Analytic function for Magnetics problem. The set up here is 
        magnetic sphere in whole-space.
        - (x0,y0,z0)
        - (x0, y0, z0 ): is the center location of sphere
        - r: is the radius of the sphere

    .. math::
    
        \mathbf{H}^p = H_0\hat{x}

                
    """
    if (~np.size(x)==np.size(y)==np.size(z)):
        print "Specify same size of x, y, z"
        return
    dim = x.shape
    x = Utils.mkvc(x)
    y = Utils.mkvc(y)
    z = Utils.mkvc(z)
    
    ind = np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2 ) < R
    r = Utils.mkvc(np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2 ))
    Bx = np.zeros(x.size)
    By = np.zeros(x.size)
    Bz = np.zeros(x.size)

    # Inside of the sphere
    rf2 = 3*mu1/(mu2+2*mu1)
    if (flag == 'total'):
        Bx[ind] = mu2*H0*(rf2)
    elif (flag == 'secondary'):
        Bx[ind] = mu2*H0*(rf2)-mu1*H0
        
    By[ind] = 0.
    Bz[ind] = 0.
    # Outside of the sphere
    rf1 = (mu2-mu1)/(mu2+2*mu1)
    if (flag == 'total'):
        Bx[~ind] = mu1*(H0+H0/r[~ind]**5*(R**3)*rf1*(2*x[~ind]**2-y[~ind]**2-z[~ind]**2))
    elif (flag == 'secondary'):
        Bx[~ind] = mu1*(H0/r[~ind]**5*(R**3)*rf1*(2*x[~ind]**2-y[~ind]**2-z[~ind]**2))
    
    By[~ind] = mu1*(H0/r[~ind]**5*(R**3)*rf1*(3*x[~ind]*y[~ind]))
    Bz[~ind] = mu1*(H0/r[~ind]**5*(R**3)*rf1*(3*x[~ind]*z[~ind]))
    
    return np.reshape(Bx, x.shape, order='F'), np.reshape(By, x.shape, order='F'), np.reshape(Bz, x.shape, order='F')
    
    
