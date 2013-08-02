import numpy as np
from utils import mkvc
import scipy.sparse.linalg.dsolve as dsl
from InnerProducts import getFaceInnerProduct, getEdgeInnerProduct

def getMisfit(m,mesh,forward,param):

    mu0   = 4*np.pi*1e-7
    omega = forward['omega'] #[param['indomega']]
    rhs   = forward['rhs']   #[:,param['indrhs']]
    mis   = 0
    dmis  = m*0

    # Maxwell's system for E
    for i in range(len(omega)):
        for j in range(rhs.shape[1]):
            Curl  = mesh.edgeCurl
            #Grad  = mesh.nodalGrad
            sigma = np.exp(m)
            Me,PP = getEdgeInnerProduct(mesh,sigma)
            Mf    = 1/mu0 * getFaceInnerProduct(mesh)   # assume mu = mu0

            A = Curl.T * Mf * Curl - 1j * omega[i] * Me
            b = mkvc(np.array(rhs[:,j]))
            e = dsl.spsolve(A,b)
            e = mkvc(e,2)
            #print np.linalg.norm(A*e-b)/np.linalg.norm(b)
            P = forward['projection']
            d = P*e
            r = mkvc(d - param.dobs[i,j,:],2)

            mis  = mis + 0.5*(r.T*r)
            # get derivatives
            lam  = dsl.spsolve(A.T,P.T*r)
            lam  = mkvc(lam,2)
            Gij  =  - 1j * omega[i] * PP.T*sp.diag((PP*e)*mesh.vol) 
            dmis = dmis - Gij.T*lam


    return mis, dmis, d


    
if __name__ == '__main__':
    from TensorMesh import TensorMesh 
    from interpmat import interpmat
    from scipy import sparse as sp

    h = [np.ones(7),np.ones(8),np.ones(9)]
    mesh = TensorMesh(h)
    xs = np.array([3.1,4.3,5.4,6.5])
    ys = np.array([3.2,4.1,5.4,6.2])
    zs = np.array([4.3,4.2,4.1,4.1]);

    xyz = mesh.gridEx
    x   = xyz[:,0]; y = xyz[:,1]; z = xyz[:,2]
    Px  = interpmat(x,y,z,xs,ys,zs)
    xyz = mesh.gridEy
    x   = xyz[:,0]; y = xyz[:,1]; z = xyz[:,2]
    Py  = interpmat(x,y,z,xs,ys,zs)
    xyz = mesh.gridEz
    x   = xyz[:,0]; y = xyz[:,1]; z = xyz[:,2]
    Pz  = interpmat(x,y,z,xs,ys,zs)
    P   = sp.hstack((Px,Py,Pz))

    ne      = np.sum(mesh.nE)
    Q       = np.matrix(np.random.randn(ne,5))
    omega   = [1,2,3]
    forward = {'omega':omega, 'rhs':Q,'projection':P}
    dobs    = np.ones(np.size(xs),np.shape(Q,2),np.size(omega))
    param   = {'dobs':dobs}



    m = np.ones(mesh.nC)
    getMisfit(m,mesh,forward,param)
