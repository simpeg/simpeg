import numpy as np
from utils import mkvc
import scipy.sparse.linalg.dsolve as dsl

def getMisfit(m,mesh,forward):

    mu0   = 4*np.pi*1e-7
    omega = forward['omega'] #[param['indomega']]
    rhs   = forward['rhs']   #[:,param['indrhs']]
    misfit = 0

    # Maxwell's system for E
    for i in range(len(omega)):
        for j in range(rhs.shape[1]):
            Curl  = mesh.edgeCurl
            #Grad  = mesh.nodalGrad
            sigma = np.exp(m)
            Me,PP = mesh.getEdgeMass(sigma)
            Mf    = 1/mu0 * mesh.getFaceMass()   # assume mu = mu0

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
            Gij  = PP.T*diag((PP*e)*mesh.vol) 
            dmis = dmis - Gij.T*lam



    
if __name__ == '__main__':
    from TensorMesh import TensorMesh 
    h = [np.ones(7),np.ones(8),np.ones(9)]
    mesh = TensorMesh(h)
    ne   = np.sum(mesh.nE)
    Q    = np.matrix(np.random.randn(ne,5))
    P    = np.matrix(Q.T)
    forward = {'omega':[1,2,3], 'rhs':Q,'projection':P}

    m = np.ones(mesh.nC)
    getMisfit(m,mesh,forward)
