try:
    import sys
    sys.path.append('/Users/haber/dropbox/simpegMaster/')
except Exception, e:
    pass

import SimPEG
import scipy.sparse as sp
import numpy as np
import SimPEG.utils as utils
import matplotlib.pylab as plt
from SimPEG.mesh import TensorMesh


def HelmholtzSol(model,mesh,w,mbc,q,P):

    n1 = mesh.nCx; n2 = mesh.nCy;
    h1 = mesh.hx;  h2 = mesh.hy;

    k = np.sqrt(w**2 * mbc)

    # D1 = utils.sdiag(1./h1) * utils.ddx(mesh.nCx)
    # D2 = utils.sdiag(1./h2) * utils.ddx(mesh.nCy)

    # L1 = - D1.T*D1
    # L2 = - D2.T*D2

    Av = mesh.aveN2CC

    B1 = utils.spzeros(n1+1,n1+1);
    B2 = utils.spzeros(n2+1,n2+1);

    B1.dtype = complex
    B2.dtype = complex
    B1[0,0] = 2*1j*k*h1[0]; B1[-1,-1] = 2*1j*k*h1[-1]
    B2[0,0] = 2*1j*k*h2[0]; B2[-1,-1] = 2*1j*k*h2[-1]

    # generate the 2D Laplacian
    # L = sp.kron(sp.identity(n2+1),L1) + sp.kron(L2,sp.identity(n1+1))
    L = mesh.nodalLaplacian
    B = sp.kron(sp.identity(n2+1),B1) + sp.kron(B2,sp.identity(n1+1))
    L = L+B
    #plt.spy(L)
    #plt.show()
    # Generate the Mass matrix
    M = utils.sdiag(Av.T*utils.mkvc(model))
    A = - L - w**2 * M

    mesh.ForModMat = A
    u = sp.linalg.spsolve(A,q);
    d = P*u
    return u, d

def HelmholtzJmatVec(v,model,u,mesh,w,mbc,P):

    Cm = -w**2 * utils.sdiag(u)*mesh.aveN2CC.T
    Cu = mesh.ForModMat
    Cmv = Cm*v;
    lam = sp.linalg.spsolve(Cu,Cmv);
    return -P*lam

def HelmholtzJTmatVec(v,model,u,mesh,w,mbc,P):

    Cm = -w**2 * utils.sdiag(u)*mesh.aveN2CC.T
    Cu = mesh.ForModMat
    Pv = -P.T*v
    z = sp.linalg.spsolve(Cu.T,Pv);
    return Cm.T*z


if __name__ == '__main__':
    # odel,mesh,w,mbc,q
    n1 = 128; n2 = 128
    h1 = np.ones(n1); h2 = np.ones(n2);
    P  = sp.identity((n1+1)*(n2+1))

    mesh  = TensorMesh([h1,h2])
    model = np.ones(mesh.nC)
    w     = 1
    mbc   = 1
    q     = np.zeros((mesh.nNx,mesh.nNy))
    q[n1/2,n2/2] = 1.0
    q     = q.reshape(mesh.nN,order = 'F')

    u, d = HelmholtzSol(model,mesh,w,mbc,q,P)
    u = u.reshape((mesh.nCx+1,mesh.nCy+1),order = 'F')

    plt.imshow(u.real)
    plt.show()

    dm = np.random.rand(mesh.nC)*1e-1+2
    u1, d1 = HelmholtzSol(model+dm,mesh,w,mbc,q,P)

    dd = HelmholtzJmatVec(dm,model,u,mesh,w,mbc,P)

    print np.linalg.norm(d1-d)
    print np.linalg.norm(d1-d-dd)




