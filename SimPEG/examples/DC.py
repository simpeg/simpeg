from SimPEG.mesh import TensorMesh
from SimPEG.forward import Problem, ModelTransforms
from SimPEG.tests import checkDerivative
from SimPEG.utils import ModelBuilder, sdiag, mkvc
from SimPEG import Solver
import numpy as np
import scipy.sparse as sp


class DCProblem(ModelTransforms.LogModel, Problem):
    """
        **DCProblem**

        Geophysical DC resistivity problem.

    """
    def __init__(self, mesh):
        Problem.__init__(self, mesh)
        self.mesh.setCellGradBC('neumann')

    def reshapeFields(self, u):
        if len(u.shape) == 1:
            u = u.reshape([-1, self.RHS.shape[1]], order='F')
        return u

    def createMatrix(self, m):
        """
            Makes the matrix A(m) for the DC resistivity problem.

            :param numpy.array m: model
            :rtype: scipy.csc_matrix
            :return: A(m)

            .. math::
                c(m,u) = A(m)u - q = G\\text{sdiag}(M(mT(m)))Du - q = 0

            Where M() is the mass matrix and mT is the model transform.
        """
        D = self.mesh.faceDiv
        G = self.mesh.cellGrad
        sigma = self.modelTransform(m)
        Msig = self.mesh.getFaceMass(sigma)
        A = D*Msig*G
        return A.tocsc()

    def dpred(self, m, u=None):
        """
            Predicted data.

            .. math::
                d_\\text{pred} = Pu(m)
        """
        if u is None:
            u = self.field(m)

        u = self.reshapeFields(u)

        return mkvc(self.P*u)

    def field(self, m):
        A = self.createMatrix(m)
        solve = Solver(A)
        phi = solve.solve(self.RHS)
        return mkvc(phi)

    def J(self, m, v, u=None):
        """
            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: Jv

            .. math::
                c(m,u) = A(m)u - q = G\\text{sdiag}(M(mT(m)))Du - q = 0

                \\nabla_u (A(m)u - q) = A(m)

                \\nabla_m (A(m)u - q) = G\\text{sdiag}(Du)\\nabla_m(M(mT(m)))

            Where M() is the mass matrix and mT is the model transform.

            .. math::
                J = - P \left( \\nabla_u c(m, u) \\right)^{-1} \\nabla_m c(m, u)

                J(v) = - P ( A(m)^{-1} ( G\\text{sdiag}(Du)\\nabla_m(M(mT(m))) v ) )
        """
        if u is None:
            u = self.field(m)

        u = self.reshapeFields(u)

        P = self.P
        D = self.mesh.faceDiv
        G = self.mesh.cellGrad
        A = self.createMatrix(m)
        Av_dm = self.mesh.getFaceMassDeriv()
        mT_dm = self.modelTransformDeriv(m)

        dCdu = A

        dCdm = np.empty_like(u)
        for i, ui in enumerate(u.T):  # loop over each column
            dCdm[:, i] = D * ( sdiag( G * ui ) * ( Av_dm * ( mT_dm * v ) ) )

        solve = Solver(dCdu)
        Jv = - P * solve.solve(dCdm)
        return mkvc(Jv)

    def Jt(self, m, v, u=None):
        """Takes data, turns it into a model..ish"""

        if u is None:
            u = self.field(m)

        u = self.reshapeFields(u)
        v = self.reshapeFields(v)

        P = self.P
        D = self.mesh.faceDiv
        G = self.mesh.cellGrad
        A = self.createMatrix(m)
        Av_dm = self.mesh.getFaceMassDeriv()
        mT_dm = self.modelTransformDeriv(m)

        dCdu = A.T
        solve = Solver(dCdu)

        w = solve.solve(P.T*v)

        Jtv = 0
        for i, ui in enumerate(u.T):  # loop over each column
            Jtv += sdiag( G * ui ) * ( D.T * w[:,i] )

        Jtv = - mT_dm.T * ( Av_dm.T * Jtv )
        return Jtv



def genTxRxmat(nelec, spacelec, surfloc, elecini, mesh):
    """ Generate projection matrix (Q) and """
    elecend = 0.5+spacelec*(nelec-1)
    elecLocR = np.linspace(elecini, elecend, nelec)
    elecLocT = elecLocR+1
    nrx = nelec-1
    ntx = nelec-1
    q = np.zeros((mesh.nC, ntx))
    Q = np.zeros((mesh.nC, nrx))

    for i in range(nrx):

        rxind1 = np.argwhere((mesh.gridCC[:,0]==surfloc) & (mesh.gridCC[:,1]==elecLocR[i]))
        rxind2 = np.argwhere((mesh.gridCC[:,0]==surfloc) & (mesh.gridCC[:,1]==elecLocR[i+1]))

        txind1 = np.argwhere((mesh.gridCC[:,0]==surfloc) & (mesh.gridCC[:,1]==elecLocT[i]))
        txind2 = np.argwhere((mesh.gridCC[:,0]==surfloc) & (mesh.gridCC[:,1]==elecLocT[i+1]))

        q[txind1,i] = 1
        q[txind2,i] = -1
        Q[rxind1,i] = 1
        Q[rxind2,i] = -1

    Q = sp.csr_matrix(Q)
    rxmidLoc = (elecLocR[0:nelec-1]+elecLocR[1:nelec])*0.5
    return q, Q, rxmidLoc



if __name__ == '__main__':

    from SimPEG import inverse
    import matplotlib.pyplot as plt

    # Create the mesh
    h1 = np.ones(20)
    h2 = np.ones(100)
    mesh = TensorMesh([h1,h2])

    # Create some parameters for the model
    sig1 = np.log(1)
    sig2 = np.log(0.01)

    # Create a synthetic model from a block in a half-space
    p0 = [5, 10]
    p1 = [15, 50]
    condVals = [sig1, sig2]
    mSynth = ModelBuilder.defineBlockConductivity(p0,p1,mesh.gridCC,condVals)
    plt.colorbar(mesh.plotImage(mSynth))
    plt.show()

    # Set up the projection
    nelec = 50
    spacelec = 2
    surfloc = 0.5
    elecini = 0.5
    elecend = 0.5+spacelec*(nelec-1)
    elecLocR = np.linspace(elecini, elecend, nelec)
    rxmidLoc = (elecLocR[0:nelec-1]+elecLocR[1:nelec])*0.5
    q, Q, rxmidloc = genTxRxmat(nelec, spacelec, surfloc, elecini, mesh)
    P = Q.T

    # Create some data
    problem = DCProblem(mesh)
    problem.P = P
    problem.RHS = q
    data = problem.createSyntheticData(mSynth, std=0.05)

    u = problem.field(mSynth)
    u = problem.reshapeFields(u)
    mesh.plotImage(u[:,10])
    # plt.show()

    # Now set up the problem to do some minimization
    # problem.dobs = dobs
    # problem.std = dobs*0 + 0.05
    m0 = mesh.gridCC[:,0]*0+sig2

    opt = inverse.InexactGaussNewton(maxIterLS=20, maxIter=10, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=6)
    reg = inverse.Regularization(mesh)
    inv = inverse.Inversion(problem, reg, opt, data, beta0=1e4)

    # Check Derivative
    derChk = lambda m: [inv.dataObj(m), inv.dataObjDeriv(m)]
    checkDerivative(derChk, mSynth)



    print inv.dataObj(m0)
    print inv.dataObj(mSynth)

    m = inv.run(m0)

    plt.colorbar(mesh.plotImage(m))
    print m
    plt.show()






