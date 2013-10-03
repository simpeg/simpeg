from SimPEG import TensorMesh
from SimPEG.forward import Problem, SyntheticProblem
from SimPEG.tests import checkDerivative
from SimPEG.utils import ModelBuilder, sdiag
import numpy as np
import scipy.sparse.linalg as linalg
import DCutils

class DCProblem(Problem):
    """
        **DCProblem**

        Geophysical DC resistivity problem.

    """
    def __init__(self, mesh):
        super(DCProblem, self).__init__(mesh)
        self.mesh.setCellGradBC('neumann')

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

    def field(self, m):
        A = self.createMatrix(m)
        solve = linalg.factorized(A)

        nRHSs = self.RHS.shape[1]  # Number of RHSs
        phi = np.zeros((self.mesh.nC, nRHSs)) + np.nan
        for ii in range(nRHSs):
            phi[:,ii] = solve(self.RHS[:,ii])

        return phi

    def J(self, m, v, u=None, solve=None):
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
        P = self.P
        D = self.mesh.faceDiv
        G = self.mesh.cellGrad
        A = self.createMatrix(m)
        Av_dm = self.mesh.getFaceMassDeriv()
        mT_dm = self.modelTransformDeriv(m)

        dCdu = A
        dCdm = D * ( sdiag( G * u ) * ( Av_dm * ( mT_dm * v ) ) )

        if solve is None:
            solve = linalg.factorized(dCdu)

        Jv = - P * solve(dCdm)
        return Jv

    def Jt(self, m, v, u=None, solve=None):
        P = self.P
        D = self.mesh.faceDiv
        G = self.mesh.cellGrad
        A = self.createMatrix(m)
        Av_dm = self.mesh.getFaceMassDeriv()
        mT_dm = self.modelTransformDeriv(m)

        dCdu = A.T

        if solve is None:
            solve = linalg.factorized(dCdu.tocsc())
        w = solve(P.T*v)

        Jtv = - mT_dm.T * ( Av_dm.T * ( sdiag( G * u ) * ( D.T * w ) ) )
        return Jtv


if __name__ == '__main__':
    # Create the mesh
    h1 = np.ones(100)
    h2 = np.ones(100)
    mesh = TensorMesh([h1,h2])

    # Create some parameters for the model
    sig1 = 1
    sig2 = 0.01

    # Create a synthetic model from a block in a half-space
    p0 = [20, 20]
    p1 = [50, 50]
    condVals = [sig1, sig2]
    mSynth = ModelBuilder.defineBlockConductivity(p0,p1,mesh.gridCC,condVals)
    mesh.plotImage(mSynth, showIt=False)


    # Set up the projection
    nelec = 50
    spacelec = 2
    surfloc = 0.5
    elecini = 0.5
    elecend = 0.5+spacelec*(nelec-1)
    elecLocR = np.linspace(elecini, elecend, nelec)
    rxmidLoc = (elecLocR[0:nelec-1]+elecLocR[1:nelec])*0.5
    q, Q, rxmidloc = DCutils.genTxRxmat(nelec, spacelec, surfloc, elecini, mesh)
    P = Q.T

    # Create some data
    class syntheticDCProblem(DCProblem, SyntheticProblem):
        pass

    synthetic = syntheticDCProblem(mesh);
    synthetic.P = P
    synthetic.RHS = q
    dobs, Wd = synthetic.createData(mSynth, std=0.05)

    u = synthetic.field(mSynth)
    mesh.plotImage(u[:,10], showIt=True)

    # Now set up the problem to do some minimization
    problem = DCProblem(mesh)
    problem.P = P
    problem.RHS = q
    problem.W = Wd
    problem.dobs = dobs
    m0 = mesh.gridCC[:,0]*0+sig1

    print problem.misfit(m0)
    print problem.misfit(mSynth)

    # Check Derivative
    derChk = lambda m: [problem.misfit(m), problem.misfitDeriv(m)]
    checkDerivative(derChk, mSynth)

    # Adjoint Test
    u = np.random.rand(mesh.nC)
    v = np.random.rand(mesh.nC)
    w = np.random.rand(dobs.shape[0])
    print w.dot(problem.J(mSynth, v, u=u))
    print v.dot(problem.Jt(mSynth, w, u=u))
