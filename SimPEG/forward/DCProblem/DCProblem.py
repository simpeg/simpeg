from SimPEG import TensorMesh
from SimPEG.forward import Problem, SyntheticProblem
from SimPEG.utils import ModelBuilder
import numpy as np
import scipy.sparse.linalg as linalg
import DCutils

class DCProblem(Problem):
    """docstring for DCProblem"""
    def __init__(self, mesh):
        super(DCProblem, self).__init__(mesh)
        self.mesh.setCellGradBC('neumann')

    def createMatrix(self, m):
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

    def J(self, m, v, u=None, RHSii=0, solve=None):
        P = self.P
        D = self.mesh.faceDiv
        G = self.mesh.cellGrad
        A = self.createMatrix(m)
        Av_dm = self.mesh.getFaceMassDeriv()
        mT_dm = self.modelTransform(m)

        dCdu = A
        dCdm = - D * ( sdiag( G * u[:, RHSii] ) * ( Av_dm * ( mT_dm * v ) ) )

        if solve is None:
            solve = linalg.factorized(dCdu)

        return - P * solve(dCdm)



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


    # Create some data
    class syntheticDCProblem(DCProblem, SyntheticProblem):
        pass

    synthetic = syntheticDCProblem(mesh);
    synthetic.P = Q.T
    synthetic.RHS = q
    dobs, Wd = synthetic.createData(mSynth)

    # Now set up the problem to do some minimization
    problem = DCProblem(mesh)



