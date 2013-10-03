import numpy as np
import unittest
from SimPEG import TensorMesh
from SimPEG.utils import ModelBuilder, sdiag
from SimPEG.forward import Problem, SyntheticProblem
from SimPEG.forward.DCProblem import DCProblem, DCutils
from TestUtils import checkDerivative
from scipy.sparse.linalg import dsolve


class DCProblemTests(unittest.TestCase):

    def setUp(self):
        # Create the mesh
        h1 = np.ones(20)
        h2 = np.ones(20)
        mesh = TensorMesh([h1,h2])

        # Create some parameters for the model
        sig1 = 1
        sig2 = 0.01

        # Create a synthetic model from a block in a half-space
        p0 = [2, 2]
        p1 = [5, 5]
        condVals = [sig1, sig2]
        mSynth = ModelBuilder.defineBlockConductivity(p0,p1,mesh.gridCC,condVals)

        # Set up the projection
        nelec = 10
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

        # Now set up the problem to do some minimization
        problem = DCProblem(mesh)
        problem.P = P
        problem.RHS = q
        problem.W = Wd
        problem.dobs = dobs

        self.p = problem
        self.mesh = mesh
        self.m0 = mSynth
        self.dobs = dobs


    def test_misfit(self):
        print 'SimPEG.forward.DCProblem: Testing Misfit'
        derChk = lambda m: [self.p.misfit(m), self.p.misfitDeriv(m)]
        passed = checkDerivative(derChk, self.m0, plotIt=False)
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        u = np.random.rand(self.mesh.nC)
        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.dobs.shape[0])
        wtJv = w.dot(self.p.J(self.m0, v, u=u))
        vtJtw = v.dot(self.p.Jt(self.m0, w, u=u))
        passed = (wtJv - vtJtw) < 1e-10
        self.assertTrue(passed)



if __name__ == '__main__':
    unittest.main()
