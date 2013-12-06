import numpy as np
import unittest
from SimPEG.mesh import TensorMesh
from SimPEG.utils import ModelBuilder, sdiag
from SimPEG.forward import Problem
from SimPEG.examples.DC import *
from TestUtils import checkDerivative
from scipy.sparse.linalg import dsolve
from SimPEG.regularization import Regularization
from SimPEG import inverse


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
        q, Q, rxmidloc = genTxRxmat(nelec, spacelec, surfloc, elecini, mesh)
        P = Q.T

        # Create some data

        problem = DCProblem(mesh)
        problem.P = P
        problem.RHS = q
        dobs, Wd = problem.createSyntheticData(mSynth, std=0.05)

        # Now set up the problem to do some minimization
        problem.W = Wd
        problem.dobs = dobs
        problem.std = dobs*0 + 0.05

        opt = inverse.InexactGaussNewton(maxIterLS=20, maxIter=10, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=6)
        reg = Regularization(mesh)
        inv = inverse.Inversion(problem, reg, opt, beta0=1e4)

        self.inv = inv
        self.reg = reg
        self.p = problem
        self.mesh = mesh
        self.m0 = mSynth
        self.dobs = dobs

    def test_misfit(self):
        derChk = lambda m: [self.p.dpred(m), lambda mx: self.p.J(self.m0, mx)]
        passed = checkDerivative(derChk, self.m0, plotIt=False)
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        u = np.random.rand(self.mesh.nC*self.p.RHS.shape[1])
        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.dobs.shape[0])
        wtJv = w.dot(self.p.J(self.m0, v, u=u))
        vtJtw = v.dot(self.p.Jt(self.m0, w, u=u))
        passed = (wtJv - vtJtw) < 1e-10
        self.assertTrue(passed)

    def test_dataObj(self):
        derChk = lambda m: [self.inv.dataObj(m), self.inv.dataObjDeriv(m)]
        checkDerivative(derChk, self.m0, plotIt=False)

    def test_modelObj(self):
        derChk = lambda m: [self.reg.modelObj(m), self.reg.modelObjDeriv(m)]
        checkDerivative(derChk, self.m0, plotIt=False)


if __name__ == '__main__':
    unittest.main()
