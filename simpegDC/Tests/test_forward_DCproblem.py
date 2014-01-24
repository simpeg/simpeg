from SimPEG import *
import unittest
import simpegDC


class DCProblemTests(unittest.TestCase):

    def setUp(self):
        # Create the mesh
        h1 = np.ones(20)
        h2 = np.ones(20)
        mesh = Mesh.TensorMesh([h1,h2])
        model = Model.BaseModel(mesh)

        # Create some parameters for the model
        sig1 = 1
        sig2 = 0.01

        # Create a synthetic model from a block in a half-space
        p0 = [2, 2]
        p1 = [5, 5]
        condVals = [sig1, sig2]
        mSynth = Utils.ModelBuilder.defineBlockConductivity(mesh.gridCC,p0,p1,condVals)

        # Set up the projection
        nelec = 10
        spacelec = 2
        surfloc = 0.5
        elecini = 0.5
        elecend = 0.5+spacelec*(nelec-1)
        elecLocR = np.linspace(elecini, elecend, nelec)
        rxmidLoc = (elecLocR[0:nelec-1]+elecLocR[1:nelec])*0.5
        q, Q, rxmidloc = simpegDC.genTxRxmat(nelec, spacelec, surfloc, elecini, mesh)
        P = Q.T
        Q = Q.toarray()

        # Create some data

        prob = simpegDC.DCProblem(mesh, model)
        data = prob.createSyntheticData(mSynth, std=0.05, P=P, RHS=Q)

        # Now set up the problem to do some minimization
        opt = Optimization.InexactGaussNewton(maxIterLS=20, maxIter=10, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=6)
        reg = Regularization.Tikhonov(model)
        objFunc = ObjFunction.BaseObjFunction(data, reg, beta=1e4)
        inv = Inversion.BaseInversion(objFunc, opt)

        self.inv = inv
        self.reg = reg
        self.p = prob
        self.mesh = mesh
        self.m0 = mSynth
        self.data = data
        self.objFunc = objFunc

    def test_misfit(self):
        derChk = lambda m: [self.data.dpred(m), lambda mx: self.p.J(self.m0, mx)]
        passed = Tests.checkDerivative(derChk, self.m0, plotIt=False)
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        u = np.random.rand(self.mesh.nC*self.data.RHS.shape[1])
        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.data.dobs.shape[0])
        wtJv = w.dot(self.p.J(self.m0, v, u=u))
        vtJtw = v.dot(self.p.Jt(self.m0, w, u=u))
        passed = (wtJv - vtJtw) < 1e-10
        self.assertTrue(passed)

    def test_dataObj(self):
        derChk = lambda m: [self.objFunc.dataObj(m), self.objFunc.dataObjDeriv(m)]
        Tests.checkDerivative(derChk, self.m0, plotIt=False)


if __name__ == '__main__':
    unittest.main()
