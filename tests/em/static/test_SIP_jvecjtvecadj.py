from __future__ import print_function
import unittest
from SimPEG import (Mesh, Utils, EM, Maps, np, Survey, DataMisfit,
                    Regularization, Optimization, Inversion, InvProblem, Tests)
from SimPEG.EM.Static import SIP, DC, IP
try:
    from pymatsolver import MumpsSolver
    Solver = MumpsSolver
except ImportError:
    from SimPEG import SolverLU
    Solver = SolverLU

np.random.seed(38)

class IPProblemTestsCC(unittest.TestCase):

    def setUp(self):

        cs = 25.
        hx = [(cs,0, -1.3),(cs,21),(cs,0, 1.3)]
        hy = [(cs,0, -1.3),(cs,21),(cs,0, 1.3)]
        hz = [(cs,0, -1.3),(cs,20)]
        mesh = Mesh.TensorMesh([hx, hy, hz],x0="CCN")
        blkind0 = Utils.ModelBuilder.getIndicesSphere(np.r_[-100., -100., -200.], 75., mesh.gridCC)
        blkind1 = Utils.ModelBuilder.getIndicesSphere(np.r_[100., 100., -200.], 75., mesh.gridCC)
        sigma = np.ones(mesh.nC)*1e-2
        eta = np.zeros(mesh.nC)
        tau = np.ones_like(sigma)*1.
        eta[blkind0] = 0.1
        eta[blkind1] = 0.1
        tau[blkind0] = 0.1
        tau[blkind1] = 0.01

        x = mesh.vectorCCx[(mesh.vectorCCx>-155.)&(mesh.vectorCCx<155.)]
        y = mesh.vectorCCx[(mesh.vectorCCy>-155.)&(mesh.vectorCCy<155.)]
        Aloc = np.r_[-200., 0., 0.]
        Bloc = np.r_[200., 0., 0.]
        M = Utils.ndgrid(x-25.,y, np.r_[0.])
        N = Utils.ndgrid(x+25.,y, np.r_[0.])

        times = np.arange(10)*1e-3 + 1e-3
        rx = SIP.Rx.Dipole(M, N, times)
        src = SIP.Src.Dipole([rx], Aloc, Bloc)
        survey = SIP.Survey([src])
        colemap = [("eta", Maps.IdentityMap(mesh)), ("taui", Maps.IdentityMap(mesh))]
        problem = SIP.Problem3D_CC(mesh, rho=1./sigma, mapping=colemap)
        problem.Solver = Solver
        problem.pair(survey)
        mSynth = np.r_[eta, 1./tau]
        survey.makeSyntheticData(mSynth)
        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = Regularization.Tikhonov(mesh)
        opt = Optimization.InexactGaussNewton(maxIterLS=20, maxIter=10, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=6)
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=1e4)
        inv = Inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p =     problem
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis

    def test_misfit(self):
        derChk = lambda m: [self.survey.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)]
        passed = Tests.checkDerivative(derChk, self.m0, plotIt=False, num=3)
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        u = np.random.rand(self.mesh.nC*self.survey.nSrc)
        v = np.random.rand(self.mesh.nC*2)
        w = np.random.rand(self.survey.dobs.shape[0])
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-10
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        derChk = lambda m: [self.dmis.eval(m), self.dmis.evalDeriv(m)]
        passed = Tests.checkDerivative(derChk, self.m0, plotIt=False, num=3)
        self.assertTrue(passed)

class IPProblemTestsN(unittest.TestCase):

    def setUp(self):

        cs = 25.
        hx = [(cs,0, -1.3),(cs,21),(cs,0, 1.3)]
        hy = [(cs,0, -1.3),(cs,21),(cs,0, 1.3)]
        hz = [(cs,0, -1.3),(cs,20)]
        mesh = Mesh.TensorMesh([hx, hy, hz],x0="CCN")
        blkind0 = Utils.ModelBuilder.getIndicesSphere(np.r_[-100., -100., -200.], 75., mesh.gridCC)
        blkind1 = Utils.ModelBuilder.getIndicesSphere(np.r_[100., 100., -200.], 75., mesh.gridCC)
        sigma = np.ones(mesh.nC)*1e-2
        eta = np.zeros(mesh.nC)
        tau = np.ones_like(sigma)*1.
        eta[blkind0] = 0.1
        eta[blkind1] = 0.1
        tau[blkind0] = 0.1
        tau[blkind1] = 0.01

        x = mesh.vectorCCx[(mesh.vectorCCx>-155.)&(mesh.vectorCCx<155.)]
        y = mesh.vectorCCx[(mesh.vectorCCy>-155.)&(mesh.vectorCCy<155.)]
        Aloc = np.r_[-200., 0., 0.]
        Bloc = np.r_[200., 0., 0.]
        M = Utils.ndgrid(x-25.,y, np.r_[0.])
        N = Utils.ndgrid(x+25.,y, np.r_[0.])

        times = np.arange(10)*1e-3 + 1e-3
        rx = SIP.Rx.Dipole(M, N, times)
        src = SIP.Src.Dipole([rx], Aloc, Bloc)
        survey = SIP.Survey([src])
        colemap = [("eta", Maps.IdentityMap(mesh)), ("taui", Maps.IdentityMap(mesh))]
        problem = SIP.Problem3D_N(mesh, sigma=sigma, mapping=colemap)
        problem.Solver = Solver
        problem.pair(survey)
        mSynth = np.r_[eta, 1./tau]
        survey.makeSyntheticData(mSynth)
        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = Regularization.Tikhonov(mesh)
        opt = Optimization.InexactGaussNewton(maxIterLS=20, maxIter=10, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=6)
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=1e4)
        inv = Inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p =     problem
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis

    def test_misfit(self):
        derChk = lambda m: [self.survey.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)]
        passed = Tests.checkDerivative(derChk, self.m0, plotIt=False, num=3)
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        u = np.random.rand(self.mesh.nC*self.survey.nSrc)
        v = np.random.rand(self.mesh.nC*2)
        w = np.random.rand(self.survey.dobs.shape[0])
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-8
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        derChk = lambda m: [self.dmis.eval(m), self.dmis.evalDeriv(m)]
        passed = Tests.checkDerivative(derChk, self.m0, plotIt=False, num=3)
        self.assertTrue(passed)

class IPProblemTestsN_air(unittest.TestCase):

    def setUp(self):

        cs = 25.
        hx = [(cs,0, -1.3),(cs,21),(cs,0, 1.3)]
        hy = [(cs,0, -1.3),(cs,21),(cs,0, 1.3)]
        hz = [(cs,0, -1.3),(cs,20),(cs,0, 1.3)]
        mesh = Mesh.TensorMesh([hx, hy, hz],x0="CCC")
        blkind0 = Utils.ModelBuilder.getIndicesSphere(np.r_[-100., -100., -200.], 75., mesh.gridCC)
        blkind1 = Utils.ModelBuilder.getIndicesSphere(np.r_[100., 100., -200.], 75., mesh.gridCC)
        sigma = np.ones(mesh.nC)*1e-2
        airind = mesh.gridCC[:,2]>0.
        sigma[airind] = 1e-8
        eta = np.zeros(mesh.nC)
        tau = np.ones_like(sigma)*1.
        eta[blkind0] = 0.1
        eta[blkind1] = 0.1
        tau[blkind0] = 0.1
        tau[blkind1] = 0.01

        actmapeta = Maps.InjectActiveCells(mesh, ~airind, 0.)
        actmaptau = Maps.InjectActiveCells(mesh, ~airind, 1.)

        x = mesh.vectorCCx[(mesh.vectorCCx>-155.)&(mesh.vectorCCx<155.)]
        y = mesh.vectorCCx[(mesh.vectorCCy>-155.)&(mesh.vectorCCy<155.)]
        Aloc = np.r_[-200., 0., 0.]
        Bloc = np.r_[200., 0., 0.]
        M = Utils.ndgrid(x-25.,y, np.r_[0.])
        N = Utils.ndgrid(x+25.,y, np.r_[0.])

        times = np.arange(10)*1e-3 + 1e-3
        rx = SIP.Rx.Dipole(M, N, times)
        src = SIP.Src.Dipole([rx], Aloc, Bloc)
        survey = SIP.Survey([src])
        colemap = [("eta", Maps.IdentityMap(mesh)*actmapeta), ("taui", Maps.IdentityMap(mesh)*actmaptau)]
        problem = SIP.Problem3D_N(mesh, sigma=sigma, mapping=colemap)
        problem.Solver = Solver
        problem.pair(survey)
        mSynth = np.r_[eta[~airind], 1./tau[~airind]]
        survey.makeSyntheticData(mSynth)
        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        regmap = Maps.IdentityMap(nP=int(mSynth[~airind].size*2))
        reg = SIP.MultiRegularization(mesh, mapping=regmap, nModels=2, indActive=~airind)
        opt = Optimization.InexactGaussNewton(maxIterLS=20, maxIter=10, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=6)
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=1e4)
        inv = Inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p =     problem
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis

    def test_misfit(self):
        derChk = lambda m: [self.survey.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)]
        passed = Tests.checkDerivative(derChk, self.m0, plotIt=False, num=3)
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        u = np.random.rand(self.mesh.nC*self.survey.nSrc)
        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.survey.dobs.shape[0])
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-8
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        derChk = lambda m: [self.dmis.eval(m), self.dmis.evalDeriv(m)]
        passed = Tests.checkDerivative(derChk, self.m0, plotIt=False, num=3)
        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
