from __future__ import print_function
import unittest
from SimPEG import (Mesh, Utils, Maps, DataMisfit,
                    Regularization, Optimization, Inversion, InvProblem, Tests)
import numpy as np
from SimPEG.EM.Static import SIP
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

np.random.seed(38)


class IPProblemTestsCC(unittest.TestCase):

    def setUp(self):

        cs = 25.
        hx = [(cs, 0, -1.3), (cs, 21), (cs, 0, 1.3)]
        hz = [(cs, 0, -1.3), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hz], x0="CN")
        blkind0 = Utils.ModelBuilder.getIndicesSphere(
            np.r_[-100., -200.], 75., mesh.gridCC
        )
        blkind1 = Utils.ModelBuilder.getIndicesSphere(
            np.r_[100., -200.], 75., mesh.gridCC
        )

        sigma = np.ones(mesh.nC) * 1e-2
        eta = np.zeros(mesh.nC)
        tau = np.ones_like(sigma) * 1.
        eta[blkind0] = 0.1
        eta[blkind1] = 0.1
        tau[blkind0] = 0.1
        tau[blkind1] = 0.1

        x = mesh.vectorCCx[(mesh.vectorCCx > -155.) & (mesh.vectorCCx < 155.)]

        Aloc = np.r_[-200., 0.]
        Bloc = np.r_[200., 0.]
        M = Utils.ndgrid(x-25., np.r_[0.])
        N = Utils.ndgrid(x+25., np.r_[0.])

        times = np.arange(10)*1e-3 + 1e-3
        rx = SIP.Rx.Dipole(M, N, times)
        src = SIP.Src.Dipole([rx], Aloc, Bloc)
        survey = SIP.Survey([src])
        wires = Maps.Wires(('eta', mesh.nC), ('taui', mesh.nC))
        problem = SIP.Problem2D_CC(
            mesh,
            rho=1./sigma,
            etaMap=wires.eta,
            tauiMap=wires.taui,
            verbose = False
        )
        problem.Solver = Solver
        problem.pair(survey)
        mSynth = np.r_[eta, 1./tau]
        problem.model = mSynth
        survey.makeSyntheticData(mSynth)
        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = Regularization.Tikhonov(mesh)
        opt = Optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=1e4)
        inv = Inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p = problem
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis

    def test_misfit(self):
        passed = Tests.checkDerivative(
            lambda m: [
                self.survey.dpred(m),
                lambda mx: self.p.Jvec(self.m0, mx)
            ],
            self.m0,
            plotIt=False,
            num=3)
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        # u = np.random.rand(self.mesh.nC*self.survey.nSrc)
        v = np.random.rand(self.mesh.nC*2)
        w = np.random.rand(self.survey.dobs.shape[0])
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-10
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = Tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)


class IPProblemTestsN(unittest.TestCase):

    def setUp(self):

        cs = 25.
        hx = [(cs, 0, -1.3), (cs, 21), (cs, 0, 1.3)]
        hz = [(cs, 0, -1.3), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hz], x0="CN")
        blkind0 = Utils.ModelBuilder.getIndicesSphere(
            np.r_[-100., -200.], 75., mesh.gridCC
        )
        blkind1 = Utils.ModelBuilder.getIndicesSphere(
            np.r_[100., -200.], 75., mesh.gridCC
        )

        sigma = np.ones(mesh.nC) * 1e-2
        eta = np.zeros(mesh.nC)
        tau = np.ones_like(sigma) * 1.
        eta[blkind0] = 0.1
        eta[blkind1] = 0.1
        tau[blkind0] = 0.1
        tau[blkind1] = 0.1

        x = mesh.vectorCCx[(mesh.vectorCCx > -155.) & (mesh.vectorCCx < 155.)]

        Aloc = np.r_[-200., 0.]
        Bloc = np.r_[200., 0.]
        M = Utils.ndgrid(x-25., np.r_[0.])
        N = Utils.ndgrid(x+25., np.r_[0.])

        times = np.arange(10)*1e-3 + 1e-3
        rx = SIP.Rx.Dipole(M, N, times)
        src = SIP.Src.Dipole([rx], Aloc, Bloc)
        survey = SIP.Survey([src])
        wires = Maps.Wires(('eta', mesh.nC), ('taui', mesh.nC))
        problem = SIP.Problem2D_N(
            mesh,
            sigma=sigma,
            etaMap=wires.eta,
            tauiMap=wires.taui,
            verbose = False
        )
        problem.Solver = Solver
        problem.pair(survey)
        mSynth = np.r_[eta, 1./tau]
        problem.model = mSynth
        survey.makeSyntheticData(mSynth)
        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = Regularization.Tikhonov(mesh)
        opt = Optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=1e4)
        inv = Inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p = problem
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis

    def test_misfit(self):
        passed = Tests.checkDerivative(
            lambda m: [
                self.survey.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)
            ],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        v = np.random.rand(self.mesh.nC*2)
        w = np.random.rand(self.survey.dobs.shape[0])
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-8
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = Tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)],
            self.m0,
            plotIt=False,
            num=2
        )
        self.assertTrue(passed)


class IPProblemTestsN_air(unittest.TestCase):

    def setUp(self):

        cs = 25.
        hx = [(cs, 0, -1.3), (cs, 21), (cs, 0, 1.3)]
        hz = [(cs, 0, -1.3), (cs, 20)]
        mesh = Mesh.TensorMesh([hx, hz], x0="CN")
        blkind0 = Utils.ModelBuilder.getIndicesSphere(
            np.r_[-100., -200.], 75., mesh.gridCC
        )
        blkind1 = Utils.ModelBuilder.getIndicesSphere(
            np.r_[100., -200.], 75., mesh.gridCC
        )

        sigma = np.ones(mesh.nC) * 1e-2
        eta = np.zeros(mesh.nC)
        tau = np.ones_like(sigma) * 1.
        c = np.ones_like(sigma)

        eta[blkind0] = 0.1
        eta[blkind1] = 0.1
        tau[blkind0] = 0.1
        tau[blkind1] = 0.1

        x = mesh.vectorCCx[(mesh.vectorCCx > -155.) & (mesh.vectorCCx < 155.)]

        Aloc = np.r_[-200., -50]
        Bloc = np.r_[200., -50]
        M = Utils.ndgrid(x-25., np.r_[0.])
        N = Utils.ndgrid(x+25., np.r_[0.])

        airind = mesh.gridCC[:, 1] > -40
        actmapeta = Maps.InjectActiveCells(mesh, ~airind, 0.)
        actmaptau = Maps.InjectActiveCells(mesh, ~airind, 1.)
        actmapc = Maps.InjectActiveCells(mesh, ~airind, 1.)

        times = np.arange(10)*1e-3 + 1e-3
        rx = SIP.Rx.Dipole(M, N, times)
        src = SIP.Src.Dipole([rx], Aloc, Bloc)
        survey = SIP.Survey([src])

        wires = Maps.Wires(('eta', actmapeta.nP), ('taui', actmaptau.nP), ('c', actmapc.nP))
        problem = SIP.Problem2D_N(
            mesh,
            sigma=sigma,
            etaMap=actmapeta*wires.eta,
            tauiMap=actmaptau*wires.taui,
            cMap=actmapc*wires.c,
            actinds = ~airind
        )

        problem.Solver = Solver
        problem.pair(survey)
        mSynth = np.r_[eta[~airind], 1./tau[~airind], c[~airind]]
        survey.makeSyntheticData(mSynth)
        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg_eta = Regularization.Simple(mesh, mapping=wires.eta, indActive=~airind)
        reg_taui = Regularization.Simple(mesh, mapping=wires.taui, indActive=~airind)
        reg_c = Regularization.Simple(mesh, mapping=wires.c, indActive=~airind)
        reg = reg_eta + reg_taui + reg_c
        opt = Optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=1e4)
        inv = Inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p = problem
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis

    def test_misfit(self):
        passed = Tests.checkDerivative(
            lambda m: [
                self.survey.dpred(m),
                lambda mx: self.p.Jvec(self.m0, mx)
            ],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        v = np.random.rand(self.reg.mapping.nP)
        w = np.random.rand(self.survey.dobs.shape[0])
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-8
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = Tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
