from __future__ import print_function
import unittest
import discretize

from SimPEG import (
    utils,
    maps,
    data_misfit,
    regularization,
    optimization,
    inversion,
    inverse_problem,
    tests,
)
import numpy as np
from SimPEG.electromagnetics import spectral_induced_polarization as sip

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

np.random.seed(38)


class SIPProblemTestsCC(unittest.TestCase):
    def setUp(self):

        cs = 25.0
        hx = [(cs, 0, -1.3), (cs, 21), (cs, 0, 1.3)]
        hz = [(cs, 0, -1.3), (cs, 20)]
        mesh = discretize.TensorMesh([hx, hz], x0="CN")
        blkind0 = utils.model_builder.getIndicesSphere(
            np.r_[-100.0, -200.0], 75.0, mesh.gridCC
        )
        blkind1 = utils.model_builder.getIndicesSphere(
            np.r_[100.0, -200.0], 75.0, mesh.gridCC
        )

        sigma = np.ones(mesh.nC) * 1e-2
        eta = np.zeros(mesh.nC)
        tau = np.ones_like(sigma) * 1.0
        eta[blkind0] = 0.1
        eta[blkind1] = 0.1
        tau[blkind0] = 0.1
        tau[blkind1] = 0.1

        x = mesh.vectorCCx[(mesh.vectorCCx > -155.0) & (mesh.vectorCCx < 155.0)]

        Aloc = np.r_[-200.0, 0.0]
        Bloc = np.r_[200.0, 0.0]
        M = utils.ndgrid(x - 25.0, np.r_[0.0])
        N = utils.ndgrid(x + 25.0, np.r_[0.0])

        times = np.arange(10) * 1e-3 + 1e-3
        rx = sip.receivers.Dipole(M, N, times)
        src = sip.sources.Dipole([rx], Aloc, Bloc)
        survey = sip.Survey([src])
        wires = maps.Wires(("eta", mesh.nC), ("taui", mesh.nC))
        problem = sip.Simulation2DCellCentered(
            mesh,
            rho=1.0 / sigma,
            etaMap=wires.eta,
            tauiMap=wires.taui,
            verbose=False,
            solver=Solver,
            survey=survey,
        )
        mSynth = np.r_[eta, 1.0 / tau]
        problem.model = mSynth
        dobs = problem.make_synthetic_data(mSynth, add_noise=True)
        # Now set up the problem to do some minimization
        dmis = data_misfit.L2DataMisfit(data=dobs, simulation=problem)
        reg = regularization.WeightedLeastSquares(mesh)
        opt = optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1e4)
        inv = inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p = problem
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis
        self.dobs = dobs

    def test_misfit(self):
        passed = tests.checkDerivative(
            lambda m: [self.p.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)],
            self.m0,
            plotIt=False,
            num=3,
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        # u = np.random.rand(self.mesh.nC*self.survey.nSrc)
        v = np.random.rand(self.mesh.nC * 2)
        w = np.random.rand(self.survey.nD)
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-10
        print("Adjoint Test", np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)], self.m0, plotIt=False, num=3
        )
        self.assertTrue(passed)


class SIPProblemTestsN(unittest.TestCase):
    def setUp(self):

        cs = 25.0
        hx = [(cs, 0, -1.3), (cs, 21), (cs, 0, 1.3)]
        hz = [(cs, 0, -1.3), (cs, 20)]
        mesh = discretize.TensorMesh([hx, hz], x0="CN")
        blkind0 = utils.model_builder.getIndicesSphere(
            np.r_[-100.0, -200.0], 75.0, mesh.gridCC
        )
        blkind1 = utils.model_builder.getIndicesSphere(
            np.r_[100.0, -200.0], 75.0, mesh.gridCC
        )

        sigma = np.ones(mesh.nC) * 1e-2
        eta = np.zeros(mesh.nC)
        tau = np.ones_like(sigma) * 1.0
        eta[blkind0] = 0.1
        eta[blkind1] = 0.1
        tau[blkind0] = 0.1
        tau[blkind1] = 0.1

        x = mesh.vectorCCx[(mesh.vectorCCx > -155.0) & (mesh.vectorCCx < 155.0)]

        Aloc = np.r_[-200.0, 0.0]
        Bloc = np.r_[200.0, 0.0]
        M = utils.ndgrid(x - 25.0, np.r_[0.0])
        N = utils.ndgrid(x + 25.0, np.r_[0.0])

        times = np.arange(10) * 1e-3 + 1e-3
        rx = sip.receivers.Dipole(M, N, times)
        src = sip.sources.Dipole([rx], Aloc, Bloc)
        survey = sip.Survey([src])
        wires = maps.Wires(("eta", mesh.nC), ("taui", mesh.nC))
        problem = sip.Simulation2DNodal(
            mesh,
            sigma=sigma,
            etaMap=wires.eta,
            tauiMap=wires.taui,
            verbose=False,
            solver=Solver,
            survey=survey,
        )
        mSynth = np.r_[eta, 1.0 / tau]
        problem.model = mSynth
        dobs = problem.make_synthetic_data(mSynth, add_noise=True)
        # Now set up the problem to do some minimization
        dmis = data_misfit.L2DataMisfit(data=dobs, simulation=problem)
        reg = regularization.WeightedLeastSquares(mesh)
        opt = optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1e4)
        inv = inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p = problem
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis
        self.dobs = dobs

    def test_misfit(self):
        passed = tests.checkDerivative(
            lambda m: [self.p.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)],
            self.m0,
            plotIt=False,
            num=3,
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        v = np.random.rand(self.mesh.nC * 2)
        w = np.random.rand(self.survey.nD)
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-8
        print("Adjoint Test", np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)], self.m0, plotIt=False, num=2
        )
        self.assertTrue(passed)


class SIPProblemTestsN_air(unittest.TestCase):
    def setUp(self):

        cs = 25.0
        hx = [(cs, 0, -1.3), (cs, 21), (cs, 0, 1.3)]
        hz = [(cs, 0, -1.3), (cs, 20)]
        mesh = discretize.TensorMesh([hx, hz], x0="CN")
        blkind0 = utils.model_builder.getIndicesSphere(
            np.r_[-100.0, -200.0], 75.0, mesh.gridCC
        )
        blkind1 = utils.model_builder.getIndicesSphere(
            np.r_[100.0, -200.0], 75.0, mesh.gridCC
        )

        sigma = np.ones(mesh.nC) * 1e-2
        eta = np.zeros(mesh.nC)
        tau = np.ones_like(sigma) * 1.0
        c = np.ones_like(sigma)

        eta[blkind0] = 0.1
        eta[blkind1] = 0.1
        tau[blkind0] = 0.1
        tau[blkind1] = 0.1

        x = mesh.vectorCCx[(mesh.vectorCCx > -155.0) & (mesh.vectorCCx < 155.0)]

        Aloc = np.r_[-200.0, -50]
        Bloc = np.r_[200.0, -50]
        M = utils.ndgrid(x - 25.0, np.r_[0.0])
        N = utils.ndgrid(x + 25.0, np.r_[0.0])

        airind = mesh.gridCC[:, 1] > -40
        actmapeta = maps.InjectActiveCells(mesh, ~airind, 0.0)
        actmaptau = maps.InjectActiveCells(mesh, ~airind, 1.0)
        actmapc = maps.InjectActiveCells(mesh, ~airind, 1.0)

        times = np.arange(10) * 1e-3 + 1e-3
        rx = sip.receivers.Dipole(M, N, times)
        src = sip.sources.Dipole([rx], Aloc, Bloc)
        survey = sip.Survey([src])

        wires = maps.Wires(
            ("eta", actmapeta.nP), ("taui", actmaptau.nP), ("c", actmapc.nP)
        )
        problem = sip.Simulation2DNodal(
            mesh,
            sigma=sigma,
            etaMap=actmapeta * wires.eta,
            tauiMap=actmaptau * wires.taui,
            cMap=actmapc * wires.c,
            actinds=~airind,
            solver=Solver,
            survey=survey,
        )
        mSynth = np.r_[eta[~airind], 1.0 / tau[~airind], c[~airind]]
        dobs = problem.make_synthetic_data(mSynth, add_noise=True)
        # Now set up the problem to do some minimization
        dmis = data_misfit.L2DataMisfit(data=dobs, simulation=problem)
        reg_eta = regularization.Simple(mesh, mapping=wires.eta, indActive=~airind)
        reg_taui = regularization.Simple(mesh, mapping=wires.taui, indActive=~airind)
        reg_c = regularization.Simple(mesh, mapping=wires.c, indActive=~airind)
        reg = reg_eta + reg_taui + reg_c
        opt = optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1e4)
        inv = inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p = problem
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis
        self.dobs = dobs

    def test_misfit(self):
        passed = tests.checkDerivative(
            lambda m: [self.p.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)],
            self.m0,
            plotIt=False,
            num=3,
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        v = np.random.rand(self.reg.mapping.nP)
        w = np.random.rand(self.survey.nD)
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-8
        print("Adjoint Test", np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)], self.m0, plotIt=False, num=3
        )
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
