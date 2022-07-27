from __future__ import print_function
import unittest
import discretize
import numpy as np

from SimPEG import utils
from SimPEG import maps
from SimPEG import data_misfit
from SimPEG import regularization
from SimPEG import optimization
from SimPEG import inversion
from SimPEG import inverse_problem
from SimPEG import tests

from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics import induced_polarization as ip

np.random.seed(30)


class IPProblemTestsCC(unittest.TestCase):
    def setUp(self):

        cs = 12.5
        hx = [(cs, 7, -1.3), (cs, 61), (cs, 7, 1.3)]
        hy = [(cs, 7, -1.3), (cs, 20)]
        mesh = discretize.TensorMesh([hx, hy], x0="CN")

        # x = np.linspace(-200, 200., 20)
        x = np.linspace(-200, 200.0, 2)
        M = utils.ndgrid(x - 12.5, np.r_[0.0])
        N = utils.ndgrid(x + 12.5, np.r_[0.0])

        A0loc = np.r_[-150, 0.0]
        A1loc = np.r_[-130, 0.0]
        B0loc = np.r_[-130, 0.0]
        B1loc = np.r_[-110, 0.0]

        rx = dc.Rx.Dipole(M, N)
        src0 = dc.Src.Dipole([rx], A0loc, B0loc)
        src1 = dc.Src.Dipole([rx], A1loc, B1loc)
        survey = ip.Survey([src0, src1])

        sigma = np.ones(mesh.nC) * 1.0
        problem = ip.Simulation2DCellCentered(
            mesh,
            survey=survey,
            sigma=sigma,
            etaMap=maps.IdentityMap(mesh),
            verbose=False,
        )

        mSynth = np.ones(mesh.nC) * 0.1
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
        v = np.random.rand(self.mesh.nC)
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


class IPProblemTestsN(unittest.TestCase):
    def setUp(self):

        cs = 12.5
        hx = [(cs, 7, -1.3), (cs, 61), (cs, 7, 1.3)]
        hy = [(cs, 7, -1.3), (cs, 20)]
        mesh = discretize.TensorMesh([hx, hy], x0="CN")

        # x = np.linspace(-200, 200., 20)
        x = np.linspace(-200, 200.0, 2)
        M = utils.ndgrid(x - 12.5, np.r_[0.0])
        N = utils.ndgrid(x + 12.5, np.r_[0.0])

        A0loc = np.r_[-150, 0.0]
        A1loc = np.r_[-130, 0.0]
        B0loc = np.r_[-130, 0.0]
        B1loc = np.r_[-110, 0.0]

        rx = dc.Rx.Dipole(M, N)
        src0 = dc.Src.Dipole([rx], A0loc, B0loc)
        src1 = dc.Src.Dipole([rx], A1loc, B1loc)
        survey = ip.Survey([src0, src1])

        sigma = np.ones(mesh.nC) * 1.0
        problem = ip.Simulation2DNodal(
            mesh,
            survey=survey,
            rho=1.0 / sigma,
            etaMap=maps.IdentityMap(mesh),
            verbose=False,
        )

        mSynth = np.ones(mesh.nC) * 0.1
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
        v = np.random.rand(self.mesh.nC)
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
