from __future__ import print_function
import unittest
import discretize
import numpy as np

from SimPEG import maps
from SimPEG import data_misfit
from SimPEG import regularization
from SimPEG import optimization
from SimPEG import inversion
from SimPEG import inverse_problem
from SimPEG import tests

from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics import induced_polarization as ip
import shutil

np.random.seed(30)


class IPProblemTestsCC(unittest.TestCase):
    def setUp(self):

        aSpacing = 2.5
        nElecs = 5

        surveySize = nElecs * aSpacing - aSpacing
        cs = surveySize / nElecs / 4

        mesh = discretize.TensorMesh(
            [
                [(cs, 10, -1.3), (cs, surveySize / cs), (cs, 10, 1.3)],
                [(cs, 3, -1.3), (cs, 3, 1.3)],
                # [(cs, 5, -1.3), (cs, 10)]
            ],
            "CN",
        )

        source_list = dc.utils.WennerSrcList(nElecs, aSpacing, in2D=True)
        survey = ip.survey.Survey(source_list)
        sigma = np.ones(mesh.nC)
        simulation = ip.simulation.Simulation3DCellCentered(
            mesh=mesh, survey=survey, sigma=sigma, etaMap=maps.IdentityMap(mesh)
        )
        mSynth = np.ones(mesh.nC) * 0.1
        dobs = simulation.make_synthetic_data(mSynth, add_noise=True)
        # Now set up the problem to do some minimization
        dmis = data_misfit.L2DataMisfit(data=dobs, simulation=simulation)
        reg = regularization.WeightedLeastSquares(mesh)
        opt = optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1e4)
        inv = inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p = simulation
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis
        # self.dobe = dobs

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
        # u = np.random.rand(self.mesh.nC*self.survey.Survey.nSrc)
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

        aSpacing = 2.5
        nElecs = 5

        surveySize = nElecs * aSpacing - aSpacing
        cs = surveySize / nElecs / 4

        mesh = discretize.TensorMesh(
            [
                [(cs, 10, -1.3), (cs, surveySize / cs), (cs, 10, 1.3)],
                [(cs, 3, -1.3), (cs, 3, 1.3)],
                # [(cs, 5, -1.3), (cs, 10)]
            ],
            "CN",
        )

        source_list = dc.utils.WennerSrcList(nElecs, aSpacing, in2D=True)
        survey = ip.survey.Survey(source_list)
        sigma = np.ones(mesh.nC)
        simulation = ip.simulation.Simulation3DNodal(
            mesh=mesh, survey=survey, sigma=sigma, etaMap=maps.IdentityMap(mesh)
        )
        mSynth = np.ones(mesh.nC) * 0.1
        dobs = simulation.make_synthetic_data(mSynth, add_noise=True)
        # Now set up the problem to do some minimization
        dmis = data_misfit.L2DataMisfit(data=dobs, simulation=simulation)
        reg = regularization.WeightedLeastSquares(mesh)
        opt = optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1e4)
        inv = inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p = simulation
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
        # u = np.random.rand(self.mesh.nC*self.survey.Survey.nSrc)
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


class IPProblemTestsCC_storeJ(unittest.TestCase):
    def setUp(self):

        aSpacing = 2.5
        nElecs = 5

        surveySize = nElecs * aSpacing - aSpacing
        cs = surveySize / nElecs / 4

        mesh = discretize.TensorMesh(
            [
                [(cs, 10, -1.3), (cs, surveySize / cs), (cs, 10, 1.3)],
                [(cs, 3, -1.3), (cs, 3, 1.3)],
                # [(cs, 5, -1.3), (cs, 10)]
            ],
            "CN",
        )

        source_list = dc.utils.WennerSrcList(nElecs, aSpacing, in2D=True)
        survey = ip.survey.Survey(source_list)
        sigma = np.ones(mesh.nC)
        simulation = ip.Simulation3DCellCentered(
            mesh=mesh,
            survey=survey,
            sigma=sigma,
            etaMap=maps.IdentityMap(mesh),
            storeJ=True,
        )
        mSynth = np.ones(mesh.nC) * 0.1
        dobs = simulation.make_synthetic_data(mSynth, add_noise=True)
        # Now set up the problem to do some minimization
        dmis = data_misfit.L2DataMisfit(data=dobs, simulation=simulation)
        reg = regularization.WeightedLeastSquares(mesh)
        opt = optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1e4)
        inv = inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p = simulation
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
        # u = np.random.rand(self.mesh.nC*self.survey.Survey.nSrc)
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

    def tearDown(self):
        # Clean up the working directory
        try:
            shutil.rmtree(self.p.sensitivity_path)
        except FileNotFoundError:
            pass


class IPProblemTestsN_storeJ(unittest.TestCase):
    def setUp(self):

        aSpacing = 2.5
        nElecs = 5

        surveySize = nElecs * aSpacing - aSpacing
        cs = surveySize / nElecs / 4

        mesh = discretize.TensorMesh(
            [
                [(cs, 10, -1.3), (cs, surveySize / cs), (cs, 10, 1.3)],
                [(cs, 3, -1.3), (cs, 3, 1.3)],
                # [(cs, 5, -1.3), (cs, 10)]
            ],
            "CN",
        )

        source_list = dc.utils.WennerSrcList(nElecs, aSpacing, in2D=True)
        survey = ip.survey.Survey(source_list)
        sigma = np.ones(mesh.nC)
        simulation = ip.simulation.Simulation3DNodal(
            mesh=mesh,
            survey=survey,
            sigma=sigma,
            etaMap=maps.IdentityMap(mesh),
            storeJ=True,
        )
        mSynth = np.ones(mesh.nC) * 0.1
        dobs = simulation.make_synthetic_data(mSynth, add_noise=True)
        # Now set up the problem to do some minimization
        dmis = data_misfit.L2DataMisfit(data=dobs, simulation=simulation)
        reg = regularization.WeightedLeastSquares(mesh)
        opt = optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1e4)
        inv = inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p = simulation
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
        # u = np.random.rand(self.mesh.nC*self.survey.Survey.nSrc)
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

    def tearDown(self):
        # Clean up the working directory
        try:
            shutil.rmtree(self.p.sensitivity_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    unittest.main()
