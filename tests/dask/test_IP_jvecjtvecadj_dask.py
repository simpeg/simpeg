import os
import shutil
import tarfile
import unittest

import discretize as ds
import numpy as np

import simpeg.dask  # noqa: F401
from simpeg import (
    data_misfit,
    inverse_problem,
    inversion,
    maps,
    optimization,
    regularization,
    tests,
    utils,
)
from simpeg.electromagnetics import induced_polarization as ip
from simpeg.electromagnetics import resistivity as dc
from simpeg.utils.io_utils.io_utils_electromagnetics import read_dcip2d_ubc

np.random.seed(30)


class IPProblemTests2DN(unittest.TestCase):
    """
    This test builds upon the 2D files used in the IP2D tutorial, with a much smaller mesh.

    It tests IP 2D with dask without calling `make_synthetic_data` first to simulate a real data case.
    """

    def setUp(self):
        # storage bucket where we have the data
        data_source = "https://storage.googleapis.com/simpeg/doc-assets/dcip2d.tar.gz"

        # download the data
        downloaded_data = utils.download(data_source, overwrite=True)

        # unzip the tarfile
        tar = tarfile.open(downloaded_data, "r")
        tar.extractall()
        tar.close()

        # path to the directory containing our data
        dir_path = downloaded_data.split(".")[0] + os.path.sep

        # files to work with
        topo_filename = dir_path + "topo_xyz.txt"
        dc_data_filename = dir_path + "dc_data.obs"
        ip_data_filename = dir_path + "ip_data.obs"

        # Load topo
        topo_xyz = np.loadtxt(str(topo_filename))
        # define the 2D topography along the survey line.
        topo_2d = np.unique(topo_xyz[:, [0, 2]], axis=0)
        # Load datas
        dc_data = read_dcip2d_ubc(dc_data_filename, "volt", "general")
        ip_data = read_dcip2d_ubc(ip_data_filename, "apparent_chargeability", "general")
        # assign uncertainties
        dc_data.standard_deviation = 0.05 * np.abs(dc_data.dobs)
        ip_data.standard_deviation = 5e-3 * np.ones_like(ip_data.dobs)
        # mesh
        cs = 10.0
        mesh = ds.TensorMesh(
            [
                [(cs, 20, -1.3), (cs, 20, 1.3)],
                [(cs, 20, -1.3), (cs, 20, 1.3)],
            ],
            "CN",
        )
        # Find cells that lie below surface topography
        ind_active = ds.utils.active_from_xyz(mesh, topo_2d)
        # Shift electrodes to the surface of discretized topography
        dc_data.survey.drape_electrodes_on_topography(mesh, ind_active, option="top")
        ip_data.survey.drape_electrodes_on_topography(mesh, ind_active, option="top")

        # Define conductivity model in S/m (or resistivity model in Ohm m)
        air_conductivity = 1e-8
        background_conductivity = 1e-2
        active_map = maps.InjectActiveCells(mesh, ind_active, air_conductivity)
        nC = int(ind_active.sum())
        # Define model
        conductivity_model = background_conductivity * np.ones(nC)

        simulation = ip.simulation.Simulation2DNodal(
            mesh=mesh,
            survey=ip_data.survey,
            sigma=conductivity_model,
            etaMap=active_map,
        )
        mSynth = np.ones(nC) * 0.1
        # test without calling make_synthetic_data first to simulate real data case
        dobs = read_dcip2d_ubc(ip_data_filename, "apparent_chargeability", "general")
        # Now set up the problem to do some minimization
        dmis = data_misfit.L2DataMisfit(data=dobs, simulation=simulation)
        reg = regularization.WeightedLeastSquares(mesh)
        opt = optimization.InexactGaussNewton(
            maxIterLS=5, maxIter=1, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=5
        )
        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1e4)
        inv = inversion.BaseInversion(invProb)

        self.inv = inv
        self.reg = reg
        self.p = simulation
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = ip_data.survey
        self.dmis = dmis
        self.conductivity_model = conductivity_model

    def test_misfit(self):
        passed = tests.check_derivative(
            lambda m: [self.p.dpred(m), lambda mx: self.p.Jvec(self.m0, mx)],
            self.m0,
            plotIt=False,
            num=3,
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        v = np.random.rand(len(self.m0))
        w = np.random.rand(self.survey.nD)
        # J = self.p.getJ(self.m0)
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-10
        print("Adjoint Test", np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = tests.check_derivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)], self.m0, plotIt=False, num=3
        )
        self.assertTrue(passed)


class IPProblemTestsCC(unittest.TestCase):
    def setUp(self):
        aSpacing = 2.5
        nElecs = 5

        surveySize = nElecs * aSpacing - aSpacing
        cs = surveySize / nElecs / 4

        mesh = ds.TensorMesh(
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
        passed = tests.check_derivative(
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
        passed = tests.check_derivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)], self.m0, plotIt=False, num=3
        )
        self.assertTrue(passed)


class IPProblemTestsN(unittest.TestCase):
    def setUp(self):
        aSpacing = 2.5
        nElecs = 5

        surveySize = nElecs * aSpacing - aSpacing
        cs = surveySize / nElecs / 4

        mesh = ds.TensorMesh(
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
        passed = tests.check_derivative(
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
        passed = tests.check_derivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)], self.m0, plotIt=False, num=3
        )
        self.assertTrue(passed)


class IPProblemTestsCC_storeJ(unittest.TestCase):
    def setUp(self):
        aSpacing = 2.5
        nElecs = 5

        surveySize = nElecs * aSpacing - aSpacing
        cs = surveySize / nElecs / 4

        mesh = ds.TensorMesh(
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
        passed = tests.check_derivative(
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
        passed = tests.check_derivative(
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

        mesh = ds.TensorMesh(
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
        passed = tests.check_derivative(
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
        passed = tests.check_derivative(
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
