import unittest

import numpy as np
import discretize

from SimPEG import maps
from SimPEG import data_misfit, simulation, survey

np.random.seed(17)


class DataMisfitTest(unittest.TestCase):
    def setUp(self):
        mesh = discretize.TensorMesh([30])
        sigma = np.ones(mesh.nC)
        model = np.log(sigma)

        # prob = DC.Simulation3DCellCentered(mesh, rhoMap=Maps.ExpMap(mesh))

        receivers = survey.BaseRx(20 * [[0.0]])
        source = survey.BaseSrc([receivers])
        sim = simulation.ExponentialSinusoidSimulation(
            mesh=mesh, survey=survey.BaseSurvey([source]), model_map=maps.ExpMap(mesh)
        )

        synthetic_data = sim.make_synthetic_data(model)
        dobs = synthetic_data.dobs

        self.relative = 0.01
        self.noise_floor = 1e-8

        synthetic_data.relative_error = self.relative
        synthetic_data.noise_floor = self.noise_floor

        dmis = data_misfit.L2DataMisfit(simulation=sim, data=synthetic_data)

        self.model = model
        self.mesh = mesh
        self.sim = sim
        self.survey = sim.survey
        # self.survey = survey
        self.data = synthetic_data
        self.dmis = dmis

    def test_DataMisfit_nP(self):
        self.assertTrue(self.dmis.nP == self.mesh.nC)

    def test_zero_uncertainties(self):
        self.data.relative_error = 0.0
        self.data.noise_floor = 0.0
        with self.assertRaises(Exception):
            Worig = self.dmis.W

    def test_setting_W(self):
        self.data.relative_error = self.relative
        self.data.noise_floor = self.noise_floor
        Worig = self.dmis.W
        v = np.random.rand(self.survey.nD)

        self.dmis.W = v
        self.assertTrue(self.dmis.W.shape == (self.survey.nD, self.survey.nD))
        self.assertTrue(np.all(self.dmis.W.diagonal() == v))

        with self.assertRaises(Exception):
            self.dmis.W = np.random.rand(self.survey.nD + 10)

        self.dmis.W = Worig

    def test_DataMisfitOrder(self):
        self.data.relative_error = self.relative
        self.data.noise_floor = self.noise_floor
        self.dmis.test(x=self.model)


if __name__ == "__main__":
    unittest.main()
