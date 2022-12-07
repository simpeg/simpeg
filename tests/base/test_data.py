import unittest

import numpy as np
import discretize

from SimPEG import maps
from SimPEG import simulation, survey
from SimPEG import Data


class DataTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(17)
        mesh = discretize.TensorMesh([30])
        sigma = np.ones(mesh.nC)
        model = np.log(sigma)

        receivers = survey.BaseRx(20 * [[0.0]])
        source = survey.BaseSrc([receivers])

        self.sim = simulation.ExponentialSinusoidSimulation(
            mesh=mesh, survey=survey.BaseSurvey([source]), model_map=maps.ExpMap(mesh)
        )

        self.dobs = self.sim.dpred(model)

    def test_instantiation_relative_error(self):
        relative = 0.5
        data = Data(self.sim.survey, dobs=self.dobs, relative_error=relative)
        np.testing.assert_equal(data.relative_error, relative * np.ones(len(self.dobs)))
        np.testing.assert_equal(data.standard_deviation, relative * np.abs(self.dobs))

    def test_instantiation_noise_floor(self):
        floor = np.min(np.abs(self.dobs))
        data = Data(self.sim.survey, dobs=self.dobs, noise_floor=floor)
        np.testing.assert_equal(data.noise_floor, floor * np.ones(len(self.dobs)))
        np.testing.assert_equal(
            data.standard_deviation, floor * np.ones(len(self.dobs))
        )

    def test_instantiation_relative_floor(self):
        relative = 0.5
        floor = np.min(np.abs(self.dobs))
        data = Data(
            self.sim.survey, dobs=self.dobs, relative_error=relative, noise_floor=floor
        )
        np.testing.assert_equal(data.relative_error, relative * np.ones(len(self.dobs)))
        np.testing.assert_equal(data.noise_floor, floor * np.ones(len(self.dobs)))
        np.testing.assert_allclose(
            data.standard_deviation,
            np.sqrt(
                (relative * np.abs(self.dobs)) ** 2
                + floor ** 2 * np.ones(len(self.dobs)),
            ),
        )

    def test_instantiation_standard_deviation(self):
        relative = 0.5
        floor = np.min(np.abs(self.dobs))
        standard_deviation = relative * np.abs(self.dobs) + floor * np.ones(
            len(self.dobs)
        )
        data = Data(
            self.sim.survey, dobs=self.dobs, standard_deviation=standard_deviation
        )

        np.testing.assert_equal(data.noise_floor, standard_deviation)
        np.testing.assert_equal(data.standard_deviation, standard_deviation)
