import unittest

import numpy as np
import scipy.sparse as sp
import discretize

from SimPEG import maps, utils
from SimPEG import data_misfit, simulation
from SimPEG import Data

np.random.seed(17)

class DataTest(unittest.TestCase):
    def setUp(self):
        mesh = discretize.TensorMesh([30])
        sigma = np.ones(mesh.nC)
        model = np.log(sigma)

        self.sim = simulation.ExponentialSinusoidSimulation(
            mesh=mesh, model_map=maps.ExpMap(mesh)
        )

        self.dobs = self.sim.dpred(model)


    def test_instantiation_standard_deviation(self):
        std = 0.5
        data = Data(self.sim.survey, dobs=self.dobs, standard_deviation=std)
        self.assertTrue(
            all(data.standard_deviation == std*np.ones(len(self.dobs)))
        )
        self.assertTrue(
            all(data.uncertainty == std*np.abs(self.dobs))
        )

    def test_instantiation_noise_floor(self):
        floor = np.min(np.abs(self.dobs))
        data = Data(self.sim.survey, dobs=self.dobs, noise_floor=floor)
        self.assertTrue(
            all(data.noise_floor == floor*np.ones(len(self.dobs)))
        )
        self.assertTrue(
            all(data.uncertainty == floor*np.ones(len(self.dobs)))
        )

    def test_instantiation_std_floor(self):
        std = 0.5
        floor = np.min(np.abs(self.dobs))
        data = Data(self.sim.survey, dobs=self.dobs, standard_deviation=std, noise_floor=floor)
        self.assertTrue(
            all(data.standard_deviation == std*np.ones(len(self.dobs)))
        )
        self.assertTrue(
            all(data.noise_floor == floor*np.ones(len(self.dobs)))
        )
        self.assertTrue(
            np.allclose(data.uncertainty, std*np.abs(self.dobs) + floor*np.ones(len(self.dobs)))
        )

    def test_instantiation_uncertainty(self):
        std = 0.5
        floor = np.min(np.abs(self.dobs))
        uncertainty = std*np.abs(self.dobs) + floor*np.ones(len(self.dobs))
        data = Data(self.sim.survey, dobs=self.dobs, uncertainty=uncertainty)

        self.assertTrue(
            all(data.noise_floor == uncertainty)
        )
        self.assertTrue(
            all(data.uncertainty == uncertainty)
        )

if __name__ == '__main__':
    unittest.main()
