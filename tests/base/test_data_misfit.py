import re
import pytest
import unittest

import numpy as np
import discretize

from simpeg import maps
from simpeg import data_misfit, simulation, survey
from simpeg import Data


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

        synthetic_data = sim.make_synthetic_data(model, random_seed=17)

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
            self.dmis.W

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
        self.dmis.test(x=self.model, random_seed=17)


class MockSimulation(simulation.BaseSimulation):
    """
    Mock simulation class that returns nans or infs in the dpred array.
    """

    def __init__(self, invalid_value=np.nan):
        self.invalid_value = invalid_value
        super().__init__()

    def dpred(self, m=None, f=None):
        a = np.arange(4, dtype=np.float64)
        a[1] = self.invalid_value
        return a


class TestNanOrInfInResidual:
    """Test errors if the simulation return dpred with nans or infs."""

    @pytest.fixture
    def n_data(self):
        return 4

    @pytest.fixture
    def sample_survey(self, n_data):
        receivers = survey.BaseRx(np.zeros(n_data)[:, np.newaxis])
        source = survey.BaseSrc([receivers])
        return survey.BaseSurvey([source])

    @pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
    def test_error(self, sample_survey, invalid_value):
        mock_simulation = MockSimulation(invalid_value)
        data = Data(sample_survey)
        dmisfit = data_misfit.BaseDataMisfit(data, mock_simulation)
        msg = re.escape(
            "The `MockSimulation.dpred()` method returned an array that contains "
            "`nan`s and/or `inf`s."
        )
        with pytest.raises(ValueError, match=msg):
            dmisfit.residual(m=None)


if __name__ == "__main__":
    unittest.main()
