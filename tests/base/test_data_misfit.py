from __future__ import print_function

import unittest

import numpy as np
import scipy.sparse as sp
import discretize

from SimPEG import maps, utils
from SimPEG import data_misfit, simulation
# from SimPEG.EM.Static import DC

np.random.seed(17)


class DataMisfitTest(unittest.TestCase):

    def setUp(self):
        mesh = discretize.TensorMesh([30])
        sigma = np.ones(mesh.nC)
        model = np.log(sigma)

        # prob = DC.Problem3D_CC(mesh, rhoMap=Maps.ExpMap(mesh))
        sim = simulation.ExponentialSinusoidSimulation(mesh=mesh, model_map=maps.ExpMap(mesh))

        # rx = DC.Rx.Pole(
        #     utils.ndgrid([mesh.vectorCCx, np.r_[mesh.vectorCCy.max()]])
        # )
        # src = DC.Src.Dipole(
        #     [rx], np.r_[-0.25, mesh.vectorCCy.max()],
        #     np.r_[0.25, mesh.vectorCCy.max()]
        # )
        # survey = DC.Survey([src])

        # prob.pair(survey)

        self.std = 0.01
        sim.survey.std = self.std
        dobs = sim.makeSyntheticData(model)
        self.eps = 1e-8 * np.min(np.abs(dobs))
        sim.survey.eps = self.eps
        dmis = data_misfit.L2DataMisfit(simulation=sim)

        self.model = model
        self.mesh = mesh
        self.survey = survey
        self.prob = prob
        self.dobs = dobs
        self.dmis = dmis

    def test_Wd_depreciation(self):
        with self.assertRaises(Exception):
            print(self.dmis.Wd)

        with self.assertRaises(Exception):
            self.dmis.Wd = utils.Identity()

    def test_DataMisfit_nP(self):
        self.assertTrue(self.dmis.nP == self.mesh.nC)

    def test_setting_W(self):
        Worig = self.dmis.W
        v = np.random.rand(self.survey.nD)

        self.dmis.W = v
        self.assertTrue(self.dmis.W.shape == (self.survey.nD, self.survey.nD))
        self.assertTrue(np.all(self.dmis.W.diagonal() == v))

        with self.assertRaises(Exception):
            self.dmis.W = np.random.rand(self.survey.nD + 10)

        self.dmis.W = Worig

    def test_DataMisfitOrder(self):
        self.dmis.test(x=self.model)

    def test_std_eps(self):
        stdtest = np.all(self.survey.std == self.dmis.std)
        epstest = (self.survey.eps == self.dmis.eps)
        Wtest = np.allclose(
            np.abs(np.dot(self.dmis.W.todense(), self.dobs)),
            1./self.std,
            atol=self.eps
        )

        self.assertTrue(stdtest)
        self.assertTrue(epstest)
        self.assertTrue(Wtest)

if __name__ == '__main__':
    unittest.main()
