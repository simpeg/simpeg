from __future__ import print_function

import unittest

import numpy as np
import scipy.sparse as sp

from SimPEG import (
    Mesh, DataMisfit, Maps, Utils, Regularization, InvProblem, Optimization,
    Directives, Inversion
)
from SimPEG.EM.Static import DC

np.random.seed(82)


class DataMisfitTest(unittest.TestCase):

    def setUp(self):
        mesh = Mesh.TensorMesh([30, 30], x0=[-0.5, -1.])
        sigma = np.random.rand(mesh.nC)
        model = np.log(sigma)

        prob = DC.Problem3D_CC(mesh, rhoMap=Maps.ExpMap(mesh))
        prob1 = DC.Problem3D_CC(mesh, rhoMap=Maps.ExpMap(mesh))

        rx = DC.Rx.Pole(
            Utils.ndgrid([mesh.vectorCCx, np.r_[mesh.vectorCCy.max()]])
        )
        rx1 = DC.Rx.Pole(
            Utils.ndgrid([mesh.vectorCCx, np.r_[mesh.vectorCCy.min()]])
        )
        src = DC.Src.Dipole(
            [rx], np.r_[-0.25, mesh.vectorCCy.max()],
            np.r_[0.25, mesh.vectorCCy.max()]
        )
        src1 = DC.Src.Dipole(
            [rx1], np.r_[-0.25, mesh.vectorCCy.max()],
            np.r_[0.25, mesh.vectorCCy.max()]
        )
        survey = DC.Survey([src])
        prob.pair(survey)

        survey1 = DC.Survey([src1])
        prob1.pair(survey1)

        dobs0 = survey.makeSyntheticData(model)
        dobs1 = survey1.makeSyntheticData(model)

        self.mesh = mesh
        self.model = model

        self.survey0 = survey
        self.prob0 = prob

        self.survey1 = survey1
        self.prob1 = prob1

        self.dmis0 = DataMisfit.l2_DataMisfit(self.survey0)
        self.dmis1 = DataMisfit.l2_DataMisfit(self.survey1)

        self.dmiscobmo = self.dmis0 + self.dmis1

    def test_multiDataMisfit(self):
        self.dmis0.test()
        self.dmis1.test()
        self.dmiscobmo.test(x=self.model)

    def test_inv(self):
        reg = Regularization.Tikhonov(self.mesh)
        opt = Optimization.InexactGaussNewton(maxIter=10)
        invProb = InvProblem.BaseInvProblem(self.dmiscobmo, reg, opt)
        directives = [
            Directives.BetaEstimate_ByEig(beta0_ratio=1e-2),
        ]
        inv = Inversion.BaseInversion(invProb, directiveList=directives)
        m0 = self.model.mean() * np.ones_like(self.model)

        mrec = inv.run(m0)

    def test_inv_mref_setting(self):
        reg1 = Regularization.Tikhonov(self.mesh)
        reg2 = Regularization.Tikhonov(self.mesh)
        reg = reg1+reg2
        opt = Optimization.InexactGaussNewton(maxIter=10)
        invProb = InvProblem.BaseInvProblem(self.dmiscobmo, reg, opt)
        directives = [
            Directives.BetaEstimate_ByEig(beta0_ratio=1e-2),
        ]
        inv = Inversion.BaseInversion(invProb, directiveList=directives)
        m0 = self.model.mean() * np.ones_like(self.model)

        mrec = inv.run(m0)

        self.assertTrue(np.all(reg1.mref == m0))
        self.assertTrue(np.all(reg2.mref == m0))

if __name__ == '__main__':
    unittest.main()
