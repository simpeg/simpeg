from __future__ import print_function

import unittest

import numpy as np
import scipy.sparse as sp

import discretize
from SimPEG import (
    data_misfit,
    maps,
    utils,
    regularization,
    inverse_problem,
    optimization,
    directives,
    inversion,
)
from SimPEG.electromagnetics import resistivity as DC

np.random.seed(82)


class DataMisfitTest(unittest.TestCase):
    def setUp(self):
        mesh = discretize.TensorMesh([30, 30], x0=[-0.5, -1.0])
        sigma = np.random.rand(mesh.nC)
        model = np.log(sigma)

        # prob = DC.Simulation3DCellCentered(mesh, rhoMap=maps.ExpMap(mesh))
        # prob1 = DC.Simulation3DCellCentered(mesh, rhoMap=maps.ExpMap(mesh))

        rx = DC.Rx.Pole(utils.ndgrid([mesh.vectorCCx, np.r_[mesh.vectorCCy.max()]]))
        rx1 = DC.Rx.Pole(utils.ndgrid([mesh.vectorCCx, np.r_[mesh.vectorCCy.min()]]))
        src = DC.Src.Dipole(
            [rx], np.r_[-0.25, mesh.vectorCCy.max()], np.r_[0.25, mesh.vectorCCy.max()]
        )
        src1 = DC.Src.Dipole(
            [rx1], np.r_[-0.25, mesh.vectorCCy.max()], np.r_[0.25, mesh.vectorCCy.max()]
        )
        survey = DC.Survey([src])
        simulation0 = DC.simulation.Simulation3DCellCentered(
            mesh=mesh, survey=survey, rhoMap=maps.ExpMap(mesh)
        )

        survey1 = DC.Survey([src1])
        simulation1 = DC.simulation.Simulation3DCellCentered(
            mesh=mesh, survey=survey1, rhoMap=maps.ExpMap(mesh)
        )

        dobs0 = simulation0.make_synthetic_data(model)
        dobs1 = simulation1.make_synthetic_data(model)

        self.mesh = mesh
        self.model = model

        self.survey0 = survey
        self.sim0 = simulation0

        self.survey1 = survey1
        self.sim1 = simulation1

        # self.dmis0 = data_misfit.L2DataMisfit(self.survey0)
        self.dmis0 = data_misfit.L2DataMisfit(data=dobs0, simulation=simulation0)
        self.dmis1 = data_misfit.L2DataMisfit(data=dobs1, simulation=simulation1)

        self.dmiscombo = self.dmis0 + self.dmis1

    def test_multiDataMisfit(self):
        self.dmis0.test()
        self.dmis1.test()
        self.dmiscombo.test(x=self.model)

    def test_inv(self):
        reg = regularization.WeightedLeastSquares(self.mesh)
        opt = optimization.InexactGaussNewton(maxIter=10, use_WolfeCurvature=True)
        invProb = inverse_problem.BaseInvProblem(self.dmiscombo, reg, opt)
        directives_list = [
            directives.ScalingMultipleDataMisfits_ByEig(verbose=True),
            directives.AlphasSmoothEstimate_ByEig(verbose=True),
            directives.BetaEstimate_ByEig(beta0_ratio=1e-2),
            directives.MultiTargetMisfits(TriggerSmall=False),
            directives.BetaSchedule(),
        ]
        inv = inversion.BaseInversion(invProb, directiveList=directives_list)
        m0 = self.model.mean() * np.ones_like(self.model)

        mrec = inv.run(m0)

    def test_inv_mref_setting(self):
        reg1 = regularization.WeightedLeastSquares(self.mesh)
        reg2 = regularization.WeightedLeastSquares(self.mesh)
        reg = reg1 + reg2
        opt = optimization.ProjectedGNCG(
            maxIter=30, lower=-10, upper=10, maxIterLS=20, maxIterCG=50, tolCG=1e-4
        )
        invProb = inverse_problem.BaseInvProblem(self.dmiscombo, reg, opt)
        directives_list = [
            directives.ScalingMultipleDataMisfits_ByEig(
                chi0_ratio=[0.01, 1.0], verbose=False
            ),
            directives.AlphasSmoothEstimate_ByEig(verbose=False),
            directives.BetaEstimate_ByEig(beta0_ratio=1e-2),
            directives.MultiTargetMisfits(TriggerSmall=False, verbose=True),
            directives.BetaSchedule(),
        ]
        inv = inversion.BaseInversion(invProb, directiveList=directives_list)
        m0 = self.model.mean() * np.ones_like(self.model)

        inv.run(m0)

        self.assertTrue(np.all(reg1.reference_model == m0))
        self.assertTrue(np.all(reg2.reference_model == m0))


if __name__ == "__main__":
    unittest.main()
