import numpy as np
import unittest

import discretize

from SimPEG import mkvc

from SimPEG import data_misfit
from SimPEG import optimization
from SimPEG import regularization
from SimPEG import inverse_problem
from SimPEG import inversion
from SimPEG.directives import BetaSchedule, TargetMisfit

from SimPEG.electromagnetics import viscous_remanent_magnetization as vrm


class VRM_inversion_tests(unittest.TestCase):

    def test_basic_inversion(self):

        """
        Test to see if inversion recovers model
        """

        h = [(2, 30)]
        meshObj = discretize.TensorMesh((h, h, [(2, 10)]), x0='CCN')

        mod = 0.00025*np.ones(meshObj.nC)
        mod[(meshObj.gridCC[:, 0] > -4.) &
            (meshObj.gridCC[:, 1] > -4.) &
            (meshObj.gridCC[:, 0] < 4.) &
            (meshObj.gridCC[:, 1] < 4.)] = 0.001

        times = np.logspace(-4, -2, 5)
        waveObj = vrm.waveforms.SquarePulse(delt=0.02)

        x, y = np.meshgrid(np.linspace(-17, 17, 16), np.linspace(-17, 17, 16))
        x, y, z = mkvc(x), mkvc(y), 0.5*np.ones(np.size(x))
        rxList = [vrm.Rx.Point(np.c_[x, y, z], times=times, fieldType='dbdt', fieldComp='z')]

        txNodes = np.array([[-20, -20, 0.001],
                            [20, -20, 0.001],
                            [20, 20, 0.001],
                            [-20, 20, 0.01],
                            [-20, -20, 0.001]])
        txList = [vrm.Src.LineCurrent(rxList, txNodes, 1., waveObj)]

        Survey = vrm.Survey(txList)
        Survey.t_active = np.zeros(Survey.nD, dtype=bool)
        Survey.set_active_interval(-1e6, 1e6)
        Problem = vrm.Simulation3DLinear(meshObj, refinement_factor=2)
        Problem.pair(Survey)
        dobs = Problem.make_synthetic_data(mod)
        Survey.eps = 1e-11

        dmis = data_misfit.L2DataMisfit(data=dobs, simulation=Problem)
        W = mkvc((np.sum(np.array(Problem.A)**2, axis=0)))**0.25
        reg = regularization.Simple(
            meshObj, alpha_s=0.01, alpha_x=1., alpha_y=1., alpha_z=1., cell_weights=W
            )
        opt = optimization.ProjectedGNCG(
            maxIter=20, lower=0., upper=1e-2, maxIterLS=20, tolCG=1e-4
            )
        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)
        directives = [
            BetaSchedule(coolingFactor=2, coolingRate=1),
            TargetMisfit()
        ]
        inv = inversion.BaseInversion(invProb, directiveList=directives)

        m0 = 1e-6*np.ones(len(mod))
        mrec = inv.run(m0)

        dmis_final = np.sum((dmis.W.diagonal()*(mkvc(dobs) - Problem.fields(mrec)))**2)
        mod_err_2 = np.sqrt(np.sum((mrec-mod)**2))/np.size(mod)
        mod_err_inf = np.max(np.abs(mrec-mod))

        self.assertTrue(dmis_final < Survey.nD and mod_err_2 < 5e-6 and mod_err_inf < np.max(mod))

if __name__ == '__main__':
    unittest.main()
