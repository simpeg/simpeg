import SimPEG.VRM as VRM
import numpy as np
from SimPEG import Mesh, mkvc
import unittest
from SimPEG import DataMisfit
from SimPEG import Directives
from SimPEG import Optimization
from SimPEG import Regularization
from SimPEG import InvProblem
from SimPEG import Inversion


class VRM_inversion_tests(unittest.TestCase):

    def test_basic_inversion(self):

        """
        Test to see if inversion recovers model
        """

        h = [(2, 30)]
        meshObj = Mesh.TensorMesh((h, h, [(2, 10)]), x0='CCN')

        mod = 0.00025*np.ones(meshObj.nC)
        mod[(meshObj.gridCC[:, 0] > -4.) &
            (meshObj.gridCC[:, 1] > -4.) &
            (meshObj.gridCC[:, 0] < 4.) &
            (meshObj.gridCC[:, 1] < 4.)] = 0.001

        times = np.logspace(-4, -2, 5)
        waveObj = VRM.WaveformVRM.SquarePulse(delt=0.02)

        x, y = np.meshgrid(np.linspace(-17, 17, 16), np.linspace(-17, 17, 16))
        x, y, z = mkvc(x), mkvc(y), 0.5*np.ones(np.size(x))
        rxList = [VRM.Rx.Point(np.c_[x, y, z], times=times, fieldType='dbdt', fieldComp='z')]

        txNodes = np.array([[-20, -20, 0.001],
                            [20, -20, 0.001],
                            [20, 20, 0.001],
                            [-20, 20, 0.01],
                            [-20, -20, 0.001]])
        txList = [VRM.Src.LineCurrent(rxList, txNodes, 1., waveObj)]

        Survey = VRM.Survey(txList)
        Survey.t_active = np.zeros(Survey.nD, dtype=bool)
        Survey.set_active_interval(-1e6, 1e6)
        Problem = VRM.Problem_Linear(meshObj, ref_factor=2)
        Problem.pair(Survey)
        Survey.makeSyntheticData(mod)
        Survey.eps = 1e-11

        dmis = DataMisfit.l2_DataMisfit(Survey)
        W = mkvc((np.sum(np.array(Problem.A)**2, axis=0)))**0.25
        reg = Regularization.Simple(
            meshObj, alpha_s=0.01, alpha_x=1., alpha_y=1., alpha_z=1., cell_weights=W
            )
        opt = Optimization.ProjectedGNCG(
            maxIter=20, lower=0., upper=1e-2, maxIterLS=20, tolCG=1e-4
            )
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
        directives = [
            Directives.BetaSchedule(coolingFactor=2, coolingRate=1),
            Directives.TargetMisfit()
        ]
        inv = Inversion.BaseInversion(invProb, directiveList=directives)

        m0 = 1e-6*np.ones(len(mod))
        mrec = inv.run(m0)

        dmis_final = np.sum((dmis.W.diagonal()*(Survey.dobs - Problem.fields(mrec)))**2)
        mod_err_2 = np.sqrt(np.sum((mrec-mod)**2))/np.size(mod)
        mod_err_inf = np.max(np.abs(mrec-mod))

        self.assertTrue(dmis_final < Survey.nD and mod_err_2 < 5e-6 and mod_err_inf < np.max(mod))

if __name__ == '__main__':
    unittest.main()
