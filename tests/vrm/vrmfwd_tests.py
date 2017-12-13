import unittest
import SimPEG.VRM as VRM
import numpy as np
from SimPEG import Mesh

# SUMMARY OF TESTS
#
# Convergence tests with sensitivity refinement (breaks if it finds zero refinement cells)
# Tensor mesh bs octree mesh
# Can you accurately approximate halfspace solution?
# Linear and log normal codes need to match


class VRM_fwd_tests(unittest.TestCase):
    """Can the code match the analytic solution for the case that the
excitation is dipolar
    """

    def test_predict_dipolar(self):

        h = [0.05, 0.05]
        meshObj = Mesh.TensorMesh((h, h, h), x0='CCC')

        dchi = 0.01
        tau1 = 1e-8
        tau2 = 1e0
        mod = (dchi/np.log(tau2/tau1))*np.ones(meshObj.nC)

        times = np.logspace(-4, -2, 3)
        waveObj = VRM.WaveformVRM.SquarePulse(0.02)

        phi = np.random.uniform(-np.pi, np.pi)
        psi = np.random.uniform(-np.pi, np.pi)
        R = 2.
        loc_rx = R*np.c_[np.sin(phi)*np.cos(psi), np.sin(phi)*np.sin(psi), np.cos(phi)]

        rxList = [VRM.Rx.Point_dhdt(loc_rx, times, 'x')]
        rxList.append(VRM.Rx.Point_dhdt(loc_rx, times, 'y'))
        rxList.append(VRM.Rx.Point_dhdt(loc_rx, times, 'z'))

        alpha = np.random.uniform(0, np.pi)
        beta = np.random.uniform(-np.pi, np.pi)
        loc_tx = [0., 0., 0.]
        Src = VRM.Src.CircLoop(rxList, loc_tx, 25., np.r_[alpha, beta], 1., waveObj)
        txList = [Src]

        Survey = VRM.Survey(txList)
        Problem = VRM.ProblemVRM.LinearVRM(meshObj, refFact=0)
        Problem.pair(Survey)
        Fields = Problem.fields(mod)

        H0 = Src.getH0(np.c_[0., 0., 0.])
        dmdtx = -H0[0, 0]*0.1**3*(dchi/np.log(tau2/tau1))*(1/times[1] - 1/(times[1]+0.02))
        dmdty = -H0[0, 1]*0.1**3*(dchi/np.log(tau2/tau1))*(1/times[1] - 1/(times[1]+0.02))
        dmdtz = -H0[0, 2]*0.1**3*(dchi/np.log(tau2/tau1))*(1/times[1] - 1/(times[1]+0.02))
        dmdot = np.dot(np.r_[dmdtx, dmdty, dmdtz], loc_rx.T)

        fx = (1/(4*np.pi))*(3*loc_rx[0, 0]*dmdot/R**5 - dmdtx/R**3)
        fy = (1/(4*np.pi))*(3*loc_rx[0, 1]*dmdot/R**5 - dmdty/R**3)
        fz = (1/(4*np.pi))*(3*loc_rx[0, 2]*dmdot/R**5 - dmdtz/R**3)

        self.assertTrue(np.all(np.abs(Fields[1:-1:3] - np.r_[fx, fy, fz]) < 1e-5*np.sqrt((Fields[1:-1:3]**2).sum())))

    def test_sources(self):
        """If all the source types are setup to product dipolar fields, will
they produce the same fields"""

        h = [0.5, 0.5]
        meshObj = Mesh.TensorMesh((h, h, h), x0='CCC')

        dchi = 0.01
        tau1 = 1e-8
        tau2 = 1e0
        mod = (dchi/np.log(tau2/tau1))*np.ones(meshObj.nC)

        times = np.logspace(-4, -2, 3)
        waveObj = VRM.WaveformVRM.SquarePulse(0.02)

        phi = np.random.uniform(-np.pi, np.pi)
        psi = np.random.uniform(-np.pi, np.pi)
        Rrx = 3.
        loc_rx = Rrx*np.c_[np.sin(phi)*np.cos(psi), np.sin(phi)*np.sin(psi), np.cos(phi)]

        rxList = [VRM.Rx.Point_dhdt(loc_rx, times, 'x')]
        rxList.append(VRM.Rx.Point_dhdt(loc_rx, times, 'y'))
        rxList.append(VRM.Rx.Point_dhdt(loc_rx, times, 'z'))

        alpha = np.random.uniform(0, np.pi)
        beta = np.random.uniform(-np.pi, np.pi)
        Rtx = 4.
        loc_tx = Rtx*np.r_[np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta), np.cos(alpha)]

        txList = [VRM.Src.MagDipole(rxList, loc_tx, [0., 0., 0.01], waveObj)]
        txList.append(VRM.Src.CircLoop(rxList, loc_tx, np.sqrt(0.01/np.pi), np.r_[0., 0.], 1., waveObj))
        px = loc_tx[0]+np.r_[-0.05, 0.05, 0.05, -0.05, -0.05]
        py = loc_tx[1]+np.r_[-0.05, -0.05, 0.05, 0.05, -0.05]
        pz = loc_tx[2]*np.ones(5)
        txList.append(VRM.Src.LineCurrent(rxList, np.c_[px, py, pz], 1., waveObj))

        Survey = VRM.Survey(txList)
        Problem = VRM.ProblemVRM.LinearVRM(meshObj, refFact=0)
        Problem.pair(Survey)
        Fields = Problem.fields(mod)

        err1 = np.all(np.abs((Fields[9:18]-Fields[0:9])/Fields[0:9]) < 0.001)
        err2 = np.all(np.abs((Fields[18:]-Fields[0:9])/Fields[0:9]) < 0.001)
        err3 = np.all(np.abs((Fields[9:18]-Fields[18:])/Fields[18:]) < 0.001)

        self.assertTrue(err1 and err2 and err3)



if __name__ == '__main__':
    unittest.main()
