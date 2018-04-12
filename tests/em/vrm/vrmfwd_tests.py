import unittest
import SimPEG.VRM as VRM
import numpy as np
from SimPEG import Mesh


class VRM_fwd_tests(unittest.TestCase):

    """
    Computed vs analytic dipole field
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

        # rxList = [VRM.Rx.Point_dhdt(loc_rx, times, 'x')]
        # rxList.append(VRM.Rx.Point_dhdt(loc_rx, times, 'y'))
        # rxList.append(VRM.Rx.Point_dhdt(loc_rx, times, 'z'))

        rxList = [VRM.Rx.Point(loc_rx, times, 'dhdt', 'x')]
        rxList.append(VRM.Rx.Point(loc_rx, times, 'dhdt', 'y'))
        rxList.append(VRM.Rx.Point(loc_rx, times, 'dhdt', 'z'))

        alpha = np.random.uniform(0, np.pi)
        beta = np.random.uniform(-np.pi, np.pi)
        loc_tx = [0., 0., 0.]
        Src = VRM.Src.CircLoop(rxList, loc_tx, 25., np.r_[alpha, beta], 1., waveObj)
        txList = [Src]

        Survey = VRM.Survey(txList)
        Problem = VRM.Problem_Linear(meshObj, refFact=0)
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
        """
        Multiple source classes are used to make a small dipole source with the
        same orientation and dipole moment. Test ensures the same fields are
        computed.
        """

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

        # rxList = [VRM.Rx.Point_dhdt(loc_rx, times, 'x')]
        # rxList.append(VRM.Rx.Point_dhdt(loc_rx, times, 'y'))
        # rxList.append(VRM.Rx.Point_dhdt(loc_rx, times, 'z'))

        rxList = [VRM.Rx.Point(loc_rx, times, 'dhdt', 'x')]
        rxList.append(VRM.Rx.Point(loc_rx, times, 'dhdt', 'y'))
        rxList.append(VRM.Rx.Point(loc_rx, times, 'dhdt', 'z'))

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
        Problem = VRM.Problem_Linear(meshObj, refFact=1)
        Problem.pair(Survey)
        Fields = Problem.fields(mod)

        err1 = np.all(np.abs(Fields[9:18]-Fields[0:9])/(np.abs(Fields[0:9])+1e-14) < 0.005)
        err2 = np.all(np.abs(Fields[18:]-Fields[0:9])/(np.abs(Fields[0:9])+1e-14) < 0.005)
        err3 = np.all(np.abs(Fields[9:18]-Fields[18:])/(np.abs(Fields[18:])+1e-14) < 0.005)

        self.assertTrue(err1 and err2 and err3)

    def test_convergence_vertical(self):

        """
        Test the convergence of the solution to analytic results from
        Cowan (2016) and test accuracy
        """

        h = [(2, 20)]
        meshObj = Mesh.TensorMesh((h, h, h), x0='CCN')

        dchi = 0.01
        tau1 = 1e-8
        tau2 = 1e0
        mod = (dchi/np.log(tau2/tau1))*np.ones(meshObj.nC)

        times = np.array([1e-3])
        waveObj = VRM.WaveformVRM.SquarePulse(0.02)

        z = 0.5
        a = 0.1
        loc_rx = np.c_[0., 0., z]
        # rxList = [VRM.Rx.Point_dhdt(loc_rx, times, 'z')]
        rxList = [VRM.Rx.Point(loc_rx, times, 'dhdt', 'z')]
        txList = [VRM.Src.CircLoop(rxList, np.r_[0., 0., z], a, np.r_[0., 0.], 1., waveObj)]

        Survey2 = VRM.Survey(txList)
        Survey3 = VRM.Survey(txList)
        Survey4 = VRM.Survey(txList)
        Survey5 = VRM.Survey(txList)
        Problem2 = VRM.Problem_Linear(meshObj, refFact=2)
        Problem3 = VRM.Problem_Linear(meshObj, refFact=3)
        Problem4 = VRM.Problem_Linear(meshObj, refFact=4)
        Problem5 = VRM.Problem_Linear(meshObj, refFact=5)
        Problem2.pair(Survey2)
        Problem3.pair(Survey3)
        Problem4.pair(Survey4)
        Problem5.pair(Survey5)
        Fields2 = Problem2.fields(mod)
        Fields3 = Problem3.fields(mod)
        Fields4 = Problem4.fields(mod)
        Fields5 = Problem5.fields(mod)

        F = -(1/np.log(tau2/tau1))*(1/times - 1/(times+0.02))
        Fields_true = (0.5*np.pi*a**2/np.pi)*(dchi/(2+dchi))*((2*z)**2 + a**2)**-1.5*F

        Errs = np.abs((np.r_[Fields2, Fields3, Fields4, Fields5] - Fields_true)/Fields_true)

        Test1 = Errs[-1] < 0.005
        Test2 = np.all(Errs[1:]-Errs[0:-1] < 0.)

        self.assertTrue(Test1 and Test2)

    def test_convergence_radial(self):

        """
        Test the convergence of the solution to analytic results from
        Cowan (2016) and test accuracy
        """

        h = [(2, 30)]
        meshObj = Mesh.TensorMesh((h, h, [(2, 20)]), x0='CCN')

        dchi = 0.01
        tau1 = 1e-8
        tau2 = 1e0
        mod = (dchi/np.log(tau2/tau1))*np.ones(meshObj.nC)

        times = np.array([1e-3])
        waveObj = VRM.WaveformVRM.SquarePulse(0.02)

        z = 0.25
        a = 5
        # rxList = [VRM.Rx.Point_dhdt(np.c_[a, 0., z], times, 'x')]
        # rxList.append(VRM.Rx.Point_dhdt(np.c_[0., a, z], times, 'y'))
        rxList = [VRM.Rx.Point(np.c_[a, 0., z], times, 'dhdt', 'x')]
        rxList.append(VRM.Rx.Point(np.c_[0., a, z], times, 'dhdt', 'y'))
        txList = [VRM.Src.CircLoop(rxList, np.r_[0., 0., z], a, np.r_[0., 0.], 1., waveObj)]

        Survey2 = VRM.Survey(txList)
        Survey3 = VRM.Survey(txList)
        Survey4 = VRM.Survey(txList)
        Survey5 = VRM.Survey(txList)
        Problem2 = VRM.Problem_Linear(meshObj, refFact=2)
        Problem3 = VRM.Problem_Linear(meshObj, refFact=3)
        Problem4 = VRM.Problem_Linear(meshObj, refFact=4)
        Problem5 = VRM.Problem_Linear(meshObj, refFact=5)
        Problem2.pair(Survey2)
        Problem3.pair(Survey3)
        Problem4.pair(Survey4)
        Problem5.pair(Survey5)
        Fields2 = Problem2.fields(mod)
        Fields3 = Problem3.fields(mod)
        Fields4 = Problem4.fields(mod)
        Fields5 = Problem5.fields(mod)

        gamma = 4*z*(2/np.pi)**1.5
        F = -(1/np.log(tau2/tau1))*(1/times - 1/(times+0.02))
        Fields_true = 0.5*(dchi/(2+dchi))*(np.pi*gamma)**-1*F

        ErrsX = np.abs((np.r_[Fields2[0], Fields3[0], Fields4[0], Fields5[0]] - Fields_true)/Fields_true)
        ErrsY = np.abs((np.r_[Fields2[1], Fields3[1], Fields4[1], Fields5[1]] - Fields_true)/Fields_true)

        Testx1 = ErrsX[-1] < 0.01
        Testy1 = ErrsY[-1] < 0.01
        Testx2 = np.all(ErrsX[1:]-ErrsX[0:-1] < 0.)
        Testy2 = np.all(ErrsY[1:]-ErrsY[0:-1] < 0.)

        self.assertTrue(Testx1 and Testx2 and Testy1 and Testy2)

    def test_vs_mesh_vs_loguniform(self):

        """
        Test to make sure OcTree matches Tensor results and linear vs
        loguniform match
        """

        h1 = [(2, 4)]
        h2 = 0.5*np.ones(16)
        meshObj_Tensor = Mesh.TensorMesh((h1, h1, h1), x0='000')
        meshObj_OcTree = Mesh.TreeMesh([h2, h2, h2], x0='000')

        meshObj_OcTree.refine(2)

        def refinefcn(cell):
            xyz = cell.center
            dist = ((xyz - [4., 4., 8.])**2).sum()**0.5
            if dist < 2.65:
                return 4
            return 2

        meshObj_OcTree.refine(refinefcn)

        chi0 = 0.
        dchi = 0.01
        tau1 = 1e-8
        tau2 = 1e0

        # Tensor Models
        mod_a = (dchi/np.log(tau2/tau1))*np.ones(meshObj_Tensor.nC)
        mod_chi0_a = chi0*np.ones(meshObj_Tensor.nC)
        mod_dchi_a = dchi*np.ones(meshObj_Tensor.nC)
        mod_tau1_a = tau1*np.ones(meshObj_Tensor.nC)
        mod_tau2_a = tau2*np.ones(meshObj_Tensor.nC)

        # OcTree Models
        mod_b = (dchi/np.log(tau2/tau1))*np.ones(meshObj_OcTree.nC)
        mod_chi0_b = chi0*np.ones(meshObj_OcTree.nC)
        mod_dchi_b = dchi*np.ones(meshObj_OcTree.nC)
        mod_tau1_b = tau1*np.ones(meshObj_OcTree.nC)
        mod_tau2_b = tau2*np.ones(meshObj_OcTree.nC)

        times = np.array([1e-3])
        waveObj = VRM.WaveformVRM.SquarePulse(0.02)

        loc_rx = np.c_[4., 4., 8.25]
        # rxList = [VRM.Rx.Point_dhdt(loc_rx, times, 'z')]
        rxList = [VRM.Rx.Point(loc_rx, times, 'dhdt', 'z')]
        txList = [VRM.Src.MagDipole(rxList, np.r_[4., 4., 8.25], [0., 0., 1.], waveObj)]

        Survey1 = VRM.Survey(txList)
        Survey2 = VRM.Survey(txList)
        Survey3 = VRM.Survey(txList)
        Survey4 = VRM.Survey(txList)
        Problem1 = VRM.Problem_Linear(meshObj_Tensor, refFact=2, refRadius=[1.9, 3.6])
        Problem2 = VRM.Problem_LogUniform(meshObj_Tensor, refFact=2, refRadius=[1.9, 3.6], chi0=mod_chi0_a, dchi=mod_dchi_a, tau1=mod_tau1_a, tau2=mod_tau2_a)
        Problem3 = VRM.Problem_Linear(meshObj_OcTree, refFact=0)
        Problem4 = VRM.Problem_LogUniform(meshObj_OcTree, refFact=0, chi0=mod_chi0_b, dchi=mod_dchi_b, tau1=mod_tau1_b, tau2=mod_tau2_b)
        Problem1.pair(Survey1)
        Problem2.pair(Survey2)
        Problem3.pair(Survey3)
        Problem4.pair(Survey4)
        Fields1 = Problem1.fields(mod_a)
        Fields2 = Problem2.fields()
        Fields3 = Problem3.fields(mod_b)
        Fields4 = Problem4.fields()
        dpred1 = Survey1.dpred(mod_a)
        dpred2 = Survey2.dpred(mod_a)

        Err1 = np.abs((Fields1-Fields2)/Fields1)
        Err2 = np.abs((Fields2-Fields3)/Fields2)
        Err3 = np.abs((Fields3-Fields4)/Fields3)
        Err4 = np.abs((Fields4-Fields1)/Fields4)
        Err5 = np.abs((dpred1-dpred2)/dpred1)

        Test1 = Err1 < 0.001
        Test2 = Err2 < 0.001
        Test3 = Err3 < 0.001
        Test4 = Err4 < 0.001
        Test5 = Err5 < 0.001

        self.assertTrue(Test1 and Test2 and Test3 and Test4 and Test5)

    def test_receiver_types(self):
        """
        Test ensures the fields predicted for each receiver type
        are correct.
        """

        h1 = [0.25, 0.25]
        meshObj_Tensor = Mesh.TensorMesh((h1, h1, h1), x0='CCN')

        chi0 = 0.
        dchi = 0.01
        tau1 = 1e-8
        tau2 = 1e0

        # Tensor Model
        mod = (dchi/np.log(tau2/tau1))*np.ones(meshObj_Tensor.nC)

        times = np.array([1e-3])
        waveObj = VRM.WaveformVRM.SquarePulse(0.02)

        phi = np.random.uniform(-np.pi, np.pi)
        psi = np.random.uniform(-np.pi, np.pi)
        R = 4.
        loc_rx = R*np.c_[np.sin(phi)*np.cos(psi), np.sin(phi)*np.sin(psi), np.cos(phi)]
        loc_tx = 0.5*np.r_[np.sin(phi)*np.cos(psi), np.sin(phi)*np.sin(psi), np.cos(phi)]

        rxList1 = [VRM.Rx.Point(loc_rx, times, 'dhdt', 'x')]
        rxList1.append(VRM.Rx.Point(loc_rx, times, 'dhdt', 'y'))
        rxList1.append(VRM.Rx.Point(loc_rx, times, 'dhdt', 'z'))

        w = 0.1
        N = 100
        rxList2 = [VRM.Rx.SquareLoop(loc_rx, times, w, N, 'dhdt', 'x')]
        rxList2.append(VRM.Rx.SquareLoop(loc_rx, times, w, N, 'dhdt', 'y'))
        rxList2.append(VRM.Rx.SquareLoop(loc_rx, times, w, N, 'dhdt', 'z'))

        txList1 = [VRM.Src.MagDipole(rxList1, loc_tx, [1., 1., 1.], waveObj)]
        txList2 = [VRM.Src.MagDipole(rxList2, loc_tx, [1., 1., 1.], waveObj)]

        Survey1 = VRM.Survey(txList1)
        Survey2 = VRM.Survey(txList2)
        Problem1 = VRM.Problem_Linear(meshObj_Tensor, refFact=2, refRadius=[1.9, 3.6])
        Problem2 = VRM.Problem_Linear(meshObj_Tensor, refFact=2, refRadius=[1.9, 3.6])
        Problem1.pair(Survey1)
        Problem2.pair(Survey2)
        Fields1 = Problem1.fields(mod)
        Fields2 = Problem2.fields(mod)

        Err = np.abs(Fields1-Fields2)

        Test = np.all(Err < 1e-7)

        self.assertTrue(Test)

if __name__ == '__main__':
    unittest.main()
