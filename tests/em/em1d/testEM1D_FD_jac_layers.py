import unittest
from SimPEG import *
from SimPEG.utils import mkvc
import matplotlib.pyplot as plt
import simpegEM1D as em1d
import numpy as np


class EM1D_FD_Jac_layers_ProblemTests(unittest.TestCase):

    def setUp(self):

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        thicknesses = np.r_[nearthick, deepthick]
        topo = np.r_[0., 0., 100.]

        src_location = np.array([0., 0., 100.+1e-5])
        rx_location = np.array([10., 0., 100.+1e-5])
        field_type = "secondary"  # "secondary", "total" or "ppm"
        frequencies = np.logspace(1, 8, 21)

        # Receiver list
        receiver_list = []
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation="x",
                field_type=field_type, component="both"
            )
        )
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation="x",
                field_type=field_type, component="imag"
            )
        )
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation="y",
                field_type=field_type, component="both"
            )
        )
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation="y",
                field_type=field_type, component="real"
            )
        )
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation="z",
                field_type=field_type, component="both"
            )
        )
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation="z",
                field_type=field_type, component="imag"
            )
        )

        I = 1.
        a = 10.
        source_list = [
            em1d.sources.HarmonicHorizontalLoopSource(
                receiver_list=receiver_list, location=src_location, I=I, a=a
            )
        ]

        # Survey
        survey = em1d.survey.EM1DSurveyFD(source_list)

        self.topo = topo
        self.survey = survey
        self.showIt = False
        self.frequencies = frequencies
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses)+1
        self.sigma_map = maps.ExpMap(nP=self.nlayers)

        sim = em1d.simulation.EM1DFMSimulation(
            survey=self.survey, thicknesses=self.thicknesses,
            sigmaMap=self.sigma_map, topo=self.topo
        )

        self.sim = sim

    def test_EM1DFDJvec_Layers(self):

        sigma_half = 0.01
        sigma_blk = 0.1
        sig = np.ones(self.nlayers)*sigma_half
        sig[3] = sigma_blk
        m_1D = np.log(sig)

        Hz = self.sim.dpred(m_1D)
        dHzdsig = self.sim.compute_integral(
            m_1D, output_type='sensitivity_sigma'
        )

        def fwdfun(m):
            resp = self.sim.dpred(m)
            return resp
            # return Hz

        def jacfun(m, dm):
            Jvec = self.sim.Jvec(m, dm)
            return Jvec

        dm = m_1D*0.5
        derChk = lambda m: [fwdfun(m), lambda mx: jacfun(m, mx)]
        passed = tests.checkDerivative(
            derChk, m_1D, num=4, dx=dm, plotIt=False, eps=1e-15
        )

        if self.showIt is True:

            ilay = 3
            temp_r = mkvc((dHzdsig[:, ilay].copy()).real)
            temp_i = mkvc((dHzdsig[:, ilay].copy()).imag)
            frequency = mkvc(self.prob.survey.frequency)

            plt.loglog(frequency[temp_r > 0], temp_r[temp_r > 0], 'b.-')
            plt.loglog(frequency[temp_r < 0], -temp_r[temp_r < 0], 'b.--')
            plt.loglog(frequency[temp_i > 0], temp_i[temp_i > 0], 'r.-')
            plt.loglog(frequency[temp_i < 0], -temp_i[temp_i < 0], 'r.--')
            plt.show()

        if passed:
            print ("EM1DFD-layers Jvec works")

    def test_EM1DFDJtvec_Layers(self):

        sigma_half = 0.01
        sigma_blk = 0.1
        sig = np.ones(self.nlayers)*sigma_half
        sig[3] = sigma_blk
        m_true = np.log(sig)

        dobs = self.sim.dpred(m_true)

        m_ini = np.log(
            np.ones(self.nlayers)*sigma_half
        )
        resp_ini = self.sim.dpred(m_ini)
        dr = resp_ini-dobs

        def misfit(m, dobs):
            dpred = self.sim.dpred(m)
            misfit = 0.5*np.linalg.norm(dpred-dobs)**2
            dmisfit = self.sim.Jtvec(m, dr)
            return misfit, dmisfit

        derChk = lambda m: misfit(m, dobs)
        passed = tests.checkDerivative(
            derChk, m_ini, num=4, plotIt=False, eps=1e-27
        )
        self.assertTrue(passed)
        if passed:
            print ("EM1DFD-layers Jtvec works")


class EM1D_FD_Jac_layers_ProblemTests_Height(unittest.TestCase):

    def setUp(self):

        topo = np.r_[0., 0., 100.]

        src_location = np.array([0., 0., 100.+20.])
        rx_location = np.array([10., 0., 100.+20.])
        field_type = "secondary"  # "secondary", "total" or "ppm"
        frequencies = np.logspace(1, 8, 21)

        # Receiver list
        receiver_list = []
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation="x",
                field_type=field_type, component="both"
            )
        )
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation="x",
                field_type=field_type, component="both"
            )
        )
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation="x",
                field_type=field_type, component="imag"
            )
        )
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation="y",
                field_type=field_type, component="imag"
            )
        )
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation="z",
                field_type=field_type, component="both"
            )
        )
        receiver_list.append(
            em1d.receivers.HarmonicPointReceiver(
                rx_location, frequencies, orientation="z",
                field_type=field_type, component="imag"
            )
        )

        I = 1.
        a = 10.
        source_list = [
            em1d.sources.HarmonicHorizontalLoopSource(
                receiver_list=receiver_list, location=src_location, I=I, a=a
            )
        ]

        # Survey
        survey = em1d.survey.EM1DSurveyFD(source_list)

        wires = maps.Wires(('sigma', 1),('height', 1))
        expmap = maps.ExpMap(nP=1)
        sigma_map = expmap * wires.sigma

        self.topo = topo
        self.survey = survey
        self.showIt = False
        self.frequencies = frequencies
        self.nlayers = 1
        self.sigma_map = sigma_map
        self.h_map = wires.height

        sim = em1d.simulation.EM1DFMSimulation(
            survey=self.survey,
            sigmaMap=self.sigma_map, hMap=wires.height, topo=self.topo
        )

        self.sim = sim

    def test_EM1DFDJvec_Layers(self):

        sigma_half = 0.01
        height = 20.

        m_1D = np.r_[np.log(sigma_half), height]

        def fwdfun(m):
            resp = self.sim.dpred(m)
            return resp
            # return Hz

        def jacfun(m, dm):
            Jvec = self.sim.Jvec(m, dm)
            return Jvec

        dm = m_1D*0.5
        derChk = lambda m: [fwdfun(m), lambda mx: jacfun(m, mx)]
        passed = tests.checkDerivative(
            derChk, m_1D, num=4, dx=dm, plotIt=False, eps=1e-15
        )

        if passed:
            print ("EM1DFD - Jvec with height works")

    def test_EM1DFDJtvec_Layers(self):

        sigma_half = 0.01
        height = 20.

        m_true = np.r_[np.log(sigma_half), height]

        dobs = self.sim.dpred(m_true)

        m_ini = m_true * 1.2
        resp_ini = self.sim.dpred(m_ini)
        dr = resp_ini-dobs

        def misfit(m, dobs):
            dpred = self.sim.dpred(m)
            misfit = 0.5*np.linalg.norm(dpred-dobs)**2
            dmisfit = self.sim.Jtvec(m, dr)
            return misfit, dmisfit

        derChk = lambda m: misfit(m, dobs)
        passed = tests.checkDerivative(
            derChk, m_ini, num=4, plotIt=False, eps=1e-27
        )
        self.assertTrue(passed)
        if passed:
            print ("EM1DFD - Jtvec with height works")



if __name__ == '__main__':
    unittest.main()
