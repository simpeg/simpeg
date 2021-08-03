import unittest
from SimPEG import maps
from discretize import tests
import numpy as np
import SimPEG.electromagnetics.time_domain as tdem


class EM1D_TD_general_Jac_layers_ProblemTests(unittest.TestCase):
    def setUp(self):

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        thicknesses = np.r_[nearthick, deepthick]
        topo = np.r_[0.0, 0.0, 100.0]

        source_location = np.array([0.0, 0.0, 100.0 + 1e-5])
        receiver_locations = np.array([[0.0, 0.0, 100.0 + 1e-5]])
        receiver_orientation = "z"  # "x", "y" or "z"
        times = np.logspace(-5, -2, 31)
        radius = 20.0

        # Waveform
        waveform = tdem.sources.TriangularWaveform(
            startTime=-0.01, peakTime=-0.005, offTime=0.0
        )

        # Receiver list

        # Define receivers at each location.
        b_receiver = tdem.receivers.PointMagneticFluxDensity(
            receiver_locations, times, receiver_orientation
        )
        dbzdt_receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
            receiver_locations, times, receiver_orientation
        )
        receivers_list = [
            b_receiver,
            dbzdt_receiver,
        ]  # Make a list containing all receivers even if just one

        # Must define the transmitter properties and associated receivers
        source_list = [
            tdem.sources.CircularLoop(
                receivers_list,
                location=source_location,
                waveform=waveform,
                radius=radius,
            )
        ]

        survey = tdem.Survey(source_list)

        sigma = 1e-2

        self.topo = topo
        self.survey = survey
        self.showIt = False
        self.sigma = sigma
        self.times = times
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses) + 1
        self.a = radius

    def test_EM1DTDJvec_Layers(self):

        sigma_map = maps.ExpMap(nP=self.nlayers)
        sim = tdem.Simulation1DLayered(
            survey=self.survey,
            thicknesses=self.thicknesses,
            sigmaMap=sigma_map,
            topo=self.topo,
        )

        m_1D = np.log(np.ones(self.nlayers) * self.sigma)

        def fwdfun(m):
            resp = sim.dpred(m)
            return resp

        def jacfun(m, dm):
            Jvec = sim.Jvec(m, dm)
            return Jvec

        dm = m_1D * 0.5
        derChk = lambda m: [fwdfun(m), lambda mx: jacfun(m, mx)]
        passed = tests.checkDerivative(
            derChk, m_1D, num=4, dx=dm, plotIt=False, eps=1e-15
        )
        self.assertTrue(passed)

    def test_EM1DTDJtvec_Layers(self):

        sigma_map = maps.ExpMap(nP=self.nlayers)
        sim = tdem.Simulation1DLayered(
            survey=self.survey,
            thicknesses=self.thicknesses,
            sigmaMap=sigma_map,
            topo=self.topo,
        )

        sigma_layer = 0.1
        sigma = np.ones(self.nlayers) * self.sigma
        sigma[3] = sigma_layer
        m_true = np.log(sigma)

        dobs = sim.dpred(m_true)

        m_ini = np.log(np.ones(self.nlayers) * self.sigma)
        resp_ini = sim.dpred(m_ini)
        dr = resp_ini - dobs

        def misfit(m, dobs):
            dpred = sim.dpred(m)
            misfit = 0.5 * np.linalg.norm(dpred - dobs) ** 2
            dmisfit = sim.Jtvec(m, dr)
            return misfit, dmisfit

        derChk = lambda m: misfit(m, dobs)
        passed = tests.checkDerivative(derChk, m_ini, num=4, plotIt=False, eps=1e-26)
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
