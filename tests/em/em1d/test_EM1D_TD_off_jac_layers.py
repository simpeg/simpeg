import unittest
from SimPEG import maps
from discretize import tests
import SimPEG.electromagnetics.time_domain as tdem
import numpy as np


class EM1D_TD_Jac_layers_ProblemTests(unittest.TestCase):
    def setUp(self):

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        thicknesses = np.r_[nearthick, deepthick]
        topo = np.r_[0.0, 0.0, 100.0]

        source_location = np.array([0.0, 0.0, 100.0 + 1e-5])
        nsrc = 1
        receiver_locations = np.array([[0.0, 0.0, 100.0 + 1e-5]])
        times = np.logspace(-5, -2, 31)
        radius = 20.0
        waveform = tdem.sources.StepOffWaveform(offTime=0.0)

        # Receiver list

        # Define receivers at each location.
        b_receiver = tdem.receivers.PointMagneticFluxDensity(
            receiver_locations, times, "z"
        )
        dbzdt_receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
            receiver_locations, times, "z"
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

        self.topo = topo
        self.survey = survey
        self.showIt = False
        self.times = times
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses) + 1
        self.a = radius

        wire_map = maps.Wires(
            ("sigma", self.nlayers),
            ("eta", self.nlayers),
            ("tau", self.nlayers),
            ("c", self.nlayers),
        )
        self.sigma_map = maps.ExpMap(nP=self.nlayers) * wire_map.sigma
        self.eta_map = maps.ExpMap(nP=self.nlayers) * wire_map.eta
        self.tau_map = maps.ExpMap(nP=self.nlayers) * wire_map.tau
        self.c_map = maps.ExpMap(nP=self.nlayers) * wire_map.c
        sim = tdem.Simulation1DLayered(
            survey=self.survey,
            sigmaMap=self.sigma_map,
            etaMap=self.eta_map,
            tauMap=self.tau_map,
            cMap=self.c_map,
            thicknesses=thicknesses,
            topo=self.topo,
        )
        self.sim = sim

    def test_EM1DTDJvec_Layers(self):

        # Conductivity
        sigma_half = 0.01
        sigma_blk = 0.1
        sig = np.ones(self.nlayers) * sigma_half
        sig[3] = sigma_blk

        eta = np.ones_like(sig) * 0.5
        tau = np.ones_like(sig) * 1e-3
        c = np.ones_like(sig) * 0.5

        # General model
        m_1D = np.r_[
            np.log(sig),
            np.log(eta),
            np.log(tau),
            np.log(c)
        ]

        def fwdfun(m):
            resp = self.sim.dpred(m)
            return resp

        def jacfun(m, dm):
            Jvec = self.sim.Jvec(m, dm)
            return Jvec

        dm = m_1D * 0.5
        derChk = lambda m: [fwdfun(m), lambda mx: jacfun(m, mx)]
        passed = tests.checkDerivative(
            derChk, m_1D, num=4, dx=dm, plotIt=False, eps=1e-15
        )
        self.assertTrue(passed)

    def test_EM1DTDJtvec_Layers(self):

        # Conductivity
        sigma_half = 0.01
        sigma_blk = 0.1
        sig = np.ones(self.nlayers) * sigma_half
        sig[3] = sigma_blk

        eta = np.ones_like(sig) * 0.5
        tau = np.ones_like(sig) * 1e-3
        c = np.ones_like(sig) * 0.5

        # General model
        m_true = np.r_[
            np.log(sig),
            np.log(eta),
            np.log(tau),
            np.log(c),
        ]

        dobs = self.sim.dpred(m_true)

        m_ini = np.r_[
            np.log(np.ones(self.nlayers) * sigma_half),
            np.log(1.1*eta),
            np.log(1.1*tau),
            np.log(1.1*c),
        ]
        resp_ini = self.sim.dpred(m_ini)
        dr = resp_ini - dobs


        def misfit(m, dobs):
            dpred = self.sim.dpred(m)
            misfit = 0.5 * np.linalg.norm(dpred - dobs) ** 2
            dmisfit = self.sim.Jtvec(m, dr)
            return misfit, dmisfit

        derChk = lambda m: misfit(m, dobs)
        passed = tests.checkDerivative(derChk, m_ini, num=4, plotIt=False, eps=1e-26)
        self.assertTrue(passed)


# class EM1D_TD_Jac_layers_ProblemTests_Height(unittest.TestCase):

#     def setUp(self):

#         topo = np.r_[0., 0., 100.]

#         src_location = np.array([0., 0., 100.+20.])
#         rx_location = np.array([0., 0., 100.+20.])
#         receiver_orientation = "z"  # "x", "y" or "z"
#         times = np.logspace(-5, -2, 31)
#         a = 20.

#         # Receiver list
#         receiver_list = []
#         receiver_list.append(
#             em1d.receivers.PointReceiver(
#                 rx_location, times, orientation=receiver_orientation,
#                 component="b"
#             )
#         )
#         receiver_list.append(
#             em1d.receivers.PointReceiver(
#                 rx_location, times, orientation=receiver_orientation,
#                 component="dbdt"
#             )
#         )

#         waveform = em1d.waveforms.StepoffWaveform()

#         source_list = [
#             em1d.sources.HorizontalLoopSource(
#                 receiver_list=receiver_list, location=src_location, waveform=waveform,
#                 radius=a, current_amplitude=1.
#             )
#         ]
#         # Survey
#         survey = em1d.survey.EM1DSurveyTD(source_list)

#         wires = maps.Wires(('sigma', 1),('height', 1))
#         expmap = maps.ExpMap(nP=1)
#         sigma_map = expmap * wires.sigma

#         self.topo = topo
#         self.survey = survey
#         self.showIt = False
#         self.times = times
#         self.nlayers = 1
#         self.a = a
#         self.sigma_map = sigma_map
#         self.h_map = wires.height


#     def test_EM1DTDJvec_Layers(self):

#         sim = em1d.simulation.EM1DTMSimulation(
#             survey=self.survey,
#             sigmaMap=self.sigma_map, hMap=self.h_map, topo=self.topo
#         )

#         sigma_half = 0.01
#         height = 20.

#         m_1D = np.r_[np.log(sigma_half), height]

#         def fwdfun(m):
#             resp = sim.dpred(m)
#             return resp

#         def jacfun(m, dm):
#             Jvec = sim.Jvec(m, dm)
#             return Jvec

#         dm = m_1D*0.5
#         derChk = lambda m: [fwdfun(m), lambda mx: jacfun(m, mx)]
#         passed = tests.checkDerivative(
#             derChk, m_1D, num=4, dx=dm, plotIt=False, eps=1e-15
#         )

#         if passed:
#             print ("EM1DTD-layers Jvec works")


#     def test_EM1DTDJtvec_Layers(self):

#         sim = em1d.simulation.EM1DTMSimulation(
#             survey=self.survey,
#             sigmaMap=self.sigma_map, hMap=self.h_map, topo=self.topo
#         )

#         sigma_half = 0.01
#         height = 20.

#         m_true = np.r_[np.log(sigma_half), height]

#         dobs = sim.dpred(m_true)

#         m_ini = 1.2 * np.r_[np.log(sigma_half), height]
#         resp_ini = sim.dpred(m_ini)
#         dr = resp_ini-dobs

#         def misfit(m, dobs):
#             dpred = sim.dpred(m)
#             misfit = 0.5*np.linalg.norm(dpred-dobs)**2
#             dmisfit = sim.Jtvec(m, dr)
#             return misfit, dmisfit

#         derChk = lambda m: misfit(m, dobs)
#         passed = tests.checkDerivative(
#             derChk, m_ini, num=4, plotIt=False, eps=1e-26
#         )
#         self.assertTrue(passed)
#         if passed:
#             print ("EM1DTD-layers Jtvec works")


if __name__ == "__main__":
    unittest.main()
