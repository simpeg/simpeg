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
            start_time=-0.01, peak_time=-0.005, off_time=0.0
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


class EM1D_TD_LineCurrent1D_Jac_layers_ProblemTests(unittest.TestCase):
    def setUp(self):
        # WalkTEM waveform
        # Low moment
        lm_waveform_times = np.r_[-1.041e-03, -9.850e-04, 0.000e00, 4.000e-06]
        lm_waveform_current = np.r_[0.0, 1.0, 1.0, 0.0]

        # High moment
        hm_waveform_times = np.r_[-8.333e-03, -8.033e-03, 0.000e00, 5.600e-06]
        hm_waveform_current = np.r_[0.0, 1.0, 1.0, 0.0]

        # Low moment
        lm_off_time = np.array(
            [
                1.149e-05,
                1.350e-05,
                1.549e-05,
                1.750e-05,
                2.000e-05,
                2.299e-05,
                2.649e-05,
                3.099e-05,
                3.700e-05,
                4.450e-05,
                5.350e-05,
                6.499e-05,
                7.949e-05,
                9.799e-05,
                1.215e-04,
                1.505e-04,
                1.875e-04,
                2.340e-04,
                2.920e-04,
                3.655e-04,
                4.580e-04,
                5.745e-04,
                7.210e-04,
            ]
        )

        # High moment
        hm_off_time = np.array(
            [
                9.810e-05,
                1.216e-04,
                1.506e-04,
                1.876e-04,
                2.341e-04,
                2.921e-04,
                3.656e-04,
                4.581e-04,
                5.746e-04,
                7.211e-04,
                9.056e-04,
                1.138e-03,
                1.431e-03,
                1.799e-03,
                2.262e-03,
                2.846e-03,
                3.580e-03,
                4.505e-03,
                5.670e-03,
                7.135e-03,
            ]
        )

        # WalkTEM geometry
        x_path = np.array([-20, -20, 20, 20, -20])
        y_path = np.array([-20, 20, 20, -20, -20])

        wire_paths = np.c_[x_path, y_path, np.zeros(5)]
        source_list = []
        receiver_list_lm = []
        receiver_list_hm = []
        receiver_location = np.array([[0, 0, 0]])
        receiver_orientation = "z"

        receiver_list_lm.append(
            tdem.receivers.PointMagneticFluxTimeDerivative(
                receiver_location, times=lm_off_time, orientation=receiver_orientation
            )
        )

        receiver_list_hm.append(
            tdem.receivers.PointMagneticFluxTimeDerivative(
                receiver_location, times=hm_off_time, orientation=receiver_orientation
            )
        )

        lm_wave = tdem.sources.PiecewiseLinearWaveform(
            lm_waveform_times, lm_waveform_current
        )
        hm_wave = tdem.sources.PiecewiseLinearWaveform(
            hm_waveform_times, hm_waveform_current
        )

        source_lm = tdem.sources.LineCurrent1D(
            receiver_list_lm, wire_paths, waveform=lm_wave
        )
        source_hm = tdem.sources.LineCurrent1D(
            receiver_list_hm, wire_paths, waveform=hm_wave
        )
        source_list.append(source_lm)
        source_list.append(source_hm)

        # Define a 1D TDEM survey
        survey = tdem.survey.Survey(source_list)

        # Physical properties
        sigma = np.array([1.0 / 10, 1.0 / 1])

        # Layer thicknesses
        thicknesses = np.array([30.0])
        n_layer = len(thicknesses) + 1

        self.survey = survey
        self.sigma = sigma
        self.thicknesses = thicknesses
        self.nlayers = len(thicknesses) + 1

    def test_EM1DTDJvec_Layers(self):

        sigma_map = maps.ExpMap(nP=self.nlayers)
        sim = tdem.Simulation1DLayered(
            survey=self.survey,
            thicknesses=self.thicknesses,
            sigmaMap=sigma_map,
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
        )

        sigma_layer = 0.1
        sigma = np.ones(self.nlayers) * self.sigma
        sigma[1] = sigma_layer
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
