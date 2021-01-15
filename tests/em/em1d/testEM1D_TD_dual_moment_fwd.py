import unittest
from SimPEG import maps
import matplotlib.pyplot as plt
import SimPEG.electromagnetics.time_domain_1d as em1d
from SimPEG.electromagnetics.time_domain_1d.known_waveforms import *
from SimPEG.electromagnetics.analytics.em1d_analytics import *
import numpy as np
from scipy import io
from scipy.interpolate import interp1d


class EM1D_TD_FwdProblemTests(unittest.TestCase):

    def setUp(self):

        wave_HM = skytem_HM_2015()
        wave_LM = skytem_LM_2015()
        time_HM = wave_HM.time_gate_center[0::2]
        time_LM = wave_LM.time_gate_center[0::2]


        source_location = np.array([0., 0., 0.])
        source_orientation = "z"  # "x", "y" or "z"
        source_current = 1.
        source_radius = 10.
        moment_amplitude=1.

        receiver_location = np.array([10., 0., 0.])
        receiver_orientation = "z"  # "x", "y" or "z"
        field_type = "secondary"  # "secondary", "total" or "ppm"

        times = np.logspace(-5, -2, 41)

        # Receiver list
        rx = em1d.receivers.PointReceiver(
                receiver_location,
                times=time_HM,
                times_dual_moment=time_LM,
                orientation=receiver_orientation,
                component="dbdt"
        )
        receiver_list = [rx]

        # Sources

        time_input_currents_HM = wave_HM.current_times[-7:]
        input_currents_HM = wave_HM.currents[-7:]
        time_input_currents_LM = wave_LM.current_times[-13:]
        input_currents_LM = wave_LM.currents[-13:]


        src = em1d.sources.HorizontalLoopSource(
            receiver_list=receiver_list,
            location=source_location,
            I=source_current,
            a=source_radius,
            wave_type="general",
            moment_type='dual',
            time_input_currents=time_input_currents_HM,
            input_currents=input_currents_HM,
            n_pulse = 1,
            base_frequency = 25.,
            time_input_currents_dual_moment = time_input_currents_LM,
            input_currents_dual_moment = input_currents_LM,
            base_frequency_dual_moment = 210
        )
        source_list = [src]

        thicknesses = np.ones(3)
        sigma = 1e-2
        n_layer = thicknesses.size + 1

        sigma_model = sigma * np.ones(n_layer)
        survey = em1d.survey.EM1DSurveyTD(source_list)
        model_mapping = maps.IdentityMap(nP=n_layer)
        simulation = em1d.simulation.EM1DTMSimulation(
            survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping,
        )


        self.survey = survey
        self.simulation = simulation
        self.showIt = False
        self.sigma_model = sigma_model
        self.sigma_halfspace = sigma

    def test_em1dtd_circular_loop_single_pulse(self):

        src = self.survey.srcList[0]
        rx = src.rxList[0]
        dBzdtTD = self.simulation.dpred(self.sigma_model)
        dBzdtTD_HM = dBzdtTD[:rx.times.size]
        dBzdtTD_LM = dBzdtTD[rx.times.size:]

        def step_func_dBzdt(time):
            return dBzdt_horizontal_circular_loop(
                src.a, time, self.sigma_halfspace
            )

        dBzdtTD_analytic_HM = piecewise_pulse(
            step_func_dBzdt, rx.times,
            src.time_input_currents,
            src.input_currents,
            src.period
        )

        dBzdtTD_analytic_LM = piecewise_pulse(
            step_func_dBzdt, rx.times_dual_moment,
            src.time_input_currents_dual_moment,
            src.input_currents_dual_moment,
            src.period_dual_moment
        )

        if self.showIt:
            plt.loglog(rx.times, -dBzdtTD_HM)
            plt.loglog(rx.times_dual_moment, -dBzdtTD_LM)
            plt.loglog(rx.times, -dBzdtTD_analytic_HM, 'x')
            plt.loglog(rx.times_dual_moment, -dBzdtTD_analytic_LM, 'x')
            plt.show()

        err = (
            np.linalg.norm(dBzdtTD_HM-dBzdtTD_analytic_HM)/
            np.linalg.norm(dBzdtTD_analytic_HM)
        )

        print ('dBzdt error (HM) = ', err)

        self.assertTrue(err < 5e-2)
        err = (
            np.linalg.norm(dBzdtTD_LM-dBzdtTD_analytic_LM)/
            np.linalg.norm(dBzdtTD_analytic_LM)
        )

        print ('dBzdt error (LM) = ', err)
        self.assertTrue(err < 5e-2)

        print ("EM1DTD-CirculurLoop-general for real conductivity works")


if __name__ == '__main__':
    unittest.main()
