import unittest
from SimPEG import maps
import matplotlib.pyplot as plt
import SimPEG.electromagnetics.time_domain_1d as em1d
from SimPEG.electromagnetics.time_domain_1d.supporting_functions.waveform_functions import *
from SimPEG.electromagnetics.analytics.em1d_analytics import *
import numpy as np
from scipy import io
from scipy.interpolate import interp1d


class EM1D_TD_FwdProblemTests(unittest.TestCase):

    def setUp(self):

        source_location = np.array([0., 0., 0.])
        source_orientation = "z"  # "x", "y" or "z"
        source_current = 1.
        source_radius = 10.
        moment_amplitude = 1.

        receiver_location = np.array([10., 0., 0.])
        receiver_orientation = "z"  # "x", "y" or "z"
        field_type = "secondary"  # "secondary", "total"

        time_HM = skytem_2015_HM_time_channels()
        time_LM = skytem_2015_LM_time_channels()
        
        # times = np.logspace(-5, -2, 41)

        # Receiver list
        rx = em1d.receivers.PointReceiver(
                receiver_location,
                times=time_HM,
                dual_times=time_LM,
                orientation=receiver_orientation,
                component="dbdt"
        )
        receiver_list = [rx]

        # Waveforms
        wave_HM = em1d.waveforms.Skytem2015HighMomentWaveform()
        wave_LM = em1d.waveforms.Skytem2015LowMomentWaveform()
        
        waveform_times_HM = skytem_2015_HM_waveform_times()
        waveform_current_HM = skytem_2015_HM_waveform_current()
        waveform_times_LM = skytem_2015_LM_waveform_times()
        waveform_current_LM = skytem_2015_LM_waveform_times()

        waveform = em1d.waveforms.DualWaveform(
            waveform_times=waveform_times_HM,
            waveform_current=waveform_current_HM,
            base_frequency = 25.,
            dual_waveform_times = waveform_times_LM,
            dual_waveform_current = waveform_current_LM,
            dual_base_frequency = 210
        )

        src = em1d.sources.HorizontalLoopSource(
            receiver_list=receiver_list,
            location=source_location,
            waveform = waveform,
            radius=source_radius,
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
                src.radius, time, self.sigma_halfspace
            )

        dBzdtTD_analytic_HM = piecewise_pulse(
            step_func_dBzdt, rx.times,
            src.waveform.waveform_times,
            src.waveform.waveform_current,
            src.waveform.period
        )

        dBzdtTD_analytic_LM = piecewise_pulse(
            step_func_dBzdt, rx.dual_times,
            src.waveform.dual_waveform_times,
            src.waveform.dual_waveform_current,
            src.waveform.dual_period
        )

        if self.showIt:
            plt.loglog(rx.times, -dBzdtTD_HM)
            plt.loglog(rx.dual_times, -dBzdtTD_LM)
            plt.loglog(rx.times, -dBzdtTD_analytic_HM, 'x')
            plt.loglog(rx.dual_times, -dBzdtTD_analytic_LM, 'x')
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
