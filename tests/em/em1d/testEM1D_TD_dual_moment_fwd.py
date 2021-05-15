import unittest
from SimPEG import maps
import matplotlib.pyplot as plt
import SimPEG.electromagnetics.time_domain_1d as em1d
import SimPEG.electromagnetics.time_domain as tdem
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

        receiver_locations = np.array([[10., 0., 0.]])
        receiver_orientation = "z"  # "x", "y" or "z"

        time_HM = skytem_2015_HM_time_channels()
        time_LM = skytem_2015_LM_time_channels()

        # Waveforms
        wave_HM = em1d.waveforms.Skytem2015HighMomentWaveform()
        wave_LM = em1d.waveforms.Skytem2015LowMomentWaveform()

        waveform_times_HM = skytem_2015_HM_waveform_times()
        waveform_current_HM = skytem_2015_HM_waveform_current()
        waveform_times_LM = skytem_2015_LM_waveform_times()
        waveform_current_LM = skytem_2015_LM_waveform_times()

        waveform_hm = tdem.sources.RawWaveform(
                waveform_times=waveform_times_HM, waveform_current=waveform_current_HM,
                n_pulse = 1, base_frequency = 25.
        )
        waveform_lm = tdem.sources.RawWaveform(
                waveform_times=waveform_times_LM, waveform_current=waveform_current_LM,
                n_pulse = 1, base_frequency = 210.
        )

        # Receiver list

        # Define receivers at each location.
        dbzdt_receiver_hm = tdem.receivers.PointMagneticFluxTimeDerivative(
            receiver_locations, time_HM, receiver_orientation
        )
        dbzdt_receiver_lm = tdem.receivers.PointMagneticFluxTimeDerivative(
            receiver_locations, time_LM, receiver_orientation
        )
        # Make a list containing all receivers even if just one

        # Must define the transmitter properties and associated receivers
        source_list = [
            tdem.sources.CircularLoop(
                [dbzdt_receiver_hm],
                location=source_location,
                waveform=waveform_hm,
                radius=source_radius,
                i_sounding=0
            ),
            tdem.sources.CircularLoop(
                [dbzdt_receiver_lm],
                location=source_location,
                waveform=waveform_lm,
                radius=source_radius,
                i_sounding=0
            )    
        ]

        survey = tdem.Survey(source_list)
        
        thicknesses = np.ones(3)
        sigma = 1e-2
        n_layer = thicknesses.size + 1

        sigma_model = sigma * np.ones(n_layer)

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

        src = self.survey.source_list[0]
        rx = src.receiver_list[0]
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

        src_lm = self.survey.source_list[1]
        rx_lm = src_lm.receiver_list[0]

        dBzdtTD_analytic_LM = piecewise_pulse(
            step_func_dBzdt, rx_lm.times,
            src_lm.waveform.waveform_times,
            src_lm.waveform.waveform_current,
            src_lm.waveform.period
        )

        if self.showIt:
            plt.loglog(rx.times, -dBzdtTD_HM)
            plt.loglog(rx_lm.times, -dBzdtTD_LM)
            plt.loglog(rx.times, -dBzdtTD_analytic_HM, 'x')
            plt.loglog(rx_lm.times, -dBzdtTD_analytic_LM, 'x')
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
