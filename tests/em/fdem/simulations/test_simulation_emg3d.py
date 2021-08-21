import pytest
import numpy as np
from numpy.testing import assert_allclose

import SimPEG.electromagnetics.frequency_domain as fdem

# Soft dependencies
try:
    import emg3d
except ImportError:
    emg3d = None


@pytest.mark.skipif(emg3d is None, reason="emg3d not installed.")
class TestSurveyToEmg3d():
    if emg3d:

        ## Receivers

        # recset1 and recset2 have one overlapping receiver
        # This means that they CANNOT be used with the same source (duplicate
        # receivers)
        recset1 = np.array([np.arange(2), np.zeros(2), np.zeros(2)]).T
        recset2 = np.array([np.arange(3)+1, np.zeros(3), np.zeros(3)]).T

        recset3 = np.array([30, 30, 30]).T

        rx_ex1 = fdem.receivers.PointElectricField(
                locations=recset1, component='complex', orientation='x')
        rx_ex2 = fdem.receivers.PointElectricField(
                locations=recset2, component='complex', orientation='x')
        rx_ey1 = fdem.receivers.PointElectricField(
                locations=recset1, component='complex', orientation='y')
        rx_hy1 = fdem.receivers.PointMagneticField(
                locations=recset1, component='complex', orientation='x')

        rx_hx3 = fdem.receivers.PointMagneticField(
                locations=recset3, component='complex', orientation='x')

        ## Sources
        wire_src_loc = ([-100, -100, 0], [-100, 100, 0])

        # 1b == 1a, to test that the conversion works even so
        pts_src_loc1a = (-10, -5, 7)
        pts_src_loc1b = (-10, -5, 7)

        pts_src_loc2 = (-20, -20, -20)

        src_list = []

        # pts_src_loc1a for two freqs
        for frequency in [1., 2.]:
            src_list.append(fdem.sources.ElectricDipole(
                location=pts_src_loc1a, azimuth=90, elevation=0,
                receiver_list=[rx_ex1, rx_ey1, rx_hy1], frequency=frequency))

        # Another source with the rx_ex2
        src_list.append(fdem.sources.ElectricDipole(
            location=pts_src_loc2, azimuth=0, elevation=0,
            receiver_list=[rx_ex2], frequency=2.))

        # 1 extra freq, for only one receiver, wire source
        src_list.append(fdem.sources.ElectricWire(
                locations=wire_src_loc, receiver_list=[rx_hx3, ],
                frequency=20.))

        # pts source for the same frequency
        src_list.append(fdem.sources.ElectricDipole(
            location=pts_src_loc1b, azimuth=90, elevation=0,
                receiver_list=[rx_ex2], frequency=20.))

        ## SimPEG survey and conversion  [test on its own]
        simpeg_survey = fdem.Survey(src_list)
        emg3d_survey, data_map = fdem.simulation_emg3d.survey_to_emg3d(
                simpeg_survey)

    def test_mapping(self):
        # Create some random numbers btw 100-999 in SimPEG-shape
        data = np.random.randint(100, 999, self.simpeg_survey.nD)

        # Create map
        emg3d_data = np.full(self.emg3d_survey.shape, np.nan)

        # Forward map
        emg3d_data[self.data_map] = data

        # Map back
        edata = emg3d_data[self.data_map]

        # Check
        assert_allclose(data, edata)

    def test_duplicate_src_rec_freq_fails(self):
        with pytest.raises(ValueError, match="Duplicate source-receiver-freq"):
            new_simpeg_survey = fdem.Survey([
                fdem.sources.ElectricDipole(
                    location=self.pts_src_loc2, azimuth=0, elevation=0,
                    receiver_list=[self.rx_ex2, self.rx_ex1], frequency=2.)
            ])
            fdem.simulation_emg3d.survey_to_emg3d(new_simpeg_survey)
