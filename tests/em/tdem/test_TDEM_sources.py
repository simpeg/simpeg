import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
from SimPEG.electromagnetics.time_domain.sources import (
    HalfSineWaveform,
    QuarterSineRampOnWaveform,
)


class TestQuarterSineRampOnWaveform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times = np.linspace(start=0, stop=1e-2, num=11)

    def test_waveform_with_plateau(self):
        quarter_sine = QuarterSineRampOnWaveform(
            ramp_on=np.r_[0.0, 3e-3], ramp_off=np.r_[7e-3, 1e-2]
        )
        result = [quarter_sine.eval(t) for t in self.times]
        expected = np.array(
            [0.0, 0.5, 0.866025, 1.0, 1.0, 1.0, 1.0, 1.0, 0.666667, 0.333333, 0.0]
        )

        assert_array_almost_equal(result, expected)

    def test_waveform_without_plateau(self):
        quarter_sine = QuarterSineRampOnWaveform(
            ramp_on=np.r_[0.0, 5e-3], ramp_off=np.r_[5e-3, 1e-2]
        )
        result = [quarter_sine.eval(t) for t in self.times]
        expected = np.array(
            [0.0, 0.309017, 0.587785, 0.809017, 0.951057, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        )

        assert_array_almost_equal(result, expected)

    def test_waveform_negative_plateau(self):
        """TODO: should this throw a ValueError instead?"""
        quarter_sine = QuarterSineRampOnWaveform(
            ramp_on=np.r_[0.0, 8e-3], ramp_off=np.r_[2e-3, 1e-2]
        )
        result = [quarter_sine.eval(t) for t in self.times]
        expected = np.array(
            [
                0.0,
                0.19509,
                0.382683,
                0.55557,
                0.707107,
                0.83147,
                0.92388,
                0.980785,
                1.0,
                0.125,
                0.0,
            ]
        )

        assert_array_almost_equal(result, expected)


class TestHalfSineWaveform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times = np.linspace(start=0, stop=1e-2, num=11)

    def test_waveform_with_plateau(self):
        half_sine = HalfSineWaveform(
            ramp_on=np.r_[0.0, 3e-3], ramp_off=np.r_[7e-3, 1e-2]
        )
        result = [half_sine.eval(t) for t in self.times]
        expected = np.array(
            [0.0, 0.5, 0.866025, 1.0, 1.0, 1.0, 1.0, 1.0, 0.866025, 0.5, 0.0]
        )

        assert_array_almost_equal(result, expected)

    def test_waveform_without_plateau(self):
        half_sine = HalfSineWaveform(
            ramp_on=np.r_[0.0, 5e-3], ramp_off=np.r_[5e-3, 1e-2]
        )
        result = [half_sine.eval(t) for t in self.times]
        expected = np.array(
            [
                0.0,
                0.309017,
                0.587785,
                0.809017,
                0.951057,
                1.0,
                0.951057,
                0.809017,
                0.587785,
                0.309017,
                0.0,
            ]
        )

        assert_array_almost_equal(result, expected)
