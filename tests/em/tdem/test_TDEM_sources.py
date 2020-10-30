import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
from SimPEG.electromagnetics.time_domain.sources import (
    StepOffWaveform,
    RampOffWaveform,
    VTEMWaveform,
    TrapezoidWaveform,
    TriangularWaveform,
    QuarterSineRampOnWaveform,
    HalfSineWaveform,
)


class TestStepOffWaveform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times = np.linspace(start=0, stop=1e-2, num=11)

    def test_waveform_with_default_off_time(self):
        step_off = StepOffWaveform()
        result = [step_off.eval(t) for t in self.times]
        expected = np.array([1.0] + [0.0] * 10)
        assert_array_almost_equal(result, expected)

    def test_waveform_with_custom_off_time(self):
        """For StepOffWaveform, offTime arg does not do anything."""
        step_off = StepOffWaveform(offTime=1e-3)
        result = [step_off.eval(t) for t in self.times]
        expected = np.array([1.0] + [0.0] * 10)
        assert_array_almost_equal(result, expected)


class TestRampOffWaveform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times = np.linspace(start=0, stop=1e-2, num=11)

    def test_waveform_with_whole_offtime(self):
        ramp_off = RampOffWaveform(offTime=1e-2)
        result = [ramp_off.eval(t) for t in self.times]
        expected = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        assert_array_almost_equal(result, expected)

    def test_waveform_with_partial_off_time(self):
        ramp_off = RampOffWaveform(offTime=5e-3)
        result = [ramp_off.eval(t) for t in self.times]
        expected = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert_array_almost_equal(result, expected)


class TestVTEMWaveform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times = np.linspace(start=0, stop=1e-2, num=11)

    def test_waveform_with_default_param(self):
        vtem = VTEMWaveform()
        result = [vtem.eval(t) for t in self.times]
        expected = np.array(
            [0.0, 0.701698, 0.93553, 0.816327, 0.136054, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        assert_array_almost_equal(result, expected)

    def test_waveform_with_custom_param(self):
        vtem = VTEMWaveform(offTime=8e-3, peakTime=4e-3, a=2.0)
        result = [vtem.eval(t) for t in self.times]
        expected = np.array(
            [0.0, 0.455054, 0.731059, 0.898464, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0]
        )
        assert_array_almost_equal(result, expected)


class TestTrapezoidWaveform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times = np.linspace(start=0, stop=1e-2, num=11)

    def test_waveform_with_symmetric_on_off(self):
        trapezoid = TrapezoidWaveform(
            ramp_on=np.r_[0.0, 2e-3], ramp_off=np.r_[6e-3, 8e-3]
        )
        result = [trapezoid.eval(t) for t in self.times]
        expected = np.array([0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0])
        assert_array_almost_equal(result, expected)

    def test_waveform_with_asymmetric_on_off(self):
        trapezoid = TrapezoidWaveform(
            ramp_on=np.r_[0.0, 2e-3], ramp_off=np.r_[6e-3, 10e-3]
        )
        result = [trapezoid.eval(t) for t in self.times]
        expected = np.array([0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0])
        assert_array_almost_equal(result, expected)


class TestTriangularWaveform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times = np.linspace(start=0, stop=1e-2, num=11)

    def test_waveform_with_symmetric_on_off(self):
        triangular = TriangularWaveform(peakTime=4e-3, offTime=8e-3)
        result = [triangular.eval(t) for t in self.times]
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0])
        assert_array_almost_equal(result, expected)

    def test_waveform_with_asymmetric_on_off(self):
        triangular = TriangularWaveform(peakTime=2e-3, offTime=6e-3)
        result = [triangular.eval(t) for t in self.times]
        expected = np.array([0.0, 0.5, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert_array_almost_equal(result, expected)


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
