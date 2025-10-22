import unittest
import re

import pytest
import numpy as np
import scipy.sparse as sp
from discretize.tests import check_derivative
from numpy.testing import assert_array_almost_equal
from simpeg.electromagnetics.time_domain.sources import (
    ExponentialWaveform,
    HalfSineWaveform,
    PiecewiseLinearWaveform,
    QuarterSineRampOnWaveform,
    RampOffWaveform,
    StepOffWaveform,
    TrapezoidWaveform,
    TriangularWaveform,
    VTEMWaveform,
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
        """For StepOffWaveform, off_time arg does not do anything."""
        step_off = StepOffWaveform(off_time=1e-3)
        result = [step_off.eval(t) for t in self.times]
        expected = np.array([1.0, 1.0] + [0.0] * 9)
        assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("ramp_start", [-1e-3, None, 0, 1e-3])
@pytest.mark.parametrize("ramp_end", [1e-2, 5e-3])
class TestRampOffWaveform:
    times = np.linspace(start=-1e-2, stop=2e-2, num=31)

    def test_waveform_evaluate(self, ramp_start, ramp_end):
        if ramp_start is None:
            args = (ramp_end,)
        else:
            args = (ramp_start, ramp_end)
        ramp_off = RampOffWaveform(*args)
        if ramp_start is None:
            assert ramp_off.ramp_start == 0.0
        result = [ramp_off.eval(t) for t in self.times]
        expected = np.interp(self.times, ramp_off.time_nodes, [1, 0])
        assert_array_almost_equal(result, expected)

    def test_waveform_derivative(self, ramp_start, ramp_end):
        # Test the waveform derivative at points between the time_nodes
        if ramp_start is None:
            args = (ramp_end,)
        else:
            args = (ramp_start, ramp_end)
        wave = RampOffWaveform(*args)

        def f(t):
            wave_eval = np.array([wave.eval(ti) for ti in t])
            dWave_dt = sp.diags(wave.eval_deriv(t))
            return wave_eval, dWave_dt

        t_nodes = wave.time_nodes
        t0 = np.concatenate(
            [
                np.linspace(t_nodes[i], t_nodes[i + 1], 6)[1:-2]
                for i in range(len(t_nodes) - 1)
            ]
        )
        dt = np.min(np.diff(t0)) * 0.5 * np.ones_like(t0)

        assert check_derivative(f, t0, dx=dt, plotIt=False, random_seed=5421)


@pytest.mark.parametrize("attr", ["ramp_end", "off_time"])
def test_ramp_off_time_is_ramp_end(attr):
    t_off = 0.01
    if attr == "ramp_end":
        ramp = RampOffWaveform(t_off)
    else:
        with pytest.warns(
            DeprecationWarning, match="`off_time` keyword arg has been deprecated.*"
        ):
            ramp = RampOffWaveform(off_time=t_off)
    assert ramp.ramp_end == t_off
    assert ramp.off_time == t_off

    t2_off = 0.02
    setattr(ramp, attr, t2_off)
    assert ramp.ramp_end == t2_off
    assert ramp.off_time == t2_off


def test_ramp_off_bad_end():
    with pytest.raises(
        ValueError,
        match=re.escape("'ramp_end' must be a value in the range (0.1, inf]"),
    ):
        RampOffWaveform(0.1, 0.0)


def test_ramp_off_good_args():
    with pytest.warns(
        DeprecationWarning, match="`off_time` keyword arg has been deprecated.*"
    ):
        ramp = RampOffWaveform(off_time=0.1)
        assert ramp.ramp_start == 0.0
        assert ramp.ramp_end == 0.1

    ramp = RampOffWaveform(0.1)
    assert ramp.ramp_start == 0.0
    assert ramp.ramp_end == 0.1

    ramp = RampOffWaveform(ramp_end=0.1)
    assert ramp.ramp_start == 0.0
    assert ramp.ramp_end == 0.1

    ramp = RampOffWaveform(0.1, 0.2)
    assert ramp.ramp_start == 0.1
    assert ramp.ramp_end == 0.2

    ramp = RampOffWaveform(0.1, ramp_end=0.2)
    assert ramp.ramp_start == 0.1
    assert ramp.ramp_end == 0.2

    ramp = RampOffWaveform(ramp_start=0.1, ramp_end=0.2)
    assert ramp.ramp_start == 0.1
    assert ramp.ramp_end == 0.2

    ramp = RampOffWaveform(ramp_end=0.2, ramp_start=0.1)
    assert ramp.ramp_start == 0.1
    assert ramp.ramp_end == 0.2

    with pytest.warns(
        DeprecationWarning, match="`off_time` keyword arg has been deprecated.*"
    ):
        ramp = RampOffWaveform(0.1, off_time=0.2)
        assert ramp.ramp_start == 0.1
        assert ramp.ramp_end == 0.2

    with pytest.warns(
        DeprecationWarning, match="`off_time` keyword arg has been deprecated.*"
    ):
        ramp = RampOffWaveform(ramp_start=0.1, off_time=0.2)
        assert ramp.ramp_start == 0.1
        assert ramp.ramp_end == 0.2


def test_ramp_off_bad_args():
    with pytest.raises(
        TypeError,
        match=re.escape("Can not specify both `off_time` and a `ramp_end` value."),
    ):
        RampOffWaveform(0.01, 0.2, off_time=0.1)
    with pytest.raises(
        TypeError,
        match=re.escape("Can not specify both `off_time` and a `ramp_end` value."),
    ):
        RampOffWaveform(ramp_end=0.2, off_time=0.1)
    with pytest.raises(
        TypeError,
        match=re.escape("RampOffWaveform() requires `ramp_end` to be specified."),
    ):
        RampOffWaveform()
    with pytest.raises(
        TypeError,
        match=re.escape("RampOffWaveform() requires `ramp_end` to be specified."),
    ):
        RampOffWaveform(ramp_start=0.0)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Must specify one or two positional arguments for the RampOffWaveform."
        ),
    ):
        RampOffWaveform(0.1, 0.2, 0.3)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "argument for RampOffWaveform() given by name ('ramp_start') and position (position 0)"
        ),
    ):
        RampOffWaveform(0.1, ramp_start=0.0)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "argument for RampOffWaveform() given by name ('ramp_start') and position (position 0)"
        ),
    ):
        RampOffWaveform(0.1, 0.2, ramp_start=0.0)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "argument for RampOffWaveform() given by name ('ramp_start') and position (position 0)"
        ),
    ):
        RampOffWaveform(0.1, 0.2, ramp_start=0.1, ramp_end=0.2)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "argument for RampOffWaveform() given by name ('ramp_end') and position (position 1)"
        ),
    ):
        RampOffWaveform(0.1, 0.2, ramp_end=0.0)


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
        vtem = VTEMWaveform(off_time=8e-3, peak_time=4e-3, ramp_on_rate=2.0)
        result = [vtem.eval(t) for t in self.times]
        expected = np.array(
            [0.0, 0.455054, 0.731059, 0.898464, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0]
        )
        assert_array_almost_equal(result, expected)

    def test_waveform_derivative(self):
        # Test the waveform derivative at points between the time_nodes
        wave = VTEMWaveform(off_time=8e-3, peak_time=4e-3, ramp_on_rate=2.0)

        def f(t):
            wave_eval = np.array([wave.eval(ti) for ti in t])
            dWave_dt = sp.diags(wave.eval_deriv(t))
            return wave_eval, dWave_dt

        t_nodes = wave.time_nodes
        t0 = np.concatenate(
            [
                np.linspace(t_nodes[i], t_nodes[i + 1], 6)[1:-2]
                for i in range(len(t_nodes) - 1)
            ]
        )
        dt = np.min(np.diff(t0)) * 0.5 * np.ones_like(t0)

        assert check_derivative(f, t0, dx=dt, plotIt=False, random_seed=643)


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

    def test_waveform_derivative(self):
        # Test the waveform derivative at points between the time_nodes
        wave = TrapezoidWaveform(ramp_on=np.r_[0.0, 2e-3], ramp_off=np.r_[6e-3, 10e-3])

        def f(t):
            wave_eval = np.array([wave.eval(ti) for ti in t])
            dWave_dt = sp.diags(wave.eval_deriv(t))
            return wave_eval, dWave_dt

        t_nodes = wave.time_nodes
        t0 = np.concatenate(
            [
                np.linspace(t_nodes[i], t_nodes[i + 1], 6)[1:-2]
                for i in range(len(t_nodes) - 1)
            ]
        )
        dt = np.min(np.diff(t0)) * 0.5 * np.ones_like(t0)

        assert check_derivative(f, t0, dx=dt, plotIt=False, random_seed=5277)


class TestTriangularWaveform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times = np.linspace(start=0, stop=1e-2, num=11)

    def test_waveform_with_symmetric_on_off(self):
        triangular = TriangularWaveform(start_time=0, peak_time=4e-3, off_time=8e-3)
        result = [triangular.eval(t) for t in self.times]
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0])
        assert_array_almost_equal(result, expected)

    def test_waveform_with_asymmetric_on_off(self):
        triangular = TriangularWaveform(start_time=0, peak_time=2e-3, off_time=6e-3)
        result = [triangular.eval(t) for t in self.times]
        expected = np.array([0.0, 0.5, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert_array_almost_equal(result, expected)

    def test_waveform_derivative(self):
        # Test the waveform derivative at points between the time_nodes
        wave = TriangularWaveform(start_time=0, peak_time=2e-3, off_time=6e-3)

        def f(t):
            wave_eval = np.array([wave.eval(ti) for ti in t])
            dWave_dt = sp.diags(wave.eval_deriv(t))
            return wave_eval, dWave_dt

        t_nodes = wave.time_nodes
        t0 = np.concatenate(
            [
                np.linspace(t_nodes[i], t_nodes[i + 1], 6)[1:-2]
                for i in range(len(t_nodes) - 1)
            ]
        )
        dt = np.min(np.diff(t0)) * 0.5 * np.ones_like(t0)

        assert check_derivative(f, t0, dx=dt, plotIt=False, random_seed=874)


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

    def test_waveform_with_plateau_derivative(self):
        # Test the waveform derivative at points between the time_nodes
        wave = QuarterSineRampOnWaveform(
            ramp_on=np.r_[0.0, 3e-3], ramp_off=np.r_[7e-3, 1e-2]
        )

        def f(t):
            wave_eval = np.array([wave.eval(ti) for ti in t])
            dWave_dt = sp.diags(wave.eval_deriv(t))
            return wave_eval, dWave_dt

        t_nodes = wave.time_nodes
        t0 = np.concatenate(
            [
                np.linspace(t_nodes[i], t_nodes[i + 1], 6)[1:-2]
                for i in range(len(t_nodes) - 1)
            ]
        )
        dt = np.min(np.diff(t0)) * 0.5 * np.ones_like(t0)

        assert check_derivative(f, t0, dx=dt, plotIt=False, random_seed=7564)

    def test_waveform_without_plateau_derivative(self):
        # Test the waveform derivative at points between the time_nodes
        wave = QuarterSineRampOnWaveform(
            ramp_on=np.r_[0.0, 5e-3], ramp_off=np.r_[5e-3, 1e-2]
        )

        def f(t):
            wave_eval = np.array([wave.eval(ti) for ti in t])
            dWave_dt = sp.diags(wave.eval_deriv(t))
            return wave_eval, dWave_dt

        t_nodes = wave.time_nodes
        t0 = np.concatenate(
            [
                np.linspace(t_nodes[i], t_nodes[i + 1], 6)[1:-2]
                for i in range(len(t_nodes) - 1)
            ]
        )
        dt = np.min(np.diff(t0)) * 0.5 * np.ones_like(t0)

        assert check_derivative(f, t0, dx=dt, plotIt=False, random_seed=12)

    def test_waveform_negative_plateau_derivative(self):
        # Test the waveform derivative at points between the time_nodes
        wave = QuarterSineRampOnWaveform(
            ramp_on=np.r_[0.0, 8e-3], ramp_off=np.r_[2e-3, 1e-2]
        )

        def f(t):
            wave_eval = np.array([wave.eval(ti) for ti in t])
            dWave_dt = sp.diags(wave.eval_deriv(t))
            return wave_eval, dWave_dt

        t_nodes = wave.time_nodes
        t0 = np.concatenate(
            [
                np.linspace(t_nodes[i], t_nodes[i + 1], 6)[1:-2]
                for i in range(len(t_nodes) - 1)
            ]
        )
        dt = np.min(np.diff(t0)) * 0.5 * np.ones_like(t0)

        assert check_derivative(f, t0, dx=dt, plotIt=False, random_seed=52)


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

    def test_waveform_with_plateau_derivative(self):
        # Test the waveform derivative at points between the time_nodes
        wave = HalfSineWaveform(ramp_on=np.r_[0.0, 3e-3], ramp_off=np.r_[7e-3, 1e-2])

        def f(t):
            wave_eval = np.array([wave.eval(ti) for ti in t])
            dWave_dt = sp.diags(wave.eval_deriv(t))
            return wave_eval, dWave_dt

        t_nodes = wave.time_nodes
        t0 = np.concatenate(
            [
                np.linspace(t_nodes[i], t_nodes[i + 1], 6)[1:-2]
                for i in range(len(t_nodes) - 1)
            ]
        )
        dt = np.min(np.diff(t0)) * 0.5 * np.ones_like(t0)

        assert check_derivative(f, t0, dx=dt, plotIt=False, random_seed=5)

    def test_waveform_without_plateau_derivative(self):
        # Test the waveform derivative at points between the time_nodes
        wave = HalfSineWaveform(ramp_on=np.r_[0.0, 5e-3], ramp_off=np.r_[5e-3, 1e-2])

        def f(t):
            wave_eval = np.array([wave.eval(ti) for ti in t])
            dWave_dt = sp.diags(wave.eval_deriv(t))
            return wave_eval, dWave_dt

        t_nodes = wave.time_nodes
        t0 = np.concatenate(
            [
                np.linspace(t_nodes[i], t_nodes[i + 1], 6)[1:-2]
                for i in range(len(t_nodes) - 1)
            ]
        )
        dt = np.min(np.diff(t0)) * 0.5 * np.ones_like(t0)

        assert check_derivative(f, t0, dx=dt, plotIt=False, random_seed=6)


class TestPiecewiseLinearWaveform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times = np.linspace(start=0, stop=1e-2, num=11)

    def test_waveform_with_default_param(self):
        wave = PiecewiseLinearWaveform(
            [0, 0.0025, 0.005, 0.0075, 0.01], [0, 0.4, 1.0, 0.6, 0.0]
        )
        result = [wave.eval(t) for t in self.times]
        expected = np.array(
            [0.0, 0.16, 0.32, 0.52, 0.76, 1.0, 0.84, 0.68, 0.48, 0.24, 0.0]
        )
        assert_array_almost_equal(result, expected)

    def test_waveform_derivative(self):
        # Test the waveform derivative at points between the time_nodes
        wave = PiecewiseLinearWaveform(
            [0, 0.0025, 0.005, 0.0075, 0.01], [0, 0.4, 1.0, 0.6, 0.0]
        )

        def f(t):
            wave_eval = np.array([wave.eval(ti) for ti in t])
            dWave_dt = sp.diags(wave.eval_deriv(t))
            return wave_eval, dWave_dt

        t_nodes = wave.time_nodes
        t0 = np.concatenate(
            [
                np.linspace(t_nodes[i], t_nodes[i + 1], 6)[1:-2]
                for i in range(len(t_nodes) - 1)
            ]
        )
        dt = np.min(np.diff(t0)) * 0.5 * np.ones_like(t0)

        assert check_derivative(f, t0, dx=dt, plotIt=False, random_seed=11)


class TestExponentialWaveform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times = np.linspace(start=-1e-2, stop=1e-2, num=21)

    def test_waveform_with_default_param(self):
        exponential_waveform = ExponentialWaveform()
        result = [exponential_waveform.eval(t) for t in self.times]
        expected = np.array(
            [
                0.0,
                0.632198578331253,
                0.8647714380371352,
                0.9503302116973793,
                0.9818055255388237,
                0.9933846464055102,
                0.9976443669192032,
                0.9992114305213273,
                0.9997879210035567,
                0.9999999999999991,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        assert_array_almost_equal(result, expected)

    def test_waveform_with_custom_param(self):
        exponential_waveform = ExponentialWaveform(
            start_time=-8e-3, off_time=8e-3, peak_time=5e-3
        )
        result = [exponential_waveform.eval(t) for t in self.times]
        expected = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.6321219876324751,
                0.8646666711948914,
                0.9502150794312231,
                0.981686580046311,
                0.9932642981054164,
                0.9975235025550406,
                0.9990903763078033,
                0.9996667969483558,
                0.999878850251482,
                0.9999568603021346,
                0.9999855585959744,
                0.9999961161082748,
                1.0,
                0.6666666666666666,
                0.33333333333333304,
                0.0,
                0.0,
                0.0,
            ]
        )
        assert_array_almost_equal(result, expected)

    def test_waveform_derivative(self):
        # Test the waveform derivative at points between the time_nodes
        exponential_waveform = ExponentialWaveform(
            start_time=-8e-3, off_time=8e-3, peak_time=5e-3
        )

        def f(t):
            wave_eval = np.array([exponential_waveform.eval(ti) for ti in t])
            dWave_dt = sp.diags(exponential_waveform.eval_deriv(t))
            return wave_eval, dWave_dt

        t_nodes = exponential_waveform.time_nodes
        t0 = np.concatenate(
            [
                np.linspace(t_nodes[i], t_nodes[i + 1], 6)[1:-2]
                for i in range(len(t_nodes) - 1)
            ]
        )
        dt = np.min(np.diff(t0)) * 0.5 * np.ones_like(t0)

        assert check_derivative(f, t0, dx=dt, plotIt=False, random_seed=5555)


def test_simple_source():
    waveform = StepOffWaveform()
    assert waveform.eval(0.0) == 1.0
