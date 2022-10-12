import numpy as np
import pytest

from SimPEG.electromagnetics import frequency_domain as fdem
from SimPEG.electromagnetics import time_domain as tdem


def test_receiver_properties_validation():

    xyz = np.c_[0.0, 0.0, 0.0]
    projComp = "Fx"
    rx = fdem.receivers.BaseRx(xyz, projComp=projComp)

    assert rx.projComp == projComp

    with pytest.raises(ValueError):
        fdem.receivers.BaseRx(xyz, component="potato")

    with pytest.raises(TypeError):
        fdem.receivers.BaseRx(xyz, component=2.0)


def test_source_properties_validation():

    xyz = np.r_[0.0, 0.0, 0.0]
    frequency = 1.0

    # Base source
    src = fdem.sources.BaseFDEMSrc([], location=xyz, frequency=frequency)
    assert src.frequency == frequency

    with pytest.raises(TypeError):
        fdem.sources.BaseFDEMSrc([], location=xyz, freq=frequency)

    with pytest.raises(TypeError):
        fdem.sources.BaseFDEMSrc([], frequency=None, location=xyz)

    # MagDipole
    with pytest.raises(TypeError):
        fdem.sources.MagDipole([], frequency, location="not_a_vector")
    with pytest.raises(ValueError):
        fdem.sources.MagDipole([], frequency, location=[0.0, 0.0, 0.0, 0.0])
    with pytest.raises(TypeError):
        fdem.sources.MagDipole([], frequency, xyz, orientation=["list", "of", "string"])
    with pytest.raises(ValueError):
        fdem.sources.MagDipole([], frequency, xyz, orientation=[1, 0, 0, 0])

    # CircularLoop
    with pytest.raises(ValueError):
        fdem.sources.CircularLoop([], frequency, location=[0.0, 0.0, 0.0], current=0.0)

    # LineCurrent
    with pytest.raises(TypeError):
        fdem.sources.LineCurrent([], frequency, location=["a", "b", "c"])
    with pytest.raises(ValueError):
        fdem.sources.LineCurrent([], frequency, location=np.random.rand(5, 3, 2))
    with pytest.raises(ValueError):
        fdem.sources.LineCurrent(
            [], frequency, location=np.random.rand(5, 3), current=0.0
        )


def test_bad_source_type():
    src = tdem.sources.MagDipole([], np.r_[0.0, 0.0, 1.0])

    with pytest.raises(TypeError):
        fdem.survey.Survey(src)
