import numpy as np
import pytest

from SimPEG.electromagnetics import frequency_domain as fdem
from SimPEG.electromagnetics import time_domain as tdem


def test_receiver_properties():

    xyz = np.c_[0.0, 0.0, 0.0]
    times = np.logspace(-5, -2, 4)
    projComp = "Fx"
    rx = tdem.receivers.BaseRx(xyz, times, projComp=projComp)

    assert rx.projComp == projComp


def test_source_properties():

    xyz = np.r_[0.0, 0.0, 0.0]

    # Base source
    src = tdem.sources.BaseTDEMSrc([], location=xyz, srcType="inductive")
    assert src.srcType == "inductive"

    # loop galvinic vs inductive
    loop_points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]

    src = tdem.sources.LineCurrent([], location=loop_points)
    assert src.srcType == "inductive"

    src = tdem.sources.LineCurrent([], location=loop_points[:-1])
    assert src.srcType == "galvanic"

    with pytest.raises(ValueError):
        tdem.sources.LineCurrent([], location=loop_points, current=0)

    print("Test source property raises passes")
