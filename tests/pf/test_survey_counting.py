import numpy as np
from SimPEG.potential_fields import gravity as grav
from SimPEG.potential_fields import magnetics as mag


def test_gravity_survey():
    rx_locs = np.random.rand(20, 3)
    rx_components = ["gx", "gz"]

    rx1 = grav.Point(rx_locs, components=rx_components)
    rx2 = grav.Point(rx_locs, components="gzz")
    src = grav.SourceField([rx1, rx2])
    survey = grav.Survey(src)

    assert rx1.nD == 40
    assert rx2.nD == 20
    assert src.nD == 60
    assert survey.nRx == 40
    np.testing.assert_equal(src.vnD, [40, 20])
    assert survey.nD == 60
    np.testing.assert_equal(survey.vnD, [40, 20])


def test_magnetics_survey():
    rx_locs = np.random.rand(20, 3)
    rx_components = ["bx", "by", "bz"]

    rx1 = mag.Point(rx_locs, components=rx_components)
    rx2 = mag.Point(rx_locs, components="tmi")
    src = mag.UniformBackgroundField([rx1, rx2])
    survey = mag.Survey(src)

    assert rx1.nD == 60
    assert rx2.nD == 20
    assert src.nD == 80
    np.testing.assert_equal(src.vnD, [60, 20])
    assert survey.nRx == 40
    assert survey.nD == 80
    np.testing.assert_equal(survey.vnD, [60, 20])
