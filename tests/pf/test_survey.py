import functools
import numpy as np
import pytest
from simpeg.potential_fields import gravity as grav
from simpeg.potential_fields import magnetics as mag
from simpeg.data import Data


@pytest.fixture(params=["gravity", "magnetics"])
def survey(request):
    rx_locs = np.random.rand(20, 3)
    if request.param == "gravity":
        rx1_components = ["gx", "gz"]
        rx2_components = "gzz"
        mod = grav
        Source = functools.partial(grav.SourceField)
    elif request.param == "magnetics":
        rx1_components = ["bx", "by"]
        rx2_components = "tmi"

        mod = mag
        Source = functools.partial(
            mag.UniformBackgroundField, amplitude=50_000, inclination=90, declination=0
        )

    rx1 = mod.Point(rx_locs, components=rx1_components)
    rx2 = mod.Point(rx_locs, components=rx2_components)
    src = Source(receiver_list=[rx1, rx2])
    return mod.Survey(src)


def test_survey_counts(survey):
    src = survey.source_field
    rx1, rx2 = src.receiver_list

    assert rx1.nD == 40
    assert rx2.nD == 20
    assert src.nD == 60
    assert survey.nRx == 40
    np.testing.assert_equal(src.vnD, [40, 20])
    assert survey.nD == 60
    np.testing.assert_equal(survey.vnD, [40, 20])


def test_survey_indexing(survey):
    src = survey.source_field
    rx1, rx2 = src.receiver_list
    d1 = -10 * np.arange(rx1.nD)
    d2 = 10 + np.arange(rx2.nD)
    data_vec = np.r_[d1, d2]

    data = Data(survey=survey, dobs=data_vec)

    np.testing.assert_equal(data[src, rx1], d1)
    np.testing.assert_equal(data[src, rx2], d2)
