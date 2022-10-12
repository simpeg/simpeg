import numpy as np
import pytest

from SimPEG.electromagnetics import frequency_domain as fdem
from SimPEG.electromagnetics import time_domain as tdem
from SimPEG.electromagnetics.utils.testing_utils import crossCheckTest

testEB = True
testHJ = True
testEJ = True
testBH = True
verbose = False

TOLEBHJ = 1e-5
TOLEJHB = 1  # averaging and more sensitive to boundary condition violations (ie. the impact of violating the boundary conditions in each case is different.)
# TODO: choose better testing parameters to lower this

SrcList_EB = ["RawVec", "MagDipole_Bfield", "MagDipole", "CircularLoop", "LineCurrent"]
SrcList_HJ = ["RawVec", "MagDipole_Bfield", "MagDipole", "CircularLoop"]


def test_src():
    src = fdem.Src.MagDipole([], location=np.array([[1.5, 3.0, 5.0]]), frequency=10)
    assert np.all(src.location == np.r_[1.5, 3.0, 5.0])
    assert src.location.shape == (3,)

    with pytest.raises(ValueError):
        src = fdem.Src.MagDipole(
            [], location=np.array([[0.0, 0.0, 0.0, 1.0]]), frequency=10
        )

    with pytest.raises(ValueError):
        src = fdem.Src.MagDipole([], location=np.r_[0.0, 0.0, 0.0, 1.0], frequency=10)

    src = tdem.Src.MagDipole(
        [],
        location=np.array([[1.5, 3.0, 5.0]]),
    )
    assert np.all(src.location == np.r_[1.5, 3.0, 5.0])

    with pytest.raises(ValueError):
        src = tdem.Src.MagDipole(
            [],
            location=np.array([[0.0, 0.0, 0.0, 1.0]]),
        )

    with pytest.raises(ValueError):
        src = tdem.Src.MagDipole(
            [],
            location=np.r_[0.0, 0.0, 0.0, 1.0],
        )


if testEB:

    @pytest.mark.parametrize("rx_type", ["ElectricField", "MagneticFluxDensity"])
    @pytest.mark.parametrize("orientation", ["x", "y", "z"])
    @pytest.mark.parametrize("component", ["i", "r"])
    def test_cross_check_EB(rx_type, orientation, component, verbose=verbose):
        assert crossCheckTest(
            SrcList_EB, "e", "b", (rx_type, orientation, component), verbose=verbose
        )


if testHJ:

    @pytest.mark.parametrize("rx_type", ["CurrentDensity", "MagneticField"])
    @pytest.mark.parametrize("orientation", ["x", "y", "z"])
    @pytest.mark.parametrize("component", ["i", "r"])
    def test_cross_check_HJ(rx_type, orientation, component, verbose=verbose):
        assert crossCheckTest(
            SrcList_HJ, "j", "h", (rx_type, orientation, component), verbose=verbose
        )
