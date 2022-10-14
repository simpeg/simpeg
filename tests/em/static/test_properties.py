import numpy as np
import pytest

from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.electromagnetics.static import spectral_induced_polarization as sip


def test_receiver_properties():

    xyz_1 = np.c_[0.0, 0.0, 0.0]
    xyz_2 = np.c_[10.0, 0.0, 0.0]
    times = np.logspace(-4, -2, 3)

    # Base DC receiver
    rx = dc.receivers.BaseRx(xyz_1)
    assert rx.orientation is None

    rx = dc.receivers.BaseRx(xyz_1, orientation="x")
    assert rx.orientation == "x"

    #####
    # DC Dipole receiver
    ####
    with pytest.raises(AttributeError):
        dc.receivers.Dipole(locations=None)
    # too many locations
    with pytest.raises(ValueError):
        dc.receivers.Dipole(locations=[xyz_1, xyz_2, xyz_1])
    # non-matching shapes of locations
    with pytest.raises(ValueError):
        dc.receivers.Dipole(locations=[xyz_1, np.r_[xyz_2, xyz_1]])

    #####
    # SIP Dipole receiver
    ####
    with pytest.raises(AttributeError):
        sip.receivers.Dipole(times=times, locations=None)
    # too many locations
    with pytest.raises(ValueError):
        sip.receivers.Dipole(times=times, locations=[xyz_1, xyz_2, xyz_1])
    # non-matching shapes of locations
    with pytest.raises(ValueError):
        sip.receivers.Dipole(times=times, locations=[xyz_1, np.r_[xyz_2, xyz_1]])


def test_SIP_zero_current():

    xyz_1 = np.c_[0.0, 0.0, 0.0]
    xyz_2 = np.c_[10.0, 0.0, 0.0]
    times = np.logspace(-4, -2, 3)

    # Base SIP source
    rx = sip.receivers.Dipole(locations=[xyz_1, xyz_2], times=times)
    with pytest.raises(ValueError):
        sip.sources.BaseSrc(rx, location=xyz_1, current=0.0)

    print("Test source property raises passes")
