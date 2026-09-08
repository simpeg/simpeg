"""
Test bugfix in EM1D simulation that ignored IP effect if at least one eta value is null.
"""

import pytest
import numpy as np
import simpeg
import simpeg.electromagnetics.time_domain as tdem
from simpeg.electromagnetics.base_1d import BaseEM1DSimulation


def build_loop(center, radius, size=200):
    """
    Build a horizontal circular loop around the center.

    Parameters
    ----------
    center : tuple of float
        The x, y, z location of the center of the loop.
    radius : float
        Loop's radius.
    size : int, optional
        Number of points that will form the loop.

    Returns
    -------
    locations : (size, 3) array
        Array with the x, y, z coordinates of the loop points.
    """
    xc, yc, zc = center
    theta = np.linspace(0, 2 * np.pi, size, endpoint=False)
    x = xc + radius * np.cos(theta)
    y = yc + radius * np.sin(theta)
    z = zc + np.zeros_like(theta)
    return np.vstack((x, y, z)).T


@pytest.fixture
def survey():
    # Define receiver
    receiver_location = (0, 0, 30)
    time_channels = 10 ** np.linspace(-2, -5, 31)  # [s]
    receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
        receiver_location, time_channels, orientation="z"
    )

    # Define source.
    # Define source as a circular loop around the receiver. Use a piecewise linear
    # waveform.
    area = 300  # [m2]
    radius = np.sqrt(area / np.pi)
    peak_current = 100  # [A]
    n_turns = 16
    waveform_times = [-10e-3, -9e-3, 0, 5e-6]  # [s]
    waveform_currents = [0, 1, 1, 0]

    waveform = tdem.sources.PiecewiseLinearWaveform(
        times=waveform_times, currents=waveform_currents
    )
    source = tdem.sources.LineCurrent(
        receiver_list=[receiver],
        location=build_loop(receiver_location, radius),
        waveform=waveform,
        current=peak_current * n_turns,
    )

    # Define survey
    survey = tdem.Survey([source])
    return survey


class TestComplexSigma:
    """
    Test the :meth:`BaseEM1DSimulation.compute_complex_sigma` method.

    Check if passing any null chargeability value in the ``eta`` array generates
    sigmas with non-null imaginary parts. Without the bugfix, a single null value in
    ``eta`` would ignore the IP effect, generating real sigmas.
    """

    thicknesses = np.array([20.0, 50.0])
    c = np.array([0, 0.5, 0])
    tau = np.array([0, 1.0e-3, 0])
    resistivities = np.array([5000.0, 500.0, 5000.0])  # [Ohm.m]
    frequencies = np.logspace(-1, -5, 11)

    def compute_sigmas(self, eta, survey):
        """
        Use the :class:`BaseEM1DSimulation` to compute sigmas for a given ``eta``.
        """
        sim = BaseEM1DSimulation(
            thicknesses=self.thicknesses,
            survey=survey,
            rho=self.resistivities,
            eta=eta,
            tau=self.tau,
            c=self.c,
        )
        return sim.compute_complex_sigma(self.frequencies)

    def test_sigmas_no_ip(self, survey):
        eta_zeros = np.array([0, 0, 0])
        eta_almost_zeros = np.array([1e-15, 1e-15, 1e-15])
        np.testing.assert_allclose(
            self.compute_sigmas(eta_zeros, survey),
            self.compute_sigmas(eta_almost_zeros, survey),
        )

    def test_sigmas_ip(self, survey):
        eta_zeros = np.array([0, 0, 0])
        eta_non_zeros = np.array([3e-3, 2e-2, 1e-1])
        assert not np.allclose(
            self.compute_sigmas(eta_zeros, survey),
            self.compute_sigmas(eta_non_zeros, survey),
        )

    def test_sigmas_ip_one_null(self, survey):
        eta = np.array([3e-3, 2e-2, 1e-10])
        eta_one_null = np.array([3e-3, 2e-2, 0])
        np.testing.assert_allclose(
            self.compute_sigmas(eta, survey),
            self.compute_sigmas(eta_one_null, survey),
        )


class TestIPEffect:
    """
    Test if the IP effect is present in the dB/dt predicted data.
    """

    thicknesses = np.array([20.0, 50.0])  # [m]
    model_map = simpeg.maps.IdentityMap(nP=len(thicknesses) + 1)

    resistivities = np.array([5000.0, 500.0, 5000.0])  # [Ohm.m]
    tau = np.array([1e-10, 1.0e-3, 1e-10])  # [s]
    c = np.array([1e-10, 0.5, 1e-10])  # -

    def get_simulation(self, eta, survey):
        simulation = tdem.Simulation1DLayered(
            survey=survey,
            thicknesses=self.thicknesses,
            rhoMap=self.model_map,
            eta=eta,
            tau=self.tau,
            c=self.c,
        )
        return simulation

    def dpred(self, eta, survey):
        return self.get_simulation(eta, survey).dpred(self.resistivities)

    def test_no_ip_effect(self, survey):
        eta_zeros = np.array([0, 0, 0])
        eta_almost_zeros = np.array([1e-15, 1e-15, 1e-15])

        dpred_zeros = self.dpred(eta_zeros, survey)
        dpred_almost_zeros = self.dpred(eta_almost_zeros, survey)
        np.testing.assert_allclose(dpred_zeros, dpred_almost_zeros)

    def test_ip_effect_vs_no_ip(self, survey):
        eta_zeros = np.array([0, 0, 0])
        eta_non_zeros = np.array([3e-3, 2e-2, 1e-1])

        dpred_zeros = self.dpred(eta_zeros, survey)
        dpred_non_zeros = self.dpred(eta_non_zeros, survey)
        assert not np.allclose(dpred_zeros, dpred_non_zeros)

    def test_ip_effect_one_null(self, survey):
        """
        Test if having one value of eta equal to zero still predicts IP effect.
        """
        eta = np.array([3e-3, 2e-2, 1e-10])
        eta_one_null = np.array([3e-3, 2e-2, 0])

        dpred = self.dpred(eta, survey)
        dpred_one_null = self.dpred(eta_one_null, survey)
        np.testing.assert_allclose(dpred, dpred_one_null)
