import unittest
from simpeg.electromagnetics import natural_source as nsem
from simpeg import discretize
import numpy as np


TOL = 1e-6


def appResNorm(conductivityHalf):
    num_frequencies = 26

    m1d = discretize.TensorMesh([[(100, 5, 1.5), (100.0, 10), (100, 5, 1.5)]], x0=["C"])
    conductivity = np.zeros(m1d.nC) + conductivityHalf
    conductivity[m1d.gridCC[:] > 200] = 1e-8

    # Calculate the analytic fields
    freqs = np.logspace(4, -4, num_frequencies)
    Z = []

    for freq in freqs:
        Ed, Eu, Hd, Hu = nsem.utils.getEHfields(
            m1d, conductivity, freq, np.array([200])
        )
        Z.append((Ed + Eu) / (Hd + Hu))

    Zarr = np.concatenate(Z)

    app_r, app_p = nsem.utils.appResPhs(freqs, Zarr)

    return np.linalg.norm(
        np.abs(app_r - np.ones(num_frequencies) / conductivityHalf)
    ) / np.log10(conductivityHalf)


class TestAnalytics(unittest.TestCase):
    def setUp(self):
        pass

    def test_appRes2en1(self):
        self.assertLess(appResNorm(2e-1), TOL)

    def test_appRes2en2(self):
        self.assertLess(appResNorm(2e-2), TOL)

    def test_appRes2en3(self):
        self.assertLess(appResNorm(2e-3), TOL)

    def test_appRes2en4(self):
        self.assertLess(appResNorm(2e-4), TOL)

    def test_appRes2en5(self):
        self.assertLess(appResNorm(2e-5), TOL)

    def test_appRes2en6(self):
        self.assertLess(appResNorm(2e-6), TOL)


if __name__ == "__main__":
    unittest.main()
