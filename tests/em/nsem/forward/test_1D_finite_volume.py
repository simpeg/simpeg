import numpy as np
from discretize import TensorMesh
from SimPEG.electromagnetics import natural_source as nsem
from SimPEG import maps
import matplotlib.pyplot as plt
from pymatsolver import Pardiso
import unittest


TOL_SIGMA = 0.2  # 20% (These are very loose tests)
TOL_PHASE = 5  # 3 degrees


class FiniteVolume1DTest(unittest.TestCase):
    def setUp(self):

        # mesh
        csz = 100
        nc = 300
        npad = 30
        pf = 1.2
        mesh = TensorMesh([[(csz, npad, -pf), (csz, nc), (csz, npad)]], "N")
        mesh.x0 = np.r_[-mesh.hx[:-npad].sum()]

        self.npad = npad
        self.mesh = mesh

        # survey
        self.frequencies = np.logspace(-2, 1, 30)

        rx_list = [
            nsem.receivers.PointNaturalSource(
                [[0]], orientation="xy", component="apparent_resistivity"
            ),
            nsem.receivers.PointNaturalSource(
                [[0]], orientation="xy", component="phase"
            ),
            nsem.receivers.PointNaturalSource(
                [[0]], orientation="yx", component="apparent_resistivity"
            ),
            nsem.receivers.PointNaturalSource(
                [[0]], orientation="yx", component="phase"
            ),
        ]
        # simulation
        src_list = [
            nsem.sources.Planewave1D(rx_list, frequency=f) for f in self.frequencies
        ]
        self.survey = nsem.survey.Survey1D(src_list)

    def get_simulation(self, formulation="e"):

        if formulation == "e":
            return nsem.simulation.Simulation1DElectricField(
                mesh=self.mesh,
                solver=Pardiso,
                survey=self.survey,
                sigmaMap=maps.IdentityMap(),
            )
        elif formulation == "h":
            return nsem.simulation.Simulation1DMagneticField(
                mesh=self.mesh,
                solver=Pardiso,
                survey=self.survey,
                sigmaMap=maps.IdentityMap(),
            )

    def get_sigma(self, sigma_back):
        # conductivity model
        sigma_air = 1e-8
        sigma = sigma_back * np.ones(self.mesh.nC)
        sigma[self.mesh.gridCC >= 0] = sigma_air

        return sigma

    def apparent_resistivity_phase_test(self, s, formulation="e"):
        sigma = self.get_sigma(s)
        sim = self.get_simulation(formulation)
        d = sim.dpred(sigma)

        rho_a_xy = d[0::4]
        phase_xy = d[1::4]
        rho_a_yx = d[2::4]
        phase_yx = d[3::4]

        np.testing.assert_allclose(rho_a_xy, 1.0 / s, rtol=TOL_SIGMA)
        np.testing.assert_allclose(rho_a_yx, 1.0 / s, rtol=TOL_SIGMA)

        np.testing.assert_allclose(phase_yx, 45, atol=TOL_PHASE)
        np.testing.assert_allclose(phase_xy, -135, atol=TOL_PHASE)

    def test_1en1_e(self):
        self.apparent_resistivity_phase_test(1e-1, "e")

    def test_3en1_e(self):
        self.apparent_resistivity_phase_test(3e-1, "e")

    def test_1en1_b(self):
        self.apparent_resistivity_phase_test(1e-1, "h")

    def test_3en1_b(self):
        self.apparent_resistivity_phase_test(3e-1, "h")


if __name__ == "__main__":
    unittest.main()
