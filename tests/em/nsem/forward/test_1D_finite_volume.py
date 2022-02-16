import numpy as np
from discretize import TensorMesh
from SimPEG.electromagnetics import natural_source as nsem
from SimPEG import maps
import matplotlib.pyplot as plt
from pymatsolver import Pardiso
import unittest


TOL_SIGMA = 0.05  # 5%
TOL_PHASE = 3  # 3 degrees


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

        # simulation
        src_list = [nsem.sources.Planewave1D([], frequency=f) for f in self.frequencies]
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
        f = sim.fields(sigma)

        sig_a = f[:, "apparent conductivity"][-self.npad - 1, :]
        phase = f[:, "phase"][-self.npad - 1, :]

        sig_a_error = np.abs(sig_a - s) / s

        if formulation == "e":
            p = 45  # Zyx = ey / hx
        elif formulation == "h":
            p = -135  # Zxy = ex / hy

        phase_error = np.abs(phase - p)

        passed_sigma = np.all(sig_a_error < TOL_SIGMA)
        passed_phase = np.all(phase_error < TOL_PHASE)

        print(f"Testing {s:1.1e} {formulation}")
        print(
            f"   App Con. max:{sig_a_error.max():1.2e}, mean:{sig_a_error.mean():1.2e}, tol:{TOL_SIGMA}, passed?: {passed_sigma}"
        )
        print(
            f"   Phase    max:{phase_error.max():1.2e}, mean:{phase_error.mean():1.2e}, tol:{TOL_PHASE}, passed?: {passed_phase}"
        )

        self.assertTrue(passed_sigma)
        self.assertTrue(passed_phase)

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
