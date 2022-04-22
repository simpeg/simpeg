import unittest
import numpy as np
from importlib import import_module
from discretize import TensorMesh
from SimPEG import utils
from SimPEG import electromagnetics as EM
from SimPEG.electromagnetics.static import resistivity as DC
import discretize
from pymatsolver import Pardiso as Solver


mesh = TensorMesh([2, 2, 2])
locs = np.array([[1.0, 2.0, 3.0]])

deprecated_modules = [
    # "SimPEG.utils.codeutils",
    # "SimPEG.utils.coordutils",
    # "SimPEG.utils.CounterUtils",
    # "SimPEG.utils.curvutils",
    # "SimPEG.utils.matutils",
    # "SimPEG.utils.meshutils",
    # "SimPEG.utils.ModelBuilder",
    # "SimPEG.utils.PlotUtils",
    # "SimPEG.utils.SolverUtils",
    # "SimPEG.electromagnetics.utils.EMUtils",
    # "SimPEG.electromagnetics.utils.AnalyticUtils",
    # "SimPEG.electromagnetics.utils.CurrentUtils",
    # "SimPEG.electromagnetics.utils.testingUtils",
    # "SimPEG.electromagnetics.static.utils.StaticUtils",
    # "SimPEG.electromagnetics.natural_source.utils.dataUtils",
    # "SimPEG.electromagnetics.natural_source.utils.ediFilesUtils",
    # "SimPEG.electromagnetics.natural_source.utils.MT1Danalytic",
    # "SimPEG.electromagnetics.natural_source.utils.MT1Dsolutions",
    # "SimPEG.electromagnetics.natural_source.utils.plotDataTypes",
    # "SimPEG.electromagnetics.natural_source.utils.plotUtils",
    # "SimPEG.electromagnetics.natural_source.utils.sourceUtils",
    # "SimPEG.electromagnetics.natural_source.utils.testUtils",
]

deprecated_problems = [
    # [
    #     "SimPEG.electromagnetics.frequency_domain",
    #     ("Problem3D_e", "Problem3D_b", "Problem3D_h", "Problem3D_j"),
    # ],
    # [
    #     "SimPEG.electromagnetics.time_domain",
    #     ("Problem3D_e", "Problem3D_b", "Problem3D_h", "Problem3D_j"),
    # ],
    # [
    #     "SimPEG.electromagnetics.natural_source",
    #     ("Problem3D_ePrimSec", "Problem1D_ePrimSec"),
    # ],
    # [
    #     "SimPEG.electromagnetics.static.induced_polarization",
    #     ("Problem3D_CC", "Problem3D_N", "Problem2D_CC", "Problem2D_N"),
    # ],
    # [
    #     "SimPEG.electromagnetics.static.resistivity",
    #     ("Problem3D_CC", "Problem3D_N", "Problem2D_CC", "Problem2D_N"),
    # ],
    # [
    #     "SimPEG.electromagnetics.static.spectral_induced_polarization",
    #     ("Problem3D_CC", "Problem3D_N", "Problem2D_CC", "Problem2D_N"),
    # ],
    # [
    #     "SimPEG.electromagnetics.viscous_remanent_magnetization",
    #     ("Problem_Linear", "Problem_LogUnifrom"),
    # ],
]

deprecated_fields = [
    # [
    #     "SimPEG.electromagnetics.frequency_domain",
    #     ("Fields3D_e", "Fields3D_b", "Fields3D_h", "Fields3D_j"),
    # ],
    # [
    #     "SimPEG.electromagnetics.time_domain",
    #     ("Fields3D_e", "Fields3D_b", "Fields3D_h", "Fields3D_j"),
    # ],
    # [
    #     "SimPEG.electromagnetics.natural_source",
    #     ("Fields1D_ePrimSec", "Fields3D_ePrimSec"),
    # ],
    # [
    #     "SimPEG.electromagnetics.static.resistivity",
    #     ("Fields_CC", "Fields_N", "Fields_ky", "Fields_ky_CC", "Fields_ky_N"),
    # ],
]

deprecated_receivers = [
    # [
    #     "SimPEG.electromagnetics.frequency_domain.receivers",
    #     ("Point_e", "Point_b", "Point_bSecondary", "Point_h", "Point_j"),
    # ],
    # [
    #     "SimPEG.electromagnetics.time_domain.receivers",
    #     ("Point_e", "Point_b", "Point_h", "Point_j", "Point_dbdt", "Point_dhdt"),
    # ],
    # [
    #     "SimPEG.electromagnetics.natural_source.receivers",
    #     ("Point_impedance1D", "Point_impedance3D", "Point_tipper3D"),
    # ],
    # ["SimPEG.electromagnetics.static.resistivity.receivers", ("Dipole_ky", "Pole_ky")],
]

deprcated_surveys = [
    # "SimPEG.electromagnetics.static.resistivity", ("Survey")
]


class DeprecateTest(unittest.TestCase):
    def test_module_deprecations(self):
        for module in deprecated_modules:
            print(module, end="...")
            with self.assertRaises(NotImplementedError):
                import_module(module)
            print("ok")

    def test_problem_deprecations(self):
        for module in deprecated_problems:
            mod = import_module(module[0])
            for Problem in module[1]:
                Prob = getattr(mod, Problem)
                print(f"{module[0]}.{Problem}...", end="")
                with self.assertRaises(NotImplementedError):
                    Prob(mesh=mesh)
                print("ok")

    def test_field_deprecations(self):
        for module in deprecated_fields:
            mod = import_module(module[0])
            for Field in module[1]:
                field = getattr(mod, Field)
                # Only testing for a deprecation warning so removing startup of Fields
                field.startup = lambda self: None
                print(f"{module[0]}.{Field}...", end="")
                with self.assertRaises(NotImplementedError):
                    field(mesh)
                print("ok")

    def test_receiver_deprecations(self):
        for module in deprecated_receivers:
            mod = import_module(module[0])
            for receiver in module[1]:
                Rx = getattr(mod, receiver)
                print(f"{module[0]}.{Rx}...", end="")
                with self.assertRaises(NotImplementedError):
                    try:
                        Rx(locs)  # for "Pole like" receiver
                    except (TypeError, ValueError):
                        Rx(locs, locs)  # for either Dipole, or Time receivers
                print("ok")


if __name__ == "__main__":
    unittest.main()
