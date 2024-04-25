import unittest
import numpy as np
from importlib import import_module
from discretize import TensorMesh


mesh = TensorMesh([2, 2, 2])
locs = np.array([[1.0, 2.0, 3.0]])

deprecated_modules = [
    # "simpeg.utils.codeutils",
    # "simpeg.utils.coordutils",
    # "simpeg.utils.CounterUtils",
    # "simpeg.utils.curvutils",
    # "simpeg.utils.matutils",
    # "simpeg.utils.meshutils",
    # "simpeg.utils.ModelBuilder",
    # "simpeg.utils.PlotUtils",
    # "simpeg.utils.SolverUtils",
    # "simpeg.electromagnetics.utils.EMUtils",
    # "simpeg.electromagnetics.utils.AnalyticUtils",
    # "simpeg.electromagnetics.utils.CurrentUtils",
    # "simpeg.electromagnetics.utils.testingUtils",
    # "simpeg.electromagnetics.static.utils.StaticUtils",
    # "simpeg.electromagnetics.natural_source.utils.dataUtils",
    # "simpeg.electromagnetics.natural_source.utils.ediFilesUtils",
    # "simpeg.electromagnetics.natural_source.utils.MT1Danalytic",
    # "simpeg.electromagnetics.natural_source.utils.MT1Dsolutions",
    # "simpeg.electromagnetics.natural_source.utils.plotDataTypes",
    # "simpeg.electromagnetics.natural_source.utils.plotUtils",
    # "simpeg.electromagnetics.natural_source.utils.sourceUtils",
    # "simpeg.electromagnetics.natural_source.utils.testUtils",
]

deprecated_problems = [
    # [
    #     "simpeg.electromagnetics.frequency_domain",
    #     ("Problem3D_e", "Problem3D_b", "Problem3D_h", "Problem3D_j"),
    # ],
    # [
    #     "simpeg.electromagnetics.time_domain",
    #     ("Problem3D_e", "Problem3D_b", "Problem3D_h", "Problem3D_j"),
    # ],
    # [
    #     "simpeg.electromagnetics.natural_source",
    #     ("Problem3D_ePrimSec", "Problem1D_ePrimSec"),
    # ],
    # [
    #     "simpeg.electromagnetics.static.induced_polarization",
    #     ("Problem3D_CC", "Problem3D_N", "Problem2D_CC", "Problem2D_N"),
    # ],
    # [
    #     "simpeg.electromagnetics.static.resistivity",
    #     ("Problem3D_CC", "Problem3D_N", "Problem2D_CC", "Problem2D_N"),
    # ],
    # [
    #     "simpeg.electromagnetics.static.spectral_induced_polarization",
    #     ("Problem3D_CC", "Problem3D_N", "Problem2D_CC", "Problem2D_N"),
    # ],
    # [
    #     "simpeg.electromagnetics.viscous_remanent_magnetization",
    #     ("Problem_Linear", "Problem_LogUnifrom"),
    # ],
]

deprecated_fields = [
    # [
    #     "simpeg.electromagnetics.frequency_domain",
    #     ("Fields3D_e", "Fields3D_b", "Fields3D_h", "Fields3D_j"),
    # ],
    # [
    #     "simpeg.electromagnetics.time_domain",
    #     ("Fields3D_e", "Fields3D_b", "Fields3D_h", "Fields3D_j"),
    # ],
    # [
    #     "simpeg.electromagnetics.natural_source",
    #     ("Fields1D_ePrimSec", "Fields3D_ePrimSec"),
    # ],
    # [
    #     "simpeg.electromagnetics.static.resistivity",
    #     ("Fields_CC", "Fields_N", "Fields_ky", "Fields_ky_CC", "Fields_ky_N"),
    # ],
]

deprecated_receivers = [
    # [
    #     "simpeg.electromagnetics.frequency_domain.receivers",
    #     ("Point_e", "Point_b", "Point_bSecondary", "Point_h", "Point_j"),
    # ],
    # [
    #     "simpeg.electromagnetics.time_domain.receivers",
    #     ("Point_e", "Point_b", "Point_h", "Point_j", "Point_dbdt", "Point_dhdt"),
    # ],
    # [
    #     "simpeg.electromagnetics.natural_source.receivers",
    #     ("Point_impedance1D", "Point_impedance3D", "Point_tipper3D"),
    # ],
    # ["simpeg.electromagnetics.static.resistivity.receivers", ("Dipole_ky", "Pole_ky")],
]

deprcated_surveys = [
    # "simpeg.electromagnetics.static.resistivity", ("Survey")
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
