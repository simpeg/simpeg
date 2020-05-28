import scooby
import unittest
from SimPEG import Report


class TestReport(unittest.TestCase):
    def test_version_defaults(self):

        # Reporting is now done by the external package scooby.
        # We just ensure the shown packages do not change (core and optional).
        out1 = Report()
        out2 = scooby.Report(
            core=[
                "SimPEG",
                "discretize",
                "pymatsolver",
                "vectormath",
                "properties",
                "numpy",
                "scipy",
                "cython",
            ],
            optional=["IPython", "matplotlib", "ipywidgets"],
            ncol=4,
        )

        # Ensure they're the same; exclude initial time to avoid errors due
        # to second-change.
        assert out1.__repr__()[115:] == out2.__repr__()[115:]


if __name__ == "__main__":
    unittest.main()
