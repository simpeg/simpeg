# import matplotlib
# matplotlib.use('Agg')

from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics.static import utils
import matplotlib.pyplot as plt
import numpy as np
import unittest


class TestsIO(unittest.TestCase):
    def setUp(self):
        self.plotIt = False
        np.random.seed(1)
        # Initiate I/O class for DC
        self.IO = dc.IO()
        # Obtain ABMN locations

        xmin, xmax = 0.0, 200.0
        ymin, ymax = 0.0, 0.0
        zmin, zmax = 0, 0
        self.endl = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
        # Generate DC survey object

    def test_flat_dpdp(self):
        self.survey = utils.gen_DCIPsurvey(
            self.endl, "dipole-dipole", dim=2, a=10, b=10, n=10
        )
        self.survey = self.IO.from_ambn_locations_to_survey(
            self.survey.locations_a,
            self.survey.locations_b,
            self.survey.locations_m,
            self.survey.locations_n,
            "dipole-dipole",
            data_dc_type="apparent_resistivity",
            data_dc=np.ones(self.survey.nD) * 100.0,
        )

        if self.plotIt:
            self.IO.plotPseudoSection(data_type="apparent_resistivity")
            plt.show()
        mesh, actind = self.IO.set_mesh()

    def test_topo_dpdp(self):
        self.survey = utils.gen_DCIPsurvey(
            self.endl, "dipole-dipole", dim=2, a=10, b=10, n=10
        )
        self.survey = self.IO.from_ambn_locations_to_survey(
            self.survey.locations_a,
            self.survey.locations_b,
            self.survey.locations_m,
            self.survey.locations_n,
            "dipole-dipole",
            data_dc_type="apparent_resistivity",
            data_dc=np.ones(self.survey.nD) * 100.0,
        )

        if self.plotIt:
            self.IO.plotPseudoSection(data_type="apparent_resistivity")
            plt.show()

        mesh, actind = self.IO.set_mesh()
        topo, mesh1D = utils.genTopography(mesh, -10, 0, its=100)
        mesh, actind = self.IO.set_mesh(topo=np.c_[mesh1D.vectorCCx, topo])
        self.survey.drape_electrodes_on_topography(mesh, actind, option="top")
        if self.plotIt:
            mesh.plotImage(actind)
            plt.plot(
                self.survey.electrode_locations[:, 0],
                self.survey.electrode_locations[:, 1],
                "k.",
            )
            plt.show()


if __name__ == "__main__":
    unittest.main()
