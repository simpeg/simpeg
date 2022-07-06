from __future__ import print_function
import unittest

from discretize import TensorMesh
from SimPEG import utils
import numpy as np
from SimPEG.electromagnetics import resistivity as dc

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

from geoana.em import fdem
from scipy.constants import mu_0, epsilon_0


class DC_CC_MultipoleFullspaceTests(unittest.TestCase):
    def setUp(self):

        cs = 0.5
        npad = 11
        hx = [(cs, npad, -1.5), (cs, 15), (cs, npad, 1.5)]
        hy = [(cs, npad, -1.5), (cs, 15), (cs, npad, 1.5)]
        hz = [(cs, npad, -1.5), (cs, 15), (cs, npad, 1.5)]
        mesh = TensorMesh([hx, hy, hz], x0="CCC")
        sigma = np.ones(mesh.nC) * 1e-2

        # Set up survey parameters for numeric solution
        x = mesh.vectorCCx[(mesh.vectorCCx > -75.0) & (mesh.vectorCCx < 75.0)]
        y = mesh.vectorCCy[(mesh.vectorCCy > -75.0) & (mesh.vectorCCy < 75.0)]

        Aloc = np.r_[2.0, 0.0, 0.0]
        Bloc = np.r_[1.0, 0.0, 0.0]
        Cloc = np.r_[0.0, 0.0, 0.0]
        M = utils.ndgrid(x - 25.0, y, np.r_[0.0])
        N = utils.ndgrid(x + 25.0, y, np.r_[0.0])

        rx = dc.receivers.Dipole(M, N)
        src = dc.sources.Multipole(
            [rx], location=[Aloc, Bloc, Cloc], current=[1.0, 1.0, -2.0]
        )
        survey = dc.survey.Survey([src])

        # Create Dipole Obj for Analytic Solution
        e1dipole = fdem.ElectricDipoleWholeSpace(
            sigma=1e-2,  # conductivity of 1 S/m
            mu=mu_0,  # permeability of free space (this is the default)
            epsilon=epsilon_0,  # permittivity of free space (this is the default)
            location=np.r_[0.5, 0.0, 0.0],  # location of the dipole
            orientation="X",  # horizontal dipole (can also be a unit-vector)
            quasistatic=True,  # don't use the quasistatic assumption
            frequency=0.0,  # DC
            length=1.0,  # length of dipole
        )
        e2dipole = fdem.ElectricDipoleWholeSpace(
            sigma=1e-2,  # conductivity of 1 S/m
            mu=mu_0,  # permeability of free space (this is the default)
            epsilon=epsilon_0,  # permittivity of free space (this is the default)
            location=np.r_[1.0, 0.0, 0.0],  # location of the dipole
            orientation="X",  # horizontal dipole (can also be a unit-vector)
            quasistatic=True,  # don't use the quasistatic assumption
            frequency=0.0,  # DC
            length=2.0,  # length of dipole
        )

        # evaluate the electric field and current density
        Ex_analytic = np.zeros_like([mesh.nFx, 1])
        Ey_analytic = np.zeros_like([mesh.nFy, 1])
        Ez_analytic = np.zeros_like([mesh.nFz, 1])
        Ex_analytic = np.real(
            e1dipole.electric_field(mesh.gridFx) + e2dipole.electric_field(mesh.gridFx)
        )[:, 0]
        Ey_analytic = np.real(
            e1dipole.electric_field(mesh.gridFy) + e2dipole.electric_field(mesh.gridFy)
        )[:, 1]
        Ez_analytic = np.real(
            e1dipole.electric_field(mesh.gridFz) + e2dipole.electric_field(mesh.gridFz)
        )[:, 2]
        E_analytic = np.hstack([Ex_analytic, Ey_analytic, Ez_analytic])

        Jx_analytic = np.zeros_like([mesh.nFx, 1])
        Jy_analytic = np.zeros_like([mesh.nFy, 1])
        Jz_analytic = np.zeros_like([mesh.nFz, 1])
        Jx_analytic = np.real(
            e1dipole.current_density(mesh.gridFx)
            + e2dipole.current_density(mesh.gridFx)
        )[:, 0]
        Jy_analytic = np.real(
            e1dipole.current_density(mesh.gridFy)
            + e2dipole.current_density(mesh.gridFy)
        )[:, 1]
        Jz_analytic = np.real(
            e1dipole.current_density(mesh.gridFz)
            + e2dipole.current_density(mesh.gridFz)
        )[:, 2]
        J_analytic = np.hstack([Jx_analytic, Jy_analytic, Jz_analytic])

        # Find faces at which to compare solutions
        faceGrid = np.vstack([mesh.gridFx, mesh.gridFy, mesh.gridFz])
        # print(faceGrid.shape)

        ROI_large_BNW = np.array([-75, 75, -75])
        ROI_large_TSE = np.array([75, -75, 75])
        ROI_largeInds = utils.model_builder.getIndicesBlock(
            ROI_large_BNW, ROI_large_TSE, faceGrid
        )[0]
        # print(ROI_largeInds.shape)

        ROI_small_BNW = np.array([-4, 4, -4])
        ROI_small_TSE = np.array([4, -4, 4])
        ROI_smallInds = utils.model_builder.getIndicesBlock(
            ROI_small_BNW, ROI_small_TSE, faceGrid
        )[0]
        # print(ROI_smallInds.shape)

        ROIfaceInds = np.setdiff1d(ROI_largeInds, ROI_smallInds)
        # print(ROIfaceInds.shape)
        # print(len(ROI_largeInds) - len(ROI_smallInds))

        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.E_analytic = E_analytic
        self.J_analytic = J_analytic
        self.ROIfaceInds = ROIfaceInds

    def test_Simulation3DCellCentered_Dirichlet(self, tolerance=0.1):
        simulation = dc.Simulation3DCellCentered(
            self.mesh, survey=self.survey, sigma=self.sigma, bc_type="Dirichlet"
        )
        simulation.solver = Solver

        #        f = simulation.fields()
        f = simulation.fields(self.sigma)
        eNumeric = utils.mkvc(f[self.survey.source_list, "e"])
        jNumeric = utils.mkvc(f[self.survey.source_list, "j"])
        # also test we can get charge and charge density
        f[:, "charge"]
        f[:, "charge_density"]
        # errJ
        # np.testing.assert_allclose(jNumeric[self.ROIfaceInds], self.J_analytic[self.ROIfaceInds], rtol=tolerance)
        # # errE
        # np.testing.assert_allclose(eNumeric[self.ROIfaceInds], self.E_analytic[self.ROIfaceInds], rtol=tolerance)
        #
        errJ = np.linalg.norm(
            jNumeric[self.ROIfaceInds] - self.J_analytic[self.ROIfaceInds]
        ) / np.linalg.norm(self.J_analytic[self.ROIfaceInds])
        errE = np.linalg.norm(
            eNumeric[self.ROIfaceInds] - self.E_analytic[self.ROIfaceInds]
        ) / np.linalg.norm(self.E_analytic[self.ROIfaceInds])
        self.assertLess(errE, tolerance)
        self.assertLess(errJ, tolerance)

    def test_Simulation3DCellCentered_Mixed(self, tolerance=0.1):
        simulation = dc.simulation.Simulation3DCellCentered(
            self.mesh, survey=self.survey, sigma=self.sigma, bc_type="Mixed"
        )
        simulation.solver = Solver

        f = simulation.fields(self.sigma)
        eNumeric = utils.mkvc(f[self.survey.source_list, "e"])
        jNumeric = utils.mkvc(f[self.survey.source_list, "j"])
        errJ = np.linalg.norm(
            jNumeric[self.ROIfaceInds] - self.J_analytic[self.ROIfaceInds]
        ) / np.linalg.norm(self.J_analytic[self.ROIfaceInds])
        errE = np.linalg.norm(
            eNumeric[self.ROIfaceInds] - self.E_analytic[self.ROIfaceInds]
        ) / np.linalg.norm(self.E_analytic[self.ROIfaceInds])
        self.assertLess(errE, tolerance)
        self.assertLess(errJ, tolerance)

    def test_Simulation3DCellCentered_Neumann(self, tolerance=0.1):
        simulation = dc.Simulation3DCellCentered(
            self.mesh, survey=self.survey, sigma=self.sigma, bc_type="Neumann"
        )
        simulation.solver = Solver

        f = simulation.fields(self.sigma)
        eNumeric = utils.mkvc(f[self.survey.source_list, "e"])
        jNumeric = utils.mkvc(f[self.survey.source_list, "j"])
        errJ = np.linalg.norm(
            jNumeric[self.ROIfaceInds] - self.J_analytic[self.ROIfaceInds]
        ) / np.linalg.norm(self.J_analytic[self.ROIfaceInds])
        errE = np.linalg.norm(
            eNumeric[self.ROIfaceInds] - self.E_analytic[self.ROIfaceInds]
        ) / np.linalg.norm(self.E_analytic[self.ROIfaceInds])
        self.assertLess(errE, tolerance)
        self.assertLess(errJ, tolerance)


class DC_N_MultipoleFullspaceTests(unittest.TestCase):
    def setUp(self):

        cs = 0.5
        npad = 11
        hx = [(cs, npad, -1.5), (cs, 15), (cs, npad, 1.5)]
        hy = [(cs, npad, -1.5), (cs, 15), (cs, npad, 1.5)]
        hz = [(cs, npad, -1.5), (cs, 15), (cs, npad, 1.5)]
        mesh = TensorMesh([hx, hy, hz], x0="CCC")
        sigma = np.ones(mesh.nC) * 1e-2

        # Set up survey parameters for numeric solution
        x = mesh.vectorNx[(mesh.vectorNx > -75.0) & (mesh.vectorNx < 75.0)]
        y = mesh.vectorNy[(mesh.vectorNy > -75.0) & (mesh.vectorNy < 75.0)]

        Aloc = np.r_[2.25, 0.0, 0.0]
        Bloc = np.r_[1.25, 0.0, 0.0]
        Cloc = np.r_[-0.25, 0.0, 0.0]

        M = utils.ndgrid(x - 25.0, y, np.r_[0.0])
        N = utils.ndgrid(x + 25.0, y, np.r_[0.0])

        rx = dc.receivers.Dipole(M, N)
        src = dc.sources.Multipole(
            [rx], location=[Aloc, Bloc, Cloc], current=[1.0, 1.0, -2.0]
        )

        survey = dc.survey.Survey([src])

        # Create Dipole Obj for Analytic Solution
        e1dipole = fdem.ElectricDipoleWholeSpace(
            sigma=1e-2,  # conductivity of 1 S/m
            mu=mu_0,  # permeability of free space (this is the default)
            epsilon=epsilon_0,  # permittivity of free space (this is the default)
            location=np.r_[0.5, 0.0, 0.0],  # location of the dipole
            orientation="X",  # horizontal dipole (can also be a unit-vector)
            quasistatic=True,  # don't use the quasistatic assumption
            frequency=0.0,  # DC
            length=1.5,  # length of dipole
        )
        e2dipole = fdem.ElectricDipoleWholeSpace(
            sigma=1e-2,  # conductivity of 1 S/m
            mu=mu_0,  # permeability of free space (this is the default)
            epsilon=epsilon_0,  # permittivity of free space (this is the default)
            location=np.r_[1.0, 0.0, 0.0],  # location of the dipole
            orientation="X",  # horizontal dipole (can also be a unit-vector)
            quasistatic=True,  # don't use the quasistatic assumption
            frequency=0.0,  # DC
            length=2.5,  # length of dipole
        )

        # evaluate the electric field and current density
        Ex_analytic = np.zeros_like([mesh.nEx, 1])
        Ey_analytic = np.zeros_like([mesh.nEy, 1])
        Ez_analytic = np.zeros_like([mesh.nEz, 1])
        Ex_analytic = np.real(
            e1dipole.electric_field(mesh.gridEx) + e2dipole.electric_field(mesh.gridEx)
        )[:, 0]
        Ey_analytic = np.real(
            e1dipole.electric_field(mesh.gridEy) + e2dipole.electric_field(mesh.gridEy)
        )[:, 1]
        Ez_analytic = np.real(
            e1dipole.electric_field(mesh.gridEz) + e2dipole.electric_field(mesh.gridEz)
        )[:, 2]
        E_analytic = np.hstack([Ex_analytic, Ey_analytic, Ez_analytic])

        Jx_analytic = np.zeros_like([mesh.nEx, 1])
        Jy_analytic = np.zeros_like([mesh.nEy, 1])
        Jz_analytic = np.zeros_like([mesh.nEz, 1])
        Jx_analytic = np.real(
            e1dipole.current_density(mesh.gridEx)
            + e2dipole.current_density(mesh.gridEx)
        )[:, 0]
        Jy_analytic = np.real(
            e1dipole.current_density(mesh.gridEy)
            + e2dipole.current_density(mesh.gridEy)
        )[:, 1]
        Jz_analytic = np.real(
            e1dipole.current_density(mesh.gridEz)
            + e2dipole.current_density(mesh.gridEz)
        )[:, 2]
        J_analytic = np.hstack([Jx_analytic, Jy_analytic, Jz_analytic])

        # Find edges at which to compare solutions
        edgeGrid = np.vstack([mesh.gridEx, mesh.gridEy, mesh.gridEz])
        # print(faceGrid.shape)

        ROI_large_BNW = np.array([-75, 75, -75])
        ROI_large_TSE = np.array([75, -75, 75])
        ROI_largeInds = utils.model_builder.getIndicesBlock(
            ROI_large_BNW, ROI_large_TSE, edgeGrid
        )[0]
        # print(ROI_largeInds.shape)

        ROI_small_BNW = np.array([-4, 4, -4])
        ROI_small_TSE = np.array([4, -4, 4])
        ROI_smallInds = utils.model_builder.getIndicesBlock(
            ROI_small_BNW, ROI_small_TSE, edgeGrid
        )[0]
        # print(ROI_smallInds.shape)

        ROIedgeInds = np.setdiff1d(ROI_largeInds, ROI_smallInds)
        # print(ROIedgeInds.shape)
        # print(len(ROI_largeInds) - len(ROI_smallInds))

        self.survey = survey
        self.mesh = mesh
        self.sigma = sigma
        self.E_analytic = E_analytic
        self.J_analytic = J_analytic
        self.ROIedgeInds = ROIedgeInds

    def test_Simulation3DNodal(self, tolerance=0.1):
        simulation = dc.simulation.Simulation3DNodal(
            self.mesh, survey=self.survey, sigma=self.sigma
        )
        simulation.solver = Solver

        f = simulation.fields(self.sigma)
        eNumeric = utils.mkvc(f[self.survey.source_list, "e"])
        jNumeric = utils.mkvc(f[self.survey.source_list, "j"])
        # also test if we can get charge and charge_density
        f[:, "charge"]
        f[:, "charge_density"]

        errJ = np.linalg.norm(
            jNumeric[self.ROIedgeInds] - self.J_analytic[self.ROIedgeInds]
        ) / np.linalg.norm(self.J_analytic[self.ROIedgeInds])
        errE = np.linalg.norm(
            eNumeric[self.ROIedgeInds] - self.E_analytic[self.ROIedgeInds]
        ) / np.linalg.norm(self.E_analytic[self.ROIedgeInds])
        self.assertLess(errE, tolerance)
        self.assertLess(errJ, tolerance)


if __name__ == "__main__":
    unittest.main()
