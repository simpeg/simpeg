"""
Test model assignment to simulation classes

Test if the `getJ` method of a few static EM simulations updates the `model`.
These tests have been added as part of the bugfix in #1361.
"""
import pytest
import numpy as np

from discretize import TensorMesh
from SimPEG import utils
from SimPEG.maps import IdentityMap, Wires
from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics import spectral_induced_polarization as sip
from SimPEG.electromagnetics.static.utils import generate_dcip_sources_line


class TestDCSimulations:
    @pytest.fixture
    def mesh_3d(self):
        """Sample mesh."""
        # TODO: Reduce number of cells so the tests run faster with less memory
        # requirements
        cs = 0.5
        npad = 11
        hx = [(cs, npad, -1.5), (cs, 15), (cs, npad, 1.5)]
        hy = [(cs, npad, -1.5), (cs, 15), (cs, npad, 1.5)]
        hz = [(cs, npad, -1.5), (cs, 15), (cs, npad, 1.5)]
        mesh = TensorMesh([hx, hy, hz], x0="CCC")
        return mesh

    @pytest.fixture
    def survey_3d(self, mesh_3d):
        """Sample survey."""
        # Set up survey parameters for numeric solution
        x = mesh_3d.nodes_x[(mesh_3d.nodes_x > -75.0) & (mesh_3d.nodes_x < 75.0)]
        y = mesh_3d.nodes_y[(mesh_3d.nodes_y > -75.0) & (mesh_3d.nodes_y < 75.0)]

        Aloc = np.r_[1.25, 0.0, 0.0]
        Bloc = np.r_[-1.25, 0.0, 0.0]
        M = utils.ndgrid(x - 25.0, y, np.r_[0.0])
        N = utils.ndgrid(x + 25.0, y, np.r_[0.0])

        rx = dc.receivers.Dipole(M, N)
        src = dc.sources.Dipole([rx], Aloc, Bloc)
        survey = dc.survey.Survey([src])
        return survey

    @pytest.fixture
    def mesh_2d(self):
        """Sample mesh."""
        cell_size = 0.5
        width = 10.0
        hx = [
            (cell_size, 10, -1.3),
            (cell_size, width / cell_size),
            (cell_size, 10, 1.3),
        ]
        hy = [(cell_size, 3, -1.3), (cell_size, 3, 1.3)]
        mesh = TensorMesh([hx, hy], "CN")
        return mesh

    @pytest.fixture
    def survey_2d(self, mesh_2d):
        """Sample survey."""
        survey_end_points = np.array([-5.0, 5.0, 0, 0])

        source_list = generate_dcip_sources_line(
            "dipole-dipole", "volt", "2D", survey_end_points, 0.0, 5, 2.5
        )
        survey = dc.survey.Survey(source_list)
        return survey

    @pytest.mark.parametrize(
        "simulation_class",
        (dc.simulation.Simulation3DNodal, dc.simulation.Simulation3DCellCentered),
    )
    def test_simulation_3d(self, mesh_3d, survey_3d, simulation_class):
        """
        Test model assignment on the ``getJ`` method of 3d simulations
        """
        mapping = IdentityMap(mesh_3d)
        simulation = simulation_class(mesh=mesh_3d, survey=survey_3d, sigmaMap=mapping)
        model_1 = np.ones(mesh_3d.nC) * 1e-2
        model_2 = np.ones(mesh_3d.nC) * 1e-1
        # Call `getJ` passing a model and check if it was correctly assigned
        j_1 = simulation.getJ(model_1)
        assert model_1 is simulation.model
        # Call `getJ` passing a different model and check if it was correctly assigned
        j_2 = simulation.getJ(model_2)
        assert model_2 is simulation.model
        # Check if the two Js are different
        assert not np.allclose(j_1, j_2)

    @pytest.mark.parametrize(
        "simulation_class",
        (dc.simulation_2d.Simulation2DNodal, dc.simulation_2d.Simulation2DCellCentered),
    )
    def test_simulation_2d(self, mesh_2d, survey_2d, simulation_class):
        """
        Test model assignment on the ``getJ`` method of 2d simulations
        """
        mapping = IdentityMap(mesh_2d)
        simulation = simulation_class(mesh=mesh_2d, survey=survey_2d, sigmaMap=mapping)
        model_1 = np.ones(mesh_2d.nC) * 1e-2
        model_2 = np.ones(mesh_2d.nC) * 1e-1
        # Call `getJ` passing a model and check if it was correctly assigned
        j_1 = simulation.getJ(model_1)
        assert model_1 is simulation.model
        # Call `getJ` passing a different model and check if it was correctly assigned
        j_2 = simulation.getJ(model_2)
        assert model_2 is simulation.model
        # Check if the two Js are different
        assert not np.allclose(j_1, j_2)


class TestSIPSimulations:
    @pytest.fixture
    def mesh_3d(self):
        """Sample mesh."""
        cs = 25.0
        hx = [(cs, 0, -1.3), (cs, 21), (cs, 0, 1.3)]
        hy = [(cs, 0, -1.3), (cs, 21), (cs, 0, 1.3)]
        hz = [(cs, 0, -1.3), (cs, 20)]
        mesh = TensorMesh([hx, hy, hz], x0="CCN")
        return mesh

    @pytest.fixture
    def survey_3d(self, mesh_3d):
        """Sample survey."""
        x = mesh_3d.cell_centers_x[
            (mesh_3d.cell_centers_x > -155.0) & (mesh_3d.cell_centers_x < 155.0)
        ]
        y = mesh_3d.cell_centers_y[
            (mesh_3d.cell_centers_y > -155.0) & (mesh_3d.cell_centers_y < 155.0)
        ]
        Aloc = np.r_[-200.0, 0.0, 0.0]
        Bloc = np.r_[200.0, 0.0, 0.0]
        M = utils.ndgrid(x - 25.0, y, np.r_[0.0])

        times = np.arange(10) * 1e-3 + 1e-3
        rx = sip.receivers.Pole(M, times)
        src = sip.sources.Dipole([rx], Aloc, Bloc)
        survey = sip.Survey([src])
        return survey

    def test_simulation_3d(self, mesh_3d, survey_3d):
        """
        Test model assignment on the ``getJ`` method of 3d simulations
        """
        wires = Wires(("eta", mesh_3d.nC), ("taui", mesh_3d.nC))
        simulation = sip.Simulation3DNodal(
            mesh_3d,
            survey=survey_3d,
            etaMap=wires.eta,
            tauiMap=wires.taui,
        )
        sigma_1 = np.ones(mesh_3d.nC) * 1e-2
        sigma_2 = np.ones(mesh_3d.nC) * 1e-1
        model_1 = np.r_[sigma_1, 1.0 / sigma_1]
        model_2 = np.r_[sigma_2, 1.0 / sigma_2]
        # Call `getJ` passing a model and check if it was correctly assigned
        j_1 = simulation.getJ(model_1)
        assert model_1 is simulation.model
        # Call `getJ` passing a different model and check if it was correctly assigned
        j_2 = simulation.getJ(model_2)
        assert model_2 is simulation.model
        # Check if the two Js are different
        assert not np.allclose(j_1, j_2)
