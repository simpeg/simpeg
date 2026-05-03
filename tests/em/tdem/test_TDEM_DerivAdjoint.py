"""Derivative and adjoint tests."""

import pytest
import numpy as np
import discretize
from simpeg import maps, data_misfit, tests
from simpeg.utils import get_default_solver
from simpeg.electromagnetics import time_domain as tdem

plotIt = False

ADJ_ATOL = 1e-11
ADJ_RTOL = 2e-5


@pytest.fixture
def mesh():
    """Return test mesh."""
    cs = 10.0
    ncx = 4
    ncy = 4
    ncz = 4
    npad = 2
    # hx = [(cs, ncx), (cs, npad, 1.3)]
    # hz = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
    return discretize.TensorMesh(
        [
            [(cs, npad, -1.5), (cs, ncx), (cs, npad, 1.5)],
            [(cs, npad, -1.5), (cs, ncy), (cs, npad, 1.5)],
            [(cs, npad, -1.5), (cs, ncz), (cs, npad, 1.5)],
        ],
        "CCC",
    )


@pytest.fixture
def locations():
    """Return receiver locations."""
    return np.array([[15, 0.0, -1e-2]])


@pytest.fixture
def times():
    """Return receiver times."""
    return np.logspace(-4, -3, 20)


def get_survey(receiver_type, orientation, locations, times):
    """Return test survey."""

    rx1 = getattr(tdem.receivers, "Point" + receiver_type)(
        locations=locations, times=times, orientation=orientation
    )
    rx2 = getattr(tdem.receivers, "Point" + receiver_type)(
        locations=locations, times=times, orientation=orientation
    )

    src1 = tdem.sources.MagDipole([rx1], location=np.array([0.0, 0.0, 0.0]))
    src2 = tdem.sources.MagDipole([rx2], location=np.array([0.0, 0.0, 8.0]))

    return tdem.Survey([src1, src2])


def get_simulation(simulation_class, mesh, survey):
    """Return test simulation."""
    sim = getattr(tdem, simulation_class)(mesh, survey=survey)

    if "Hierarchical" in simulation_class:
        sigma_map, tau_map, kappa_map = get_wire_mappings(mesh)
        sim.sigmaMap = sigma_map
        sim.tauMap = tau_map
        sim.kappaMap = kappa_map
    else:
        sigma_map = get_sigma_mapping(mesh)
        sim.sigmaMap = sigma_map

    sim.solver = get_default_solver()
    sim.time_steps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]

    return sim


def get_sigma_mapping(mesh):
    """Return mapping for sigma."""
    active = mesh.cell_centers_z < 0.0
    activeMap = maps.InjectActiveCells(
        mesh, active, np.log(1e-8), nC=mesh.shape_cells[2]
    )
    return maps.ExpMap(mesh) * maps.SurjectVertical1D(mesh) * activeMap


def get_wire_mappings(mesh):
    """Return mapping for cell, faces and edges."""
    active_cells = mesh.cell_centers[:, -1] < 0.0
    active_faces = mesh.faces[:, -1] < 0.0
    active_edges = mesh.edges[:, -1] < 0.0
    n_active_cells = np.sum(active_cells)
    n_active_faces = np.sum(active_faces)
    n_active_edges = np.sum(active_edges)

    wire_map = maps.Wires(
        ("log_sigma", n_active_cells),
        ("log_tau", n_active_faces),
        ("log_kappa", n_active_edges),
    )

    sigma_map = (
        maps.InjectActiveCells(mesh, active_cells, 1e-8)
        * maps.ExpMap(nP=n_active_cells)
        * wire_map.log_sigma
    )
    tau_map = (
        maps.InjectActiveFaces(mesh, active_faces, 0)
        * maps.ExpMap(nP=n_active_faces)
        * wire_map.log_tau
    )
    kappa_map = (
        maps.InjectActiveEdges(mesh, active_edges, 0)
        * maps.ExpMap(nP=n_active_edges)
        * wire_map.log_kappa
    )

    return sigma_map, tau_map, kappa_map


CASES_LIST = [
    ("Simulation3DElectricField", "MagneticFluxTimeDerivative", "z"),
    ("Simulation3DElectricField", "MagneticFluxTimeDerivative", "x"),
    ("Simulation3DElectricField", "ElectricField", "y"),
    ("Simulation3DElectricField", "MagneticFieldTimeDerivative", "z"),
    ("Simulation3DElectricField", "MagneticFieldTimeDerivative", "x"),
    ("Simulation3DElectricField", "CurrentDensity", "y"),
    ("Simulation3DHierarchicalElectricField", "MagneticFluxTimeDerivative", "z"),
    ("Simulation3DHierarchicalElectricField", "MagneticFluxTimeDerivative", "x"),
    ("Simulation3DHierarchicalElectricField", "ElectricField", "y"),
    ("Simulation3DHierarchicalElectricField", "MagneticFieldTimeDerivative", "z"),
    ("Simulation3DHierarchicalElectricField", "MagneticFieldTimeDerivative", "x"),
    ("Simulation3DHierarchicalElectricField", "CurrentDensity", "y"),
    ("Simulation3DMagneticFluxDensity", "MagneticFluxTimeDerivative", "z"),
    ("Simulation3DMagneticFluxDensity", "MagneticFluxTimeDerivative", "x"),
    ("Simulation3DMagneticFluxDensity", "MagneticFluxDensity", "z"),
    ("Simulation3DMagneticFluxDensity", "MagneticFluxDensity", "x"),
    ("Simulation3DMagneticFluxDensity", "MagneticFieldTimeDerivative", "z"),
    ("Simulation3DMagneticFluxDensity", "MagneticFieldTimeDerivative", "x"),
    ("Simulation3DMagneticFluxDensity", "MagneticField", "z"),
    ("Simulation3DMagneticFluxDensity", "MagneticField", "x"),
    ("Simulation3DMagneticFluxDensity", "CurrentDensity", "y"),
    ("Simulation3DHierarchicalMagneticFluxDensity", "MagneticFluxDensity", "z"),
    ("Simulation3DHierarchicalMagneticFluxDensity", "MagneticFluxDensity", "x"),
    ("Simulation3DHierarchicalMagneticFluxDensity", "MagneticFluxTimeDerivative", "z"),
    ("Simulation3DHierarchicalMagneticFluxDensity", "MagneticFluxTimeDerivative", "x"),
    ("Simulation3DHierarchicalMagneticFluxDensity", "MagneticField", "z"),
    ("Simulation3DHierarchicalMagneticFluxDensity", "MagneticField", "x"),
    ("Simulation3DHierarchicalMagneticFluxDensity", "MagneticFieldTimeDerivative", "z"),
    ("Simulation3DHierarchicalMagneticFluxDensity", "MagneticFieldTimeDerivative", "x"),
    ("Simulation3DHierarchicalMagneticFluxDensity", "CurrentDensity", "y"),
    ("Simulation3DMagneticField", "MagneticField", "z"),
    ("Simulation3DMagneticField", "MagneticField", "x"),
    ("Simulation3DMagneticField", "MagneticFieldTimeDerivative", "z"),
    ("Simulation3DMagneticField", "MagneticFieldTimeDerivative", "x"),
    ("Simulation3DMagneticField", "MagneticFluxDensity", "z"),
    ("Simulation3DMagneticField", "MagneticFluxDensity", "x"),
    ("Simulation3DMagneticField", "MagneticFluxTimeDerivative", "z"),
    ("Simulation3DMagneticField", "MagneticFluxTimeDerivative", "x"),
    ("Simulation3DMagneticField", "ElectricField", "y"),
    ("Simulation3DMagneticField", "CurrentDensity", "y"),
    ("Simulation3DCurrentDensity", "MagneticFieldTimeDerivative", "z"),
    ("Simulation3DCurrentDensity", "MagneticFieldTimeDerivative", "x"),
    ("Simulation3DCurrentDensity", "MagneticFluxTimeDerivative", "z"),
    ("Simulation3DCurrentDensity", "MagneticFluxTimeDerivative", "x"),
    ("Simulation3DCurrentDensity", "CurrentDensity", "y"),
]


@pytest.mark.parametrize("simulation_class, receiver_type, orientation", CASES_LIST)
class TestDerivatives:
    """Derivative and adjoint test for primary-secondary formulation."""

    def get_setup_objects(
        self,
        simulation_class,
        receiver_type,
        orientation,
        mesh,
        locations,
        times,
    ):
        """Setup test."""
        survey = get_survey(
            receiver_type,
            orientation,
            locations,
            times,
        )

        sim = get_simulation(simulation_class, mesh, survey)

        if "Hierarchical" in simulation_class:
            n_sigma = np.sum(mesh.cell_centers[:, -1] < 0.0)
            n_tau = np.sum(mesh.faces[:, -1] < 0.0)
            n_kappa = np.sum(mesh.edges[:, -1] < 0.0)
            m0 = np.r_[
                np.log(1e-1) * np.ones(n_sigma) + 1e-3 * np.random.randn(n_sigma),
                np.log(10 * 1e-1) * np.ones(n_tau) + 1e-3 * np.random.randn(n_tau),
                np.log(100 * 1e-1) * np.ones(n_kappa) + 1e-3 * np.random.randn(n_kappa),
            ]
        else:
            n_param = sim.sigmaMap.nP
            m0 = np.log(1e-1) * np.ones(n_param) + (1e-3 * np.random.randn(n_param))

        # Define data and misfit
        data = sim.make_synthetic_data(m0, add_noise=True, seed=4412)
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=data)

        return m0, dmis

    def test_misfit(
        self,
        simulation_class,
        receiver_type,
        orientation,
        mesh,
        locations,
        times,
    ):
        """Test derivative."""
        m0, dmis = self.get_setup_objects(
            simulation_class,
            receiver_type,
            orientation,
            mesh,
            locations,
            times,
        )
        sim = dmis.simulation

        passed = tests.check_derivative(
            lambda m: (sim.dpred(m), lambda mx: sim.Jvec(m0, mx)),
            m0,
            plotIt=False,
            random_seed=42,
            num=3,
        )

        assert passed

    def test_adjoint(
        self,
        simulation_class,
        receiver_type,
        orientation,
        mesh,
        locations,
        times,
    ):
        """Test adjoint."""
        m0, dmis = self.get_setup_objects(
            simulation_class,
            receiver_type,
            orientation,
            mesh,
            locations,
            times,
        )
        sim = dmis.simulation
        n_data = sim.survey.nD

        f = sim.fields(m0)

        tests.assert_isadjoint(
            lambda u: sim.Jvec(m0, u, f=f),
            lambda v: sim.Jtvec(m0, v, f=f),
            m0.shape,
            (n_data,),
            rtol=ADJ_RTOL,
            atol=ADJ_ATOL,
            random_seed=44,
        )
