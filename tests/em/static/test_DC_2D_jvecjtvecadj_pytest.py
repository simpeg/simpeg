import pytest
import numpy as np
from discretize import TensorMesh
from SimPEG import (
    maps,
    data_misfit,
    tests,
)
from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics.static.resistivity.simulation_2d import (
    Simulation2DCellCentered,
    Simulation2DNodal,
)

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


@pytest.fixture
def mesh():
    # Return mesh for testing
    npad = 15
    cs = 12.5
    hx = [(cs, npad, -1.4), (cs, 61), (cs, npad, 1.4)]
    hy = [(cs, npad, -1.4), (cs, 20)]
    return TensorMesh([hx, hy], x0="CN")


def get_setup_objects(formulation, model_type, bc_type, storeJ, mesh):
    # Define the survey
    x = np.linspace(-135, 250.0, 20)
    M = utils.ndgrid(x - 12.5, np.r_[0.0])
    N = utils.ndgrid(x + 12.5, np.r_[0.0])
    A0loc = np.r_[-150, 0.0]
    A1loc = np.r_[-130, 0.0]
    # rxloc = [np.c_[M, np.zeros(20)], np.c_[N, np.zeros(20)]]
    rx1 = dc.receivers.Dipole(M, N)
    rx2 = dc.receivers.Dipole(M, N, data_type="apparent_resistivity")
    src0 = dc.sources.Pole([rx1, rx2], A0loc)
    src1 = dc.sources.Pole([rx1, rx2], A1loc)
    survey = dc.survey.Survey([src0, src1])
    survey.set_geometric_factor()

    # Define the model
    if model_type == "isotropic":
        m0 = np.ones(mesh.nC)
    elif model_type == "anisotropic":
        m0 = np.ones(3 * mesh.nC)
    else:
        raise ValueError("model_type not recognized.")

    # Define the simulation
    sim = formulation(
        mesh, survey=survey, solver=Solver, storeJ=storeJ, bc_type=bc_type
    )
    if isinstance(sim, Simulation2DCellCentered):
        sim.rhoMap = maps.IdentityMap(nP=len(m0))
    elif isinstance(sim, Simulation2DNodal):
        sim.sigmaMap = maps.IdentityMap(nP=len(m0))

    data = sim.make_synthetic_data(m0, add_noise=True)
    dmis = data_misfit.L2DataMisfit(simulation=sim, data=data)

    return m0, dmis


# Cases we want to test
CASES_LIST = [
    (Simulation2DCellCentered, "isotropic", "Robin", False, 1e-10),
    (Simulation2DCellCentered, "isotropic", "Robin", True, 1e-10),
    (Simulation2DCellCentered, "anisotropic", "Dirichlet", False, 1e-8),
    (Simulation2DCellCentered, "anisotropic", "Dirichlet", True, 1e-8),
    (Simulation2DNodal, "isotropic", "Neumann", False, 1e-8),
    (Simulation2DNodal, "isotropic", "Neumann", True, 1e-8),
    (Simulation2DNodal, "isotropic", "Robin", False, 1e-8),
    (Simulation2DNodal, "isotropic", "Robin", True, 1e-8),
    (Simulation2DNodal, "anisotropic", "Neumann", False, 1e-10),
    (Simulation2DNodal, "anisotropic", "Neumann", True, 1e-10),
]


@pytest.mark.parametrize(
    "formulation, model_type, bc_type, storeJ, adjoint_tol", CASES_LIST
)
def test_misfit(formulation, model_type, bc_type, storeJ, adjoint_tol, mesh):
    m0, dmis = get_setup_objects(formulation, model_type, bc_type, storeJ, mesh)
    sim = dmis.simulation

    assert tests.check_derivative(
        lambda m: (sim.dpred(m), lambda mx: sim.Jvec(m0, mx)),
        m0,
        plotIt=False,
        num=3,
    )


@pytest.mark.parametrize(
    "formulation, model_type, bc_type, storeJ, adjoint_tol", CASES_LIST
)
def test_adjoint(formulation, model_type, bc_type, storeJ, adjoint_tol, mesh):
    m0, dmis = get_setup_objects(formulation, model_type, bc_type, storeJ, mesh)
    sim = dmis.simulation
    n_data = sim.survey.nD

    rng = np.random.default_rng(seed=41)
    v = rng.random(len(m0))
    w = rng.random(n_data)
    wtJv = w.dot(sim.Jvec(m0, v))
    vtJtw = v.dot(sim.Jtvec(m0, w))
    passed = np.abs(wtJv - vtJtw) < adjoint_tol
    print("Adjoint Test", np.abs(wtJv - vtJtw), passed)
    assert passed


@pytest.mark.parametrize(
    "formulation, model_type, bc_type, storeJ, adjoint_tol", CASES_LIST
)
def test_dataObj(formulation, model_type, bc_type, storeJ, adjoint_tol, mesh):
    m0, dmis = get_setup_objects(formulation, model_type, bc_type, storeJ, mesh)

    assert tests.check_derivative(
        lambda m: [dmis(m), dmis.deriv(m)], m0, plotIt=False, num=3
    )
