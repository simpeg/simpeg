import pytest
import numpy as np
from discretize import TensorMesh, CylindricalMesh
from SimPEG import (
    maps,
    data_misfit,
    tests,
)
from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics.static.resistivity.simulation import (
    Simulation3DCellCentered,
    Simulation3DNodal,
)
import shutil

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


np.random.seed(40)

TOL = 1e-5
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order


# Return mesh for testing
def get_mesh(mesh_type):
    if mesh_type == TensorMesh:
        aSpacing = 2.5
        nElecs = 5
        surveySize = nElecs * aSpacing - aSpacing
        cs = surveySize / nElecs / 4
        mesh = TensorMesh(
            [
                [(cs, 10, -1.3), (cs, surveySize / cs), (cs, 10, 1.3)],
                [(cs, 3, -1.3), (cs, 3, 1.3)],
                # [(cs, 5, -1.3), (cs, 10)]
            ],
            "CN",
        )

    elif mesh_type == CylindricalMesh:
        cs = 10
        nc = 20
        npad = 10
        mesh = CylindricalMesh(
            [
                [(cs, nc), (cs, npad, 1.3)],
                np.r_[2 * np.pi],
                [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)],
            ]
        )
        mesh.x0 = np.r_[0.0, 0.0, -mesh.h[2][: npad + nc].sum()]

    return mesh


def get_setup_objects(formulation, model_type, bc_type, storeJ, mesh):
    # Define the survey
    if isinstance(mesh, TensorMesh):
        aSpacing = 2.5
        nElecs = 8
        source_list = dc.utils.WennerSrcList(nElecs, aSpacing, in2D=True)

    else:
        raise TypeError("Tests only written for TensorMesh class")

    survey = dc.survey.Survey(source_list)

    # Define the model
    if model_type == "isotropic":
        m0 = np.ones(mesh.nC)
    elif model_type == "anisotropic":
        m0 = np.ones(2 * mesh.nC)  # YES, 2*nC!!!
    else:
        raise ValueError("model_type not recognized.")

    # Define the simulation
    sim = formulation(
        mesh,
        survey=survey,
        rhoMap=maps.IdentityMap(nP=len(m0)),
        solver=Solver,
        storeJ=storeJ,
        bc_type=bc_type,
    )

    data = sim.make_synthetic_data(m0, add_noise=True)
    dmis = data_misfit.L2DataMisfit(simulation=sim, data=data)

    return m0, dmis


def tear_down(sim):
    try:
        shutil.rmtree(sim.sensitivity_path)
    except FileNotFoundError:
        pass


CASES_LIST_DATA = [
    (Simulation3DCellCentered, TensorMesh, "isotropic", "Robin", False, 1e-10),
    (Simulation3DCellCentered, TensorMesh, "isotropic", "Robin", True, 1e-10),
    (Simulation3DCellCentered, TensorMesh, "isotropic", "Neumann", False, 1e-10),
    (Simulation3DCellCentered, TensorMesh, "isotropic", "Dirichlet", False, 1e-10),
    (Simulation3DNodal, TensorMesh, "isotropic", "Robin", False, 1e-10),
    (Simulation3DNodal, TensorMesh, "isotropic", "Robin", True, 1e-10),
    (Simulation3DNodal, TensorMesh, "isotropic", "Neumann", False, 1e-10),
    (Simulation3DCellCentered, TensorMesh, "anisotropic", "Neumann", False, 1e-10),
    (Simulation3DCellCentered, TensorMesh, "anisotropic", "Neumann", True, 1e-10),
    (Simulation3DNodal, TensorMesh, "anisotropic", "Neumann", False, 1e-10),
    (Simulation3DNodal, TensorMesh, "anisotropic", "Neumann", True, 1e-10),
]


@pytest.mark.parametrize(
    "formulation, mesh_type, model_type, bc_type, storeJ, adjoint_tol", CASES_LIST_DATA
)
def test_misfit(formulation, mesh_type, model_type, bc_type, storeJ, adjoint_tol):
    mesh = get_mesh(mesh_type)

    m0, dmis = get_setup_objects(formulation, model_type, bc_type, storeJ, mesh)
    sim = dmis.simulation

    passed = tests.check_derivative(
        lambda m: (sim.dpred(m), lambda mx: sim.Jvec(m0, mx)),
        m0,
        plotIt=False,
        num=3,
    )

    if storeJ:
        tear_down(sim)

    assert passed


@pytest.mark.parametrize(
    "formulation, mesh_type, model_type, bc_type, storeJ, adjoint_tol", CASES_LIST
)
def test_adjoint(formulation, mesh_type, model_type, bc_type, storeJ, adjoint_tol):
    mesh = get_mesh(mesh_type)

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

    if storeJ:
        tear_down(sim)


@pytest.mark.parametrize(
    "formulation, mesh_type, model_type, bc_type, storeJ, adjoint_tol", CASES_LIST
)
def test_dataObj(formulation, mesh_type, model_type, bc_type, storeJ, adjoint_tol):
    mesh = get_mesh(mesh_type)

    m0, dmis = get_setup_objects(formulation, model_type, bc_type, storeJ, mesh)
    sim = dmis.simulation

    passed = tests.check_derivative(
        lambda m: [dmis(m), dmis.deriv(m)], m0, plotIt=False, num=3
    )

    if storeJ:
        tear_down(sim)

    assert passed
