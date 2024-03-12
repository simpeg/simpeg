import numpy as np
import pytest

from discretize import TensorMesh

from SimPEG import utils, SolverLU
from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics.static.resistivity.simulation_2d import (
    Simulation2DCellCentered,
    Simulation2DNodal,
)
from SimPEG.electromagnetics import analytics

try:
    from pymatsolver import Pardiso

    SOLVER = Pardiso
except ImportError:
    SOLVER = SolverLU


@pytest.fixture
def mesh():
    # Return mesh for testing
    npad = 15
    cs = 12.5
    hx = [(cs, npad, -1.4), (cs, 61), (cs, npad, 1.4)]
    hy = [(cs, npad, -1.4), (cs, 20)]
    return TensorMesh([hx, hy], x0="CN")


@pytest.fixture
def electrode_locations(mesh):
    x = mesh.cell_centers_x[
        np.logical_and(mesh.cell_centers_x > -150, mesh.cell_centers_x < 250)
    ]
    M = utils.ndgrid(x, np.r_[0.0])
    N = utils.ndgrid(x + 12.5 * 4, np.r_[0.0])
    M = np.c_[M, np.zeros(x.size)]  # xyz electrode locations
    N = np.c_[N, np.zeros(x.size)]  # xyz electrode locations

    A = np.r_[-200, 0.0, 0.0]  # xyz electrode locations
    B = np.r_[-250, 0.0, 0.0]  # xyz electrode locations

    return A, B, M, N


def get_analytic_solution(survey_type, sighalf, A, B, M, N):
    # Tensor mesh, survey type, halfspace conductivity

    if survey_type == "dipole-dipole":
        data_ana_A = analytics.DCAnalytic_Pole_Dipole(
            A, [M, N], sighalf, earth_type="halfspace"
        )
        data_ana_B = analytics.DCAnalytic_Pole_Dipole(
            B, [M, N], sighalf, earth_type="halfspace"
        )
        return data_ana_A - data_ana_B
    elif survey_type == "pole-dipole":
        return analytics.DCAnalytic_Pole_Dipole(
            A, [M, N], sighalf, earth_type="halfspace"
        )
    elif survey_type == "dipole-pole":
        return analytics.DCAnalytic_Dipole_Pole(
            [A, B], M, sighalf, earth_type="halfspace"
        )
    elif survey_type == "pole-pole":
        return analytics.DCAnalytic_Pole_Pole(A, M, sighalf, earth_type="halfspace")
    else:
        raise ValueError(
            "survey_type not recognized. {'pole-pole', 'pole-dipole', 'dipole-pole', 'dipole-dipole'}"
        )


def get_survey(survey_type, data_type, A, B, M, N):
    # Remove y coordinate
    A = A[[0, 2]]
    B = B[[0, 2]]
    M = M[:, [0, 2]]
    N = N[:, [0, 2]]

    if survey_type == "dipole-dipole":
        rx = dc.receivers.Dipole(M, N, data_type=data_type)
        src = dc.sources.Dipole([rx], A, B)
    elif survey_type == "pole-dipole":
        rx = dc.receivers.Dipole(M, N, data_type=data_type)
        src = dc.sources.Pole([rx], A)
    elif survey_type == "dipole-pole":
        rx = dc.receivers.Pole(M, data_type=data_type)
        src = dc.sources.Dipole([rx], A, B)
    elif survey_type == "pole-pole":
        rx = dc.receivers.Pole(M, data_type=data_type)
        src = dc.sources.Pole([rx], A)
    else:
        raise ValueError(
            "survey_type not recognized. {'pole-pole', 'pole-dipole', 'dipole-pole', 'dipole-dipole'}"
        )

    return dc.Survey([src])


CASES_LIST_DATA = [
    ("dipole-dipole", "volt", Simulation2DCellCentered, "isotropic", "Robin"),
    (
        "dipole-dipole",
        "apparent_resistivity",
        Simulation2DCellCentered,
        "isotropic",
        "Robin",
    ),
    ("dipole-dipole", "volt", Simulation2DCellCentered, "anisotropic", "Neumann"),
    ("dipole-dipole", "volt", Simulation2DNodal, "isotropic", "Robin"),
    ("dipole-dipole", "apparent_resistivity", Simulation2DNodal, "isotropic", "Robin"),
    ("dipole-dipole", "volt", Simulation2DNodal, "anisotropic", "Neumann"),
    ("pole-dipole", "volt", Simulation2DCellCentered, "isotropic", "Robin"),
    ("pole-dipole", "volt", Simulation2DCellCentered, "anisotropic", "Neumann"),
    ("pole-dipole", "volt", Simulation2DNodal, "isotropic", "Robin"),
    ("pole-dipole", "volt", Simulation2DNodal, "anisotropic", "Neumann"),
    ("dipole-pole", "volt", Simulation2DCellCentered, "isotropic", "Robin"),
    ("dipole-pole", "volt", Simulation2DCellCentered, "anisotropic", "Neumann"),
    ("dipole-pole", "volt", Simulation2DNodal, "isotropic", "Robin"),
    ("dipole-pole", "volt", Simulation2DNodal, "anisotropic", "Neumann"),
    ("pole-pole", "volt", Simulation2DCellCentered, "isotropic", "Robin"),
    ("pole-pole", "volt", Simulation2DCellCentered, "anisotropic", "Neumann"),
    ("pole-pole", "volt", Simulation2DNodal, "isotropic", "Robin"),
    # ('pole-pole', 'volt', Simulation2DNodal, 'anisotropic', 'Neumann'),  # Would fail even for isotropy
]
TOLERANCE = 0.05


@pytest.mark.parametrize(
    "survey_type, data_type, formulation, model_type, bc_type", CASES_LIST_DATA
)
def test_numerical_solution_data(
    survey_type, data_type, formulation, model_type, bc_type, mesh, electrode_locations
):
    sighalf = 1e-2

    # Numerical solution
    survey = get_survey(survey_type, data_type, *electrode_locations)
    if data_type == "apparent_resistivity":
        survey.set_geometric_factor()

    if model_type == "isotropic":
        sigma = sighalf * np.ones(mesh.nC)
    else:
        sigma = sighalf * np.ones(3 * mesh.nC)
    sim = formulation(mesh, survey, sigma=sigma, bc_type=bc_type, solver=SOLVER)
    numeric_solution = sim.dpred()

    # Analytic solution
    if data_type == "apparent_resistivity":
        analytic_solution = np.ones_like(numeric_solution) / sighalf
    else:
        analytic_solution = get_analytic_solution(
            survey_type, sighalf, *electrode_locations
        )

    # Error
    err = np.sqrt(
        np.linalg.norm((numeric_solution - analytic_solution) / analytic_solution) ** 2
        / analytic_solution.size
    )

    assert err < TOLERANCE


CASES_LIST_FIELDS = [
    ("dipole-dipole", Simulation2DCellCentered, "isotropic", "Dirichlet"),
    ("dipole-dipole", Simulation2DNodal, "isotropic", "Neumann"),
]


@pytest.mark.parametrize(
    "survey_type, formulation, model_type, bc_type", CASES_LIST_FIELDS
)
def test_numerical_solution_fields(
    survey_type, formulation, model_type, bc_type, mesh, electrode_locations
):
    sighalf = 1e-2

    # Compute fields
    survey = get_survey(survey_type, "volt", *electrode_locations)
    if model_type == "isotropic":
        sigma = sighalf * np.ones(mesh.nC)
    else:
        sigma = sighalf * np.ones(3 * mesh.nC)

    sim = formulation(mesh, survey, sigma=sigma, bc_type=bc_type, solver=SOLVER)
    fields = sim.fields()

    for field_type in ["phi", "j", "e", "charge", "charge_density"]:
        try:
            temp = fields[:, field_type]
            if field_type == "phi":
                temp = temp[:, 0]
        except AttributeError:
            print("Could not extract {} from fields".format(field_type))
