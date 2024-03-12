import numpy as np
import pytest

from discretize import TensorMesh

from SimPEG import utils, SolverLU
from SimPEG.electromagnetics import resistivity as dc
from SimPEG.electromagnetics.static.resistivity.simulation import (
    Simulation3DCellCentered,
    Simulation3DNodal,
)
from SimPEG.electromagnetics import analytics

try:
    from pymatsolver import Pardiso

    SOLVER = Pardiso
except ImportError:
    SOLVER = SolverLU


def get_mesh(earth_type):
    if earth_type == "halfspace":
        cs = 25.0
        npad = 10
        hx = [(cs, npad, -1.3), (cs, 21), (cs, npad, 1.3)]
        hy = [(cs, npad, -1.3), (cs, 21), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, 20)]
        mesh = TensorMesh([hx, hy, hz], x0="CCN")
    elif earth_type == "wholespace":
        cs = 25.0
        npad = 10
        hx = [(cs, npad, -1.3), (cs, 21), (cs, npad, 1.3)]
        hy = [(cs, npad, -1.3), (cs, 21), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, 20), (cs, npad, -1.3)]
        mesh = TensorMesh([hx, hy, hz], x0="CCC")

    return mesh


def get_electrode_locations(mesh):
    x = mesh.cell_centers_x[
        (mesh.cell_centers_x > -125.0) & (mesh.cell_centers_x < 125.0)
    ]
    y = mesh.cell_centers_y[
        (mesh.cell_centers_y > -125.0) & (mesh.cell_centers_y < 125.0)
    ]
    M = utils.ndgrid(x - 25.0, y, np.r_[0.0])
    N = utils.ndgrid(x + 25.0, y, np.r_[0.0])

    A = np.r_[-200, 0.0, 0.0]  # xyz electrode locations
    B = np.r_[200, 0.0, 0.0]  # xyz electrode locations

    return A, B, M, N


def get_analytic_solution(survey_type, earth_type, sigma_val, A, B, M, N):
    # Tensor mesh, survey type, halfspace conductivity

    if survey_type == "dipole-dipole":
        data_ana_A = analytics.DCAnalytic_Pole_Dipole(
            A, [M, N], sigma_val, earth_type=earth_type
        )
        data_ana_B = analytics.DCAnalytic_Pole_Dipole(
            B, [M, N], sigma_val, earth_type=earth_type
        )
        return data_ana_A - data_ana_B
    elif survey_type == "pole-dipole":
        return analytics.DCAnalytic_Pole_Dipole(
            A, [M, N], sigma_val, earth_type=earth_type
        )
    elif survey_type == "dipole-pole":
        return analytics.DCAnalytic_Dipole_Pole(
            [A, B], M, sigma_val, earth_type=earth_type
        )
    elif survey_type == "pole-pole":
        return analytics.DCAnalytic_Pole_Pole(A, M, sigma_val, earth_type=earth_type)
    else:
        raise ValueError(
            "survey_type not recognized. {'pole-pole', 'pole-dipole', 'dipole-pole', 'dipole-dipole'}"
        )


def get_survey(survey_type, data_type, A, B, M, N):
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


# APPARENT RESISTIVITY CASES DON'T WORK!!!
CASES_LIST_DATA = [
    (
        Simulation3DCellCentered,
        "Neumann",
        "dipole-dipole",
        "volt",
        "halfspace",
        "isotropic",
    ),
    (
        Simulation3DCellCentered,
        "Neumann",
        "dipole-dipole",
        "volt",
        "halfspace",
        "anisotropic",
    ),
    (Simulation3DCellCentered, "Mixed", "pole-pole", "volt", "halfspace", "isotropic"),
    (
        Simulation3DCellCentered,
        "Robin",
        "dipole-dipole",
        "volt",
        "halfspace",
        "isotropic",
    ),
    (
        Simulation3DCellCentered,
        "Dirichlet",
        "dipole-dipole",
        "volt",
        "wholespace",
        "isotropic",
    ),
    (
        Simulation3DCellCentered,
        "Dirichlet",
        "dipole-dipole",
        "volt",
        "wholespace",
        "anisotropic",
    ),
    (Simulation3DNodal, "Neumann", "dipole-dipole", "volt", "halfspace", "isotropic"),
    (Simulation3DNodal, "Neumann", "dipole-dipole", "volt", "halfspace", "anisotropic"),
    (Simulation3DNodal, "Robin", "dipole-dipole", "volt", "halfspace", "isotropic"),
]
TOLERANCE = 0.05


@pytest.mark.parametrize(
    "formulation, bc_type, survey_type, data_type, earth_type, model_type",
    CASES_LIST_DATA,
)
def test_numerical_solution_data(
    formulation, bc_type, survey_type, data_type, earth_type, model_type
):
    mesh = get_mesh(earth_type)
    A, B, M, N = get_electrode_locations(mesh)
    survey = get_survey(survey_type, data_type, A, B, M, N)
    if data_type == "apparent_resistivity":
        survey.set_geometric_factor(space_type=earth_type)

    sigma_val = 1e-2
    if model_type == "isotropic":
        sigma = sigma_val * np.ones(mesh.nC)
    else:
        sigma = sigma_val * np.ones(3 * mesh.nC)
    sim = formulation(mesh, survey, sigma=sigma, bc_type=bc_type, solver=SOLVER)
    numeric_solution = sim.dpred()

    # Analytic solution
    if data_type == "apparent_resistivity":
        analytic_solution = np.ones_like(numeric_solution) / sigma_val
    else:
        analytic_solution = get_analytic_solution(
            survey_type, earth_type, sigma_val, A, B, M, N
        )

    # Error
    err = np.sqrt(
        np.linalg.norm((numeric_solution - analytic_solution) / analytic_solution) ** 2
        / analytic_solution.size
    )

    assert err < TOLERANCE
