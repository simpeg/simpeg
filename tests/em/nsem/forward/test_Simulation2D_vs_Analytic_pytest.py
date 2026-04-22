import pytest
from scipy.constants import mu_0
import numpy as np
from discretize import TensorMesh
from simpeg.electromagnetics import natural_source as nsem
from simpeg.utils import model_builder
from simpeg import maps

REL_TOLERANCE = 0.05
ABS_TOLERANCE = 1e-7


@pytest.fixture
def mesh():
    # Mesh for testing
    return TensorMesh(
        [
            [(40.0, 10, -1.4), (40.0, 50), (40.0, 10, 1.4)],
            [(40.0, 10, -1.4), (40.0, 50), (40.0, 10, 1.4)],
        ],
        "CC",
    )


@pytest.fixture
def mapping(mesh):
    return maps.IdentityMap(mesh)


def get_model(mesh, model_type):
    # Model used for testing
    model = 1e-8 * np.ones(mesh.nC)

    if mesh.dim == 1:
        model[mesh.cell_centers < 0.0] = 1e-2
        return model

    model[mesh.cell_centers[:, 1] < 0.0] = 1e-2

    if model_type == "layer":
        model[mesh.cell_centers[:, 1] < -500.0] = 1e-1
    elif model_type == "block":
        ind_block = model_builder.get_block_indices(
            mesh.cell_centers,
            np.array([-500, -800]),
            np.array([500, -400]),
        )
        model[ind_block] = 1e-1

    return model


@pytest.fixture
def locations():
    # Receiver locations
    elevation = 0.0
    rx_x = np.arange(-350, 350, 200)
    return np.c_[rx_x, elevation + np.zeros_like(rx_x)]


@pytest.fixture
def frequencies():
    # Frequencies being evaluated
    return [1e1, 2e1]


def get_survey(simulation_type, locations, frequencies, survey_type, component, orientation):
    source_list = []

    for f in frequencies:
        # MT data types (Zxy, Zyx)
        if survey_type == "impedance":
            rx_list = [
                nsem.receivers.Impedance(
                    locations_e=locations,
                    locations_h=locations,
                    orientation=orientation,
                    component=component,
                )
            ]

        if survey_type == "admittance":
            rx_list = [
                nsem.receivers.Admittance(
                    locations_e=locations,
                    locations_h=locations,
                    orientation=orientation,
                    component=component,
                )
            ]

        # ZTEM data types (Tzx, Tzy)
        elif survey_type == "tipper":
            rx_list = [
                nsem.receivers.Tipper(
                    locations_h=locations,
                    locations_base=locations,
                    orientation=orientation,
                    component=component,
                )
            ]

        if simulation_type == "PS":
            source_list.append(nsem.sources.Planewave(rx_list, f))
        else:
            source_list.append(nsem.sources.FictitiousSource(rx_list, f))

    return nsem.survey.Survey(source_list)


def get_analytic_halfspace_solution(sigma, f, survey_type, component, orientation):
    # MT data types (Zxy, Zyx)
    if survey_type == "impedance":
        if component in ["real", "imag"]:
            ampl = np.sqrt(np.pi * f * mu_0 / sigma)
            if orientation == "xy":
                return -ampl
            else:
                return ampl
        elif component == "app_res":
            return 1 / sigma
        elif component == "phase":
            if orientation == "xy":
                return -135.0
            else:
                return 45

    # Admittance data types (Yxy, Yyx)
    elif survey_type == "admittance":
        if component in ["real", "imag"]:
            ampl = 0.5 * np.sqrt(sigma / (np.pi * f * mu_0))
            if orientation == "yx":
                ampl *= -1
            if component == "imag":
                ampl *= -1
            return ampl

    # ZTEM data types (Tzx, Tzy)
    elif survey_type == "tipper":
        return 0.0


# Validate impedances, tippers and admittances against analytic
# solution for a halfspace.

CASES_LIST_HALFSPACE = [
    ("PS", "impedance", "real", "xy"),
    ("PS", "impedance", "real", "yx"),
    ("PS", "impedance", "imag", "xy"),
    ("PS", "impedance", "imag", "yx"),
    ("PS", "impedance", "app_res", "xy"),
    ("PS", "impedance", "app_res", "yx"),
    ("PS", "impedance", "phase", "xy"),
    ("PS", "impedance", "phase", "yx"),
    ("PS", "admittance", "real", "xy"),
    ("PS", "admittance", "real", "yx"),
    ("PS", "admittance", "imag", "xy"),
    ("PS", "admittance", "imag", "yx"),
    ("PS", "tipper", "real", "zx"),
    ("PS", "tipper", "imag", "zx"),
    ("FS", "impedance", "real", "xy"),
    ("FS", "impedance", "real", "yx"),
    ("FS", "impedance", "imag", "xy"),
    ("FS", "impedance", "imag", "yx"),
    ("FS", "impedance", "app_res", "xy"),
    ("FS", "impedance", "app_res", "yx"),
    ("FS", "impedance", "phase", "xy"),
    ("FS", "impedance", "phase", "yx"),
    ("FS", "admittance", "real", "xy"),
    ("FS", "admittance", "real", "yx"),
    ("FS", "admittance", "imag", "xy"),
    ("FS", "admittance", "imag", "yx"),
    ("FS", "tipper", "real", "zx"),
    ("FS", "tipper", "imag", "zx"),
]


@pytest.mark.parametrize("solution_type, survey_type, component, orientation", CASES_LIST_HALFSPACE)
def test_analytic_halfspace_solution(
    solution_type, survey_type, component, orientation, frequencies, locations, mesh, mapping
):
    # Numerical solution
    survey = get_survey(solution_type, locations, frequencies, survey_type, component, orientation)
    model_hs = get_model(mesh, "halfspace")  # 1e-2 halfspace
    if (orientation == "xy" and survey_type == "impedance") or (
        orientation == "yx" and survey_type == "admittance"
    ):
        if solution_type == "PS":
            sim = nsem.simulation.Simulation2DElectricField(
                mesh, survey=survey, sigmaMap=mapping
            )
        else:
            mesh_1d = TensorMesh([mesh.h[-1]], origin=[mesh.origin[-1]])
            sigma_1d = get_model(mesh_1d, "halfspace")
            sim = nsem.simulation.Simulation2DElectricFieldFictitious(
                mesh, survey=survey, sigmaMap=mapping, sigma_background=sigma_1d,
            )
    elif (
        (orientation == "yx" and survey_type == "impedance")
        or (orientation == "xy" and survey_type == "admittance")
        or (orientation == "zx" and survey_type == "tipper")
    ):
        if solution_type == "PS":
            sim = nsem.simulation.Simulation2DMagneticField(
                mesh, survey=survey, sigmaMap=mapping
            )
        else:
            mesh_1d = TensorMesh([mesh.h[-1]], origin=[mesh.origin[-1]])
            sigma_1d = get_model(mesh_1d, "halfspace")
            sim = nsem.simulation.Simulation2DMagneticFieldFictitious(
                mesh, survey=survey, sigmaMap=mapping, sigma_background=sigma_1d,
            )

    numeric_solution = sim.dpred(model_hs)

    # Analytic solution
    sigma_hs = 1e-2
    n_locations = np.shape(locations)[0]
    analytic_solution = np.hstack(
        [
            get_analytic_halfspace_solution(
                sigma_hs, f, survey_type, component, orientation
            )
            for f in frequencies
        ]
    )
    analytic_solution = np.repeat(analytic_solution, n_locations)

    # Error
    np.testing.assert_allclose(
        analytic_solution, numeric_solution, rtol=REL_TOLERANCE, atol=ABS_TOLERANCE
    )
