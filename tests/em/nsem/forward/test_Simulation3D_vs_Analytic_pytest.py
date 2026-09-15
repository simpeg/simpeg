import pytest
from scipy.constants import mu_0
import numpy as np
from discretize import TensorMesh
from simpeg.electromagnetics import natural_source as nsem
from simpeg.utils import model_builder, mkvc, get_default_solver
from simpeg import maps

REL_TOLERANCE = 0.05
ABS_TOLERANCE = 1e-9
REL_TOLERANCE_2 = 0.1
ABS_TOLERANCE_2 = 1e-7


@pytest.fixture
def mesh():
    """Get test mesh."""
    mesh = TensorMesh(
        [
            [(200, 10, -1.5), (200.0, 6), (200, 10, 1.5)],
            [(200, 10, -1.5), (200.0, 6), (200, 10, 1.5)],
            [(200, 10, -1.5), (200.0, 10), (200, 10, 1.5)],
        ],
        "CCC",
    )
    mesh.origin[-1] -= 200.0
    return mesh


@pytest.fixture
def mapping(mesh):
    """Get test mapping."""
    return maps.IdentityMap(mesh)


def get_model(mesh, model_type):
    """Get test model."""
    model = 1e-8 * np.ones(mesh.nC)

    if mesh.dim == 1:
        model[mesh.cell_centers < 0.0] = 1e-2
        return model

    model[mesh.cell_centers[:, 2] < 0.0] = 1e-2

    if model_type == "layer":
        model[mesh.cell_centers[:, 2] < -3000.0] = 1e-1
    elif model_type == "block":
        ind_block = model_builder.get_indices_block(
            np.array([-200, -200, -200]),
            np.array([200, 200, -600]),
            mesh.cell_centers,
        )
        model[ind_block] = 1e-1
        # pass

    return model


@pytest.fixture
def locations():
    """Get test locations."""
    elevation = 0.0
    v = np.r_[-350.0, -150.0, 150.0, 350.0]  # needs to be symmetric
    rx_x, rx_y = np.meshgrid(v, v)
    return np.hstack(
        (mkvc(rx_x, 2), mkvc(rx_y, 2), elevation + np.zeros((np.prod(rx_x.shape), 1)))
    )


@pytest.fixture
def frequencies():
    """Get test frequencies."""
    return [1e-1, 2e-1]


def get_survey(locations, frequencies, survey_type, component, base_type):
    """Get test survey."""
    source_list = []

    for f in frequencies:
        # MT data types (Zxx, Zxy, Zyx, Zyy)
        if survey_type == "impedance":
            if component == "phase":
                orientations = ["xy", "yx"]  # off-diagonal only!!!
            else:
                orientations = ["xx", "xy", "yx", "yy"]
            rx_list = [
                nsem.receivers.Impedance(
                    locations_e=locations,
                    locations_h=locations,
                    orientation=ij,
                    component=component,
                )
                for ij in orientations
            ]

        # ZTEM data types (Txx, Tyx, Tzx, Txy, Tyy, Tzy)
        elif survey_type == "tipper":
            rx_list = [
                nsem.receivers.Tipper(
                    locations_h=locations,
                    locations_base=locations,
                    orientation=ij,
                    component=component,
                )
                for ij in ["xx", "yx", "zx", "xy", "yy", "zy"]
            ]

        # Admittance data types (Yxx, Yyx, Yzx, Yxy, Yyy, Yzy)
        elif survey_type == "admittance":
            rx_list = [
                nsem.receivers.Admittance(
                    locations_e=locations,
                    locations_h=locations,
                    orientation=ij,
                    component=component,
                )
                for ij in ["xx", "yx", "zx", "xy", "yy", "zy"]
            ]

        elif survey_type == "apparent_conductivity":
            rx_list = [
                nsem.receivers.ApparentConductivity(
                    locations_h=locations,
                    locations_e=locations,
                    component=component,
                )
            ]

        elif survey_type == "det_horizontal":
            rx_list = [
                nsem.receivers.HorizontalDeterminant(
                    locations_h=locations,
                    locations_base=locations,
                    base_type=base_type,
                    component=component,
                )
            ]

        elif survey_type == "gram_amp":
            rx_list = [
                nsem.receivers.RootGramDeterminant(
                    locations_h=locations,
                    locations_base=locations,
                    base_type=base_type,
                )
            ]

        elif survey_type == "cross_amp":
            rx_list = [
                nsem.receivers.CrossProductAmplitude(
                    locations_h=locations,
                    locations_base=locations,
                    base_type=base_type,
                )
            ]

        source_list.append(nsem.sources.PlanewaveXYPrimary(rx_list, f))

    return nsem.survey.Survey(source_list)


def get_analytic_halfspace_solution(sigma, f, survey_type, component, base_type):
    """Get analytic halfpsace solution."""
    # MT data types (Zxx, Zxy, Zyx, Zyy)
    if survey_type == "impedance":
        if component in ["real", "imag"]:
            ampl = np.sqrt(np.pi * f * mu_0 / sigma)
            return np.r_[0.0, -ampl, ampl, 0.0]
        elif component == "app_res":
            return np.r_[0.0, 1 / sigma, 1 / sigma, 0.0]
        elif component == "phase":
            return np.r_[-135.0, 45.0]  # off-diagonal only!

    # ZTEM data types (Txx, Tyx, Tzx, Txy, Tyy, Tzy)
    elif survey_type == "tipper":
        if component == "real":
            return np.r_[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        else:
            return np.r_[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Admittance data types (Yxx, Yyx, Yzx, Yxy, Yyy, Yzy)
    elif survey_type == "admittance":
        ampl = 0.5 * np.sqrt(sigma / (np.pi * f * mu_0))
        if component == "real":
            return np.r_[0.0, -ampl, 0.0, ampl, 0.0, 0.0]
        else:
            return np.r_[0.0, ampl, 0.0, -ampl, 0.0, 0.0]

    # MobileMT data type (app_cond)
    elif survey_type == "apparent_conductivity":
        return sigma

    elif survey_type == "det_horizontal":
        if base_type == "magnetic":
            if component == "imag":
                return np.r_[0.0]
            else:
                return np.r_[1.0]
        elif base_type == "electric":
            if component == "real":
                return (
                    0.025 * sigma / (2 * np.pi * f * mu_0)
                )  # np.r_[0.] <-- analytically
            elif component == "imag":
                return -sigma / (2 * np.pi * f * mu_0)
            else:
                return sigma / (2 * np.pi * f * mu_0)

    elif survey_type == "gram_amp":
        if base_type == "magnetic":
            return np.r_[1.0]
        elif base_type == "electric":
            return sigma / (2 * np.pi * f * mu_0)

    elif survey_type == "cross_amp":
        if base_type == "magnetic":
            return np.r_[1.0]
        elif base_type == "electric":
            return sigma / (2 * np.pi * f * mu_0)


# Validate impedances, tippers and admittances against analytic
# solution for a halfspace.

CASES_LIST_HALFSPACE = [
    ("impedance", "real", None),
    ("impedance", "imag", None),
    ("tipper", "real", None),
    ("tipper", "imag", None),
    ("admittance", "real", None),
    ("admittance", "imag", None),
    ("det_horizontal", "real", "magnetic"),
    ("det_horizontal", "imag", "magnetic"),
    ("det_horizontal", "amp", "magnetic"),
    ("det_horizontal", "real", "electric"),
    ("det_horizontal", "imag", "electric"),
    ("det_horizontal", "amp", "electric"),
    ("cross_amp", None, "electric"),
    ("gram_amp", None, "magnetic"),
    ("cross_amp", None, "electric"),
    ("gram_amp", None, "magnetic"),
]


@pytest.mark.parametrize("survey_type, component, base_type", CASES_LIST_HALFSPACE)
def test_analytic_halfspace_solution(
    survey_type, component, base_type, frequencies, locations, mesh, mapping
):
    """Analytic halfspace solution tests."""
    # Numerical solution
    survey = get_survey(locations, frequencies, survey_type, component, base_type)
    model_hs = get_model(mesh, "halfspace")  # 1e-2 halfspace
    sim = nsem.simulation.Simulation3DPrimarySecondary(
        mesh,
        survey=survey,
        sigmaPrimary=model_hs,
        sigmaMap=mapping,
        solver=get_default_solver(),
    )
    numeric_solution = sim.dpred(model_hs)

    # Analytic solution
    sigma_hs = 1e-2
    n_locations = np.shape(locations)[0]
    analytic_solution = np.hstack(
        [
            get_analytic_halfspace_solution(
                sigma_hs, f, survey_type, component, base_type
            )
            for f in frequencies
        ]
    )
    analytic_solution = np.repeat(analytic_solution, n_locations)

    if (
        (survey_type == "det_horizontal")
        & (component == "real")
        & (base_type == "electric")
    ):
        np.testing.assert_array_less(
            np.abs(numeric_solution), np.abs(analytic_solution)
        )
    else:
        np.testing.assert_allclose(
            analytic_solution, numeric_solution, rtol=REL_TOLERANCE, atol=ABS_TOLERANCE
        )


CASES_LIST_CROSSCHECK = [
    ("gram_amp", "root_gram_determinant"),
    ("cross_amp", "cross_product_amplitude"),
    ("det_horizontal", "horizontal_determinant"),
]


# PRIMARY-SECONDARY DOESN'T SEEM TO WORK UNLESS THE PADDING IS EXTREME.
@pytest.mark.parametrize("survey_type, component", CASES_LIST_CROSSCHECK)
def test_apparent_conductivity_crosscheck(
    survey_type, component, frequencies, locations, mesh, mapping
):
    """Cross check test for apparent conductivity data."""
    # Numerical solution
    survey_1 = get_survey(
        locations,
        frequencies,
        "apparent_conductivity",
        component,
        None,
    )
    survey_2 = get_survey(locations, frequencies, survey_type, "amp", "electric")

    model_block = get_model(mesh, "block")
    model_hs = get_model(mesh, "halfspace")  # 1e-2 halfspace

    sim_1 = nsem.simulation.Simulation3DPrimarySecondary(
        mesh,
        survey=survey_1,
        sigmaPrimary=model_hs,
        sigmaMap=mapping,
        solver=get_default_solver(),
    )
    sim_2 = nsem.simulation.Simulation3DPrimarySecondary(
        mesh,
        survey=survey_2,
        sigmaPrimary=model_hs,
        sigmaMap=mapping,
        solver=get_default_solver(),
    )

    dpred_1 = sim_1.dpred(model_block)

    alpha = 2 * np.pi * np.kron(frequencies, np.ones(len(locations))) * mu_0
    dpred_2 = alpha * sim_2.dpred(model_block)

    np.testing.assert_allclose(
        dpred_1, dpred_2, rtol=REL_TOLERANCE_2, atol=ABS_TOLERANCE_2
    )


def test_symmetry_for_appcon(frequencies, locations, mesh, mapping):
    """Test the app con is symmetric across the y-axis."""
    # Numerical solution
    survey = get_survey(
        locations,
        frequencies,
        "apparent_conductivity",
        "cross_product_amplitude",
        None,
    )
    model_hs = get_model(mesh, "halfspace")  # 1e-2 halfspace
    model_block = get_model(mesh, "block")
    sim = nsem.simulation.Simulation3DPrimarySecondary(
        mesh,
        survey=survey,
        sigmaPrimary=model_hs,
        sigmaMap=mapping,
        solver=get_default_solver(),
    )
    solution = sim.dpred(model_block)

    n_pt = int(np.sqrt(np.shape(locations)[0]))
    n_freq = len(frequencies)

    solution = solution.reshape((n_freq, n_pt, n_pt))
    solution_flipped = np.flip(solution, axis=-1)

    # Error
    np.testing.assert_allclose(solution, solution_flipped, atol=ABS_TOLERANCE)
