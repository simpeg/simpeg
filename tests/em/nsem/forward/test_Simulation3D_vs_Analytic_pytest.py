import pytest
from scipy.constants import mu_0
import numpy as np
from discretize import TensorMesh
from simpeg.electromagnetics import natural_source as nsem
from simpeg.utils import model_builder, mkvc
from simpeg import maps

REL_TOLERANCE_TEST_1 = 0.05
ABS_TOLERANCE_TEST_1 = 1e-13

REL_TOLERANCE_TEST_2 = 0.08
ABS_TOLERANCE_TEST_2 = 1e-13


@pytest.fixture
def mesh():
    # Mesh for testing
    return TensorMesh(
        [
            [(250, 10, -1.5), (250.0, 8), (250, 10, 1.5)],
            [(250, 10, -1.5), (250.0, 8), (250, 10, 1.5)],
            [(250, 10, -1.5), (250.0, 8), (250, 10, 1.5)],
        ],
        "CCC",
    )


@pytest.fixture
def mapping(mesh):
    return maps.IdentityMap(mesh)


def get_model(mesh, model_type):
    # Model used for testing

    if model_type == "halfspace":
        model = 1e-8 * np.ones(mesh.nC)
        model[mesh.cell_centers[:, 2] < 0.0] = 1e-2

    elif model_type == "layer":
        model = 1e-8 * np.ones(mesh.nC)
        model[mesh.cell_centers[:, 2] < 0.0] = 1e-2
        model[mesh.cell_centers[:, 2] < -3000.0] = 1e-1

    elif model_type == "block":
        model = 1e-8 * np.ones(mesh.nC)
        model[mesh.cell_centers[:, 2] < 0.0] = 1e-2

        ind_block = model_builder.get_indices_block(
            np.array([-500, -500, -1000]),
            np.array([500, 500, -500]),
            mesh.cell_centers,
        )
        model[ind_block] = 1e-1

    elif model_type == "1d":
        model = 1e-8 * np.ones(len(mesh.h[2]))
        model[mesh.cell_centers_z < 0.0] = 1e-2

    return model


@pytest.fixture
def locations():
    # Receiver locations
    elevation = 0.0
    rx_x, rx_y = np.meshgrid(np.arange(-300, 301, 200), np.arange(-300, 301, 200))
    return np.hstack(
        (mkvc(rx_x, 2), mkvc(rx_y, 2), elevation + np.zeros((np.prod(rx_x.shape), 1)))
    )


@pytest.fixture
def frequencies():
    # Frequencies being evaluated
    return [1e-1, 5e-1]


def get_survey(source_type, locations, frequencies, survey_type, component):
    source_list = []

    for f in frequencies:
        # MT data types (Zxx, Zxy, Zyx, Zyy)
        if survey_type == "impedance":
            if component == "phase":
                rx_list = [
                    nsem.receivers.Impedance(
                        locations_e=locations,
                        locations_h=locations,
                        orientation=ij,
                        component=component,
                    )
                    for ij in ["xy", "yx"]
                ]  # off-diagonal only!!!
            else:
                rx_list = [
                    nsem.receivers.Impedance(
                        locations_e=locations,
                        locations_h=locations,
                        orientation=ij,
                        component=component,
                    )
                    for ij in ["xx", "xy", "yx", "yy"]
                ]

        # ZTEM data types (Txx, Tyx, Tzx, Txy, Tyy, Tzy)
        elif survey_type == "tipper":
            rx_list = [
                nsem.receivers.Tipper(
                    locations=locations,
                    locations_bs=locations,
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
                    locations_e=locations,
                    locations_h=locations,
                )
            ]

        if source_type == "primary_secondary":
            source_list.append(nsem.sources.PlanewaveXYPrimary(rx_list, f))
        else:
            source_list.append(nsem.sources.FictitiousSource3D(rx_list, f))

    return nsem.survey.Survey(source_list)


def get_analytic_halfspace_solution(sigma, f, survey_type, component):
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


# Validate impedances, tippers and admittances against analytic
# solution for a halfspace.

CASES_LIST_HALFSPACE = [
    ("impedance", "real"),
    ("impedance", "imag"),
    ("impedance", "app_res"),
    ("impedance", "phase"),
    ("tipper", "real"),
    ("tipper", "imag"),
    ("admittance", "real"),
    ("admittance", "imag"),
    ("apparent_conductivity", None),
]


@pytest.mark.parametrize("survey_type, component", CASES_LIST_HALFSPACE)
def test_analytic_halfspace_solution(
    survey_type, component, frequencies, locations, mesh, mapping
):
    # Numerical solution
    survey = get_survey(
        "primary_secondary", locations, frequencies, survey_type, component
    )
    model_hs = get_model(mesh, "halfspace")  # 1e-2 halfspace
    sim = nsem.simulation.Simulation3DPrimarySecondary(
        mesh, survey=survey, sigmaPrimary=model_hs, sigmaMap=mapping
    )
    numeric_solution = sim.dpred(model_hs)

    # Analytic solution
    sigma_hs = 1e-2
    n_locations = np.shape(locations)[0]
    analytic_solution = np.hstack(
        [
            get_analytic_halfspace_solution(sigma_hs, f, survey_type, component)
            for f in frequencies
        ]
    )
    analytic_solution = np.repeat(analytic_solution, n_locations)

    # # Error
    err = np.abs(
        (numeric_solution - analytic_solution)
        / (analytic_solution + ABS_TOLERANCE_TEST_1)
    )

    assert np.all(err < REL_TOLERANCE_TEST_1)


CASES_LIST_CROSSCHECK = [
    ("impedance", "real"),
    ("impedance", "imag"),
    ("tipper", "real"),
    ("tipper", "imag"),
    ("admittance", "real"),
    ("admittance", "imag"),
]


@pytest.mark.parametrize("survey_type, component", CASES_LIST_CROSSCHECK)
def test_simulation_3d_crosscheck(
    survey_type, component, frequencies, locations, mesh, mapping
):
    # Numerical solution
    survey_ps = get_survey(
        "primary_secondary", locations, frequencies, survey_type, component
    )
    survey_1d = get_survey(
        "fictitious_source", locations, frequencies, survey_type, component
    )
    survey_3d = get_survey(
        "fictitious_source", locations, frequencies, survey_type, component
    )

    model_block = get_model(mesh, "block")
    model_hs = get_model(mesh, "halfspace")
    model_1d = get_model(mesh, "1d")

    sim_ps = nsem.simulation.Simulation3DPrimarySecondary(
        mesh, survey=survey_ps, sigmaPrimary=model_hs, sigmaMap=mapping
    )
    sim_1d = nsem.simulation.Simulation3DFictitiousSource(
        mesh, survey=survey_1d, sigma_background=model_1d, sigmaMap=mapping
    )
    sim_3d = nsem.simulation.Simulation3DFictitiousSource(
        mesh, survey=survey_3d, sigma_background=model_hs, sigmaMap=mapping
    )

    dpred_ps = sim_ps.dpred(model_block)
    dpred_1d = sim_1d.dpred(model_block)
    dpred_3d = sim_3d.dpred(model_block)

    print(np.c_[dpred_ps, dpred_1d, dpred_3d])

    # Error
    err_1 = np.abs((dpred_1d - dpred_ps) / (dpred_1d + ABS_TOLERANCE_TEST_2))
    err_2 = np.abs((dpred_1d - dpred_3d) / (dpred_1d + ABS_TOLERANCE_TEST_2))
    print(np.c_[err_1, err_2])

    assert (np.all(err_1 < REL_TOLERANCE_TEST_2)) & (np.all(err_2 < REL_TOLERANCE_TEST_2))
