import pytest
from scipy.constants import mu_0
import numpy as np
from discretize import TensorMesh
from simpeg.electromagnetics import natural_source as nsem
from simpeg.electromagnetics.natural_source.utils.test_utils import (
    PlanewaveXYPrimaryDeprecated,
)
from simpeg.utils import model_builder, mkvc, get_default_solver
from simpeg import maps

REL_TOLERANCE = 0.05
ABS_TOLERANCE = 1e-9
REL_TOLERANCE_2 = 0.1
ABS_TOLERANCE_2 = 1e-7


@pytest.fixture
def mesh():
    # Mesh for testing
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
    return maps.IdentityMap(mesh)


def get_model(mesh, model_type):
    # Model used for testing
    model = 1e-8 * np.ones(mesh.nC)
    
    if mesh.dim == 1:
        model[mesh.cell_centers < 0.0] = 1e-2
        return model

    model[mesh.cell_centers[:, 2] < 0.0] = 1e-2

    if model_type == "layer":
        model[mesh.cell_centers[:, 2] < -3000.0] = 1e-1
    elif model_type == "block":
        ind_block = model_builder.get_indices_block(
            np.array([-200, -200, -800]),
            np.array([200, 200, -400]),
            mesh.cell_centers,
        )
        model[ind_block] = 1e-1
        # pass

    return model


@pytest.fixture
def locations():
    # Receiver locations
    elevation = 0.0
    rx_x, rx_y = np.meshgrid(np.arange(-350, 350, 200), np.arange(-350, 350, 200))
    return np.hstack(
        (mkvc(rx_x, 2), mkvc(rx_y, 2), elevation + np.zeros((np.prod(rx_x.shape), 1)))
    )


@pytest.fixture
def frequencies():
    # Frequencies being evaluated
    return [1e-1, 2e-1]


def get_survey(
    source_type, locations, frequencies, survey_type, component
):
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
            rx_list = [nsem.receivers.ApparentConductivity(locations)]


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
    ("primary_secondary", "impedance", "real"),
    ("primary_secondary", "impedance", "imag"),
    ("primary_secondary", "impedance", "app_res"),
    ("primary_secondary", "impedance", "phase"),
    ("primary_secondary", "tipper", "real"),
    ("primary_secondary", "tipper", "imag"),
    ("primary_secondary", "admittance", "real"),
    ("primary_secondary", "admittance", "imag"),
    ("primary_secondary", "apparent_conductivity", None),
    ("fictitious_source", "impedance", "real"),
    ("fictitious_source", "impedance", "imag"),
    ("fictitious_source", "tipper", "real"),
    ("fictitious_source", "tipper", "imag"),
    ("fictitious_source", "admittance", "real"),
    ("fictitious_source", "admittance", "imag"),
    ("fictitious_source", "apparent_conductivity", None),
]


@pytest.mark.parametrize("source_type, survey_type, component", CASES_LIST_HALFSPACE)
def test_analytic_halfspace_solution(
    source_type, survey_type, component, frequencies, locations, mesh, mapping
):
    # Numerical solution
    survey = get_survey(source_type, locations, frequencies, survey_type, component)
    model_hs = get_model(mesh, "halfspace")  # 1e-2 halfspace
    mesh_1d = TensorMesh([mesh.h[-1]], origin=[mesh.origin[-1]])
    model_1d = get_model(mesh_1d, "halfspace")

    if source_type == "primary_secondary":
        sim = nsem.simulation.Simulation3DPrimarySecondary(
            mesh, survey=survey, sigmaPrimary=model_hs, sigmaMap=mapping, solver=get_default_solver()
        )
    else:
        sim = nsem.simulation.Simulation3DFictitiousSource(
            mesh, survey=survey, sigma_background=model_1d, sigmaMap=mapping, solver=get_default_solver()
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

    np.testing.assert_allclose(
        analytic_solution, numeric_solution, rtol=REL_TOLERANCE, atol=ABS_TOLERANCE
    )


CASES_LIST_CROSSCHECK = [
    ("impedance", "real"),
    ("impedance", "imag"),
    ("tipper", "real"),
    ("tipper", "imag"),
    ("admittance", "real"),
    ("admittance", "imag"),
    ("apparent_conductivity", None),
]


# PRIMARY-SECONDARY DOESN'T SEEM TO WORK UNLESS THE PADDING IS EXTREME.
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
    # survey_3d = get_survey(
    #     "fictitious_source", locations, frequencies, survey_type, component
    # )

    model_block = get_model(mesh, "block")
    model_hs = get_model(mesh, "halfspace")
    model_1d = get_model(mesh, "1d")

    sim_ps = nsem.simulation.Simulation3DPrimarySecondary(
        mesh, survey=survey_ps, sigmaPrimary=model_hs, sigmaMap=mapping, solver=get_default_solver(),
    )
    sim_1d = nsem.simulation.Simulation3DFictitiousSource(
        mesh, survey=survey_1d, sigma_background=model_1d, sigmaMap=mapping, solver=get_default_solver(),
    )
    # sim_3d = nsem.simulation.Simulation3DFictitiousSource(
    #     mesh, survey=survey_3d, sigma_background=model_hs, sigmaMap=mapping, solver=get_default_solver(),
    # )

    dpred_ps = sim_ps.dpred(model_block)
    dpred_1d = sim_1d.dpred(model_block)
    # dpred_3d = sim_3d.dpred(model_block)

    np.testing.assert_allclose(
        dpred_ps, dpred_1d, rtol=REL_TOLERANCE_2, atol=ABS_TOLERANCE_2
    )

