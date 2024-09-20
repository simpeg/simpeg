import pytest
import numpy as np
from discretize import TensorMesh, tests
from simpeg import (
    maps,
    data_misfit,
)
from simpeg.utils import mkvc, model_builder
from simpeg.electromagnetics import natural_source as nsem

REL_TOLERANCE = 5e-2
ABS_TOLERANCE = 1e-4
ADJ_TOLERANCE = 1e-10


@pytest.fixture
def mesh():
    return TensorMesh(
        [
            [(200, 6, -1.5), (200.0, 4), (200, 6, 1.5)],
            [(200, 8, -1.5), (200.0, 8), (200, 8, 1.5)],
        ],
        "CC",
    )


@pytest.fixture
def active_cells(mesh):
    return mesh.cell_centers[:, 1] < 0.0


@pytest.fixture
def mapping(mesh, active_cells):
    return maps.InjectActiveCells(mesh, active_cells, 1e-8) * maps.ExpMap(
        nP=np.sum(active_cells)
    )


@pytest.fixture
def sigma_hs(mesh, active_cells):
    sigma_hs = 1e-8 * np.ones(mesh.nC)
    sigma_hs[active_cells] = 1e1
    return sigma_hs


@pytest.fixture
def locations():
    # Receiver locations
    elevation = 0.0
    rx_x = np.arange(-350, 350, 200)
    return np.c_[rx_x, elevation + np.zeros_like(rx_x)]


@pytest.fixture
def frequencies():
    # Frequencies being evaluated
    return [1e-1, 2e-1]


def get_survey(survey_type, orientation, components, locations, frequencies):

    if not isinstance(components, list):
        components = [components]

    source_list = []

    for f in frequencies:

        # MT data types (Zxy or Zyx)
        if survey_type == "impedance":
            rx_list = [
                nsem.receivers.Impedance(
                    locations,
                    orientation=orientation,
                    component=comp,
                )
                for comp in components
            ]

        # ZTEM data types (Tzx or Tzy)
        elif survey_type == "tipper":
            rx_list = [
                nsem.receivers.Tipper(
                    locations_h=locations,
                    locations_base=np.zeros_like(locations),
                    orientation=orientation,
                    component=comp,
                )
                for comp in components
            ]

        # Admittance data types (Yxy or Yyx)
        elif survey_type == "admittance":
            rx_list = [
                nsem.receivers.Admittance(
                    locations,
                    orientation=orientation,
                    component=comp,
                )
                for comp in components
            ]

        source_list.append(nsem.sources.PlanewaveXYPrimary(rx_list, f))

    return nsem.survey.Survey(source_list)


CASES_LIST = [
    ("impedance", "xy", ["real", "imag"]),
    ("impedance", "yx", ["real", "imag"]),
    ("impedance", "xy", ["app_res"]),
    ("impedance", "yx", ["app_res"]),
    ("impedance", "xy", ["phase"]),
    ("impedance", "yx", ["phase"]),
]


@pytest.mark.parametrize("survey_type, orientation, components", CASES_LIST)
class TestDerivatives:
    def get_setup_objects(
        self,
        survey_type,
        orientation,
        components,
        locations,
        frequencies,
        mesh,
        active_cells,
        mapping,
        sigma_hs,
    ):
        survey = get_survey(
            survey_type, orientation, components, locations, frequencies
        )

        # Define the simulation
        if orientation in ['xy', 'zy']:
            sim = nsem.simulation.Simulation2DElectricField(
                mesh, survey=survey, sigmaMap=mapping
            )
        elif orientation in ['yx', 'zx']:
            sim = nsem.simulation.Simulation2DMagneticField(
                mesh, survey=survey, sigmaMap=mapping
            )

        n_active = np.sum(active_cells)
        np.random.seed(1983)

        # Model
        m0 = np.log(1e1) * np.ones(n_active)
        ind = model_builder.get_indices_block(
            np.r_[-200.0, -600.0],
            np.r_[200.0, -200.0],
            mesh.cell_centers[active_cells, :],
        )
        m0[ind] = np.log(1e0)
        m0 += 0.01 * np.random.uniform(low=-1, high=1, size=n_active)

        # Define data and misfit
        data = sim.make_synthetic_data(m0, add_noise=True)
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=data)

        return m0, dmis

    def test_misfit(
        self,
        survey_type,
        orientation,
        components,
        locations,
        frequencies,
        mesh,
        active_cells,
        mapping,
        sigma_hs,
    ):
        m0, dmis = self.get_setup_objects(
            survey_type,
            orientation,
            components,
            locations,
            frequencies,
            mesh,
            active_cells,
            mapping,
            sigma_hs,
        )
        sim = dmis.simulation

        passed = tests.check_derivative(
            lambda m: (sim.dpred(m), lambda mx: sim.Jvec(m0, mx)),
            m0,
            plotIt=False,
            num=3,
        )

        assert passed

    def test_adjoint(
        self,
        survey_type,
        orientation,
        components,
        locations,
        frequencies,
        mesh,
        active_cells,
        mapping,
        sigma_hs,
    ):
        m0, dmis = self.get_setup_objects(
            survey_type,
            orientation,
            components,
            locations,
            frequencies,
            mesh,
            active_cells,
            mapping,
            sigma_hs,
        )
        sim = dmis.simulation
        n_data = sim.survey.nD

        f = sim.fields(m0)
        tests.assert_isadjoint(
            lambda u: sim.Jvec(m0, u, f=f),
            lambda v: sim.Jtvec(m0, v, f=f),
            m0.shape,
            (n_data,),
            rtol=ADJ_TOLERANCE,
        )
