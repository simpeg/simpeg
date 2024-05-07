import pytest
import numpy as np
from discretize import TensorMesh
from simpeg import (
    maps,
    data_misfit,
    tests,
)
from simpeg.utils import mkvc
from simpeg.electromagnetics import natural_source as nsem

REL_TOLERANCE = 5e-2
ABS_TOLERANCE = 1e-4
ADJ_TOLERANCE = 1e-10


@pytest.fixture
def mesh():
    return TensorMesh(
        [
            [(200, 6, -1.5), (200.0, 4), (200, 6, 1.5)],
            [(200, 6, -1.5), (200.0, 4), (200, 6, 1.5)],
            [(200, 8, -1.5), (200.0, 8), (200, 8, 1.5)],
        ],
        "CCC",
    )


@pytest.fixture
def active_cells(mesh):
    return mesh.cell_centers[:, 2] < 0.0


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
    rx_x, rx_y = np.meshgrid(np.arange(-350, 350, 200), np.arange(-350, 350, 200))
    return np.hstack(
        (mkvc(rx_x, 2), mkvc(rx_y, 2), elevation + np.zeros((np.prod(rx_x.shape), 1)))
    )


@pytest.fixture
def frequencies():
    # Frequencies being evaluated
    return [1e-1, 2e-1]


def get_survey(survey_type, orientations, components, locations, frequencies):
    if not isinstance(orientations, list):
        orientations = [orientations]

    if not isinstance(components, list):
        components = [components]

    source_list = []

    for f in frequencies:
        rx_list = []

        # MT data types (Zxx, Zxy, Zyx, Zyy)
        if survey_type == "impedance":
            for orient in orientations:
                rx_list.extend(
                    [
                        nsem.receivers.PointImpedance(
                            locations_e=locations,
                            locations_h=locations,
                            orientation=orient,
                            component=comp,
                        )
                        for comp in components
                    ]
                )

        # ZTEM data types (Txx, Tyx, Tzx, Txy, Tyy, Tzy)
        elif survey_type == "tipper":
            for orient in orientations:
                rx_list.extend(
                    [
                        nsem.receivers.Point3DTipper(
                            locations=locations,
                            locations_bs=locations,
                            orientation=orient,
                            component=comp,
                        )
                        for comp in components
                    ]
                )

        # Admittance data types (Yxx, Yyx, Yzx, Yxy, Yyy, Yzy)
        elif survey_type == "admittance":
            for orient in orientations:
                rx_list.extend(
                    [
                        nsem.receivers.Point3DAdmittance(
                            locations_e=locations,
                            locations_h=locations,
                            orientation=orient,
                            component=comp,
                        )
                        for comp in components
                    ]
                )

        # MobileMT is app_cond
        elif survey_type == "mobilemt":
            rx_list.extend(
                [
                    nsem.receivers.Point3DMobileMT(
                        locations_e=locations, locations_h=locations
                    )
                ]
            )

        source_list.append(nsem.sources.PlanewaveXYPrimary(rx_list, f))

    return nsem.survey.Survey(source_list)


CASES_LIST = [
    ("impedance", ["xy", "yx"], ["real", "imag"]),
    ("impedance", ["xx", "yy"], ["real", "imag"]),
    ("impedance", ["xy", "yx"], ["app_res"]),
    ("impedance", ["xx", "yy"], ["app_res"]),
    ("impedance", ["xy", "yx"], ["phase"]),
    ("tipper", ["zx", "zy"], ["real", "imag"]),
    ("tipper", ["xx", "yx", "xy", "yy"], ["real", "imag"]),
    ("admittance", ["xy", "yx"], ["real", "imag"]),
    ("admittance", ["xx", "yy"], ["real", "imag"]),
    ("admittance", ["zx", "zy"], ["real", "imag"]),
    ("mobilemt", None, None),
]


@pytest.mark.parametrize("survey_type, orientations, components", CASES_LIST)
class TestDerivatives:
    def get_setup_objects(
        self,
        survey_type,
        orientations,
        components,
        locations,
        frequencies,
        mesh,
        active_cells,
        mapping,
        sigma_hs,
    ):
        survey = get_survey(
            survey_type, orientations, components, locations, frequencies
        )

        # Define the simulation
        sim = nsem.simulation.Simulation3DPrimarySecondary(
            mesh, survey=survey, sigmaMap=mapping, sigmaPrimary=sigma_hs
        )

        n_active = np.sum(active_cells)
        np.random.seed(1983)

        m0 = np.log(1e1) + 0.01 * np.random.uniform(low=-1, high=1, size=n_active)
        data = sim.make_synthetic_data(m0, add_noise=True)
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=data)

        return m0, dmis

    def test_misfit(
        self,
        survey_type,
        orientations,
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
            orientations,
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
        orientations,
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
            orientations,
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

        rng = np.random.default_rng(seed=41)
        v = rng.random(len(m0))
        w = rng.random(n_data)
        wtJv = w.dot(sim.Jvec(m0, v))
        vtJtw = v.dot(sim.Jtvec(m0, w))
        passed = np.abs(wtJv - vtJtw) < ADJ_TOLERANCE
        print("Adjoint Test", np.abs(wtJv - vtJtw), passed)
        assert passed
