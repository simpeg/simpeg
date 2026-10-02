"""Derivative and adjoint tests for 3D simulations."""

import pytest
import numpy as np
from discretize import TensorMesh, tests
from simpeg import maps, data_misfit
from simpeg.utils import mkvc, model_builder, get_default_solver
from simpeg.electromagnetics import natural_source as nsem

ADJ_RTOL = 1e-8
ADJ_ATOL = 1e-11


@pytest.fixture
def mesh():
    """Return test mesh."""
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
    """Return active cells."""
    return mesh.cell_centers[:, 2] < 0.0


@pytest.fixture
def mapping(mesh, active_cells):
    """Return mapping."""
    return maps.InjectActiveCells(mesh, active_cells, 1e-8) * maps.ExpMap(
        nP=np.sum(active_cells)
    )


@pytest.fixture
def locations():
    """Return receiver locations."""
    elevation = 0.0
    rx_x, rx_y = np.meshgrid(np.arange(-350, 350, 200), np.arange(-350, 350, 200))
    return np.hstack(
        (mkvc(rx_x, 2), mkvc(rx_y, 2), elevation + np.zeros((np.prod(rx_x.shape), 1)))
    )


@pytest.fixture
def frequencies():
    """Return frequencies."""
    return [1e-1, 2e-1]


@pytest.fixture
def sigma_hs(mesh, active_cells):
    """Return background conductivity."""
    sigma_hs = 1e-8 * np.ones(mesh.nC)
    sigma_hs[active_cells] = 1e1
    return sigma_hs


def get_survey(survey_type, orientations, components, locations, frequencies):
    """Return test survey."""
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
                        nsem.receivers.Impedance(
                            locations,
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
                        nsem.receivers.Tipper(
                            locations_h=locations,
                            locations_base=np.zeros_like(locations),
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
                        nsem.receivers.Admittance(
                            locations_h=locations,
                            locations_e=np.zeros_like(locations),
                            orientation=orient,
                            component=comp,
                        )
                        for comp in components
                    ]
                )

        # MobileMT is app_cond
        elif survey_type == "apparent_conductivity":
            rx_list.extend(
                [
                    nsem.receivers.ApparentConductivity(
                        locations_h=locations,
                        locations_e=np.zeros_like(locations),
                        component=components[0],
                    )
                ]
            )

        # Horizontal determinanet
        elif survey_type == "det_horizontal":
            rx_list = [
                nsem.receivers.HorizontalDeterminant(
                    locations_h=locations,
                    locations_base=np.zeros_like(locations),
                    base_type=orientations[0],
                    component=comp,
                )
                for comp in components
            ]

        # Determinant ampliutde of the Gram matrix
        elif survey_type == "gram_amp":
            rx_list = [
                nsem.receivers.RootGramDeterminant(
                    locations,
                    locations_base=np.zeros_like(locations),
                    base_type=orientations[0],
                )
            ]

        # Determinant amplitude of the cross product
        elif survey_type == "cross_amp":
            rx_list = [
                nsem.receivers.CrossProductAmplitude(
                    locations,
                    locations_base=np.zeros_like(locations),
                    base_type=orientations[0],
                )
            ]

        source_list.append(nsem.sources.PlanewaveXYPrimary(rx_list, f))

    return nsem.survey.Survey(source_list)


CASES_LIST = [
    ("impedance", ["xy", "yx"], ["real", "imag"]),
    ("impedance", ["xx", "yy"], ["real", "imag"]),
    ("impedance", ["xy", "yx"], ["app_res"]),
    ("impedance", ["xx", "yy"], ["app_res"]),
    ("impedance", ["xy", "yx"], ["phase"]),
    ("tipper", ["zx", "zy"], ["real", "imag"]),
    ("tipper", ["xx", "yy"], ["real", "imag"]),
    ("tipper", ["xy", "yx"], ["real", "imag"]),
    ("admittance", ["xy", "yx"], ["real", "imag"]),
    ("admittance", ["xx", "yy"], ["real", "imag"]),
    ("admittance", ["zx", "zy"], ["real", "imag"]),
    ("det_horizontal", "electric", ["real", "imag", "amp"]),
    ("det_horizontal", "magnetic", ["real", "imag", "amp"]),
    ("apparent_conductivity", None, "root_gram_determinant"),
    ("apparent_conductivity", None, "cross_product_amplitude"),
    ("apparent_conductivity", None, "horizontal_determinant"),
]


@pytest.mark.parametrize("survey_type, orientations, components", CASES_LIST)
class TestDerivatives:
    """Perform derivative convergence test."""

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
        """Get test objections."""
        survey = get_survey(
            survey_type, orientations, components, locations, frequencies
        )

        # Define the simulation
        sim = nsem.simulation.Simulation3DPrimarySecondary(
            mesh,
            survey=survey,
            sigmaMap=mapping,
            sigmaPrimary=sigma_hs,
            solver=get_default_solver(),
        )

        n_active = np.sum(active_cells)
        rng = np.random.default_rng(4412)

        # Model
        m0 = np.log(1e1) * np.ones(n_active)
        ind = model_builder.get_indices_block(
            np.r_[-200.0, -200.0, -600.0],
            np.r_[200.0, 200.0, -200.0],
            mesh.cell_centers[active_cells, :],
        )
        m0[ind] = np.log(1e0)
        m0 += 0.01 * rng.uniform(low=-1, high=1, size=n_active)

        # Define data and misfit
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
        """Test data misfit."""
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
            lambda m: (sim.dpred(m), lambda mx: sim.Jvec(m, mx)),
            m0,
            plotIt=False,
            num=3,
            random_seed=412,
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
        """Perform adjoint test."""
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

        f = sim.fields(m0)
        tests.assert_isadjoint(
            lambda u: sim.Jvec(m0, u, f=f),
            lambda v: sim.Jtvec(m0, v, f=f),
            m0.shape,
            (n_data,),
            rtol=ADJ_RTOL,
            random_seed=32,
        )
