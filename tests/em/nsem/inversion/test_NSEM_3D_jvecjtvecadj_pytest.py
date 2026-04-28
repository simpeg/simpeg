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

def get_sigma_background(mesh, simulation_type, active_cells):
    """Return background conductivity."""
    if simulation_type == "PS":
        sigma_hs = 1e-8 * np.ones(mesh.nC)
        sigma_hs[active_cells] = 1e1
        return sigma_hs
    else:
        n_layers = len(mesh.h[2])
        sigma_1d = 1e-8 * np.ones(n_layers)
        sigma_1d[0 : int(n_layers / 2)] = 1e1
        return sigma_1d

def get_survey(
    simulation_type, survey_type, orientations, components, locations, frequencies
):
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
                            locations,
                            orientation=orient,
                            component=comp,
                        )
                        for comp in components
                    ]
                )

        elif survey_type == "amplitude_ratio":
            # The orientations variable carries the base type
            rx_list.append(
                nsem.receivers.AmplitudeRatio(
                    locations,
                    base_type=orientations[0],
                    component=components[0],
                )
            )

        # MobileMT is app_cond
        elif survey_type == "apparent_conductivity":
            rx_list.extend([nsem.receivers.ApparentConductivity(locations)])

        if simulation_type == "PS":
            source_list.append(nsem.sources.PlanewaveXYPrimary(rx_list, f))
        else:
            source_list.append(nsem.sources.FictitiousSource(rx_list, f))

    return nsem.survey.Survey(source_list)


CASES_LIST = [
    ("PS", "impedance", ["xy", "yx"], ["real", "imag"]),
    ("PS", "impedance", ["xx", "yy"], ["real", "imag"]),
    ("PS", "impedance", ["xy", "yx"], ["app_res"]),
    ("PS", "impedance", ["xx", "yy"], ["app_res"]),
    ("PS", "impedance", ["xy", "yx"], ["phase"]),
    ("PS", "tipper", ["zx", "zy"], ["real", "imag"]),
    ("PS", "tipper", ["xx", "yy"], ["real", "imag"]),
    ("PS", "tipper", ["xy", "yx"], ["real", "imag"]),
    ("PS", "admittance", ["xy", "yx"], ["real", "imag"]),
    ("PS", "admittance", ["xx", "yy"], ["real", "imag"]),
    ("PS", "admittance", ["zx", "zy"], ["real", "imag"]),
    ("PS", "apparent_conductivity", None, None),
    ("PS", "tipper", ["det"], ["real", "imag"]),
    ("PS", "admittance", ["det"], ["real", "imag"]),
    ("PS", "tipper", ["sqrt_det"], ["real", "imag"]),
    ("PS", "admittance", ["sqrt_det"], ["real", "imag"]),
    ("PS", "amplitude_ratio", "magnetic", "amp"),
    ("PS", "amplitude_ratio", "electric", "amp"),
    ("PS", "amplitude_ratio", "magnetic", "amp_squared"),
    ("PS", "amplitude_ratio", "electric", "amp_squared"),
    ("FS_e", "impedance", ["xy", "yx"], ["real", "imag"]),
    ("FS_e", "impedance", ["xx", "yy"], ["real", "imag"]),
    ("FS_e", "impedance", ["xy", "yx"], ["app_res"]),
    ("FS_e", "impedance", ["xx", "yy"], ["app_res"]),
    ("FS_e", "impedance", ["xy", "yx"], ["phase"]),
    ("FS_e", "tipper", ["zx", "zy"], ["real", "imag"]),
    ("FS_e", "tipper", ["xx", "yy"], ["real", "imag"]),
    ("FS_e", "tipper", ["xy", "yx"], ["real", "imag"]),
    ("FS_e", "admittance", ["xy", "yx"], ["real", "imag"]),
    ("FS_e", "admittance", ["xx", "yy"], ["real", "imag"]),
    ("FS_e", "admittance", ["zx", "zy"], ["real", "imag"]),
    ("FS_e", "apparent_conductivity", None, None),
    ("FS_e", "tipper", ["det"], ["real", "imag"]),
    ("FS_e", "admittance", ["det"], ["real", "imag"]),
    ("FS_e", "tipper", ["sqrt_det"], ["real", "imag"]),
    ("FS_e", "admittance", ["sqrt_det"], ["real", "imag"]),
    ("FS_e", "amplitude_ratio", "magnetic", "amp"),
    ("FS_e", "amplitude_ratio", "electric", "amp"),
    ("FS_e", "amplitude_ratio", "magnetic", "amp_squared"),
    ("FS_e", "amplitude_ratio", "electric", "amp_squared"),
    ("FS_h", "impedance", ["xy", "yx"], ["real", "imag"]),
    ("FS_h", "impedance", ["xx", "yy"], ["real", "imag"]),
    ("FS_h", "impedance", ["xy", "yx"], ["app_res"]),
    ("FS_h", "impedance", ["xx", "yy"], ["app_res"]),
    ("FS_h", "impedance", ["xy", "yx"], ["phase"]),
    ("FS_h", "tipper", ["zx", "zy"], ["real", "imag"]),
    ("FS_h", "tipper", ["xx", "yy"], ["real", "imag"]),
    ("FS_h", "tipper", ["xy", "yx"], ["real", "imag"]),
    ("FS_h", "admittance", ["xy", "yx"], ["real", "imag"]),
    ("FS_h", "admittance", ["xx", "yy"], ["real", "imag"]),
    ("FS_h", "admittance", ["zx", "zy"], ["real", "imag"]),
    ("FS_h", "apparent_conductivity", None, None),
    ("FS_h", "tipper", ["det"], ["real", "imag"]),
    ("FS_h", "admittance", ["det"], ["real", "imag"]),
    ("FS_h", "tipper", ["sqrt_det"], ["real", "imag"]),
    ("FS_h", "admittance", ["sqrt_det"], ["real", "imag"]),
    ("FS_h", "amplitude_ratio", "magnetic", "amp"),
    ("FS_h", "amplitude_ratio", "electric", "amp"),
    ("FS_h", "amplitude_ratio", "magnetic", "amp_squared"),
    ("FS_h", "amplitude_ratio", "electric", "amp_squared"),
]

@pytest.mark.parametrize("simulation_type, survey_type, orientations, components", CASES_LIST)
class TestDerivatives:
    """Derivative and adjoint test for primary-secondary formulation."""
    def get_setup_objects(
        self,
        simulation_type,
        survey_type,
        orientations,
        components,
        locations,
        frequencies,
        mesh,
        active_cells,
        mapping,
    ):
        survey = get_survey(
            simulation_type,
            survey_type,
            orientations,
            components,
            locations,
            frequencies,
        )

        sigma_background = get_sigma_background(mesh, simulation_type, active_cells)

        if simulation_type == "PS":
            sim = nsem.simulation.Simulation3DPrimarySecondary(
                mesh,
                survey=survey,
                sigmaMap=mapping,
                sigmaPrimary=sigma_background,
                solver=get_default_solver(),
            )
        elif simulation_type == "FS_e":
            sim = nsem.simulation.Simulation3DElectricFieldFictitious(
                mesh,
                survey=survey,
                sigmaMap=mapping,
                sigma_background=sigma_background,
                solver=get_default_solver(),
            )
        elif simulation_type == "FS_h":
            sim = nsem.simulation.Simulation3DMagneticFieldFictitious(
                mesh,
                survey=survey,
                sigmaMap=mapping,
                sigma_background=sigma_background,
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
        simulation_type,
        survey_type,
        orientations,
        components,
        locations,
        frequencies,
        mesh,
        active_cells,
        mapping,
    ):
        m0, dmis = self.get_setup_objects(
            simulation_type,
            survey_type,
            orientations,
            components,
            locations,
            frequencies,
            mesh,
            active_cells,
            mapping,
        )
        sim = dmis.simulation

        passed = tests.check_derivative(
            lambda m: (sim.dpred(m), lambda mx: sim.Jvec(m0, mx)),
            m0,
            plotIt=False,
            random_seed=42,
            num=3,
        )

        assert passed

    def test_adjoint(
        self,
        simulation_type,
        survey_type,
        orientations,
        components,
        locations,
        frequencies,
        mesh,
        active_cells,
        mapping,
    ):
        m0, dmis = self.get_setup_objects(
            simulation_type,
            survey_type,
            orientations,
            components,
            locations,
            frequencies,
            mesh,
            active_cells,
            mapping,
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
            atol=ADJ_ATOL,
            random_seed=44,
        )