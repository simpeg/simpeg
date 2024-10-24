import unittest
import pytest

import discretize
import numpy as np
from pymatsolver import Pardiso as Solver
from scipy.constants import mu_0
from simpeg import maps
from simpeg.electromagnetics import analytics
from simpeg.electromagnetics import time_domain as tdem


def analytic_wholespace_dipole_comparison(
    mesh_type,
    formulation_type="MagneticFluxDensity",
    src_type="MagDipole",
    rx_type="MagneticFluxDensity",
    rx_orientation="Z",
    sigma=1e-2,
    rx_offset=None,
    bounds=None,
):
    if bounds is None:
        bounds = [1e-5, 1e-3]
    if mesh_type == "CYL":
        cs, ncx, ncz, npad = 5.0, 30, 10, 15
        hx = [(cs, ncx), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
        mesh = discretize.CylindricalMesh([hx, 1, hz], "00C")

    elif mesh_type == "TENSOR":
        cs, nc, npad = 8.0, 14, 8
        hx = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]
        hy = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]
        mesh = discretize.TensorMesh([hx, hy, hz], "CCC")

    mapping = maps.IdentityMap(mesh)
    sigma_model = np.ones(mesh.nC) * sigma

    times = np.logspace(-5, -4, 21)
    rx = getattr(tdem.receivers, "Point{}".format(rx_type))(
        np.array([rx_offset]), times, orientation=rx_orientation
    )

    if src_type == "MagDipole":
        # Vertical magnetic dipole
        src = tdem.sources.MagDipole(
            [rx],
            waveform=tdem.sources.StepOffWaveform(),
            location=np.array([0.0, 0.0, 0.0]),
            orientation="Z",
        )
    elif src_type == "ElectricDipole":
        # Vertical electric dipole along x
        ds = 1.0  # Dipole length
        locations = np.zeros((2, 3))
        locations[0, -1] = -ds / 2
        locations[1, -1] = ds / 2
        src = tdem.sources.LineCurrent(
            [rx], waveform=tdem.sources.StepOffWaveform(), location=locations
        )

    survey = tdem.Survey([src])

    if type(rx_orientation) is str:
        ind = ["X", "Y", "Z"].index(rx_orientation)
        projection_vector = np.zeros(3)
        projection_vector[ind] = 1.0
    else:
        projection_vector = rx_orientation

    if src_type == "MagDipole":
        if rx_type == "MagneticFluxDensity":
            analytic_solution = (
                mu_0
                * np.c_[
                    analytics.TDEM.TransientMagneticDipoleWholeSpace(
                        np.c_[rx_offset].T,
                        np.r_[0.0, 0.0, 0.0],
                        sigma,
                        times,
                        "Z",
                        fieldType="h",
                        mu_r=1,
                    )
                ]
            ).dot(projection_vector)
        elif rx_type == "MagneticFluxTimeDerivative":
            analytic_solution = (
                mu_0
                * np.c_[
                    analytics.TDEM.TransientMagneticDipoleWholeSpace(
                        np.c_[rx_offset].T,
                        np.r_[0.0, 0.0, 0.0],
                        sigma,
                        times,
                        "Z",
                        fieldType="dhdt",
                        mu_r=1,
                    )
                ]
            ).dot(projection_vector)
        else:
            analytic_solution = np.c_[
                analytics.TDEM.TransientMagneticDipoleWholeSpace(
                    np.c_[rx_offset].T,
                    np.r_[0.0, 0.0, 0.0],
                    sigma,
                    times,
                    "Z",
                    fieldType="e",
                    mu_r=1,
                )
            ].dot(projection_vector)

    elif src_type == "ElectricDipole":
        if rx_type == "MagneticFluxDensity":
            analytic_solution = (
                mu_0
                * np.c_[
                    analytics.TDEM.TransientElectricDipoleWholeSpace(
                        np.c_[rx_offset].T,
                        np.r_[0.0, 0.0, 0.0],
                        sigma,
                        times,
                        "Z",
                        fieldType="h",
                        mu_r=1,
                    )
                ]
            ).dot(projection_vector)
        elif rx_type == "MagneticFluxTimeDerivative":
            analytic_solution = (
                mu_0
                * np.c_[
                    analytics.TDEM.TransientElectricDipoleWholeSpace(
                        np.c_[rx_offset].T,
                        np.r_[0.0, 0.0, 0.0],
                        sigma,
                        times,
                        "Z",
                        fieldType="dhdt",
                        mu_r=1,
                    )
                ]
            ).dot(projection_vector)
        else:
            analytic_solution = np.c_[
                analytics.TDEM.TransientElectricDipoleWholeSpace(
                    np.c_[rx_offset].T,
                    np.r_[0.0, 0.0, 0.0],
                    sigma,
                    times,
                    "Z",
                    fieldType="e",
                    mu_r=1,
                )
            ].dot(projection_vector)

    sim = getattr(tdem.simulation, "Simulation3D{}".format(formulation_type))(
        mesh=mesh, survey=survey, sigmaMap=mapping
    )

    sim.solver = Solver
    sim.time_steps = [
        (1e-06, 40),
        (5e-06, 40),
        (1e-05, 40),
        (5e-05, 40),
        (0.0001, 40),
        (0.0005, 40),
    ]
    numeric_solution = sim.dpred(sigma_model)

    ind = np.logical_and(rx.times > bounds[0], rx.times < bounds[1])
    log10diff = np.linalg.norm(
        np.log10(np.abs(numeric_solution[ind]))
        - np.log10(np.abs(analytic_solution[ind]))
    ) / np.linalg.norm(np.log10(np.abs(analytic_solution[ind])))

    print(
        " |bz_ana| = {ana} |bz_num| = {num} |bz_ana-bz_num| = {diff}".format(
            ana=np.linalg.norm(analytic_solution),
            num=np.linalg.norm(numeric_solution),
            diff=np.linalg.norm(analytic_solution - numeric_solution),
        )
    )
    print("Difference: {}".format(log10diff))

    return log10diff


def analytic_halfspace_mag_dipole_comparison(
    mesh_type,
    src_type="MagDipole",
    sig_half=1e-2,
    rxOffset=50.0,
    bounds=None,
    rx_type="MagneticFluxDensityz",
):
    if bounds is None:
        bounds = [1e-5, 1e-3]
    if mesh_type == "CYL":
        cs, ncx, ncz, npad = 5.0, 30, 10, 15
        hx = [(cs, ncx), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
        mesh = discretize.CylindricalMesh([hx, 1, hz], "00C")

    elif mesh_type == "TENSOR":
        cs, nc, npad = 20.0, 13, 5
        hx = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]
        hy = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]
        mesh = discretize.TensorMesh([hx, hy, hz], "CCC")

    active = mesh.cell_centers_z < 0.0
    actMap = maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.shape_cells[2])
    mapping = maps.ExpMap(mesh) * maps.SurjectVertical1D(mesh) * actMap

    rx = getattr(tdem.receivers, "Point{}".format(rx_type[:-1]))(
        np.array([[rxOffset, 0.0, 0.0]]), np.logspace(-5, -4, 21), rx_type[-1]
    )

    if src_type == "MagDipole":
        src = tdem.Src.MagDipole(
            [rx],
            waveform=tdem.Src.StepOffWaveform(),
            location=np.array([0.0, 0.0, 0.0]),
        )
    elif src_type == "CircularLoop":
        src = tdem.sources.CircularLoop(
            [rx],
            waveform=tdem.Src.StepOffWaveform(),
            location=np.array([0.0, 0.0, 0.0]),
            radius=0.1,
            # test number of turns and current
            n_turns=2,
            current=0.5,
        )

    survey = tdem.Survey([src])
    time_steps = [
        (1e-06, 40),
        (5e-06, 40),
        (1e-05, 40),
        (5e-05, 40),
        (0.0001, 40),
        (0.0005, 40),
    ]

    sim = tdem.Simulation3DMagneticFluxDensity(
        mesh, survey=survey, time_steps=time_steps, sigmaMap=mapping
    )
    sim.solver = Solver

    sigma = np.ones(mesh.shape_cells[2]) * 1e-8
    sigma[active] = sig_half
    sigma = np.log(sigma[active])

    if src_type == "MagDipole":
        bz_ana = mu_0 * analytics.hzAnalyticDipoleT(
            rx.locations[0][0] + 1e-3, rx.times, sig_half
        )
    elif src_type == "CircularLoop":
        bz_ana = mu_0 * analytics.hzAnalyticDipoleT(13, rx.times, sig_half)

    bz_calc = sim.dpred(sigma)
    ind = np.logical_and(rx.times > bounds[0], rx.times < bounds[1])
    log10diff = np.linalg.norm(
        np.log10(np.abs(bz_calc[ind])) - np.log10(np.abs(bz_ana[ind]))
    ) / np.linalg.norm(np.log10(np.abs(bz_ana[ind])))

    print(
        " |bz_ana| = {ana} |bz_num| = {num} |bz_ana-bz_num| = {diff}".format(
            ana=np.linalg.norm(bz_ana),
            num=np.linalg.norm(bz_calc),
            diff=np.linalg.norm(bz_ana - bz_calc),
        )
    )
    print("Difference: {}".format(log10diff))

    return log10diff


###########################################################
# ANALYTIC WHOLESPACE TESTS FOR MAG AND ELECTRIC DIPOLES
###########################################################


class WholespaceTests(unittest.TestCase):
    # WORKING
    def test_cyl_Bform_MagDipole_Bfield_Z(self):
        assert (
            analytic_wholespace_dipole_comparison(
                "CYL",
                "MagneticFluxDensity",
                "MagDipole",
                "MagneticFluxDensity",
                "Z",
                1e-2,
                [0, 0, 48],
            )
            < 0.01
        )

    # WORKING
    def test_tensor_Bform_MagDipole_Bfield_Z(self):
        assert (
            analytic_wholespace_dipole_comparison(
                "TENSOR",
                "MagneticFluxDensity",
                "MagDipole",
                "MagneticFluxDensity",
                "Z",
                1e-2,
                [0, 0, 48],
            )
            < 0.01
        )

    # WORKING
    def test_cyl_Bform_MagDipole_Bfield_Z_vector_orientation(self):
        assert (
            analytic_wholespace_dipole_comparison(
                "CYL",
                "MagneticFluxDensity",
                "MagDipole",
                "MagneticFluxDensity",
                np.r_[0.0, 0.0, 1.0],
                1e-2,
                [0, 0, 48],
            )
            < 0.01
        )

    # WORKING
    def test_tensor_Bform_MagDipole_Bfield_Z_vector_orientation(self):
        assert (
            analytic_wholespace_dipole_comparison(
                "TENSOR",
                "MagneticFluxDensity",
                "MagDipole",
                "MagneticFluxDensity",
                np.r_[0.0, 0.0, 1.0],
                1e-2,
                [0, 0, 48],
            )
            < 0.01
        )

    # WORKING
    def test_tensor_Bform_MagDipole_Bfield_TotalField_vector_orientation(self):
        # TMI orientation with significant values for all 3 components [-0.5, 0.5, -0.707] for testing
        inclination = np.radians(45.0)
        declination = np.radians(-45.0)
        tmi_orientation = np.r_[
            np.cos(inclination) * np.sin(declination),
            np.cos(inclination) * np.cos(declination),
            -np.sin(inclination),
        ]
        assert np.isclose(np.linalg.norm(tmi_orientation), 1.0)
        assert (
            analytic_wholespace_dipole_comparison(
                "TENSOR",
                "MagneticFluxDensity",
                "MagDipole",
                "MagneticFluxDensity",
                tmi_orientation,
                1e-2,
                [0, 0, 48],
            )
            < 0.01
        )

    # NOT IMPLEMENTED (NO PHI_0 BECAUSE NO GRAD?)
    # def test_cyl_Eform_EletricDipole_Efield_Z(self):
    #     self.assertTrue(
    #         analytic_wholespace_dipole_comparison(
    #             'CYL', 'ElectricField', 'ElectricDipole', 'ElectricField', 'Z', 1e-2, [0,0,48]
    #         )
    #     ) < 0.01

    # WORKING
    def test_tensor_Eform_ElectricDipole_Efield_Z(self):
        assert (
            analytic_wholespace_dipole_comparison(
                "TENSOR",
                "ElectricField",
                "ElectricDipole",
                "ElectricField",
                "Z",
                1e-2,
                [0, 0, 48],
            )
            < 0.01
        )

    # NOT IMPLEMENTED (NO BO)
    # def test_cyl_Bform_ElectricDipole_Bfield_Z(self):
    #     self.assertTrue(
    #         analytic_wholespace_dipole_comparison(
    #             'CYL', 'MagneticFluxDensity', 'ElectricDipole', 'MagneticFluxDensity', 'X', 1e-2, [0,48,0]
    #         ) < 0.01
    #     )

    # NOT IMPLEMENTED (NO B0)
    # def test_tensor_Bform_ElectricDipole_Bfield_Z(self):
    #     self.assertTrue(
    #         analytic_wholespace_dipole_comparison(
    #             'TENSOR', 'MagneticFluxDensity', 'ElectricDipole', 'MagneticFluxDensity', 'X', 1e-2, [0,48,0]
    #         ) < 0.01
    #     )

    # NOT IMPLEMENTED (NO BO)
    # def test_cyl_Eform_ElectricDipole_dBdtfield_Z(self):
    #     self.assertTrue(
    #         analytic_wholespace_dipole_comparison(
    #             'CYL', 'ElectricField', 'ElectricDipole', 'MagneticFluxTimeDerivative', 'X', 1e-2, [0,48,0]
    #         ) < 0.01
    #     )

    # NOT IMPLEMENTED NOT ACCURATE
    def test_tensor_Eform_ElectricDipole_dBdtfield_Z(self):
        assert (
            analytic_wholespace_dipole_comparison(
                "TENSOR",
                "ElectricField",
                "ElectricDipole",
                "MagneticFluxTimeDerivative",
                "X",
                1e-2,
                [0, 48, 0],
            )
            < 0.01
        )

    def test_tensor_Eform_ElectricDipole_dBdtfield_X_vector_orientation(self):
        assert (
            analytic_wholespace_dipole_comparison(
                "TENSOR",
                "ElectricField",
                "ElectricDipole",
                "MagneticFluxTimeDerivative",
                np.r_[1.0, 0.0, 0.0],
                1e-2,
                [0, 48, 0],
            )
            < 0.01
        )

    def test_tensor_Eform_ElectricDipole_dBdtfield_TotalField_vector_orientation(self):
        # TMI orientation with significant values for all 3 components [-0.5, 0.5, -0.707] for testing
        inclination = np.radians(45.0)
        declination = np.radians(-45.0)
        tmi_orientation = np.r_[
            np.cos(inclination) * np.sin(declination),
            np.cos(inclination) * np.cos(declination),
            -np.sin(inclination),
        ]
        assert np.isclose(np.linalg.norm(tmi_orientation), 1.0)
        assert (
            analytic_wholespace_dipole_comparison(
                "TENSOR",
                "ElectricField",
                "ElectricDipole",
                "MagneticFluxTimeDerivative",
                tmi_orientation,
                1e-2,
                [0, 48, 0],
            )
            < 0.01
        )


###########################################################
# ANALYTIC HALFSPACE TESTS FOR MAG DIPOLE
###########################################################


class TDEM_bTests(unittest.TestCase):
    def test_analytic_p2_CYL_50_MagDipolem(self):
        assert (
            analytic_halfspace_mag_dipole_comparison("CYL", rxOffset=50.0, sig_half=1e2)
            < 0.01
        )

    def test_analytic_p1_CYL_50_MagDipolem(self):
        assert (
            analytic_halfspace_mag_dipole_comparison("CYL", rxOffset=50.0, sig_half=1e1)
            < 0.01
        )

    def test_analytic_p0_CYL_50_MagDipolem(self):
        assert (
            analytic_halfspace_mag_dipole_comparison("CYL", rxOffset=50.0, sig_half=1e0)
            < 0.01
        )

    def test_analytic_m1_CYL_50_MagDipolem(self):
        assert (
            analytic_halfspace_mag_dipole_comparison(
                "CYL", rxOffset=50.0, sig_half=1e-1
            )
            < 0.01
        )

    def test_analytic_m2_CYL_50_MagDipolem(self):
        assert (
            analytic_halfspace_mag_dipole_comparison(
                "CYL", rxOffset=50.0, sig_half=1e-2
            )
            < 0.01
        )

    def test_analytic_m3_CYL_50_MagDipolem(self):
        assert (
            analytic_halfspace_mag_dipole_comparison(
                "CYL", rxOffset=50.0, sig_half=1e-3
            )
            < 0.02
        )

    def test_analytic_p0_CYL_1m_MagDipole(self):
        assert (
            analytic_halfspace_mag_dipole_comparison("CYL", rxOffset=1.0, sig_half=1e0)
            < 0.01
        )

    def test_analytic_m1_CYL_1m_MagDipole(self):
        assert (
            analytic_halfspace_mag_dipole_comparison("CYL", rxOffset=1.0, sig_half=1e-1)
            < 0.01
        )

    def test_analytic_m2_CYL_1m_MagDipole(self):
        assert (
            analytic_halfspace_mag_dipole_comparison("CYL", rxOffset=1.0, sig_half=1e-2)
            < 0.01
        )

    def test_analytic_m3_CYL_1m_MagDipole(self):
        assert (
            analytic_halfspace_mag_dipole_comparison("CYL", rxOffset=1.0, sig_half=1e-3)
            < 0.02
        )

    def test_analytic_p0_CYL_0m_CircularLoop(self):
        assert (
            analytic_halfspace_mag_dipole_comparison(
                "CYL", src_type="CircularLoop", rxOffset=0.0, sig_half=1e0
            )
            < 0.15
        )

    def test_analytic_m1_CYL_0m_CircularLoop(self):
        assert (
            analytic_halfspace_mag_dipole_comparison(
                "CYL", src_type="CircularLoop", rxOffset=0.0, sig_half=1e-1
            )
            < 0.15
        )

    def test_analytic_m2_CYL_0m_CircularLoop(self):
        assert (
            analytic_halfspace_mag_dipole_comparison(
                "CYL", src_type="CircularLoop", rxOffset=0.0, sig_half=1e-2
            )
            < 0.15
        )

    def test_analytic_m3_CYL_0m_CircularLoop(self):
        assert (
            analytic_halfspace_mag_dipole_comparison(
                "CYL", src_type="CircularLoop", rxOffset=0.0, sig_half=1e-3
            )
            < 0.15
        )


@pytest.mark.parametrize(
    ("mesh_type", "rx_type", "orientation"),
    [
        ("TENSOR", "MagneticFluxDensity", "x"),
        ("TENSOR", "MagneticFluxDensity", "z"),
        ("CYL", "MagneticFluxDensity", "z"),
        ("CYL", "MagneticFluxTimeDerivative", "z"),
    ],
)
def test_layer_conductance_to_analytic(
    mesh_type,
    rx_type,
    orientation,
    check_time_bounds=None,
):
    # Some static parameters
    PHI = np.linspace(0, 2 * np.pi, 21)
    loop_radius = np.pi**-0.5
    receiver_location = np.c_[40.0, 0.0, 1.0]
    source_location = np.r_[0.0, 0.0, 1.0]

    if orientation == "x":
        source_nodes = np.c_[
            np.zeros_like(PHI),
            loop_radius * np.cos(PHI),
            1.0 + loop_radius * np.sin(PHI),
        ]
    elif orientation == "z":
        source_nodes = np.c_[
            loop_radius * np.cos(PHI), loop_radius * np.sin(PHI), np.ones_like(PHI)
        ]

    layer_depth = 24.0
    layer_thickness = 0.1
    layer_conductivity = 10.0
    background_conductivity = 2.5e-3

    tau = layer_thickness * layer_conductivity

    if check_time_bounds is None:
        check_time_bounds = [1e-5, 1e-3]

    # 1D LAYER MODEL
    thicknesses = np.array([layer_depth - layer_thickness / 2, layer_thickness])
    n_layer = len(thicknesses) + 1

    sigma_1d = background_conductivity * np.ones(n_layer)
    sigma_1d[1] = layer_conductivity

    sigma_map_1d = maps.IdentityMap(nP=n_layer)

    # 3D LAYER MODEL
    if mesh_type == "CYL":
        cs, ncx, ncz, npad = 4.0, 40, 20, 20
        hx = [(cs, ncx), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
        mesh = discretize.CylindricalMesh([hx, 1, hz], "00C")

    elif mesh_type == "TENSOR":
        cs, nc, npad = 8.0, 14, 8
        hx = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]
        hy = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]
        mesh = discretize.TensorMesh([hx, hy, hz], "CCC")

    sigma_3d = 1e-8 * np.ones(mesh.nC)
    sigma_3d[mesh.cell_centers[:, -1] < 0.0] = background_conductivity

    tau_3d = np.zeros(mesh.nF)
    tau_3d[np.isclose(mesh.faces[:, -1], -layer_depth)] = tau
    tau_map = maps.IdentityMap(nP=mesh.n_faces)

    # DEFINE SURVEY
    times = np.logspace(-5, -4, 21)
    rx = getattr(tdem.receivers, f"Point{rx_type}")(
        receiver_location, times, orientation=orientation
    )

    # 1D SURVEY AND SIMULATION
    src_1d = tdem.sources.MagDipole(
        [rx],
        location=np.r_[0.0, 0.0, 1.0],
        orientation=orientation,
        waveform=tdem.sources.StepOffWaveform(),
    )
    survey_1d = tdem.Survey([src_1d])

    sim_1d = tdem.Simulation1DLayered(
        survey=survey_1d,
        thicknesses=thicknesses,
        sigmaMap=sigma_map_1d,
    )

    # 3D SURVEY AND SIMULATION
    if mesh_type == "CYL":
        src_3d = tdem.sources.CircularLoop(
            [rx],
            radius=loop_radius,
            location=source_location,
            waveform=tdem.sources.StepOffWaveform(),
        )
    else:
        if rx_type == "MagneticFluxDensity":
            src_3d = tdem.sources.MagDipole(
                [rx],
                location=source_location,
                orientation=orientation,
                waveform=tdem.sources.StepOffWaveform(),
            )
        else:
            src_3d = tdem.sources.LineCurrent(
                [rx], location=source_nodes, waveform=tdem.sources.StepOffWaveform()
            )

    survey_3d = tdem.Survey([src_3d])

    # DEFINE THE SIMULATIONS
    if rx_type == "MagneticFluxDensity":
        sim_3d = tdem.Simulation3DHierarchicalMagneticFluxDensity(
            mesh=mesh, survey=survey_3d, sigma=sigma_3d, tauMap=tau_map
        )
    else:
        sim_3d = tdem.simulation.Simulation3DHierarchicalElectricField(
            mesh=mesh, survey=survey_3d, sigma=sigma_3d, tauMap=tau_map
        )

    sim_3d.time_steps = [
        (1e-06, 40),
        (5e-06, 40),
        (1e-05, 40),
        (5e-05, 40),
        (0.0001, 40),
        (0.0005, 40),
    ]

    # COMPUTE SOLUTIONS
    analytic_solution = sim_1d.dpred(sigma_1d)
    numeric_solution = sim_3d.dpred(tau_3d)

    time_inds = (rx.times > check_time_bounds[0]) & (rx.times < check_time_bounds[1])
    # compare them on a log scale over these times?
    analytic_solution = analytic_solution[time_inds]
    numeric_solution = numeric_solution[time_inds]

    print(
        " |bz_ana| = {ana} |bz_num| = {num} |bz_ana-bz_num| = {diff}".format(
            ana=np.linalg.norm(analytic_solution),
            num=np.linalg.norm(numeric_solution),
            diff=np.linalg.norm(analytic_solution - numeric_solution),
        )
    )

    np.testing.assert_allclose(numeric_solution, analytic_solution, rtol=0.2)


# class LayerConductanceTests(unittest.TestCase):
#     # Compares analytic 1D layered Earth solution to a plate of equivalent
#     # conductance.
#
#     def test_tensor_magdipole_b_x(self):
#         assert (
#             analytic_layer_small_loop_face_conductivity_comparison(
#                 mesh_type="TENSOR",
#                 rx_type="MagneticFluxDensity",
#                 orientation="X",
#                 bounds=None,
#             )
#             < 0.01
#         )
#
#     def test_tensor_magdipole_b_z(self):
#         assert (
#             analytic_layer_small_loop_face_conductivity_comparison(
#                 mesh_type="TENSOR",
#                 rx_type="MagneticFluxDensity",
#                 orientation="Z",
#                 bounds=None,
#             )
#             < 0.01
#         )
#
#     def test_cyl_magdipole_b_z(self):
#         assert (
#             analytic_layer_small_loop_face_conductivity_comparison(
#                 mesh_type="CYL",
#                 rx_type="MagneticFluxDensity",
#                 orientation="Z",
#                 bounds=None,
#             )
#             < 0.01
#         )
#
#     def test_tensor_linecurrent_dbdt_x(self):
#         assert (
#             analytic_layer_small_loop_face_conductivity_comparison(
#                 mesh_type="TENSOR",
#                 rx_type="MagneticFluxTimeDerivative",
#                 orientation="X",
#                 bounds=None,
#             )
#             < 0.01
#         )
#
#     def test_tensor_linecurrent_dbdt_z(self):
#         assert (
#             analytic_layer_small_loop_face_conductivity_comparison(
#                 mesh_type="TENSOR",
#                 rx_type="MagneticFluxTimeDerivative",
#                 orientation="Z",
#                 bounds=None,
#             )
#             < 0.01
#         )
#
#     def test_cyl_circularloop_dbdt_z(self):
#         assert (
#             analytic_layer_small_loop_face_conductivity_comparison(
#                 mesh_type="CYL",
#                 rx_type="MagneticFluxTimeDerivative",
#                 orientation="Z",
#                 bounds=None,
#             )
#             < 0.01
#         )
