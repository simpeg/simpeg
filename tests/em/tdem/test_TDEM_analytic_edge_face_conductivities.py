import discretize
import numpy as np
import pytest
from scipy.constants import mu_0
from simpeg import maps
from simpeg.electromagnetics import time_domain as tdem

REL_TOL = 0.07

CASES_LIST = [
    ("ElectricField", "CYL"),
    ("ElectricField", "TREE"),
    ("MagneticFluxDensity", "TREE")
]

@pytest.mark.parametrize("formulation, mesh_type", CASES_LIST)
def test_layer_conductance_to_analytic(formulation, mesh_type):
    # Some static parameters
    times = np.logspace(-3, -2, 6)

    loop_radius = np.pi**-0.5
    receiver_location = np.c_[16.0, 0.0, 1.0]
    source_location = np.r_[0.0, 0.0, 1.0]

    layer_depth = 100.0
    layer_thickness = 1.
    layer_conductivity = 100
    background_conductivity = 1e-2

    tau = layer_thickness * layer_conductivity

    # 1D LAYER MODEL
    thicknesses = np.array([layer_depth - layer_thickness / 2, layer_thickness])
    n_layer = len(thicknesses) + 1

    sigma_1d = background_conductivity * np.ones(n_layer)
    sigma_1d[1] = layer_conductivity

    sigma_map_1d = maps.IdentityMap(nP=n_layer)

    # 3D LAYER MODEL
    if mesh_type == "CYL":
        hr = [(2.5, 120), (2.5, 25, 1.3)]
        hz = [(2.5, 25, -1.3), (2.5, 200), (2.5, 25, 1.3)]

        mesh = discretize.CylindricalMesh([hr, 1, hz], x0="00C")

        ind = np.where(mesh.h[2] == np.min(mesh.h[2]))[0]
        ind = ind[int(len(ind) / 2)]

        mesh.origin = mesh.origin - np.r_[0.0, 0.0, mesh.nodes_z[ind] - 50]

    elif mesh_type == "TREE":
        dh = 5.0  # base cell width
        dom_width = 8000.0  # domain width
        nbc = 2 ** int(
            np.round(np.log(dom_width / dh) / np.log(2.0))
        )  # num. base cells

        h = [(dh, nbc)]
        mesh = discretize.TreeMesh([h, h, h], x0="CCC")
        mesh.refine_points(
            np.reshape(source_location, (1, 3)),
            level=-1,
            padding_cells_by_level=[8, 4, 4, 4],
            finalize=False,
        )
        x0s = np.vstack([ii * np.c_[-120, -120, -120] for ii in range(1, 5)])
        x1s = np.vstack([ii * np.c_[120, 120, 20] for ii in range(1, 5)])

        mesh.refine_box(x0s, x1s, levels=[-2, -3, -4, -5], finalize=False)
        mesh.finalize()

    sigma_3d = 1e-8 * np.ones(mesh.nC)
    sigma_3d[mesh.cell_centers[:, -1] < 0.0] = background_conductivity

    tau_3d = np.zeros(mesh.nF)
    tau_3d[np.isclose(mesh.faces[:, -1], -layer_depth)] = tau
    tau_map = maps.IdentityMap(nP=mesh.n_faces)

    # DEFINE SURVEY
    waveform = tdem.sources.StepOffWaveform()

    if formulation == 'ElectricField':
        rx_list = [
            tdem.receivers.PointMagneticFluxTimeDerivative(
                receiver_location, orientation='z', times=times
            )
        ]
    else:
        rx_list = [
            tdem.receivers.PointMagneticFluxDensity(
                receiver_location, orientation='z', times=times
            )
        ]

    # 1D SURVEY AND SIMULATION
    src_1d = [
        tdem.sources.MagDipole(
            rx_list, location=np.r_[0.0, 0.0, 1.0], orientation='z', waveform=waveform
        )
    ]
    survey_1d = tdem.Survey(src_1d)

    sim_1d = tdem.Simulation1DLayered(
        survey=survey_1d,
        thicknesses=thicknesses,
        sigmaMap=sigma_map_1d,
    )

    # 3D SURVEY AND SIMULATION
    time_steps = [(5e-5, 20), (2.5e-4, 40)]

    if mesh_type == "CYL":
        src_3d = [
            tdem.sources.CircularLoop(
                rx_list,
                radius=loop_radius,
                location=source_location,
                waveform=waveform,
            )
        ]
    else:
        src_3d = [
            tdem.sources.MagDipole(
                rx_list,
                location=source_location,
                orientation='z',
                waveform=waveform,
            )
        ]

    survey_3d = tdem.Survey(src_3d)

    # DEFINE THE SIMULATIONS
    if formulation == "MagneticFluxDensity":
        sim_3d = tdem.Simulation3DHierarchicalMagneticFluxDensity(
            mesh=mesh, survey=survey_3d, sigma=sigma_3d, tauMap=tau_map
        )
    else:
        sim_3d = tdem.Simulation3DHierarchicalElectricField(
            mesh=mesh, survey=survey_3d, sigma=sigma_3d, tauMap=tau_map
        )
    sim_3d.time_steps = time_steps

    # COMPUTE SOLUTIONS
    analytic_solution = sim_1d.dpred(sigma_1d)  # ALWAYS RETURNS H-FIELD
    numeric_solution = sim_3d.dpred(tau_3d)

    np.testing.assert_allclose(
        numeric_solution, analytic_solution, rtol=REL_TOL
    )


# NEED A TEST FOR THE EDGE CONDUCTIVITIES