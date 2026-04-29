"""Numerical tests for hierarchical properties."""
import discretize
import numpy as np
import pytest
from scipy.constants import mu_0
from simpeg import maps
from simpeg.utils import ndgrid
from simpeg.electromagnetics import time_domain as tdem

REL_TOL = 0.1
ABS_TOL = 1e-13

CASES_LIST = [
    ("ElectricField", "CYL"),
    ("ElectricField", "TREE"),
    ("MagneticFluxDensity", "TREE"),
]

@pytest.mark.parametrize("formulation, mesh_type", CASES_LIST)
def test_layer_conductance_to_analytic(formulation, mesh_type):
    """Validate 1D analytic solution for thin layer against layer as face conductances."""
    # Some static parameters
    times = np.logspace(-3, -2, 6)

    loop_radius = np.pi**-0.5
    receiver_location = np.c_[16.0, 0.0, 1.0]
    source_location = np.r_[0.0, 0.0, 1.0]

    layer_depth = 100.0
    layer_thickness = 1.0
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

    if formulation == "ElectricField":
        rx_list = [
            tdem.receivers.PointMagneticFluxTimeDerivative(
                receiver_location, orientation="z", times=times
            )
        ]
    else:
        rx_list = [
            tdem.receivers.PointMagneticFluxDensity(
                receiver_location, orientation="z", times=times
            )
        ]

    # 1D SURVEY AND SIMULATION
    src_1d = [
        tdem.sources.MagDipole(
            rx_list, location=np.r_[0.0, 0.0, 1.0], orientation="z", waveform=waveform
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
                orientation="z",
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

    analytic_solution = sim_1d.dpred(sigma_1d)  # ALWAYS RETURNS H-FIELD
    numeric_solution = sim_3d.dpred(tau_3d)

    np.testing.assert_allclose(numeric_solution, analytic_solution, rtol=REL_TOL)

def test_edge_conductivity():
    """Cross check for a thin conductive wire."""
    # Some static parameters
    location_a = np.r_[-40, 0, 0]
    location_b = np.r_[40, 0, 0]
    locations_rx = ndgrid(np.linspace(-10, 10, 4), 0, 8)
    times = np.logspace(-4, -3, 3)

    wire_depth = 16.0
    wire_width = 1.0
    wire_length = 20.0
    wire_conductivity = 1e0  # Only 1 order of magnitude (linear regime
    background_conductivity = 1e-1

    kappa_value = wire_width**2 * wire_conductivity

    # Mesh
    dh = 0.25  # base cell width
    dom_width = 2000.0  # domain width
    nbc = 2 ** int(np.round(np.log(dom_width / dh) / np.log(2.0)))  # num. base cells

    h = [(dh, nbc)]
    mesh = discretize.TreeMesh([h, h, h], x0="CCC")

    pts = ndgrid(
        np.arange(-wire_length / 2, wire_length / 2 + 1e-6, dh), 0, -wire_depth
    )
    mesh.refine_points(
        pts,
        level=-1,
        padding_cells_by_level=[4, 4, 4, 4],
        finalize=False,
    )

    pts = np.vstack([locations_rx, location_a, location_b])
    mesh.refine_points(
        pts,
        level=-2,
        padding_cells_by_level=[4, 4, 4, 4],
        finalize=False,
    )
    x0s = np.vstack([ii * np.c_[-60, -60, -30] for ii in range(1, 4)])
    x1s = np.vstack([ii * np.c_[60, 60, 10] for ii in range(1, 4)])

    mesh.refine_box(x0s, x1s, levels=[-5, -6, -7], finalize=False)
    mesh.finalize()
    print(mesh.n_cells)

    # Models
    ccs = mesh.cell_centers
    sigma_0 = 1e-8 * np.ones(mesh.n_cells)
    sigma_0[ccs[:, -1] < 0.0] = background_conductivity

    sigma_voxel = sigma_0.copy()
    inds_block = (
        (np.abs(ccs[:, 0]) <= wire_length / 2)
        & (np.abs(ccs[:, 1]) <= wire_width / 2)
        & (np.abs(ccs[:, 2] + wire_depth) <= wire_width / 2)
    )
    sigma_voxel[inds_block] = wire_conductivity
    print(sum(inds_block))

    kappa = np.zeros(mesh.n_edges)
    edges = mesh.edges
    inds_kappa = (
        (np.abs(edges[:, 0]) <= wire_length / 2)
        & (np.isclose(edges[:, 1], 0.0))
        & (np.isclose(edges[:, 2], -wire_depth))
    )
    kappa[inds_kappa] = kappa_value
    print(sum(inds_kappa))

    # DEFINE SURVEY
    rx_sigma = tdem.receivers.PointElectricField(
        locations=locations_rx, orientation="x", times=times
    )
    rx_kappa = tdem.receivers.PointElectricField(
        locations=locations_rx, orientation="x", times=times
    )

    waveform = tdem.sources.StepOffWaveform()
    src_sigma = tdem.sources.LineCurrent(
        [rx_sigma], location=np.c_[location_a, location_b].T, waveform=waveform
    )
    src_kappa = tdem.sources.LineCurrent(
        [rx_kappa], location=np.c_[location_a, location_b].T, waveform=waveform
    )

    survey_sigma = tdem.Survey([src_sigma])
    survey_kappa = tdem.Survey([src_kappa])

    time_steps = [(5e-6, 20), (5e-5, 21)]

    sim_sigma = tdem.simulation.Simulation3DElectricField(
        mesh=mesh,
        survey=survey_sigma,
        sigmaMap=maps.IdentityMap(nP=mesh.n_cells),
        time_steps=time_steps,
    )
    sim_kappa = tdem.simulation.Simulation3DHierarchicalElectricField(
        mesh=mesh,
        survey=survey_kappa,
        sigma=sigma_0,
        kappaMap=maps.IdentityMap(nP=mesh.n_edges),
        time_steps=time_steps,
    )

    # COMPUTE SOLUTIONS
    dpred_0 = sim_sigma.dpred(sigma_0)
    true_solution = sim_sigma.dpred(sigma_voxel) - dpred_0
    approx_solution = sim_kappa.dpred(kappa) - dpred_0

    np.testing.assert_allclose(
        true_solution, approx_solution, atol=ABS_TOL, rtol=REL_TOL
    )
