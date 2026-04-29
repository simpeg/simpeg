import discretize
import numpy as np
import pytest
from scipy.constants import mu_0
from simpeg.utils import ndgrid, model_builder
from simpeg import maps
from simpeg.electromagnetics.static import resistivity as dcr

ABS_TOL = 1e-13
REL_TOL = 0.08

import matplotlib.pyplot as plt


# ONLY PASSES IN THE LINEAR REGIME
def test_layer_conductance_to_analytic():
    """Validate 1D analytic solution for thin layer against layer as face conductances."""

    # Some static parameters
    location_a = np.r_[-20, 0, 0]
    locations_m = ndgrid(np.linspace(0, 30, 4), 0, 0)

    layer_depth = 40.0
    layer_thickness = 0.1
    layer_conductivity = 1e0  # Only 1 order of magnitude (linear regime)
    background_conductivity = 1e-1

    tau = layer_thickness * layer_conductivity

    # 1D LAYER MODEL
    thicknesses = np.array([layer_depth - layer_thickness / 2, layer_thickness])
    n_layer = len(thicknesses) + 1

    sigma_1d = background_conductivity * np.ones(n_layer)
    sigma_1d[1] = layer_conductivity

    sigma_map_1d = maps.IdentityMap(nP=n_layer)

    # Mesh
    dh = 2.5  # base cell width
    dom_width = 8000.0  # domain width
    nbc = 2 ** int(np.round(np.log(dom_width / dh) / np.log(2.0)))  # num. base cells

    h = [(dh, nbc)]
    mesh = discretize.TreeMesh([h, h, h], x0="CCN")
    pts = np.vstack([locations_m, location_a])
    mesh.refine_points(
        pts,
        level=-1,
        padding_cells_by_level=[8, 6, 6, 6],
        finalize=False,
    )
    x0s = np.vstack([ii * np.c_[-60, -60, -60] for ii in range(1, 5)])
    x1s = np.vstack([ii * np.c_[60, 60, 10] for ii in range(1, 5)])

    mesh.refine_box(x0s, x1s, levels=[-2, -3, -4, -5], finalize=False)
    mesh.finalize()

    sigma_3d = 1e-8 * np.ones(mesh.nC)
    sigma_3d[mesh.cell_centers[:, -1] < 0.0] = background_conductivity

    tau_3d = np.zeros(mesh.nF)
    tau_3d[np.isclose(mesh.faces[:, -1], -layer_depth)] = tau
    tau_map = maps.IdentityMap(nP=mesh.n_faces)

    # DEFINE SURVEY
    rx_list = [dcr.receivers.Pole(locations=locations_m)]

    # 1D SURVEY AND SIMULATION
    src_1d = [dcr.sources.Pole(rx_list, location=location_a)]
    survey_1d = dcr.survey.Survey(src_1d)

    sim_1d = dcr.Simulation1DLayers(
        survey=survey_1d,
        thicknesses=thicknesses,
        sigmaMap=sigma_map_1d,
    )

    # 3D SURVEY AND SIMULATION
    src_3d = [dcr.sources.Pole(rx_list, location=location_a)]

    survey_3d = dcr.Survey(src_3d)

    sim_3d = dcr.simulation.Simulation3DHierarchicalNodal(
        mesh=mesh, survey=survey_3d, sigma=sigma_3d, tauMap=tau_map
    )

    # COMPUTE ANOMALY
    analytic_solution = sim_1d.dpred(sigma_1d) - sim_1d.dpred(
        background_conductivity * np.ones(3)
    )
    numeric_solution = sim_3d.dpred(tau_3d) - sim_3d.dpred(np.zeros_like(tau_3d))

    np.testing.assert_allclose(
        numeric_solution, analytic_solution, atol=ABS_TOL, rtol=REL_TOL
    )


def test_edge_conductivity():
    """Cross check for a thin conductive wire."""

    # Some static parameters
    location_a = np.r_[-40, 0, 0]
    location_b = np.r_[40, 0, 0]
    locations_m = ndgrid(np.linspace(-20, 20, 5), 0, 8)

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
    mesh = discretize.TreeMesh([h, h, h], x0="CCN")

    pts = ndgrid(
        np.arange(-wire_length / 2, wire_length / 2 + 1e-6, dh), 0, -wire_depth
    )
    mesh.refine_points(
        pts,
        level=-1,
        padding_cells_by_level=[6, 6, 6, 8],
        finalize=False,
    )

    pts = np.vstack([locations_m, location_a, location_b])
    mesh.refine_points(
        pts,
        level=-2,
        padding_cells_by_level=[6, 6, 6, 6],
        finalize=False,
    )
    x0s = np.vstack([ii * np.c_[-60, -60, -30] for ii in range(1, 5)])
    x1s = np.vstack([ii * np.c_[60, 60, 10] for ii in range(1, 5)])

    mesh.refine_box(x0s, x1s, levels=[-5, -6, -7, -8], finalize=False)
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
    rx_sigma = dcr.receivers.Pole(locations=locations_m)
    rx_kappa = dcr.receivers.Pole(locations=locations_m)

    src_sigma = dcr.sources.Dipole(
        [rx_sigma], location_a=location_a, location_b=location_b
    )
    src_kappa = dcr.sources.Dipole(
        [rx_kappa], location_a=location_a, location_b=location_b
    )

    survey_sigma = dcr.Survey([src_sigma])
    survey_kappa = dcr.Survey([src_kappa])

    sim_sigma = dcr.simulation.Simulation3DNodal(
        mesh=mesh, survey=survey_sigma, sigmaMap=maps.IdentityMap(nP=mesh.n_cells)
    )
    sim_kappa = dcr.simulation.Simulation3DHierarchicalNodal(
        mesh=mesh,
        survey=survey_kappa,
        sigma=sigma_0,
        kappaMap=maps.IdentityMap(nP=mesh.n_edges),
    )

    # COMPUTE SOLUTIONS
    dpred_0 = sim_sigma.dpred(sigma_0)
    true_solution = sim_sigma.dpred(sigma_voxel) - dpred_0
    approx_solution = sim_kappa.dpred(kappa) - dpred_0

    np.testing.assert_allclose(
        true_solution, approx_solution, atol=ABS_TOL, rtol=REL_TOL
    )
