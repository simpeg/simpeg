import unittest
import discretize
from SimPEG import maps
from SimPEG.potential_fields import gravity
from geoana.gravity import Prism
import numpy as np
import os


def test_ana_grav_forward(tmp_path):
    nx = 5
    ny = 5

    # Define a mesh
    cs = 0.2
    hxind = [(cs, 41)]
    hyind = [(cs, 41)]
    hzind = [(cs, 41)]
    mesh = discretize.TensorMesh([hxind, hyind, hzind], "CCC")

    # create a model of two blocks, 1 inside the other
    block1 = np.array([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
    block2 = np.array([[-0.7, 0.7], [-0.7, 0.7], [-0.7, 0.7]])

    def get_block_inds(grid, block):
        return np.where(
            (grid[:, 0] > block[0, 0])
            & (grid[:, 0] < block[0, 1])
            & (grid[:, 1] > block[1, 0])
            & (grid[:, 1] < block[1, 1])
            & (grid[:, 2] > block[2, 0])
            & (grid[:, 2] < block[2, 1])
        )

    block1_inds = get_block_inds(mesh.cell_centers, block1)
    block2_inds = get_block_inds(mesh.cell_centers, block2)

    rho1 = 1.0
    rho2 = 2.0

    model = np.zeros(mesh.n_cells)
    model[block1_inds] = rho1
    model[block2_inds] = rho2

    active_cells = model != 0.0
    model_reduced = model[active_cells]

    # Create reduced identity map for Linear Pproblem
    idenMap = maps.IdentityMap(nP=int(sum(active_cells)))

    # Create plane of observations
    xr = np.linspace(-20, 20, nx)
    yr = np.linspace(-20, 20, ny)
    X, Y = np.meshgrid(xr, yr)
    Z = np.ones_like(X) * 3.0
    locXyz = np.c_[X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]

    receivers = gravity.Point(locXyz, components=["gx", "gy", "gz"])
    sources = gravity.SourceField([receivers])
    survey = gravity.Survey(sources)

    sim = gravity.Simulation3DIntegral(
        mesh,
        survey=survey,
        rhoMap=idenMap,
        ind_active=active_cells,
        store_sensitivities="disk",
        sensitivity_path=str(tmp_path) + os.sep,
        n_processes=4,
    )

    data = sim.dpred(model_reduced)
    d_x = data[0::3]
    d_y = data[1::3]
    d_z = data[2::3]

    # Compute analytical response from dense prism
    prism_1 = Prism(block1[:, 0], block1[:, 1], rho1 * 1000)  # g/cc to kg/m**3
    prism_2 = Prism(block2[:, 0], block2[:, 1], -rho1 * 1000)
    prism_3 = Prism(block2[:, 0], block2[:, 1], rho2 * 1000)

    d = (
        prism_1.gravitational_field(locXyz)
        + prism_2.gravitational_field(locXyz)
        + prism_3.gravitational_field(locXyz)
    ) * 1e5  # convert to mGal from m/s^2
    np.testing.assert_allclose(d_x, d[:, 0], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(d_y, d[:, 1], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(d_z, d[:, 2], rtol=1e-10, atol=1e-14)


def test_ana_gg_forward():
    nx = 5
    ny = 5

    # Define a mesh
    cs = 0.2
    hxind = [(cs, 41)]
    hyind = [(cs, 41)]
    hzind = [(cs, 41)]
    mesh = discretize.TensorMesh([hxind, hyind, hzind], "CCC")

    # create a model of two blocks, 1 inside the other
    block1 = np.array([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
    block2 = np.array([[-0.7, 0.7], [-0.7, 0.7], [-0.7, 0.7]])

    def get_block_inds(grid, block):
        return np.where(
            (grid[:, 0] > block[0, 0])
            & (grid[:, 0] < block[0, 1])
            & (grid[:, 1] > block[1, 0])
            & (grid[:, 1] < block[1, 1])
            & (grid[:, 2] > block[2, 0])
            & (grid[:, 2] < block[2, 1])
        )

    block1_inds = get_block_inds(mesh.cell_centers, block1)
    block2_inds = get_block_inds(mesh.cell_centers, block2)

    rho1 = 1.0
    rho2 = 2.0

    model = np.zeros(mesh.n_cells)
    model[block1_inds] = rho1
    model[block2_inds] = rho2

    active_cells = model != 0.0
    model_reduced = model[active_cells]

    # Create reduced identity map for Linear Pproblem
    idenMap = maps.IdentityMap(nP=int(sum(active_cells)))

    # Create plane of observations
    xr = np.linspace(-20, 20, nx)
    yr = np.linspace(-20, 20, ny)
    X, Y = np.meshgrid(xr, yr)
    Z = np.ones_like(X) * 3.0
    locXyz = np.c_[X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]

    receivers = gravity.Point(
        locXyz, components=["gxx", "gxy", "gxz", "gyy", "gyz", "gzz"]
    )
    sources = gravity.SourceField([receivers])
    survey = gravity.Survey(sources)

    sim = gravity.Simulation3DIntegral(
        mesh,
        survey=survey,
        rhoMap=idenMap,
        ind_active=active_cells,
        store_sensitivities="forward_only",
    )

    data = sim.dpred(model_reduced)
    d_xx = data[0::6]
    d_xy = data[1::6]
    d_xz = data[2::6]
    d_yy = data[3::6]
    d_yz = data[4::6]
    d_zz = data[5::6]

    # Compute analytical response from dense prism
    prism_1 = Prism(block1[:, 0], block1[:, 1], rho1 * 1000)  # g/cc to kg/m**3
    prism_2 = Prism(block2[:, 0], block2[:, 1], -rho1 * 1000)
    prism_3 = Prism(block2[:, 0], block2[:, 1], rho2 * 1000)

    d = (
        prism_1.gravitational_gradient(locXyz)
        + prism_2.gravitational_gradient(locXyz)
        + prism_3.gravitational_gradient(locXyz)
    ) * 1e9  # convert to Eotvos from 1/s^2

    np.testing.assert_allclose(d_xx, d[..., 0, 0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(d_xy, d[..., 0, 1], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(d_xz, d[..., 0, 2], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(d_yy, d[..., 1, 1], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(d_yz, d[..., 1, 2], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(d_zz, d[..., 2, 2], rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
