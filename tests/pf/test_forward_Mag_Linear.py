import unittest
import discretize
from SimPEG import utils, maps
from SimPEG.potential_fields import magnetics as mag
from geoana.em.static import MagneticPrism
from scipy.constants import mu_0
import numpy as np


def test_ana_mag_forward():
    nx = 5
    ny = 5

    H0 = (50000.0, 60.0, 250.0)
    b0 = mag.analytics.IDTtoxyz(-H0[1], H0[2], H0[0])
    chi1 = 0.01
    chi2 = 0.02

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

    model = np.zeros(mesh.n_cells)
    model[block1_inds] = chi1
    model[block2_inds] = chi2

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
    components = ["bx", "by", "bz", "tmi"]

    rxLoc = mag.Point(locXyz, components=components)
    srcField = mag.SourceField([rxLoc], parameters=H0)
    survey = mag.Survey(srcField)

    # Creat reduced identity map for Linear Pproblem
    idenMap = maps.IdentityMap(nP=int(sum(active_cells)))

    sim = mag.Simulation3DIntegral(
        mesh,
        survey=survey,
        chiMap=idenMap,
        ind_active=active_cells,
        store_sensitivities="forward_only",
    )

    data = sim.dpred(model_reduced)
    d_x = data[0::4]
    d_y = data[1::4]
    d_z = data[2::4]
    d_t = data[3::4]

    tmi = sim.tmi_projection
    d_t2 = d_x * tmi[0] + d_y * tmi[1] + d_z * tmi[2]
    np.testing.assert_allclose(d_t, d_t2)  # double check internal projection

    # Compute analytical response from magnetic prism
    prism_1 = MagneticPrism(block1[:, 0], block1[:, 1], chi1 * b0 / mu_0)
    prism_2 = MagneticPrism(block2[:, 0], block2[:, 1], -chi1 * b0 / mu_0)
    prism_3 = MagneticPrism(block2[:, 0], block2[:, 1], chi2 * b0 / mu_0)

    d = (
        prism_1.magnetic_flux_density(locXyz)
        + prism_2.magnetic_flux_density(locXyz)
        + prism_3.magnetic_flux_density(locXyz)
    )

    np.testing.assert_allclose(d_x, d[:, 0])
    np.testing.assert_allclose(d_y, d[:, 1])
    np.testing.assert_allclose(d_z, d[:, 2])
    np.testing.assert_allclose(d_t, d @ tmi)


def test_ana_mag_grad_forward():
    nx = 5
    ny = 5

    H0 = (50000.0, 60.0, 250.0)
    b0 = mag.analytics.IDTtoxyz(-H0[1], H0[2], H0[0])
    chi1 = 0.01
    chi2 = 0.02

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

    model = np.zeros(mesh.n_cells)
    model[block1_inds] = chi1
    model[block2_inds] = chi2

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
    components = ["bxx", "bxy", "bxz", "byy", "byz", "bzz"]

    rxLoc = mag.Point(locXyz, components=components)
    srcField = mag.SourceField([rxLoc], parameters=H0)
    survey = mag.Survey(srcField)

    # Creat reduced identity map for Linear Pproblem
    idenMap = maps.IdentityMap(nP=int(sum(active_cells)))

    sim = mag.Simulation3DIntegral(
        mesh,
        survey=survey,
        chiMap=idenMap,
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

    # Compute analytical response from magnetic prism
    prism_1 = MagneticPrism(block1[:, 0], block1[:, 1], chi1 * b0 / mu_0)
    prism_2 = MagneticPrism(block2[:, 0], block2[:, 1], -chi1 * b0 / mu_0)
    prism_3 = MagneticPrism(block2[:, 0], block2[:, 1], chi2 * b0 / mu_0)

    d = (
        prism_1.magnetic_field_gradient(locXyz)
        + prism_2.magnetic_field_gradient(locXyz)
        + prism_3.magnetic_field_gradient(locXyz)
    ) * mu_0

    np.testing.assert_allclose(d_xx, d[..., 0, 0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(d_xy, d[..., 0, 1], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(d_xz, d[..., 0, 2], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(d_yy, d[..., 1, 1], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(d_yz, d[..., 1, 2], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(d_zz, d[..., 2, 2], rtol=1e-10, atol=1e-12)


def test_ana_mag_vec_forward():
    nx = 5
    ny = 5

    H0 = (50000.0, 60.0, 250.0)
    b0 = mag.analytics.IDTtoxyz(-H0[1], H0[2], H0[0])

    M1 = utils.mat_utils.dip_azimuth2cartesian(45, -40) * 0.05
    M2 = utils.mat_utils.dip_azimuth2cartesian(120, 32) * 0.1

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

    model = np.zeros((mesh.n_cells, 3))
    model[block1_inds] = M1
    model[block2_inds] = M2

    active_cells = np.any(model != 0.0, axis=1)
    model_reduced = model[active_cells].reshape(-1, order="F")

    # Create plane of observations
    xr = np.linspace(-20, 20, nx)
    yr = np.linspace(-20, 20, ny)
    X, Y = np.meshgrid(xr, yr)
    Z = np.ones_like(X) * 3.0
    locXyz = np.c_[X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]
    components = ["bx", "by", "bz", "tmi"]

    rxLoc = mag.Point(locXyz, components=components)
    srcField = mag.SourceField([rxLoc], parameters=H0)
    survey = mag.Survey(srcField)

    # Create reduced identity map for Linear Pproblem
    idenMap = maps.IdentityMap(nP=int(sum(active_cells)) * 3)

    sim = mag.Simulation3DIntegral(
        mesh,
        survey=survey,
        chiMap=idenMap,
        ind_active=active_cells,
        store_sensitivities="forward_only",
        model_type="vector",
    )

    data = sim.dpred(model_reduced).reshape(-1, 4)

    # Compute analytical response from magnetic prism
    prism_1 = MagneticPrism(block1[:, 0], block1[:, 1], M1 * np.linalg.norm(b0) / mu_0)
    prism_2 = MagneticPrism(block2[:, 0], block2[:, 1], -M1 * np.linalg.norm(b0) / mu_0)
    prism_3 = MagneticPrism(block2[:, 0], block2[:, 1], M2 * np.linalg.norm(b0) / mu_0)

    d = (
        prism_1.magnetic_flux_density(locXyz)
        + prism_2.magnetic_flux_density(locXyz)
        + prism_3.magnetic_flux_density(locXyz)
    )
    tmi = sim.tmi_projection

    np.testing.assert_allclose(data[:, 0], d[:, 0])
    np.testing.assert_allclose(data[:, 1], d[:, 1])
    np.testing.assert_allclose(data[:, 2], d[:, 2])
    np.testing.assert_allclose(data[:, 3], d @ tmi)


def test_ana_mag_amp_forward():
    nx = 5
    ny = 5

    H0 = (50000.0, 60.0, 250.0)
    b0 = mag.analytics.IDTtoxyz(-H0[1], H0[2], H0[0])

    M1 = utils.mat_utils.dip_azimuth2cartesian(45, -40) * 0.05
    M2 = utils.mat_utils.dip_azimuth2cartesian(120, 32) * 0.1

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

    model = np.zeros((mesh.n_cells, 3))
    model[block1_inds] = M1
    model[block2_inds] = M2

    active_cells = np.any(model != 0.0, axis=1)
    model_reduced = model[active_cells].reshape(-1, order="F")

    # Create plane of observations
    xr = np.linspace(-20, 20, nx)
    yr = np.linspace(-20, 20, ny)
    X, Y = np.meshgrid(xr, yr)
    Z = np.ones_like(X) * 3.0
    locXyz = np.c_[X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]
    components = ["bx", "by", "bz"]

    rxLoc = mag.Point(locXyz, components=components)
    srcField = mag.SourceField([rxLoc], parameters=H0)
    survey = mag.Survey(srcField)

    # Create reduced identity map for Linear Pproblem
    idenMap = maps.IdentityMap(nP=int(sum(active_cells)) * 3)

    sim = mag.Simulation3DIntegral(
        mesh,
        survey=survey,
        chiMap=idenMap,
        ind_active=active_cells,
        store_sensitivities="forward_only",
        model_type="vector",
        is_amplitude_data=True,
    )

    data = sim.dpred(model_reduced)

    # Compute analytical response from magnetic prism
    prism_1 = MagneticPrism(block1[:, 0], block1[:, 1], M1 * np.linalg.norm(b0) / mu_0)
    prism_2 = MagneticPrism(block2[:, 0], block2[:, 1], -M1 * np.linalg.norm(b0) / mu_0)
    prism_3 = MagneticPrism(block2[:, 0], block2[:, 1], M2 * np.linalg.norm(b0) / mu_0)

    d = (
        prism_1.magnetic_flux_density(locXyz)
        + prism_2.magnetic_flux_density(locXyz)
        + prism_3.magnetic_flux_density(locXyz)
    )
    d_amp = np.linalg.norm(d, axis=1)

    np.testing.assert_allclose(data, d_amp)


if __name__ == "__main__":
    unittest.main()
