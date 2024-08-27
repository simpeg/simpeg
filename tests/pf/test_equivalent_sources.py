import pytest

import numpy as np
from discretize import TensorMesh
from discretize.utils import mesh_builder_xyz, mkvc, refine_tree_xyz

import numba
import choclo

from simpeg import (
    data_misfit,
    directives,
    inverse_problem,
    inversion,
    maps,
    optimization,
    regularization,
    utils,
)
from simpeg.potential_fields import gravity, magnetics


def create_topography(x, y, amplitude=50, scale_factor=100):
    """Create synthetic Gaussian topography from a function"""
    return amplitude * np.exp(
        -0.5 * ((x / scale_factor) ** 2.0 + (y / scale_factor) ** 2.0)
    )


def create_grid(x_range, y_range, spacing):
    """Create a 2D horizontal coordinates grid."""
    x_start, x_end = x_range
    y_start, y_end = y_range
    x, y = np.meshgrid(
        np.linspace(x_start, x_end, spacing), np.linspace(y_start, y_end, spacing)
    )
    return x, y


@pytest.fixture
def mesh(topography, coordinates):
    """Sample 2D mesh to use with equivalent sources."""
    # Define parameters for building the mesh
    h = [5, 5]
    padDist = np.ones((2, 2)) * 100
    nCpad = [2, 4]
    # Build mesh
    mesh = mesh_builder_xyz(
        topography[:, :2], h, padding_distance=padDist, mesh_type="TREE"
    )
    # Refine mesh
    mesh = refine_tree_xyz(
        mesh,
        coordinates[:, :2],
        method="radial",
        octree_levels=nCpad,
        octree_levels_padding=nCpad,
        finalize=True,
    )
    return mesh


@pytest.fixture
def topography():
    """Synthetic topography grid."""
    x, y = create_grid(x_range=(-200, 200), y_range=(-200, 200), spacing=50)
    z = create_topography(x, y)
    return np.c_[mkvc(x), mkvc(y), mkvc(z)]


@pytest.fixture
def coordinates():
    """Synthetic observation points grid on 5m above the topography."""
    x, y = create_grid(x_range=(-100, 100), y_range=(-100, 100), spacing=50)
    z = create_topography(x, y) + 5.0
    return np.c_[mkvc(x), mkvc(y), mkvc(z)]


@pytest.fixture
def coordinates_constant_height():
    """Synthetic observation points grid with all points at the same height."""
    x, y = create_grid(x_range=(-100, 100), y_range=(-100, 100), spacing=50)
    z = np.full_like(x, 5.0)
    return np.c_[mkvc(x), mkvc(y), mkvc(z)]


@pytest.fixture
def block_model(mesh):
    """Build a block model."""
    model = utils.model_builder.add_block(
        mesh.cell_centers,
        np.zeros(mesh.n_cells),
        np.r_[-20, -20],
        np.r_[20, 20],
        1.0,
    )
    return model


class TestGravityEquivalentSources:

    # @pytest.fixture
    # def synthetic_data(self, block, coordinates):
    #     """Synthetic gravity data."""
    #     data = gravity_forward(coordinates, block, 3200.0)
    #     data += np.random.default_rng(seed=42).normal(scale=1.0, size=data.size)
    #     return data

    def test_gravity_equivalent_sources(self, coordinates, mesh, block_model):
        receivers = gravity.Point(coordinates)
        source_field = gravity.SourceField([receivers])
        survey = gravity.Survey(source_field)
        identity_map = maps.IdentityMap(nP=mesh.n_cells)

        bottom, top = -5.0, 0.0
        eq_sources_simulation = gravity.SimulationEquivalentSourceLayer(
            mesh, top, bottom, survey=survey, rhoMap=identity_map
        )

        # Create the gravity forward model operator
        eq_sources_simulation = gravity.SimulationEquivalentSourceLayer(
            mesh,
            0.0,
            -5.0,
            survey=survey,
            rhoMap=identity_map,
            store_sensitivities="ram",
        )

        # Define model
        model = 2.67 * block_model

        synthetic_data = eq_sources_simulation.make_synthetic_data(
            model,
            relative_error=0.0,
            noise_floor=1e-6,
            add_noise=True,
            random_seed=1,
        )

        # Create a regularization
        reg = regularization.Sparse(mesh, mapping=identity_map)
        reg.norms = [0, 0, 0]
        reg.gradient_type = "components"

        # Data misfit function
        dmis = data_misfit.L2DataMisfit(
            simulation=eq_sources_simulation, data=synthetic_data
        )

        # Add directives to the inversion
        opt = optimization.ProjectedGNCG(
            maxIter=40,
            lower=-5.0,
            upper=5.0,
            maxIterLS=5,
            maxIterCG=20,
            tolCG=1e-4,
        )

        inversion_prob = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=1e3)

        # Build directives
        update_irls = directives.Update_IRLS(
            f_min_change=1e-3,
            max_irls_iterations=30,
            beta_tol=1e-1,
            beta_search=False,
        )
        sensitivity_weights = directives.UpdateSensitivityWeights()
        update_jacobi = directives.UpdatePreconditioner()
        inv = inversion.BaseInversion(
            inversion_prob,
            directiveList=[update_irls, sensitivity_weights, update_jacobi],
        )

        recovered_model = inv.run(np.zeros(mesh.n_cells))

        # Compute predicted data
        prediction = eq_sources_simulation.dpred(recovered_model)

        # -------------------------------
        import matplotlib.pyplot as plt

        x, y = coordinates[:, 0].ravel(), coordinates[:, 1].ravel()
        size_per_side = int(np.sqrt(x.size))
        shape = (size_per_side, size_per_side)
        x, y = x.reshape(shape), y.reshape(shape)

        vmin = min([prediction.min(), synthetic_data.dobs.min()])
        vmax = max([prediction.max(), synthetic_data.dobs.max()])
        _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        tmp = ax1.pcolormesh(x, y, prediction.reshape(shape), vmin=vmin, vmax=vmax)
        ax2.pcolormesh(x, y, synthetic_data.dobs.reshape(shape), vmin=vmin, vmax=vmax)

        for ax in (ax1, ax2):
            ax.set_aspect("equal")
        plt.colorbar(tmp, ax=(ax1, ax2))
        plt.show()
        # -------------------------------
        diff = prediction - synthetic_data.dobs
        maxabs = np.max(np.abs(diff))

        tmp = plt.pcolormesh(x, y, diff.reshape(shape), vmin=-maxabs, vmax=maxabs)
        plt.gca().set_aspect("equal")
        plt.colorbar(tmp)
        plt.show()
        # -------------------------------

        dmisfit = inv.invProb.dmisfit(recovered_model)
        print(dmisfit, prediction.shape[0])
        assert dmisfit < prediction.shape[0] * 1.15

        np.testing.assert_almost_equal(prediction, synthetic_data.dobs, decimal=3)
