"""
Test NSEM solutions against propagator solution.
"""

import re
import pytest
from scipy.constants import mu_0
import numpy as np
from discretize import TensorMesh
from simpeg.electromagnetics.natural_source.utils import (
    primary_e_1d_solution,
    primary_h_1d_solution,
    getEHfields,
)
from simpeg import maps

REL_TOLERANCE = 0.05
ABS_TOLERANCE = 1e-5


@pytest.fixture
def mesh():
    """Mesh for testing."""
    hz = 5
    nc = 500
    npad = 30
    pf = 1.1
    mesh = TensorMesh([[(hz, npad, -pf), (hz, nc), (hz, npad)]], "N")
    mesh.x0 = -np.r_[mesh.h[0][: npad + 1].sum()] - 2000
    return mesh


@pytest.fixture
def mapping(mesh):
    """Return mapping."""
    return maps.IdentityMap(mesh)


def get_model(mesh, model_type):
    """Model used for testing."""
    model = 1e-8 * np.ones(mesh.nC)
    model[mesh.cell_centers < 0.0] = 1e-2

    if model_type == "layer":
        model[mesh.cell_centers < -1000.0] = 1e-1

    return model


CASES_LIST_HALFSPACE = [
    ("e", "dirichlet", "dirichlet"),
    ("e", "dirichlet", "robin"),
    ("e", "neumann", "dirichlet"),
    ("e", "neumann", "robin"),
    ("h", "dirichlet", "dirichlet"),
    ("h", "dirichlet", "robin"),
    ("h", "neumann", "dirichlet"),
    ("h", "neumann", "robin"),
]


@pytest.mark.parametrize("solution_type, top_bc, bot_bc", CASES_LIST_HALFSPACE)
def test_propagator_fv1d_crosscheck(solution_type, top_bc, bot_bc, mesh, mapping):
    """Validate 1d fields against propagator solution."""
    sig_1d = get_model(mesh, "halfspace")
    freq = 100.0

    # Propagator solution on nodes
    Ed, Eu, Hd, Hu = getEHfields(
        mesh, sig_1d, freq, mesh.nodes, scaleUD=True, scaleValue=1
    )

    if solution_type == "e":
        u1 = primary_e_1d_solution(mesh, sig_1d, freq, top_bc, bot_bc, 500)
        if top_bc == "dirichlet":
            u0 = Ed + Eu
            u0 /= u0[-1]
        elif top_bc == "neumann":
            u0 = Hd + Hu
            u0 /= u0[-1]
            u0 = mesh.average_node_to_cell @ u0

            u1 = mesh.nodal_gradient @ u1
            u1 /= -1.0j * 2 * np.pi * freq * mu_0

    elif solution_type == "h":
        u1 = primary_h_1d_solution(mesh, sig_1d, freq, top_bc, bot_bc, 500)
        if top_bc == "dirichlet":
            u0 = Hd + Hu
            u0 /= u0[-1]
        elif top_bc == "neumann":
            u0 = Ed + Eu
            u0 /= u0[-1]
            u0 = mesh.average_node_to_cell @ u0

            u1 = mesh.nodal_gradient @ u1
            u1 /= sig_1d

    np.testing.assert_allclose(
        np.real(u1), np.real(u0), rtol=REL_TOLERANCE, atol=ABS_TOLERANCE
    )
    np.testing.assert_allclose(
        np.imag(u1), np.imag(u0), rtol=REL_TOLERANCE, atol=ABS_TOLERANCE
    )


@pytest.mark.parametrize(
    "primary_solution_func", (primary_e_1d_solution, primary_h_1d_solution)
)
class TestErrors:
    """
    Test errors raised by ``primary_e_1d_solution`` and ``primary_h_1d_solution``.
    """

    def test_invalid_top_bc(self, mesh, primary_solution_func):
        sig_1d = get_model(mesh, "halfspace")
        top_bc = "invalid bc"
        msg = re.escape(f"Invalid 'top_bc' equal to '{top_bc}'")
        with pytest.raises(ValueError, match=msg):
            primary_solution_func(mesh, sigma_1d=sig_1d, freq=100, top_bc=top_bc)

    def test_invalid_bottom_bc(self, mesh, primary_solution_func):
        sig_1d = get_model(mesh, "halfspace")
        bottom_bc = "invalid bc"
        msg = re.escape(f"Invalid 'bot_bc' equal to '{bottom_bc}'")
        with pytest.raises(ValueError, match=msg):
            primary_solution_func(mesh, sigma_1d=sig_1d, freq=100, bot_bc=bottom_bc)

    def test_invalid_sigma_size(self, mesh, primary_solution_func):
        invalid_sigma = np.ones(3)
        msg = re.escape(
            "Number of cells in vertical direction must match length of 'sigma_1d'."
        )
        with pytest.raises(ValueError, match=msg):
            primary_solution_func(mesh, sigma_1d=invalid_sigma, freq=100)
