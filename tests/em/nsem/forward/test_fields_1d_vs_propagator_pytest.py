# noqa: D100
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

ABS_TOLERANCE = 3e-3


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
    ("e", "dirichlet"),
    ("e", "neumann"),
    ("h", "dirichlet"),
    ("h", "neumann"),
]


@pytest.mark.parametrize("solution_type, boundary_condition", CASES_LIST_HALFSPACE)
def test_propagator_fv1d_crosscheck(solution_type, boundary_condition, mesh, mapping):
    """Validate 1d fields against propagator solution."""
    sig_1d = get_model(mesh, "halfspace")
    freq = 100.0

    # Propagator solution on nodes
    Ed, Eu, Hd, Hu = getEHfields(
        mesh, sig_1d, freq, mesh.nodes, scaleUD=True, scaleValue=1
    )

    if solution_type == "e":
        u1 = primary_e_1d_solution(mesh, sig_1d, freq, boundary_condition, 500)
        if boundary_condition == "dirichlet":
            u0 = Ed + Eu
            u0 /= u0[-1]
        elif boundary_condition == "neumann":
            u0 = Hd + Hu
            u0 /= u0[-1]
            u0 = mesh.average_node_to_cell @ u0

            u1 = mesh.nodal_gradient @ u1
            u1 /= -1.0j * 2 * np.pi * freq * mu_0

    elif solution_type == "h":
        u1 = primary_h_1d_solution(mesh, sig_1d, freq, boundary_condition, 500)
        if boundary_condition == "dirichlet":
            u0 = Hd + Hu
            u0 /= u0[-1]
        elif boundary_condition == "neumann":
            u0 = Ed + Eu
            u0 /= u0[-1]
            u0 = mesh.average_node_to_cell @ u0

            u1 = mesh.nodal_gradient @ u1
            u1 /= sig_1d

    assert np.real(u1) == pytest.approx(np.real(u0), abs=ABS_TOLERANCE)
    assert np.imag(u1) == pytest.approx(np.imag(u0), abs=ABS_TOLERANCE)
