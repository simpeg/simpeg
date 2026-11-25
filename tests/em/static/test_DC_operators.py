"""
Tests for the discrete operators of the 3D DC resistivity problem.
"""

import numpy as np
import pytest
import discretize

from simpeg import maps
from simpeg.electromagnetics import time_domain as tdem
from simpeg.electromagnetics.static import resistivity as dc
from simpeg.electromagnetics.static.resistivity._operators import (
    CellCenteredDCOperator,
    NodalDCOperator,
)

CELL_CENTERED = (dc.Simulation3DCellCentered, CellCenteredDCOperator)
NODAL = (dc.Simulation3DNodal, NodalDCOperator)
SIM_CLASSES = [dc.Simulation3DCellCentered, dc.Simulation3DNodal]

CLASSES_AND_BCS = [
    (*CELL_CENTERED, "Dirichlet"),
    (*CELL_CENTERED, "Neumann"),
    (*CELL_CENTERED, "Robin"),
    (*NODAL, "Neumann"),
    (*NODAL, "Robin"),
]


@pytest.fixture
def mesh():
    return discretize.TensorMesh([6, 7, 8], origin="CCN")


@pytest.fixture
def model(mesh):
    return np.random.default_rng(seed=42).normal(size=mesh.n_cells)


@pytest.mark.parametrize("sim_class, operator_class, bc_type", CLASSES_AND_BCS)
@pytest.mark.parametrize("prop_map", ["sigmaMap", "rhoMap"])
def test_operator_without_dc_simulation(
    mesh, model, sim_class, operator_class, bc_type, prop_map
):
    """
    The operators only need an object that holds the electrical properties.

    Use a TDEM simulation as that object, and compare against the matrices of
    the DC simulation.
    """
    kwargs = {prop_map: maps.ExpMap(mesh)}
    physprops = tdem.Simulation3DElectricField(mesh, **kwargs)
    physprops.model = model
    simulation = sim_class(mesh, bc_type=bc_type, **kwargs)
    simulation.model = model

    operator = operator_class(mesh, bc_type=bc_type)
    A = operator.system_matrix(physprops)
    np.testing.assert_allclose(A.toarray(), simulation.getA().toarray())

    rng = np.random.default_rng(seed=5)
    phi = rng.normal(size=A.shape[0])
    v = rng.normal(size=model.size)
    w = rng.normal(size=A.shape[0])
    np.testing.assert_allclose(
        operator.system_matrix_deriv(physprops, phi, v),
        simulation.getADeriv(phi, v),
    )
    np.testing.assert_allclose(
        operator.system_matrix_deriv(physprops, phi, w, adjoint=True),
        simulation.getADeriv(phi, w, adjoint=True),
    )


@pytest.mark.parametrize(
    "sim_class, operator_class, bc_type",
    [(*CELL_CENTERED, "invalid"), (*NODAL, "invalid"), (*NODAL, "Dirichlet")],
)
def test_invalid_bc_type(mesh, sim_class, operator_class, bc_type):
    with pytest.raises(ValueError, match="bc_type"):
        operator_class(mesh, bc_type=bc_type)
    with pytest.raises(ValueError, match="bc_type"):
        sim_class(mesh, bc_type=bc_type)


@pytest.mark.parametrize("sim_class", SIM_CLASSES)
def test_bc_type_setter(mesh, model, sim_class):
    """Test that updating ``bc_type`` updates the system matrix."""
    sigma = np.exp(model)
    simulation = sim_class(mesh, bc_type="Robin", sigma=sigma)
    simulation.bc_type = "Neumann"
    assert simulation.bc_type == "Neumann"
    expected = sim_class(mesh, bc_type="Neumann", sigma=sigma)
    np.testing.assert_equal(simulation.getA().toarray(), expected.getA().toarray())


@pytest.mark.parametrize("sim_class", SIM_CLASSES)
def test_set_bc_deprecated(mesh, sim_class):
    simulation = sim_class(mesh, bc_type="Robin")
    with pytest.warns(FutureWarning, match="`setBC` has been deprecated"):
        simulation.setBC()
    assert simulation.bc_type == "Robin"


@pytest.mark.parametrize("sim_class, operator_class, bc_type", CLASSES_AND_BCS)
def test_resistivity_argument_deprecated(
    mesh, model, sim_class, operator_class, bc_type
):
    """
    Test the deprecated ``resistivity`` argument of ``getA``.

    For the Neumann and Dirichlet conditions, the system matrix only depends on
    the resistivity through the inner product matrix, so we can compare against
    a simulation with that resistivity.
    """
    resistivity = np.exp(model)
    simulation = sim_class(mesh, bc_type=bc_type, sigma=np.ones(mesh.n_cells))
    with pytest.warns(FutureWarning, match="`resistivity` argument"):
        A = simulation.getA(resistivity=resistivity)
    if bc_type != "Robin":
        expected = sim_class(mesh, bc_type=bc_type, rho=resistivity)
        np.testing.assert_allclose(A.toarray(), expected.getA().toarray())


def test_symmetric_null_space_fix(mesh, model):
    """Test the two ways of removing the null space give the same fields."""
    physprops = dc.Simulation3DNodal(mesh, bc_type="Neumann", sigma=np.exp(model))
    operator = NodalDCOperator(mesh, bc_type="Neumann")
    symmetric = NodalDCOperator(mesh, bc_type="Neumann", symmetric_null_space_fix=True)
    A = operator.system_matrix(physprops)
    A_symmetric = symmetric.system_matrix(physprops)
    assert abs(A - A.T).max() > 0
    assert abs(A_symmetric - A_symmetric.T).max() == 0

    # a source term that adds up to zero
    q = np.zeros(mesh.n_nodes)
    q[[10, -10]] = [1.0, -1.0]
    Grad = mesh.nodal_gradient
    e = Grad @ (physprops.solver(A) * q)
    e_symmetric = Grad @ (physprops.solver(A_symmetric) * q)
    np.testing.assert_allclose(e, e_symmetric, atol=1e-10 * np.abs(e).max())


def test_tdem_galvanic_not_implemented(mesh):
    """Test the B formulation does not support the initial DC problem."""
    simulation = tdem.Simulation3DMagneticFluxDensity(mesh, sigma=1.0)
    with pytest.raises(NotImplementedError, match="galvanic sources"):
        simulation.Adcinv
