"""
Tests for the DC resistivity problem solved for the initial fields of galvanic sources.
"""

import numpy as np
import pytest
import discretize

from simpeg import maps
from simpeg.electromagnetics import time_domain as tdem


class CountingSolver:
    """Wrap a solver object to count the number of solves."""

    def __init__(self, solver):
        self.solver = solver
        self.n_solves = 0

    def __mul__(self, rhs):
        self.n_solves += 1
        return self.solver * rhs


@pytest.fixture
def mesh():
    h = [(5.0, 2, -1.3), (5.0, 6), (5.0, 2, 1.3)]
    return discretize.TensorMesh([h, h, h], origin="CCC")


@pytest.fixture
def source():
    location = np.array([[-10.0, 0.0, -2.5], [10.0, 0.0, -2.5]])
    return tdem.sources.LineCurrent(
        [], location=location, waveform=tdem.sources.StepOffWaveform()
    )


def get_simulation(mesh, source, formulation):
    simulation = getattr(tdem, f"Simulation3D{formulation}")(
        mesh,
        survey=tdem.Survey([source]),
        time_steps=[(1e-5, 2)],
        sigmaMap=maps.ExpMap(mesh),
    )
    simulation.model = np.full(mesh.n_cells, np.log(0.1))
    return simulation


@pytest.mark.parametrize(
    "formulation, field",
    [("ElectricField", "e"), ("MagneticField", "j"), ("CurrentDensity", "j")],
)
def test_number_of_dc_solves(mesh, source, formulation, field):
    """Test the DC problem is solved only once for the potentials of a source."""
    simulation = get_simulation(mesh, source, formulation)
    simulation._Adcinv = solver = CountingSolver(simulation.Adcinv)
    rng = np.random.default_rng(seed=42)
    initial = getattr(source, f"{field}Initial")
    initial_deriv = getattr(source, f"{field}InitialDeriv")

    fields = initial(simulation)
    assert solver.n_solves == 1
    initial(simulation)
    assert solver.n_solves == 1

    # one more solve for each derivative, none for the potentials
    initial_deriv(simulation, rng.normal(size=mesh.n_cells))
    assert solver.n_solves == 2
    initial_deriv(simulation, rng.normal(size=fields.size), adjoint=True)
    assert solver.n_solves == 3
