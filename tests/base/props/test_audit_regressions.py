"""Regression tests for issues found while auditing the parametrized physical properties."""

import discretize
import numpy as np
import pytest

from simpeg import maps
from simpeg.electromagnetics.static import resistivity as dc


def _dc_sim(**kwargs):
    mesh = discretize.TensorMesh([4, 4, 4])
    return mesh, dc.Simulation3DCellCentered(mesh, survey=dc.Survey([]), **kwargs)


def test_failed_parametrize_keeps_value():
    mesh, sim = _dc_sim(sigma=np.ones(64))
    with pytest.raises(ValueError):
        sim.parametrize(sigma=maps.IdentityMap(nP=mesh.n_cells + 3))
    np.testing.assert_allclose(sim.sigma[:2], [1.0, 1.0])
    assert not sim.is_parametrized("sigma")


def test_failed_parametrize_keeps_mapping():
    mesh, sim = _dc_sim(sigma=maps.ExpMap(nP=64))
    old = sim.parametrizations.sigma
    with pytest.raises(ValueError):
        sim.parametrize(rho=maps.IdentityMap(nP=mesh.n_cells + 3))
    assert sim.parametrizations.sigma is old
    assert "rho" not in sim.parametrizations
