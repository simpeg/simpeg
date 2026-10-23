"""Regression tests for issues found while auditing the parametrized physical properties."""

import discretize
from discretize.utils import inverse_property_tensor
import numpy as np
import pytest

from simpeg import maps, props
from simpeg.simulation import LinearSimulation
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


@pytest.mark.parametrize("n_components", [3, 6])
@pytest.mark.parametrize("name", ["sigma", "rho"])
def test_flattened_anisotropic_values(name, n_components):
    mesh = discretize.TensorMesh([3, 3, 3])
    other = "rho" if name == "sigma" else "sigma"
    rng = np.random.default_rng(0)
    value = rng.uniform(1.0, 2.0, mesh.n_cells * n_components)
    sim = dc.Simulation3DNodal(mesh, survey=dc.Survey([]), **{name: value})
    np.testing.assert_allclose(getattr(sim, name), value)
    np.testing.assert_allclose(
        getattr(sim, other), inverse_property_tensor(mesh, value)
    )


def test_deprecated_map_set_to_none_removes_parametrization():
    G = np.random.default_rng(0).normal(size=(3, 10))
    sim = LinearSimulation(G=G, linear_model=maps.ExpMap())
    assert sim.needs_model
    with pytest.warns(FutureWarning):
        sim.model_map = None
    assert not sim.needs_model


def test_deprecated_recip_map_set_to_none_removes_parametrization():
    mesh = discretize.TensorMesh([3, 3, 3])
    sim = dc.Simulation3DCellCentered(mesh, survey=dc.Survey([]), rho=maps.ExpMap(mesh))
    with pytest.warns(FutureWarning):
        sim.sigmaMap = None
    assert not sim.needs_model


def test_get_without_objtype_uses_reciprocal():
    mesh = discretize.TensorMesh([3, 3, 3])
    sim = dc.Simulation3DCellCentered(
        mesh, survey=dc.Survey([]), rho=np.full(mesh.n_cells, 4.0)
    )
    value = type(sim).sigma.__get__(sim)
    np.testing.assert_allclose(value, 0.25)


def test_shadowing_one_of_a_reciprocal_pair_errors_clearly():
    class Child(dc.Simulation3DCellCentered):
        rho = 1.0

    mesh = discretize.TensorMesh([3, 3, 3])
    with pytest.raises(TypeError, match="reciprocal"):
        Child(mesh, survey=dc.Survey([]), sigma=np.ones(mesh.n_cells))


def test_warnings_point_at_user_code():
    import warnings

    mesh = discretize.TensorMesh([3, 3, 3])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        sim = dc.Simulation3DNodal(
            mesh, survey=dc.Survey([]), sigmaMap=maps.IdentityMap(nP=mesh.n_cells)
        )
        sim.sigma = np.ones(mesh.n_cells)
    record = [w for w in record if w.category in (FutureWarning, UserWarning)]
    record = [w for w in record if "pymatsolver" not in str(w.message)]
    assert len(record) == 2
    for w in record:
        assert w.filename == __file__, (str(w.message), w.filename)
    assert "0.28.0" in str(record[0].message)


def test_reparametrize_clears_hierarchical_caches():
    from simpeg.electromagnetics import frequency_domain as fdem

    mesh = discretize.TensorMesh([[(10.0, 4)]] * 3, "CCC")
    n = mesh.n_cells

    def make():
        return fdem.Simulation3DHierarchicalElectricField(
            mesh, face_conductance=0.1 * np.ones(mesh.n_faces), sigma=maps.ExpMap()
        )

    m = np.log(0.01) * np.ones(n)
    sim = make()
    sim.model = m
    sim.MeSigma
    sim.parametrize(sigma=maps.ExpMap() * maps.LinearMap(np.eye(n), np.ones(n)))
    fresh = make()
    fresh.model = m + 1.0
    np.testing.assert_allclose(
        sim.MeSigma.diagonal().sum(), fresh.MeSigma.diagonal().sum()
    )


def test_model_can_be_passed_with_parametrization_to_constructor():
    mesh = discretize.TensorMesh([3, 3, 3])
    m = np.zeros(mesh.n_cells)
    sim = dc.Simulation3DNodal(
        mesh, survey=dc.Survey([]), sigma=maps.ExpMap(mesh), model=m
    )
    np.testing.assert_allclose(sim.model, m)
    np.testing.assert_allclose(sim.sigma, 1.0)


def test_delete_model_sets_none():
    mesh = discretize.TensorMesh([3, 3, 3])
    sim = dc.Simulation3DNodal(mesh, survey=dc.Survey([]), sigma=maps.ExpMap(mesh))
    sim.model = np.zeros(mesh.n_cells)
    del sim.model
    assert sim.model is None


def test_scalar_values_are_python_scalars():
    mesh = discretize.TensorMesh([3, 3, 3])
    sim = dc.Simulation3DNodal(mesh, survey=dc.Survey([]), sigma=2.0)
    assert np.isscalar(sim.sigma)
    assert not isinstance(sim.sigma, np.ndarray)


def test_set_model_reports_updates():
    mesh = discretize.TensorMesh([3, 3, 3])
    sim = dc.Simulation3DNodal(mesh, survey=dc.Survey([]), sigma=maps.ExpMap(mesh))
    m = np.zeros(mesh.n_cells)
    assert sim._set_model(m) is True
    assert sim._set_model(m.copy()) is False
    assert sim._set_model(m + 1) is True


def test_model_without_parametrization_errors():
    mesh = discretize.TensorMesh([3, 3, 3])
    sim = dc.Simulation3DNodal(mesh, survey=dc.Survey([]), sigma=np.ones(27))
    with pytest.raises(AttributeError, match="no parametrized properties"):
        sim.model = np.zeros(27)


@pytest.mark.parametrize(
    "removed",
    [
        lambda: props.Mapping("a"),
        lambda: props.Derivative("a", "b"),
        lambda: props.Invertible("a"),
        lambda: props.Reciprocal("a", "b"),
    ],
)
def test_removed_apis_raise_not_implemented(removed):
    with pytest.raises(NotImplementedError):
        removed()


@pytest.mark.parametrize("old", ["mapping", "optional"])
def test_physical_property_removed_arguments(old):
    with pytest.raises(TypeError, match=old):
        props.PhysicalProperty("details", **{old: True})


def test_meta_simulation_needs_model():
    from simpeg.meta import MetaSimulation

    mesh = discretize.TensorMesh([3, 3, 3])
    sim = dc.Simulation3DNodal(mesh, survey=dc.Survey([]), sigma=maps.ExpMap(mesh))
    meta = MetaSimulation([sim], [maps.IdentityMap(nP=mesh.n_cells)])
    assert meta.needs_model


def test_parametrization_list():
    import pickle

    mesh = discretize.TensorMesh([3, 3, 3])
    sim = dc.Simulation3DNodal(mesh, survey=dc.Survey([]), sigma=maps.ExpMap(mesh))
    plist = sim.parametrizations
    assert "sigma" in plist
    assert list(plist) == ["sigma"]
    assert len(plist) == 1
    assert "sigma" in str(plist)
    with pytest.raises(AttributeError, match="read-only"):
        plist.sigma = maps.IdentityMap()
    with pytest.raises(AttributeError, match="no attribute"):
        plist.rho

    clone = pickle.loads(pickle.dumps(sim))
    assert list(clone.parametrizations) == ["sigma"]
    clone.model = np.zeros(mesh.n_cells)
    np.testing.assert_allclose(clone.sigma, 1.0)


def test_assigning_mapping_directly_errors():
    mesh = discretize.TensorMesh([3, 3, 3])
    sim = dc.Simulation3DNodal(mesh, survey=dc.Survey([]), sigma=np.ones(27))
    with pytest.raises(TypeError, match="Cannot assign a mapping directly"):
        sim.sigma = maps.ExpMap(mesh)


def test_assigning_value_over_parametrization_warns():
    mesh = discretize.TensorMesh([3, 3, 3])
    sim = dc.Simulation3DNodal(mesh, survey=dc.Survey([]), sigma=maps.ExpMap(mesh))
    with pytest.warns(UserWarning, match="removes the existing parametrization"):
        sim.sigma = np.ones(27)
    assert not sim.is_parametrized("sigma")


def test_invertible_properties():
    inv = dc.Simulation3DNodal.invertible_properties
    assert set(inv) == {"sigma", "rho"}
    assert set(inv) <= set(dc.Simulation3DNodal.physical_properties)


def test_positional_property_arguments_are_rejected():
    from simpeg.potential_fields import gravity

    mesh = discretize.TensorMesh([3, 3, 3])
    # this used to be `rhoMap`, and would now silently be the `engine`
    with pytest.raises(TypeError, match="positional"):
        gravity.Simulation3DIntegral(mesh, None, maps.IdentityMap())
