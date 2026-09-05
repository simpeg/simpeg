import re
import unittest
import numpy as np
import numpy.testing as npt
import inspect
import pytest

from simpeg import maps
from simpeg import props
from simpeg.props import _add_deprecated_physical_property_functions


@_add_deprecated_physical_property_functions("sigma")
class SingleExample(props.HasModel):
    sigma = props.PhysicalProperty("Electrical conductivity (S/m)")

    def __init__(self, sigma=None, **kwargs):
        super().__init__(**kwargs)
        self._init_property(sigma=sigma)


class SingleNotInvertible(props.HasModel):
    rho = props.PhysicalProperty("Electrical resistivity (Ohm m)", invertible=False)

    def __init__(self, rho=None, **kwargs):
        super().__init__(**kwargs)
        self._init_property(rho=rho)


@_add_deprecated_physical_property_functions("sigma")
class OptionalInvertible(props.HasModel):
    sigma = props.PhysicalProperty("Electrical conductivity (S/m)", default=None)

    def __init__(self, sigma=None, **kwargs):
        super().__init__(**kwargs)
        self._init_property(sigma=sigma)


@_add_deprecated_physical_property_functions("sigma")
@_add_deprecated_physical_property_functions("rho")
class ReciprocalExample(props.HasModel):
    sigma = props.PhysicalProperty("Electrical conductivity (S/m)", reciprocal="rho")
    rho = props.PhysicalProperty("Electrical resistivity (Ohm m)", reciprocal="sigma")

    def __init__(self, sigma=None, rho=None, **kwargs):
        super().__init__(**kwargs)
        self._init_recip_properties(sigma=sigma, rho=rho)


@_add_deprecated_physical_property_functions("sigma")
@_add_deprecated_physical_property_functions("rho")
class ReciprocalWithDefault(props.HasModel):
    sigma = props.PhysicalProperty(
        "Electrical conductivity (S/m)", default=0.1, reciprocal="rho"
    )
    rho = props.PhysicalProperty("Electrical resistivity (Ohm m)", reciprocal="sigma")

    def __init__(self, sigma=None, rho=None, **kwargs):
        super().__init__(**kwargs)
        self._init_recip_properties(sigma=sigma, rho=rho)


@_add_deprecated_physical_property_functions("sigma")
@_add_deprecated_physical_property_functions("rho")
class ReciprocalWithNoneDefault(props.HasModel):
    sigma = props.PhysicalProperty(
        "Electrical conductivity (S/m)", default=None, reciprocal="rho"
    )
    rho = props.PhysicalProperty("Electrical resistivity (Ohm m)", reciprocal="sigma")

    def __init__(self, sigma=None, rho=None, **kwargs):
        super().__init__(**kwargs)
        self._init_recip_properties(sigma=sigma, rho=rho)


@_add_deprecated_physical_property_functions("sigma")
@_add_deprecated_physical_property_functions("rho")
class ReciprocalSingleInvertible(props.HasModel):
    sigma = props.PhysicalProperty("Electrical conductivity (S/m)", reciprocal="rho")
    rho = props.PhysicalProperty(
        "Electrical resistivity (Ohm m)", invertible=False, reciprocal="sigma"
    )

    def __init__(self, sigma=None, rho=None, **kwargs):
        super().__init__(**kwargs)
        self._init_recip_properties(sigma=sigma, rho=rho)


class ReciprocalNotInvertible(props.HasModel):
    sigma = props.PhysicalProperty(
        "Electrical conductivity (S/m)", invertible=False, reciprocal="rho"
    )
    rho = props.PhysicalProperty(
        "Electrical resistivity (Ohm m)", invertible=False, reciprocal="sigma"
    )

    def __init__(self, sigma=None, rho=None, **kwargs):
        super().__init__(**kwargs)
        self._init_recip_properties(sigma=sigma, rho=rho)


@_add_deprecated_physical_property_functions("sigma")
class MultipleInvertible(props.HasModel):
    sigma = props.PhysicalProperty("Saturated hydraulic conductivity")
    rho = props.PhysicalProperty("fitting parameter")
    gamma = props.PhysicalProperty("fitting parameter")

    def __init__(
        self,
        sigma=24.96,
        rho=1.175e06,
        gamma=4.74,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._init_property(sigma=sigma, rho=rho, gamma=gamma)


ALL_PROP_CLASSES = {
    SingleExample,
    SingleNotInvertible,
    ReciprocalExample,
    ReciprocalWithDefault,
    ReciprocalSingleInvertible,
    ReciprocalNotInvertible,
    MultipleInvertible,
    OptionalInvertible,
    ReciprocalWithNoneDefault,
}

SIGMA_CLASSES = {cls for cls in ALL_PROP_CLASSES if "sigma" in cls.physical_properties}
RHO_CLASSES = {cls for cls in ALL_PROP_CLASSES if "rho" in cls.physical_properties}
INVERTIBLE_SIGMA_CLASSES = {cls for cls in SIGMA_CLASSES if cls.sigma.invertible}
INVERTIBLE_RHO_CLASSES = {cls for cls in RHO_CLASSES if cls.rho.invertible}

RECIPROCAL_CLASSES = {
    cls
    for cls in ALL_PROP_CLASSES
    if any(prop.has_reciprocal for prop in cls.physical_properties.values())
}

DEFAULT_CLASSES = {
    cls
    for cls in ALL_PROP_CLASSES
    if any(prop.optional for prop in cls.physical_properties.values())
}


@pytest.mark.parametrize("modeler", ALL_PROP_CLASSES)
def test_internal_property_dictionary(modeler):
    params = inspect.signature(modeler).parameters
    params = set(name for name in params if name != "kwargs")

    assert params == set(modeler.physical_properties.keys())


@pytest.mark.parametrize("modeler", RECIPROCAL_CLASSES)
@pytest.mark.parametrize("assign, retrieve", [("sigma", "rho"), ("sigma", "rho")])
def test_get_recip_prop(modeler, assign, retrieve):
    prop1 = getattr(modeler, assign)
    assert prop1._reciprocal == retrieve

    prop2 = getattr(modeler, retrieve)
    assert prop2._reciprocal == assign

    assert prop1.get_reciprocal(modeler) is prop2


@pytest.mark.parametrize("modeler", INVERTIBLE_SIGMA_CLASSES)
@pytest.mark.parametrize("dep", [False, "init", "set attrMap"])
def test_invertible_map_assignment(modeler, dep):
    exp_map = maps.ExpMap()
    cls_name = modeler.__name__
    if dep:
        if dep == "init":
            with pytest.warns(
                FutureWarning,
                match=f"Passing mapping argument sigmaMap to {cls_name} is deprecated.*",
            ):
                pm = modeler(sigmaMap=exp_map)
        else:
            pm = modeler()
            with pytest.warns(
                FutureWarning,
                match=f"Setting `{cls_name}\\.sigmaMap` directly is deprecated.*",
            ):
                pm.sigmaMap = exp_map
    else:
        pm = modeler(sigma=exp_map)
    assert "sigma" in pm.parametrizations
    assert pm.parametrizations.sigma is exp_map

    with pytest.warns(
        FutureWarning, match=f"Getting `{cls_name}\\.sigmaMap` directly is deprecated.*"
    ):
        assert pm.sigmaMap is exp_map


@pytest.mark.parametrize("modeler", INVERTIBLE_SIGMA_CLASSES)
@pytest.mark.parametrize("value", [None, np.array([1, 2, 3])])
def test_invertible_map_deletion(modeler, value):
    pm = modeler(sigma=maps.ExpMap())
    pm.sigma = value
    assert pm._prop_deriv("sigma") == 0
    assert "sigma" not in pm.parametrizations


def test_reciprocal_deletion():
    exp_map = maps.ExpMap()
    pm = ReciprocalExample(sigma=exp_map)
    assert "sigma" in pm.parametrizations

    pm.rho = np.array([1, 2, 3])
    assert not hasattr(pm, ReciprocalExample.sigma.private_name)
    assert "sigma" not in pm.parametrizations
    assert pm._prop_deriv("sigma") == 0


@pytest.mark.parametrize("modeler", INVERTIBLE_SIGMA_CLASSES)
def test_invertible_needs_model(modeler):
    assert modeler.sigma.invertible
    pm = modeler(sigma=maps.ExpMap())
    assert pm.needs_model

    # There is currently no model, so sigma, which is mapped, fails
    with pytest.raises(AttributeError):
        pm.sigma


@pytest.mark.parametrize("modeler", INVERTIBLE_SIGMA_CLASSES)
def test_retrieve_mapped_property(modeler):
    assert modeler.sigma.invertible
    pm = modeler(sigma=maps.ExpMap())
    pm.model = np.array([1, 2, 3])
    desired = np.exp(np.array([1, 2, 3]))
    npt.assert_equal(pm.sigma, desired)


@pytest.mark.parametrize("modeler", INVERTIBLE_SIGMA_CLASSES)
def test_derivative_mapped_property(modeler):
    assert modeler.sigma.invertible
    pm = modeler(sigma=maps.ExpMap())
    pm.model = np.array([1, 2, 3])

    deriv = pm._prop_deriv("sigma").todense()
    desired = np.diag(np.exp(np.r_[1.0, 2.0, 3.0]))
    npt.assert_equal(deriv, desired)

    cls_name = modeler.__name__
    with pytest.warns(
        FutureWarning, match=f"Getting `{cls_name}\\.sigmaDeriv` is deprecated.*"
    ):
        deriv = pm.sigmaDeriv.todense()
    npt.assert_equal(deriv, desired)


@pytest.mark.parametrize("modeler", RHO_CLASSES - INVERTIBLE_RHO_CLASSES)
def test_not_assign_map_invertible(modeler):
    assert not modeler.rho.invertible
    with pytest.raises(TypeError):
        modeler(rho=maps.ExpMap())


@pytest.mark.parametrize("modeler", RECIPROCAL_CLASSES)
@pytest.mark.parametrize("assign, retrieve", [("sigma", "rho"), ("rho", "sigma")])
def test_reciprocal_assigned(modeler, assign, retrieve):
    pm = modeler()
    setattr(pm, assign, np.array([1, 2, 3]))
    value = getattr(pm, retrieve)
    desired = 1.0 / np.array([1, 2, 3])
    npt.assert_equal(value, desired)


@pytest.mark.parametrize("assign, retrieve", [("sigma", "rho"), ("rho", "sigma")])
def test_reciprocal_two_mapped_retrieve(assign, retrieve):
    pm = ReciprocalExample()
    pm.parametrize(**{assign: maps.ExpMap()})
    pm.model = np.array([1, 2, 3])

    value = getattr(pm, retrieve)
    desired = 1.0 / np.exp(np.array([1, 2, 3]))
    npt.assert_equal(value, desired)

    match_string = "Getting `ReciprocalExample.{0}Map` directly is deprecated. If this is still necessary use `ReciprocalExample.parametrizations.{0}` instead"
    with pytest.warns(FutureWarning, match=re.escape(match_string.format(assign))):
        assert getattr(pm, assign + "Map") is getattr(pm.parametrizations, assign)

    with pytest.warns(FutureWarning, match=re.escape(match_string.format(retrieve))):
        rMap = getattr(pm, retrieve + "Map")
    assert rMap is not None
    npt.assert_equal(rMap @ pm.model, desired)


@pytest.mark.parametrize("modeler", RECIPROCAL_CLASSES & INVERTIBLE_SIGMA_CLASSES)
def test_reciprocal_sigma_mapped_retrieve(modeler):
    pm = modeler()
    pm.parametrize(sigma=maps.ExpMap())
    assert pm.is_parametrized("rho")
    pm.model = np.array([1, 2, 3])
    value = pm.rho
    desired = 1.0 / np.exp(np.array([1, 2, 3]))
    npt.assert_equal(value, desired)

    cls_name = modeler.__name__
    with pytest.warns(
        FutureWarning, match=f"Getting `{cls_name}\\.sigmaMap` directly is deprecated.*"
    ):
        assert pm.sigmaMap is pm.parametrizations.sigma

    with pytest.warns(
        FutureWarning, match=f"Getting `{cls_name}\\.rhoMap` directly is deprecated.*"
    ):
        rho_map = pm.rhoMap
    assert rho_map is not None
    npt.assert_equal(rho_map @ pm.model, desired)


@pytest.mark.parametrize("modeler", RECIPROCAL_CLASSES & INVERTIBLE_SIGMA_CLASSES)
def test_reciprocal_sigma_derivative(modeler):
    pm = modeler(sigma=maps.ExpMap())
    pm.model = np.array([1, 2, 3])

    deriv = pm._prop_deriv("rho").todense()
    desired = np.diag(-1 / np.exp(np.r_[1.0, 2.0, 3.0]))
    npt.assert_allclose(deriv, desired)


def test_multi_parameter_inversion():
    """The setup of the defaults should not invalidate the
    mappings or other defaults.
    """
    PM = MultipleInvertible()
    params = inspect.signature(MultipleInvertible).parameters

    np.testing.assert_equal(PM.sigma, params["sigma"].default)
    np.testing.assert_equal(PM.rho, params["rho"].default)
    np.testing.assert_equal(PM.gamma, params["gamma"].default)


@pytest.mark.parametrize("modeler", RECIPROCAL_CLASSES - DEFAULT_CLASSES)
def test_reciprocal_not_assigned(modeler):
    pm = modeler()
    with pytest.raises(AttributeError):
        pm.sigma
    with pytest.raises(AttributeError):
        pm.rho


@pytest.mark.parametrize("modeler", RECIPROCAL_CLASSES & INVERTIBLE_SIGMA_CLASSES)
def test_reciprocal_map_no_model(modeler):
    pm = modeler(sigma=maps.ExpMap())
    with pytest.raises(AttributeError):
        pm.rho


def test_no_value():
    pm = SingleNotInvertible()
    with pytest.raises(AttributeError):
        pm.rho


@pytest.mark.parametrize("modeler", RECIPROCAL_CLASSES & DEFAULT_CLASSES)
@pytest.mark.parametrize("assign, retrieve", [("sigma", "rho"), ("rho", "sigma")])
def test_reciprocal_default(modeler, assign, retrieve):
    pm = modeler()
    v1 = getattr(pm, assign)
    v2 = getattr(pm, retrieve)
    if v1 is None:
        assert v2 is None
    else:
        assert v1 == 1.0 / v2


@pytest.mark.parametrize("modeler", ALL_PROP_CLASSES)
def test_no_map_yet(modeler):
    pm = modeler()
    # all HasModel classes should error if assigning a model
    # before any maps have been assigned
    # regardless if they have invertible properties
    with pytest.raises(AttributeError):
        pm.model = 10


def test_optional_inverted():
    modeler = OptionalInvertible()
    assert "sigma" not in modeler.parametrizations
    assert modeler.sigma is None

    modeler.sigma = 10
    assert modeler.sigma == 10


# --- Location / AnisotropyLevel / fshape override mechanism -----------------


class _MeshHasModel(props.HasModel):
    """A HasModel subclass with a plain (non-PhysicalProperty) mesh attribute."""

    def __init__(self, mesh=None, **kwargs):
        self.mesh = mesh
        super().__init__(**kwargs)


class CellCenteredIsotropic(_MeshHasModel):
    sigma = props.PhysicalProperty(
        "Electrical conductivity (S/m)", location=props.Location.CELL_CENTERS
    )

    def __init__(self, mesh=None, sigma=None, **kwargs):
        super().__init__(mesh=mesh, **kwargs)
        self._init_property(sigma=sigma)


class CellCenteredFullAnisotropic(_MeshHasModel):
    sigma = props.PhysicalProperty(
        "Electrical conductivity (S/m)",
        reciprocal="rho",
        location=props.Location.CELL_CENTERS,
        anisotropy=props.AnisotropyLevel.FULL,
    )
    rho = props.PhysicalProperty(
        "Electrical resistivity (Ohm m)",
        reciprocal="sigma",
        location=props.Location.CELL_CENTERS,
        anisotropy=props.AnisotropyLevel.FULL,
    )

    def __init__(self, mesh=None, sigma=None, rho=None, **kwargs):
        super().__init__(mesh=mesh, **kwargs)
        self._init_recip_properties(sigma=sigma, rho=rho)


@pytest.fixture
def small_2d_mesh():
    import discretize

    return discretize.TensorMesh([3, 3])


def test_location_isotropic_rejects_anisotropic_shape(small_2d_mesh):
    modeler = CellCenteredIsotropic(mesh=small_2d_mesh)
    n_cells = small_2d_mesh.n_cells

    # scalar, and per-cell isotropic values are still accepted
    modeler.sigma = 1.0
    modeler.sigma = np.full(n_cells, 1.0)

    # diagonal/full-tensor anisotropic shapes are rejected (anisotropy=ISOTROPIC)
    with pytest.raises(ValueError):
        modeler.sigma = np.ones((n_cells, small_2d_mesh.dim))
    with pytest.raises(ValueError):
        modeler.sigma = np.ones((n_cells, 3))


def test_location_full_anisotropy_accepts_all_shapes(small_2d_mesh):
    modeler = CellCenteredFullAnisotropic(mesh=small_2d_mesh)
    n_cells = small_2d_mesh.n_cells

    modeler.rho = 1.0
    modeler.rho = np.full(n_cells, 1.0)
    modeler.rho = np.ones((n_cells, small_2d_mesh.dim))  # diagonal
    modeler.rho = np.ones((n_cells, 3))  # full-tensor (2D: 3 params)


def test_location_full_anisotropy_reciprocal_inverts_via_discretize(small_2d_mesh):
    from discretize.utils import inverse_property_tensor

    modeler = CellCenteredFullAnisotropic(mesh=small_2d_mesh)
    n_cells = small_2d_mesh.n_cells
    rho = np.tile(np.array([1.0, 2.0, 0.5]), (n_cells, 1))
    modeler.rho = rho

    npt.assert_allclose(modeler.sigma, inverse_property_tensor(small_2d_mesh, rho))


class _ReshapeMap(maps.IdentityMap):
    """A trivial map that reshapes its (flat) input to `out_shape`."""

    def __init__(self, nP, out_shape, **kwargs):
        super().__init__(nP=nP, **kwargs)
        self.out_shape = out_shape

    def _transform(self, m):
        return np.asarray(m).reshape(self.out_shape)

    def deriv(self, m, v=None):
        import scipy.sparse as sp

        return sp.identity(self.nP)


def test_reciprocal_deriv_errors_for_full_tensor_parametrization(small_2d_mesh):
    from discretize.utils import inverse_property_tensor

    n_cells = small_2d_mesh.n_cells
    modeler = CellCenteredFullAnisotropic(mesh=small_2d_mesh)
    full_map = _ReshapeMap(nP=n_cells * 3, out_shape=(n_cells, 3))
    modeler.parametrize(rho=full_map)
    modeler.model = np.tile(np.array([1.0, 2.0, 0.5]), n_cells)

    # the VALUE is still computed correctly (inverts a concrete evaluated array)
    npt.assert_allclose(
        modeler.sigma,
        inverse_property_tensor(small_2d_mesh, full_map * modeler.model),
    )
    # but the DERIVATIVE through `maps.ReciprocalMap` is not correct for a
    # genuine full tensor, and should error instead of silently being wrong
    with pytest.raises(NotImplementedError, match="full-tensor anisotropic"):
        modeler._prop_deriv("sigma")


def test_reciprocal_deriv_still_works_for_isotropic_parametrization(small_2d_mesh):
    n_cells = small_2d_mesh.n_cells
    modeler = CellCenteredFullAnisotropic(mesh=small_2d_mesh)
    modeler.parametrize(rho=maps.ExpMap(nP=n_cells))
    modeler.model = np.zeros(n_cells)

    # a property merely CAPABLE of full anisotropy, but currently parametrized
    # isotropically, must not be blocked -- this is the overwhelmingly common case
    modeler._prop_deriv("sigma")


def test_reciprocal_deriv_falls_back_to_model_length_for_ambiguous_shape(
    small_2d_mesh,
):
    n_cells = small_2d_mesh.n_cells

    # a wildcard-shaped map (no mesh/nP) can't declare its own output size, so
    # the guard falls back to the model's length as a proxy
    modeler = CellCenteredFullAnisotropic(mesh=small_2d_mesh)
    modeler.parametrize(rho=maps.ExpMap())
    modeler.model = np.zeros(n_cells * 3)
    with pytest.raises(NotImplementedError, match="full-tensor anisotropic"):
        modeler._prop_deriv("sigma")

    # same wildcard map, but an isotropic-length model: not blocked
    modeler2 = CellCenteredFullAnisotropic(mesh=small_2d_mesh)
    modeler2.parametrize(rho=maps.ExpMap())
    modeler2.model = np.zeros(n_cells)
    modeler2._prop_deriv("sigma")


def test_location_and_shape_mutually_exclusive():
    with pytest.raises(ValueError):
        props.PhysicalProperty("x", shape=(), location=props.Location.CELL_CENTERS)


def test_anisotropy_requires_location():
    with pytest.raises(ValueError):
        props.PhysicalProperty("x", anisotropy=props.AnisotropyLevel.FULL)


@pytest.mark.parametrize("location", [props.Location.FACES, props.Location.EDGES])
@pytest.mark.parametrize(
    "anisotropy", [props.AnisotropyLevel.DIAGONAL, props.AnisotropyLevel.FULL]
)
def test_anisotropy_not_supported_off_cell_centers(location, anisotropy):
    with pytest.raises(ValueError):
        props.PhysicalProperty("x", location=location, anisotropy=anisotropy)


def test_manual_invert_overrides_auto_derived(small_2d_mesh):
    calls = []

    def custom_invert(obj, value):
        calls.append(value)
        return value * 0 + 42.0

    class Custom(_MeshHasModel):
        sigma = props.PhysicalProperty(
            "sigma",
            reciprocal="rho",
            location=props.Location.CELL_CENTERS,
            anisotropy=props.AnisotropyLevel.FULL,
            invert=custom_invert,
        )
        rho = props.PhysicalProperty(
            "rho",
            reciprocal="sigma",
            location=props.Location.CELL_CENTERS,
            anisotropy=props.AnisotropyLevel.FULL,
        )

        def __init__(self, mesh=None, sigma=None, rho=None, **kwargs):
            super().__init__(mesh=mesh, **kwargs)
            self._init_recip_properties(sigma=sigma, rho=rho)

    modeler = Custom(mesh=small_2d_mesh, rho=2.0)
    result = modeler.sigma
    assert len(calls) == 1
    npt.assert_allclose(result, 42.0)


def test_set_feature_rederives_shape_and_invert_for_anisotropy(small_2d_mesh):
    """set_feature(anisotropy=...) must rebuild fshape/invert, not just the attribute.

    Regression test: `fshape`/`invert` are closures baked in from `location`/
    `anisotropy` at construction time. A naive `set_feature` (plain setattr)
    would change `.anisotropy` cosmetically while the shape validator/inverter
    kept silently using the OLD level.
    """
    full = props.PhysicalProperty(
        "x",
        reciprocal="y",
        location=props.Location.CELL_CENTERS,
        anisotropy=props.AnisotropyLevel.FULL,
    )
    restricted = full.set_feature(anisotropy=props.AnisotropyLevel.ISOTROPIC)
    assert restricted.anisotropy is props.AnisotropyLevel.ISOTROPIC

    class Restricted(_MeshHasModel):
        x = restricted
        y = props.PhysicalProperty(
            "y",
            reciprocal="x",
            location=props.Location.CELL_CENTERS,
            anisotropy=props.AnisotropyLevel.ISOTROPIC,
        )

        def __init__(self, mesh=None, x=None, y=None, **kwargs):
            super().__init__(mesh=mesh, **kwargs)
            self._init_recip_properties(x=x, y=y)

    modeler = Restricted(mesh=small_2d_mesh)
    n_cells = small_2d_mesh.n_cells
    modeler.x = np.full(n_cells, 1.0)  # isotropic still accepted
    with pytest.raises(ValueError):
        modeler.x = np.ones((n_cells, 3))  # full-tensor must now be rejected

    # `invert` must also have been rebuilt to the plain elementwise fallback,
    # not the tensor-aware one inherited from the original FULL declaration.
    modeler.y = 2.0
    npt.assert_allclose(modeler.x, 0.5)


def test_shapes_decorator_generic_override():
    """The core `.shapes()` mechanism, with no `location` involved at all."""

    class ActiveCellsLike(props.HasModel):
        sigma = props.PhysicalProperty("sigma")

        def __init__(self, active_subset=None, sigma=None, **kwargs):
            self.active_subset = active_subset
            super().__init__(**kwargs)
            self._init_property(sigma=sigma)

        @sigma.shapes
        def sigma(self):
            size = len(self.active_subset) if self.active_subset is not None else 5
            return [(), (1,), (size,)]

    modeler = ActiveCellsLike(active_subset=[0, 1, 2])
    modeler.sigma = np.ones(3)
    with pytest.raises(ValueError):
        modeler.sigma = np.ones(5)


def test_shapes_decorator_survives_setter_shallow_copy():
    """`.setter()` must not silently drop a `.shapes()` override (or vice versa)."""

    class Custom(props.HasModel):
        sigma = props.PhysicalProperty("sigma")

        def __init__(self, sigma=None, **kwargs):
            super().__init__(**kwargs)
            self._init_property(sigma=sigma)

        @sigma.shapes
        def sigma(self):
            return [(3,)]

        @sigma.setter
        def sigma(self, value):
            type(self).sigma._fset(self, value)

    modeler = Custom()
    modeler.sigma = np.ones(3)
    with pytest.raises(ValueError):
        modeler.sigma = np.ones(4)


if __name__ == "__main__":
    unittest.main()
