import re
import sys
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


def _ordered(classes):
    """Sort a set of classes, so that test collection does not depend on hashing."""
    return sorted(classes, key=lambda cls: cls.__name__)


@pytest.mark.parametrize("modeler", _ordered(ALL_PROP_CLASSES))
def test_internal_property_dictionary(modeler):
    params = inspect.signature(modeler).parameters
    params = set(name for name in params if name != "kwargs")

    assert params == set(modeler.physical_properties.keys())


@pytest.mark.parametrize("modeler", _ordered(RECIPROCAL_CLASSES))
@pytest.mark.parametrize("assign, retrieve", [("sigma", "rho"), ("rho", "sigma")])
def test_get_recip_prop(modeler, assign, retrieve):
    prop1 = getattr(modeler, assign)
    assert prop1._reciprocal == retrieve

    prop2 = getattr(modeler, retrieve)
    assert prop2._reciprocal == assign

    assert prop1.get_reciprocal(modeler) is prop2


@pytest.mark.parametrize("modeler", _ordered(INVERTIBLE_SIGMA_CLASSES))
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


@pytest.mark.parametrize("modeler", _ordered(INVERTIBLE_SIGMA_CLASSES))
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


@pytest.mark.parametrize("modeler", _ordered(INVERTIBLE_SIGMA_CLASSES))
def test_invertible_needs_model(modeler):
    assert modeler.sigma.invertible
    pm = modeler(sigma=maps.ExpMap())
    assert pm.needs_model

    # There is currently no model, so sigma, which is mapped, fails
    with pytest.raises(AttributeError):
        pm.sigma


@pytest.mark.parametrize("modeler", _ordered(INVERTIBLE_SIGMA_CLASSES))
def test_retrieve_mapped_property(modeler):
    assert modeler.sigma.invertible
    pm = modeler(sigma=maps.ExpMap())
    pm.model = np.array([1, 2, 3])
    desired = np.exp(np.array([1, 2, 3]))
    npt.assert_equal(pm.sigma, desired)


@pytest.mark.parametrize("modeler", _ordered(INVERTIBLE_SIGMA_CLASSES))
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


@pytest.mark.parametrize("modeler", _ordered(RHO_CLASSES - INVERTIBLE_RHO_CLASSES))
def test_not_assign_map_invertible(modeler):
    assert not modeler.rho.invertible
    with pytest.raises(TypeError):
        modeler(rho=maps.ExpMap())


@pytest.mark.parametrize("modeler", _ordered(RECIPROCAL_CLASSES))
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


@pytest.mark.parametrize(
    "modeler", _ordered(RECIPROCAL_CLASSES & INVERTIBLE_SIGMA_CLASSES)
)
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


@pytest.mark.parametrize(
    "modeler", _ordered(RECIPROCAL_CLASSES & INVERTIBLE_SIGMA_CLASSES)
)
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


@pytest.mark.parametrize("modeler", _ordered(RECIPROCAL_CLASSES - DEFAULT_CLASSES))
def test_reciprocal_not_assigned(modeler):
    pm = modeler()
    with pytest.raises(AttributeError):
        pm.sigma
    with pytest.raises(AttributeError):
        pm.rho


@pytest.mark.parametrize(
    "modeler", _ordered(RECIPROCAL_CLASSES & INVERTIBLE_SIGMA_CLASSES)
)
def test_reciprocal_map_no_model(modeler):
    pm = modeler(sigma=maps.ExpMap())
    with pytest.raises(AttributeError):
        pm.rho


def test_no_value():
    pm = SingleNotInvertible()
    with pytest.raises(AttributeError):
        pm.rho


@pytest.mark.parametrize("modeler", _ordered(RECIPROCAL_CLASSES & DEFAULT_CLASSES))
@pytest.mark.parametrize("assign, retrieve", [("sigma", "rho"), ("rho", "sigma")])
def test_reciprocal_default(modeler, assign, retrieve):
    pm = modeler()
    v1 = getattr(pm, assign)
    v2 = getattr(pm, retrieve)
    if v1 is None:
        assert v2 is None
    else:
        assert v1 == 1.0 / v2


@pytest.mark.parametrize("modeler", _ordered(ALL_PROP_CLASSES))
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


@pytest.mark.parametrize("h", [[3, 3], [3, 3, 3]])
def test_location_full_anisotropy_accepts_all_shapes(h):
    from discretize import TensorMesh

    mesh = TensorMesh(h)
    modeler = CellCenteredFullAnisotropic(mesh=mesh)
    n_cells = mesh.n_cells
    n_full = 3 if mesh.dim == 2 else 6

    modeler.rho = 1.0
    modeler.rho = np.full(n_cells, 1.0)
    modeler.rho = np.ones((n_cells, mesh.dim))  # diagonal
    modeler.rho = np.ones((n_cells, n_full))  # full-tensor
    # and the flattened layouts of those (this is what a mapping's output looks like)
    modeler.rho = np.ones(n_cells * mesh.dim)
    modeler.rho = np.ones(n_cells * n_full)
    with pytest.raises(ValueError):
        modeler.rho = np.ones(n_cells * (n_full + 1))


def test_location_full_anisotropy_reciprocal_inverts_via_discretize(small_2d_mesh):
    from discretize.utils import inverse_property_tensor

    modeler = CellCenteredFullAnisotropic(mesh=small_2d_mesh)
    n_cells = small_2d_mesh.n_cells
    rho = np.tile(np.array([1.0, 2.0, 0.5]), (n_cells, 1))
    modeler.rho = rho

    npt.assert_allclose(modeler.sigma, inverse_property_tensor(small_2d_mesh, rho))


@pytest.mark.parametrize("h", [[3, 3], [3, 3, 3]])
@pytest.mark.parametrize("assign, retrieve", [("sigma", "rho"), ("rho", "sigma")])
def test_flattened_full_tensor_reciprocal(h, assign, retrieve):
    from discretize import TensorMesh
    from discretize.utils import inverse_property_tensor

    mesh = TensorMesh(h)
    n_full = 3 if mesh.dim == 2 else 6
    value = np.random.default_rng(0).uniform(1.0, 2.0, mesh.n_cells * n_full)

    modeler = CellCenteredFullAnisotropic(mesh=mesh, **{assign: value})

    npt.assert_allclose(getattr(modeler, assign), value)
    npt.assert_allclose(
        getattr(modeler, retrieve), inverse_property_tensor(mesh, value)
    )


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


def test_reciprocal_deriv_catches_combomap_resolved_full_tensor_shape(
    small_2d_mesh,
):
    """A ComboMap chaining a wildcard-shaped map with a size-fixing map now
    resolves a concrete shape (via ComboMap's shape propagation), so this
    is caught by the primary shape check -- no need for the model-length
    fallback at all.
    """
    n_cells = small_2d_mesh.n_cells
    n_full = 3  # 2D full-tensor parameter count

    fixed_map = maps.Projection(n_cells * n_full, np.arange(n_cells * n_full))
    combo = maps.ComboMap([maps.ExpMap(), fixed_map])
    assert combo.shape == (n_cells * n_full, n_cells * n_full)

    modeler = CellCenteredFullAnisotropic(mesh=small_2d_mesh)
    modeler.parametrize(rho=combo)
    modeler.model = np.zeros(n_cells * n_full)
    with pytest.raises(NotImplementedError, match="full-tensor anisotropic"):
        modeler._prop_deriv("sigma")


def test_location_and_shape_mutually_exclusive():
    with pytest.raises(ValueError):
        props.PhysicalProperty("x", shape=(), location=props.Location.CELL_CENTERS)


def test_parametrize_validates_mapping_output_length(small_2d_mesh):
    n_cells = small_2d_mesh.n_cells
    modeler = CellCenteredIsotropic(mesh=small_2d_mesh)

    with pytest.raises(ValueError, match="output length"):
        modeler.parametrize(sigma=maps.IdentityMap(nP=n_cells + 1))

    # a matching output length is accepted
    modeler.parametrize(sigma=maps.IdentityMap(nP=n_cells))

    # a wildcard-shaped mapping can't be checked, so it's accepted regardless
    modeler2 = CellCenteredIsotropic(mesh=small_2d_mesh)
    modeler2.parametrize(sigma=maps.ExpMap())


def test_parametrize_validates_consistent_input_length():
    modeler = MultipleInvertible()
    modeler.parametrize(sigma=maps.IdentityMap(nP=5))

    with pytest.raises(ValueError, match="model of length"):
        modeler.parametrize(rho=maps.IdentityMap(nP=7))

    # a consistent input length across independently-parametrized properties
    # is accepted
    modeler2 = MultipleInvertible()
    modeler2.parametrize(sigma=maps.IdentityMap(nP=5))
    modeler2.parametrize(rho=maps.IdentityMap(nP=5))

    # re-parametrizing the SAME attribute must not conflict with itself
    modeler2.parametrize(sigma=maps.IdentityMap(nP=5))


def test_model_validates_unresolved_shape_against_model_length(small_2d_mesh):
    """A wildcard-shaped mapping's output shape can't be checked at
    `parametrize()` time, but once a concrete model is assigned, its length
    is used as a size-preserving proxy (without evaluating the mapping) and
    checked against the property's expected sizes.
    """
    n_cells = small_2d_mesh.n_cells

    modeler = CellCenteredFullAnisotropic(mesh=small_2d_mesh)
    modeler.parametrize(rho=maps.ExpMap())
    with pytest.raises(ValueError, match="unresolved output shape"):
        modeler.model = np.zeros(n_cells + 2)

    # a length matching one of rho's expected sizes (here, full-tensor) is accepted
    modeler2 = CellCenteredFullAnisotropic(mesh=small_2d_mesh)
    modeler2.parametrize(rho=maps.ExpMap())
    modeler2.model = np.zeros(n_cells * 3)


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


# --- parametrize() ---------------------------------------------------------


def test_failed_parametrize_keeps_value(small_2d_mesh):
    n_cells = small_2d_mesh.n_cells
    modeler = CellCenteredFullAnisotropic(mesh=small_2d_mesh, sigma=np.ones(n_cells))

    with pytest.raises(ValueError):
        modeler.parametrize(sigma=maps.IdentityMap(nP=n_cells + 3))

    npt.assert_allclose(modeler.sigma, 1.0)
    assert not modeler.is_parametrized("sigma")


def test_failed_parametrize_keeps_reciprocal_mapping(small_2d_mesh):
    n_cells = small_2d_mesh.n_cells
    modeler = CellCenteredFullAnisotropic(
        mesh=small_2d_mesh, sigma=maps.ExpMap(nP=n_cells)
    )
    old_map = modeler.parametrizations.sigma

    # replacing it (through the reciprocal) with a bad one, must not remove the old
    with pytest.raises(ValueError):
        modeler.parametrize(rho=maps.IdentityMap(nP=n_cells + 3))

    assert modeler.parametrizations.sigma is old_map
    assert "rho" not in modeler.parametrizations


# --- deprecated xMap interface ---------------------------------------------


def test_deprecated_map_set_to_none_removes_parametrization():
    pm = SingleExample(sigma=maps.ExpMap())
    assert pm.needs_model
    with pytest.warns(FutureWarning, match="sigmaMap"):
        pm.sigmaMap = None
    assert not pm.needs_model


def test_deprecated_map_set_to_none_only_removes_its_own_parametrization():
    pm = ReciprocalExample(rho=maps.ExpMap())
    with pytest.warns(FutureWarning, match="sigmaMap"):
        pm.sigmaMap = None
    assert pm.is_parametrized("rho")

    with pytest.warns(FutureWarning, match="rhoMap"):
        pm.rhoMap = None
    assert not pm.needs_model


def test_deprecated_map_set_to_none_keeps_fixed_value():
    pm = ReciprocalExample(sigma=np.ones(3))
    with pytest.warns(FutureWarning, match="rhoMap"):
        pm.rhoMap = None
    npt.assert_equal(pm.sigma, 1.0)


def test_deprecation_warnings_point_at_caller():
    with pytest.warns(FutureWarning) as record:
        pm = ReciprocalExample(sigmaMap=maps.ExpMap())
        pm.rhoMap = maps.ExpMap()
        pm.model = np.ones(3)
        pm.sigmaMap
        pm.sigmaDeriv
    assert len(record) == 4
    assert all(w.filename == __file__ for w in record)
    assert "0.29.0" in str(record[0].message)


def test_assigning_over_parametrization_warns_at_caller():
    pm = SingleExample(sigma=maps.ExpMap())
    with pytest.warns(UserWarning, match="removes the existing parametrization") as w:
        pm.sigma = np.ones(3)
    assert w[0].filename == __file__
    assert not pm.is_parametrized("sigma")


def test_assigning_mapping_directly_errors():
    pm = SingleExample(sigma=np.ones(3))
    with pytest.raises(TypeError, match="Cannot assign a mapping directly"):
        pm.sigma = maps.ExpMap()


# --- descriptor details ----------------------------------------------------


def test_get_without_objtype_uses_reciprocal():
    pm = ReciprocalExample(rho=np.full(3, 4.0))
    npt.assert_allclose(ReciprocalExample.sigma.__get__(pm), 0.25)


def test_replacing_one_of_a_reciprocal_pair_with_an_attribute_errors():
    with pytest.raises(TypeError, match="Override both"):

        class Child(ReciprocalExample):
            rho = 1.0


def test_replacing_one_of_a_reciprocal_pair_without_reciprocal_errors():
    with pytest.raises(TypeError, match="not the reciprocal"):

        class Child(ReciprocalExample):
            rho = props.PhysicalProperty("Electrical resistivity (Ohm m)")


def test_mixin_with_one_half_of_a_reciprocal_pair_is_allowed():
    class SigmaMixin(props.HasModel):
        sigma = props.PhysicalProperty("Conductivity", reciprocal="rho")

    class RhoMixin(props.HasModel):
        rho = props.PhysicalProperty("Resistivity", reciprocal="sigma")

    class Both(SigmaMixin, RhoMixin):
        pass

    modeler = Both(rho=np.full(3, 2.0))
    npt.assert_allclose(modeler.sigma, 0.5)


def test_scalars_are_stored_as_python_scalars():
    pm = SingleExample(sigma=2.0)
    assert not isinstance(pm.sigma, np.ndarray)
    assert np.isscalar(pm.sigma)


def test_invertible_properties():
    assert set(ReciprocalSingleInvertible.invertible_properties) == {"sigma"}
    assert set(MultipleInvertible.invertible_properties) == {"sigma", "rho", "gamma"}


def test_parametrization_list():
    import pickle

    pm = SingleExample(sigma=maps.ExpMap())
    plist = pm.parametrizations

    assert "sigma" in plist
    assert list(plist) == ["sigma"]
    assert len(plist) == 1
    assert plist.get("other") is None
    assert "sigma" in str(plist)
    with pytest.raises(AttributeError, match="read-only"):
        plist.sigma = maps.IdentityMap()
    with pytest.raises(AttributeError, match="no attribute"):
        plist.other

    clone = pickle.loads(pickle.dumps(pm))
    assert list(clone.parametrizations) == ["sigma"]
    clone.model = np.zeros(3)
    npt.assert_allclose(clone.sigma, 1.0)


# --- removed interface -----------------------------------------------------


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
    with pytest.raises(TypeError, match=f"no longer takes a `{old}` argument"):
        props.PhysicalProperty("details", **{old: True})


# --- constructors ----------------------------------------------------------


def test_parametrizing_both_of_a_reciprocal_pair_errors():
    pm = ReciprocalExample()
    with pytest.raises(TypeError, match="both `rho` and its reciprocal"):
        pm.parametrize(sigma=maps.ExpMap(), rho=maps.ExpMap())
    assert not pm.parametrizations


def test_neither_of_a_reciprocal_pair_warns_once():
    with pytest.warns(FutureWarning) as record:
        ReciprocalExample()
    assert len(record) == 1


def test_model_setter_returns_if_updated():
    pm = SingleExample(sigma=maps.ExpMap())
    assert props.HasModel.model.fset(pm, np.ones(3)) is True
    assert props.HasModel.model.fset(pm, np.ones(3)) is False


def test_no_parametrizations_error_lists_sorted_choices():
    pm = MultipleInvertible()
    with pytest.raises(AttributeError, match="'gamma', 'rho', 'sigma'"):
        pm.model = np.ones(3)


def test_both_value_and_deprecated_map_errors():
    with pytest.raises(TypeError, match="Cannot pass both `sigma`"):
        SingleExample(sigma=np.ones(3), sigmaMap=maps.ExpMap())


class ForwardsEverything(ReciprocalExample):
    """A subclass written against the old signature, that forwards everything."""

    def __init__(self, sigma=None, sigmaMap=None, model=None, **kwargs):
        super().__init__(sigma=sigma, sigmaMap=sigmaMap, model=model, **kwargs)


def test_subclass_forwarding_unused_deprecated_arguments():
    pm = ForwardsEverything(sigma=np.ones(3))
    npt.assert_equal(pm.sigma, 1.0)


def test_subclass_forwarding_deprecated_map_and_model():
    with pytest.warns(FutureWarning, match="sigmaMap"):
        pm = ForwardsEverything(sigmaMap=maps.ExpMap(), model=np.zeros(3))
    npt.assert_equal(pm.sigma, 1.0)


def test_subclass_adding_deprecated_map_to_super_call():
    class AddsMap(ReciprocalExample):
        def __init__(self, **kwargs):
            super().__init__(sigmaMap=maps.ExpMap(), **kwargs)

    with pytest.warns(FutureWarning, match="sigmaMap"):
        pm = AddsMap(model=np.zeros(3))
    npt.assert_equal(pm.sigma, 1.0)


class AddsMapToSuperCall(ReciprocalExample):
    def __init__(self, **kwargs):
        super().__init__(sigmaMap=maps.ExpMap(), **kwargs)


def test_subclass_adding_deprecated_map_only_warns_about_the_map():
    with pytest.warns(FutureWarning) as record:
        pm = AddsMapToSuperCall()
    assert len(record) == 1
    assert "sigmaMap" in str(record[0].message)
    assert pm.is_parametrized("sigma")


@pytest.mark.parametrize("name", ["sigma", "rho"])
def test_subclass_adding_deprecated_map_and_value_errors(name):
    with pytest.warns(FutureWarning, match="sigmaMap"):
        with pytest.raises(TypeError, match="Cannot pass both `sigma`"):
            AddsMapToSuperCall(**{name: np.ones(3)})


def test_subclass_naming_a_deprecated_argument_receives_none():
    saw = []

    class NamesTheMap(ReciprocalExample):
        def __init__(self, sigmaMap=None, **kwargs):
            saw.append(sigmaMap)
            super().__init__(sigmaMap=sigmaMap, **kwargs)

    with pytest.warns(FutureWarning, match="sigmaMap"):
        pm = NamesTheMap(sigmaMap=maps.ExpMap())

    # the deprecated arguments are translated before any `__init__` runs
    assert saw == [None]
    assert pm.is_parametrized("sigma")


@pytest.mark.skipif(
    sys.version_info < (3, 13), reason="property.__set_name__ needs python 3.13"
)
def test_deprecated_properties_know_their_names():
    assert ReciprocalExample.sigmaMap.__name__ == "sigmaMap"
    assert ReciprocalExample.sigmaDeriv.__name__ == "sigmaDeriv"


if __name__ == "__main__":
    unittest.main()
