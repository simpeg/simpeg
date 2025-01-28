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

SIGMA_CLASSES = {
    cls for cls in ALL_PROP_CLASSES if "sigma" in cls.physical_properties()
}
RHO_CLASSES = {cls for cls in ALL_PROP_CLASSES if "rho" in cls.physical_properties()}
INVERTIBLE_SIGMA_CLASSES = {cls for cls in SIGMA_CLASSES if cls.sigma.invertible}
INVERTIBLE_RHO_CLASSES = {cls for cls in RHO_CLASSES if cls.rho.invertible}

RECIPROCAL_CLASSES = {
    cls
    for cls in ALL_PROP_CLASSES
    if any(prop.has_reciprocal for prop in cls.physical_properties().values())
}

DEFAULT_CLASSES = {
    cls
    for cls in ALL_PROP_CLASSES
    if any(prop.optional for prop in cls.physical_properties().values())
}


@pytest.mark.parametrize("modeler", ALL_PROP_CLASSES)
def test_internal_property_dictionary(modeler):
    params = inspect.signature(modeler).parameters
    params = set(name for name in params if name != "kwargs")

    assert params == set(modeler.physical_properties().keys())


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
                match=f"Passing argument sigmaMap to {cls_name} is deprecated.*",
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
    pm.parametrize(assign, maps.ExpMap())
    pm.model = np.array([1, 2, 3])

    value = getattr(pm, retrieve)
    desired = 1.0 / np.exp(np.array([1, 2, 3]))
    npt.assert_equal(value, desired)

    with pytest.warns(FutureWarning, match="Dd"):
        assert getattr(pm, assign + "Map") is getattr(pm.parametrizations, assign)

    with pytest.warns(FutureWarning, match="Dd"):
        rMap = getattr(pm, retrieve + "Map")
    assert rMap is not None
    assert npt.assert_equal(rMap @ pm.model, desired)


@pytest.mark.parametrize("modeler", RECIPROCAL_CLASSES & INVERTIBLE_SIGMA_CLASSES)
def test_reciprocal_sigma_mapped_retrieve(modeler):
    pm = modeler()
    pm.parametrize("sigma", maps.ExpMap())
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


if __name__ == "__main__":
    unittest.main()
