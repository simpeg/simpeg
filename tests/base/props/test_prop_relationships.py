import unittest
import numpy as np
import numpy.testing as npt
import inspect
import pytest

from simpeg import maps
from simpeg import props


class SingleExample(props.HasModel):
    sigma = props.PhysicalProperty("Electrical conductivity (S/m)")

    def __init__(self, sigma=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma


class SingleNotInvertible(props.HasModel):
    rho = props.PhysicalProperty("Electrical resistivity (Ohm m)", invertible=False)

    def __init__(self, rho=None, **kwargs):
        super().__init__(**kwargs)
        self.rho = rho


class OptionalInvertible(props.HasModel):
    sigma = props.PhysicalProperty("Electrical conductivity (S/m)", default=None)

    def __init__(self, sigma=None, **kwargs):
        self.sigma = sigma
        super().__init__(**kwargs)


class ReciprocalExample(props.HasModel):
    sigma = props.PhysicalProperty("Electrical conductivity (S/m)")
    rho = props.PhysicalProperty("Electrical resistivity (Ohm m)", reciprocal=sigma)

    def __init__(self, sigma=None, rho=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.rho = rho


class ReciprocalWithDefault(props.HasModel):
    sigma = props.PhysicalProperty("Electrical conductivity (S/m)", default=0.1)
    rho = props.PhysicalProperty("Electrical resistivity (Ohm m)", reciprocal=sigma)

    def __init__(self, sigma=None, rho=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.rho = rho


class ReciprocalWithNoneDefault(props.HasModel):
    sigma = props.PhysicalProperty("Electrical conductivity (S/m)", default=None)
    rho = props.PhysicalProperty("Electrical resistivity (Ohm m)", reciprocal=sigma)

    def __init__(self, sigma=None, rho=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.rho = rho


class ReciprocalSingleInvertible(props.HasModel):
    sigma = props.PhysicalProperty("Electrical conductivity (S/m)")
    rho = props.PhysicalProperty(
        "Electrical resistivity (Ohm m)", invertible=False, reciprocal=sigma
    )

    def __init__(self, sigma=None, rho=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.rho = rho


class ReciprocalNotInvertible(props.HasModel):
    sigma = props.PhysicalProperty("Electrical conductivity (S/m)", invertible=False)
    rho = props.PhysicalProperty(
        "Electrical resistivity (Ohm m)", invertible=False, reciprocal=sigma
    )

    def __init__(self, sigma=None, rho=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.rho = rho


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
        self.sigma = sigma
        self.rho = rho
        self.gamma = gamma


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

SIGMA_CLASSES = {cls for cls in ALL_PROP_CLASSES if "sigma" in cls._physical_properties}
RHO_CLASSES = {cls for cls in ALL_PROP_CLASSES if "rho" in cls._physical_properties}
INVERTIBLE_SIGMA_CLASSES = {cls for cls in SIGMA_CLASSES if cls.sigma.invertible}
INVERTIBLE_RHO_CLASSES = {cls for cls in RHO_CLASSES if cls.rho.invertible}

RECIPROCAL_CLASSES = {
    cls
    for cls in ALL_PROP_CLASSES
    if any(prop.reciprocal for prop in cls._physical_properties.values())
}

DEFAULT_CLASSES = {
    cls
    for cls in ALL_PROP_CLASSES
    if any(prop.optional for prop in cls._physical_properties.values())
}


@pytest.mark.parametrize("modeler", ALL_PROP_CLASSES)
def test_internal_property_dictionary(modeler):
    params = inspect.signature(modeler).parameters
    params = set(name for name in params if name != "kwargs")

    assert params == set(modeler._physical_properties.keys())


@pytest.mark.parametrize("modeler", INVERTIBLE_SIGMA_CLASSES)
def test_invertible_map_assignment(modeler):
    exp_map = maps.ExpMap()
    pm = modeler(sigma=exp_map)
    assert pm._prop_map("sigma") is exp_map
    assert "sigma" in pm._mapped_properties


@pytest.mark.parametrize("modeler", INVERTIBLE_SIGMA_CLASSES)
@pytest.mark.parametrize("value", [None, np.array([1, 2, 3])])
def test_invertible_map_deletion(modeler, value):
    exp_map = maps.ExpMap()
    pm = modeler(sigma=exp_map)
    pm.sigma = value
    assert pm._prop_map("sigma") is None
    assert pm._prop_deriv("sigma") == 0
    assert "sigma" not in pm._mapped_properties


def test_reciprocal_deletion():
    exp_map = maps.ExpMap()
    pm = ReciprocalExample(sigma=exp_map)
    assert "sigma" in pm._mapped_properties
    assert hasattr(pm, ReciprocalExample.sigma.cached_name)

    pm.rho = np.array([1, 2, 3])
    assert not hasattr(pm, ReciprocalExample.sigma.cached_name)
    assert pm._prop_map("sigma") is None
    assert pm._prop_deriv("sigma") == 0
    assert "sigma" not in pm._mapped_properties


@pytest.mark.parametrize("modeler", INVERTIBLE_SIGMA_CLASSES)
def test_invertible_needs_model(modeler):
    assert len(modeler._invertible_properties) > 0
    pm = modeler(sigma=maps.ExpMap())
    assert pm.needs_model

    # There is currently no model, so sigma, which is mapped, fails
    with pytest.raises(AttributeError):
        pm.sigma


@pytest.mark.parametrize("modeler", INVERTIBLE_SIGMA_CLASSES)
def test_retrieve_mapped_property(modeler):
    exp_map = maps.ExpMap()
    pm = modeler(sigma=exp_map)
    pm.model = np.array([1, 2, 3])
    desired = np.exp(np.array([1, 2, 3]))
    npt.assert_equal(pm.sigma, desired)


@pytest.mark.parametrize("modeler", INVERTIBLE_SIGMA_CLASSES)
def test_derivative_mapped_property(modeler):
    exp_map = maps.ExpMap()
    pm = modeler(sigma=exp_map)
    pm.model = np.array([1, 2, 3])

    deriv = pm._prop_deriv("sigma").todense()
    desired = np.diag(np.exp(np.r_[1.0, 2.0, 3.0]))
    npt.assert_equal(deriv, desired)


@pytest.mark.parametrize("modeler", RHO_CLASSES - INVERTIBLE_RHO_CLASSES)
def test_no_assign_map_invertible(modeler):
    assert "rho" not in modeler._invertible_properties
    with pytest.raises(ValueError):
        modeler(rho=maps.ExpMap())


@pytest.mark.parametrize("modeler", RECIPROCAL_CLASSES)
@pytest.mark.parametrize("assign_retrieve", [("sigma", "rho"), ("rho", "sigma")])
def test_reciprocal_assigned(modeler, assign_retrieve):
    pm = modeler()
    assign, retrieve = assign_retrieve
    setattr(pm, assign, np.array([1, 2, 3]))
    value = getattr(pm, retrieve)
    desired = 1.0 / np.array([1, 2, 3])
    npt.assert_equal(value, desired)


@pytest.mark.parametrize("assign_retrieve", [("sigma", "rho"), ("rho", "sigma")])
def test_reciprocal_two_mapped_retrieve(assign_retrieve):
    pm = ReciprocalExample()
    assign, retrieve = assign_retrieve
    setattr(pm, assign, maps.ExpMap())
    pm.model = np.array([1, 2, 3])

    value = getattr(pm, retrieve)
    desired = 1.0 / np.exp(np.array([1, 2, 3]))
    npt.assert_equal(value, desired)


def test_reciprocal_single_mapped_retrieve():
    pm = ReciprocalSingleInvertible()
    pm.sigma = maps.ExpMap()
    pm.model = np.array([1, 2, 3])
    value = pm.rho
    desired = 1.0 / np.exp(np.array([1, 2, 3]))
    npt.assert_equal(value, desired)


@pytest.mark.parametrize("modeler", RECIPROCAL_CLASSES & INVERTIBLE_SIGMA_CLASSES)
def test_reciprocal_sigma_mapped_retrieve(modeler):
    pm = modeler()
    pm.sigma = maps.ExpMap()
    pm.model = np.array([1, 2, 3])
    value = pm.rho
    desired = 1.0 / np.exp(np.array([1, 2, 3]))
    npt.assert_equal(value, desired)


@pytest.mark.parametrize("modeler", RECIPROCAL_CLASSES & INVERTIBLE_SIGMA_CLASSES)
def test_reciprocal_sigma_derivative(modeler):
    pm = modeler(sigma=maps.ExpMap())
    pm.model = np.array([1, 2, 3])

    deriv = pm._prop_deriv("rho").todense()
    desired = np.diag(-1 / np.exp(np.r_[1.0, 2.0, 3.0]))
    npt.assert_allclose(deriv, desired)


def test_multi_parameter_inversion():
    """The setup of the defaults should not invalidated the
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
@pytest.mark.parametrize("assign_retrieve", [("sigma", "rho"), ("rho", "sigma")])
def test_reciprocal_default(modeler, assign_retrieve):
    pm = modeler()
    assign, retrieve = assign_retrieve
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
    assert modeler._prop_map("sigma") is None
    assert modeler.sigma is None

    modeler.sigma = 10
    assert modeler.sigma == 10


if __name__ == "__main__":
    unittest.main()
