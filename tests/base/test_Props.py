import unittest
import numpy as np
import inspect
import pytest

import discretize

from simpeg import maps
from simpeg import utils
from simpeg import props


class SimpleExample(props.HasModel):
    conductivity_map = props.Mapping("Mapping to the inversion model.")

    sigma = props.PhysicalProperty(
        "Electrical conductivity (S/m)", mapping=conductivity_map
    )

    _con_deriv = props.Derivative(
        "Derivative of sigma wrt the model.", physical_property=sigma
    )

    def __init__(self, sigma=None, conductivity_map=None, **kwargs):
        super().__init__(**kwargs)
        self.conductivity = sigma
        self.conductivity_map = conductivity_map


class ShortcutExample(props.HasModel):
    sigma, conductivity_map, _con_deriv = props.Invertible(
        "Electrical conductivity (S/m)"
    )

    def __init__(self, sigma=None, conductivity_map=None, **kwargs):
        super().__init__(**kwargs)
        self.conductivity = sigma
        self.conductivity_map = conductivity_map


class ReciprocalMappingExample(props.HasModel):
    sigma, conductivity_map, _con_deriv = props.Invertible(
        "Electrical conductivity (S/m)"
    )

    rho, rhoMap, rhoDeriv = props.Invertible("Electrical resistivity (Ohm m)")

    props.Reciprocal(sigma, rho)

    def __init__(
        self, sigma=None, conductivity_map=None, rho=None, rhoMap=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.conductivity = sigma
        self.rho = rho
        self.conductivity_map = conductivity_map
        self.rhoMap = rhoMap


class ReciprocalExample(props.HasModel):
    sigma, conductivity_map, _con_deriv = props.Invertible(
        "Electrical conductivity (S/m)"
    )

    rho = props.PhysicalProperty("Electrical resistivity (Ohm m)")

    props.Reciprocal(sigma, rho)

    def __init__(self, sigma=None, conductivity_map=None, rho=None, **kwargs):
        super().__init__(**kwargs)
        self.conductivity = sigma
        self.rho = rho
        self.conductivity_map = conductivity_map


class ReciprocalPropExample(props.HasModel):
    sigma = props.PhysicalProperty("Electrical conductivity (S/m)")

    rho = props.PhysicalProperty("Electrical resistivity (Ohm m)")

    props.Reciprocal(sigma, rho)

    def __init__(self, sigma=None, rho=None, rhoMap=None, **kwargs):
        super().__init__(**kwargs)
        self.conductivity = sigma
        self.rho = rho


class ReciprocalPropExampleDefaults(props.HasModel):
    sigma = props.PhysicalProperty("Electrical conductivity (S/m)")

    rho = props.PhysicalProperty("Electrical resistivity (Ohm m)")

    props.Reciprocal(sigma, rho)

    def __init__(
        self, sigma=None, conductivity_map=None, rho=None, rhoMap=None, **kwargs
    ):
        super().__init__(**kwargs)
        if sigma is None:
            sigma = np.r_[1.0, 2.0, 3.0]
        self.conductivity = sigma
        self.rho = rho


class ComplicatedInversion(props.HasModel):
    Ks, KsMap, KsDeriv = props.Invertible("Saturated hydraulic conductivity")

    A, AMap, ADeriv = props.Invertible("fitting parameter")

    gamma, gammaMap, gammaDeriv = props.Invertible("fitting parameter")

    def __init__(
        self,
        Ks=24.96,
        KsMap=None,
        A=1.175e06,
        AMap=None,
        gamma=4.74,
        gammaMap=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.Ks = Ks
        self.KsMap = KsMap
        self.A = A
        self.AMap = AMap
        self.gamma = gamma
        self.gammaMap = gammaMap


class NestedModels(props.HasModel):
    nest_model = props.NestedModeler(ComplicatedInversion, "Nested models")

    def __init__(self, nest_model=None, **kwargs):
        super().__init__(**kwargs)
        self.nest_model = nest_model


class OptionalInvertible(props.HasModel):
    sigma, conductivity_map, _con_deriv = props.Invertible(
        "Electrical conductivity (S/m)", optional=True
    )


@pytest.mark.parametrize("example", [SimpleExample, ShortcutExample])
def test_basic(example):
    expMap = maps.ExpMap(discretize.TensorMesh((3,)))
    assert expMap.nP == 3

    PM = example(conductivity_map=expMap)
    assert PM.conductivityMap is not None
    assert PM.conductivityMap is expMap

    # There is currently no model, so sigma, which is mapped, fails
    with pytest.raises(AttributeError):
        PM.conductivity

    PM.model = np.r_[1.0, 2.0, 3.0]
    assert np.all(PM.conductivity == np.exp(np.r_[1.0, 2.0, 3.0]))
    # PM = pickle.loads(pickle.dumps(PM))
    # PM = maps.ExpMap.deserialize(PM.serialize())

    assert np.all(
        PM._con_deriv.todense() == utils.sdiag(np.exp(np.r_[1.0, 2.0, 3.0])).todense()
    )

    # If we set sigma, we should delete the mapping
    PM.conductivity = np.r_[1.0, 2.0, 3.0]
    assert np.all(PM.conductivity == np.r_[1.0, 2.0, 3.0])
    # PM = pickle.loads(pickle.dumps(PM))
    assert PM.conductivityMap is None
    assert PM._con_deriv == 0

    del PM.model
    # sigma is not changed
    assert np.all(PM.conductivity == np.r_[1.0, 2.0, 3.0])


def test_reciprocal():
    expMap = maps.ExpMap(discretize.TensorMesh((3,)))

    PM = ReciprocalMappingExample()

    with pytest.raises(AttributeError):
        PM.conductivity
    PM.conductivityMap = expMap
    PM.model = np.r_[1.0, 2.0, 3.0]
    assert np.all(PM.conductivity == np.exp(np.r_[1.0, 2.0, 3.0]))
    assert np.all(PM.rho == 1.0 / np.exp(np.r_[1.0, 2.0, 3.0]))

    PM.rho = np.r_[1.0, 2.0, 3.0]
    assert PM.rhoMap is None
    assert PM.conductivityMap is None
    assert PM.rhoDeriv == 0
    assert PM._con_deriv == 0
    assert np.all(PM.conductivity == 1.0 / np.r_[1.0, 2.0, 3.0])

    PM.conductivityMap = expMap
    # change your mind?
    # PM = pickle.loads(pickle.dumps(PM))
    PM.rhoMap = expMap
    assert PM._conductivity_map is None
    assert len(PM.rhoMap) == 1
    assert len(PM.conductivityMap) == 2
    # PM = pickle.loads(pickle.dumps(PM))
    assert np.all(PM.rho == np.exp(np.r_[1.0, 2.0, 3.0]))
    assert np.all(PM.conductivity == 1.0 / np.exp(np.r_[1.0, 2.0, 3.0]))
    # PM = pickle.loads(pickle.dumps(PM))
    assert isinstance(PM._con_deriv.todense(), np.ndarray)


def test_reciprocal_no_map():
    expMap = maps.ExpMap(discretize.TensorMesh((3,)))

    PM = ReciprocalExample()
    with pytest.raises(AttributeError):
        PM.conductivity

    PM.conductivityMap = expMap
    # PM = pickle.loads(pickle.dumps(PM))
    PM.model = np.r_[1.0, 2.0, 3.0]
    assert np.all(PM.conductivity == np.exp(np.r_[1.0, 2.0, 3.0]))
    assert np.all(PM.rho == 1.0 / np.exp(np.r_[1.0, 2.0, 3.0]))

    PM.rho = np.r_[1.0, 2.0, 3.0]
    assert PM.conductivityMap is None
    assert PM._con_deriv == 0
    assert np.all(PM.conductivity == 1.0 / np.r_[1.0, 2.0, 3.0])

    PM.conductivityMap = expMap
    assert len(PM.conductivityMap) == 1
    # PM = pickle.loads(pickle.dumps(PM))
    assert np.all(PM.rho == 1.0 / np.exp(np.r_[1.0, 2.0, 3.0]))
    assert np.all(PM.conductivity == np.exp(np.r_[1.0, 2.0, 3.0]))
    assert isinstance(PM._con_deriv.todense(), np.ndarray)


def test_reciprocal_no_maps():
    PM = ReciprocalPropExample()
    with pytest.raises(AttributeError):
        PM.conductivity

    # PM = pickle.loads(pickle.dumps(PM))
    PM.conductivity = np.r_[1.0, 2.0, 3.0]
    # PM = pickle.loads(pickle.dumps(PM))

    assert np.all(PM.conductivity == np.r_[1.0, 2.0, 3.0])
    # PM = pickle.loads(pickle.dumps(PM))
    assert np.all(PM.rho == 1.0 / np.r_[1.0, 2.0, 3.0])

    PM.rho = np.r_[1.0, 2.0, 3.0]
    assert np.all(PM.conductivity == 1.0 / np.r_[1.0, 2.0, 3.0])


def test_reciprocal_defaults():
    PM = ReciprocalPropExampleDefaults()
    assert np.all(PM.conductivity == np.r_[1.0, 2.0, 3.0])
    assert np.all(PM.rho == 1.0 / np.r_[1.0, 2.0, 3.0])

    rho = np.r_[2.0, 4.0, 6.0]
    PM.rho = rho
    assert np.all(PM.rho == rho)
    assert np.all(PM.conductivity == 1.0 / rho)


def test_multi_parameter_inversion():
    """The setup of the defaults should not invalidated the
    mappings or other defaults.
    """
    PM = ComplicatedInversion()
    params = inspect.signature(ComplicatedInversion).parameters

    np.testing.assert_equal(PM.Ks, params["Ks"].default)
    np.testing.assert_equal(PM.gamma, params["gamma"].default)
    np.testing.assert_equal(PM.A, params["A"].default)


def test_nested():
    PM = NestedModels()
    assert PM._has_nested_models is True


def test_optional_inverted():
    modeler = OptionalInvertible()
    assert modeler.conductivityMap is None
    assert modeler.conductivity is None

    modeler.conductivity = 10
    assert modeler.conductivity == 10


if __name__ == "__main__":
    unittest.main()
