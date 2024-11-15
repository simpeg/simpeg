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

    conductivity = props.PhysicalProperty(
        "Electrical conductivity (S/m)", mapping=conductivity_map
    )

    _con_deriv = props.Derivative(
        "Derivative of conductivity wrt the model.", physical_property=conductivity
    )

    def __init__(self, conductivity=None, conductivity_map=None, **kwargs):
        super().__init__(**kwargs)
        self.conductivity = conductivity
        self.conductivity_map = conductivity_map


class ShortcutExample(props.HasModel):
    conductivity, conductivity_map, _con_deriv = props.Invertible(
        "Electrical conductivity (S/m)"
    )

    def __init__(self, conductivity=None, conductivity_map=None, **kwargs):
        super().__init__(**kwargs)
        self.conductivity = conductivity
        self.conductivity_map = conductivity_map


class ReciprocalMappingExample(props.HasModel):
    conductivity, conductivity_map, _con_deriv = props.Invertible(
        "Electrical conductivity (S/m)"
    )

    resistivity, resistivity_map, _res_deriv = props.Invertible(
        "Electrical resistivity (Ohm m)"
    )

    props.Reciprocal(conductivity, resistivity)

    def __init__(
        self,
        conductivity=None,
        conductivity_map=None,
        resistivity=None,
        resistivity_map=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conductivity = conductivity
        self.resistivity = resistivity
        self.conductivity_map = conductivity_map
        self.resistivity_map = resistivity_map


class ReciprocalExample(props.HasModel):
    conductivity, conductivity_map, _con_deriv = props.Invertible(
        "Electrical conductivity (S/m)"
    )

    resistivity = props.PhysicalProperty("Electrical resistivity (Ohm m)")

    props.Reciprocal(conductivity, resistivity)

    def __init__(
        self, conductivity=None, conductivity_map=None, resistivity=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.conductivity = conductivity
        self.resistivity = resistivity
        self.conductivity_map = conductivity_map


class ReciprocalPropExample(props.HasModel):
    conductivity = props.PhysicalProperty("Electrical conductivity (S/m)")

    resistivity = props.PhysicalProperty("Electrical resistivity (Ohm m)")

    props.Reciprocal(conductivity, resistivity)

    def __init__(
        self, conductivity=None, resistivity=None, resistivity_map=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.conductivity = conductivity
        self.resistivity = resistivity


class ReciprocalPropExampleDefaults(props.HasModel):
    conductivity = props.PhysicalProperty("Electrical conductivity (S/m)")

    resistivity = props.PhysicalProperty("Electrical resistivity (Ohm m)")

    props.Reciprocal(conductivity, resistivity)

    def __init__(
        self,
        conductivity=None,
        conductivity_map=None,
        resistivity=None,
        resistivity_map=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if conductivity is None:
            conductivity = np.r_[1.0, 2.0, 3.0]
        self.conductivity = conductivity
        self.resistivity = resistivity


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
    conductivity, conductivity_map, _con_deriv = props.Invertible(
        "Electrical conductivity (S/m)", optional=True
    )


@pytest.mark.parametrize("example", [SimpleExample, ShortcutExample])
def test_basic(example):
    expMap = maps.ExpMap(discretize.TensorMesh((3,)))
    assert expMap.nP == 3

    PM = example(conductivity_map=expMap)
    assert PM.conductivity_map is not None
    assert PM.conductivity_map is expMap

    # There is currently no model, so conductivity, which is mapped, fails
    with pytest.raises(AttributeError):
        PM.conductivity

    PM.model = np.r_[1.0, 2.0, 3.0]
    assert np.all(PM.conductivity == np.exp(np.r_[1.0, 2.0, 3.0]))
    # PM = pickle.loads(pickle.dumps(PM))
    # PM = maps.ExpMap.deserialize(PM.serialize())

    assert np.all(
        PM._con_deriv.todense() == utils.sdiag(np.exp(np.r_[1.0, 2.0, 3.0])).todense()
    )

    # If we set conductivity, we should delete the mapping
    PM.conductivity = np.r_[1.0, 2.0, 3.0]
    assert np.all(PM.conductivity == np.r_[1.0, 2.0, 3.0])
    # PM = pickle.loads(pickle.dumps(PM))
    assert PM.conductivity_map is None
    assert PM._con_deriv == 0

    del PM.model
    # conductivity is not changed
    assert np.all(PM.conductivity == np.r_[1.0, 2.0, 3.0])


def test_reciprocal():
    expMap = maps.ExpMap(discretize.TensorMesh((3,)))

    PM = ReciprocalMappingExample()

    with pytest.raises(AttributeError):
        PM.conductivity
    PM.conductivity_map = expMap
    PM.model = np.r_[1.0, 2.0, 3.0]
    assert np.all(PM.conductivity == np.exp(np.r_[1.0, 2.0, 3.0]))
    assert np.all(PM.resistivity == 1.0 / np.exp(np.r_[1.0, 2.0, 3.0]))

    PM.resistivity = np.r_[1.0, 2.0, 3.0]
    assert PM.resistivity_map is None
    assert PM.conductivity_map is None
    assert PM._res_deriv == 0
    assert PM._con_deriv == 0
    assert np.all(PM.conductivity == 1.0 / np.r_[1.0, 2.0, 3.0])

    PM.conductivity_map = expMap
    # change your mind?
    # PM = pickle.loads(pickle.dumps(PM))
    PM.resistivity_map = expMap
    assert PM._conductivity_map is None
    assert len(PM.resistivity_map) == 1
    assert len(PM.conductivity_map) == 2
    # PM = pickle.loads(pickle.dumps(PM))
    assert np.all(PM.resistivity == np.exp(np.r_[1.0, 2.0, 3.0]))
    assert np.all(PM.conductivity == 1.0 / np.exp(np.r_[1.0, 2.0, 3.0]))
    # PM = pickle.loads(pickle.dumps(PM))
    assert isinstance(PM._con_deriv.todense(), np.ndarray)


def test_reciprocal_no_map():
    expMap = maps.ExpMap(discretize.TensorMesh((3,)))

    PM = ReciprocalExample()
    with pytest.raises(AttributeError):
        PM.conductivity

    PM.conductivity_map = expMap
    # PM = pickle.loads(pickle.dumps(PM))
    PM.model = np.r_[1.0, 2.0, 3.0]
    assert np.all(PM.conductivity == np.exp(np.r_[1.0, 2.0, 3.0]))
    assert np.all(PM.resistivity == 1.0 / np.exp(np.r_[1.0, 2.0, 3.0]))

    PM.resistivity = np.r_[1.0, 2.0, 3.0]
    assert PM.conductivity_map is None
    assert PM._con_deriv == 0
    assert np.all(PM.conductivity == 1.0 / np.r_[1.0, 2.0, 3.0])

    PM.conductivity_map = expMap
    assert len(PM.conductivity_map) == 1
    # PM = pickle.loads(pickle.dumps(PM))
    assert np.all(PM.resistivity == 1.0 / np.exp(np.r_[1.0, 2.0, 3.0]))
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
    assert np.all(PM.resistivity == 1.0 / np.r_[1.0, 2.0, 3.0])

    PM.resistivity = np.r_[1.0, 2.0, 3.0]
    assert np.all(PM.conductivity == 1.0 / np.r_[1.0, 2.0, 3.0])


def test_reciprocal_defaults():
    PM = ReciprocalPropExampleDefaults()
    assert np.all(PM.conductivity == np.r_[1.0, 2.0, 3.0])
    assert np.all(PM.resistivity == 1.0 / np.r_[1.0, 2.0, 3.0])

    resistivity = np.r_[2.0, 4.0, 6.0]
    PM.resistivity = resistivity
    assert np.all(PM.resistivity == resistivity)
    assert np.all(PM.conductivity == 1.0 / resistivity)


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
    assert modeler.conductivity_map is None
    assert modeler.conductivity is None

    modeler.conductivity = 10
    assert modeler.conductivity == 10


if __name__ == "__main__":
    unittest.main()
