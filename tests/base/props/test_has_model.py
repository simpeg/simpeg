import re

import pytest
import numpy as np
import numpy.testing as npt
from simpeg import props, maps


@pytest.fixture(scope="module")
def mock_model_class():
    class MockModel(props.HasModel):
        prop = props.PhysicalProperty("test physical property")
        other = props.PhysicalProperty("another physical property")

        def __init__(self, prop=None, other=None, **kwargs):
            self._init_property(prop=prop, other=other)
            super().__init__(**kwargs)

        @property
        def prop_dependent_property(self):
            if (thing := getattr(self, "_prop_dependent_property", None)) is None:
                thing = 2 * self.prop
                self._prop_dependent_property = thing
            return thing

        @property
        def _delete_on_model_update(self):
            if "prop" in self.parametrizations:
                return ["_prop_dependent_property"]
            return []

    return MockModel


@pytest.fixture()
def modeler(mock_model_class):
    modeler = mock_model_class(prop=maps.ExpMap())
    return modeler


def test_clear_on_model_change(modeler):
    x = np.ones(10)
    modeler.model = x
    npt.assert_array_equal(np.exp(x), modeler.prop)

    item_1 = modeler.prop_dependent_property
    assert item_1 is not None

    x2 = x + 1
    modeler.model = x2
    assert getattr(modeler, "_prop_dependent_property", None) is None

    item_2 = modeler.prop_dependent_property
    assert item_2 is not None
    assert item_1 is not item_2


def test_no_clear_on_model_reassign(modeler):
    x = np.ones(10)

    modeler.model = x
    npt.assert_array_equal(np.exp(x), modeler.prop)

    item_1 = modeler.prop_dependent_property
    assert item_1 is not None

    modeler.model = x.copy()
    assert getattr(modeler, "_prop_dependent_property", None) is not None

    item_2 = modeler.prop_dependent_property

    assert item_1 is item_2


def test_map_clearing(modeler):
    modeler.prop = np.ones(10)
    assert "prop" not in modeler.parametrizations


def test_no_clear_without_mapping(modeler):
    modeler.parametrize("other", maps.ExpMap())
    modeler.prop = np.ones(10)
    item1 = modeler.prop_dependent_property
    assert item1 is not None

    modeler.model = np.zeros(10)

    assert getattr(modeler, "_prop_dependent_property", None) is not None
    assert modeler.prop_dependent_property is item1


def test_model_needed(modeler):
    assert modeler.needs_model


def test_no_model_needed(modeler):
    modeler.prop = None
    assert not modeler.needs_model

    modeler.parametrize("prop", maps.ExpMap())
    assert modeler.needs_model

    modeler.prop = np.ones(10)
    assert not modeler.needs_model


def test_deletion_deprecation(modeler):
    msg = re.escape("HasModel.deleteTheseOnModelUpdate has been deprecated") + ".*"
    with pytest.warns(FutureWarning, match=msg):
        modeler.deleteTheseOnModelUpdate

    msg = "clean_on_model_update has been deprecated due to repeated functionality.*"
    with pytest.warns(FutureWarning, match=msg):
        modeler.clean_on_model_update
