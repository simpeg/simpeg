from simpeg import props, maps
import pytest
import numpy.testing as npt


class InnerProp(props.HasModel):
    phys_prop = props.PhysicalProperty("An invertible physical property")


class NotInvertibleInnerProp(InnerProp):
    phys_prop = InnerProp.phys_prop.update_invertible(False)


class NestedModels(props.HasModel):
    nest_model = props.NestedModeler(InnerProp, "Nested modeler")
    nest_two = props.NestedModeler(InnerProp, "Second Nested modeler")

    def __init__(self, nest_model=None, nest_two=None, **kwargs):
        super().__init__(**kwargs)
        self.nest_model = nest_model
        if nest_two is None:
            nest_two = nest_model
        self.nest_two = nest_two


def test_has_nested():
    assert NestedModels._has_nested_models is True


def test_unset_nested():
    pm = NestedModels()
    with pytest.raises(AttributeError):
        pm.nest_model


@pytest.mark.parametrize("prop1", [maps.ExpMap(), [1, 2, 3]])
@pytest.mark.parametrize("prop2", [maps.ExpMap(), [1, 2, 3]])
def test_model_passthrough(prop1, prop2):
    needs_model = isinstance(prop1, maps.IdentityMap) or isinstance(
        prop2, maps.IdentityMap
    )
    if needs_model:
        inner1 = InnerProp()
        inner1.phys_prop = prop1
        inner2 = InnerProp()
        inner2.phys_prop = prop2
        pm = NestedModels(nest_model=inner1, nest_two=inner2)
        pm.model = [1, 2, 3]
        npt.assert_equal(pm.model, [1, 2, 3])

        m1 = pm.nest_model.model
        m2 = pm.nest_two.model
        if isinstance(prop1, maps.IdentityMap):
            npt.assert_equal(m1, [1, 2, 3])
        else:
            assert m1 is None
        if isinstance(prop2, maps.IdentityMap):
            npt.assert_equal(m2, [1, 2, 3])
        else:
            assert m2 is None


@pytest.mark.parametrize("prop1", [maps.ExpMap(), [1, 2, 3]])
@pytest.mark.parametrize("prop2", [maps.ExpMap(), [1, 2, 3]])
def test_needs_model(prop1, prop2):
    inner1 = InnerProp()
    inner1.phys_prop = prop1
    inner2 = InnerProp()
    inner2.phys_prop = prop2
    pm = NestedModels(nest_model=inner1, nest_two=inner2)
    assert pm.needs_model == isinstance(prop1, maps.IdentityMap) or isinstance(
        prop2, maps.IdentityMap
    )


@pytest.mark.parametrize("prop1", [maps.ExpMap(nP=3), [1, 2, 3]])
def test_bad_model_passthrough(prop1):
    prop2 = maps.ExpMap(nP=5)

    inner1 = InnerProp()
    inner1.phys_prop = prop1
    inner2 = InnerProp()
    inner2.phys_prop = prop2
    pm = NestedModels(nest_model=inner1, nest_two=inner2)

    with pytest.raises(ValueError):
        pm.model = [1, 2, 3]
