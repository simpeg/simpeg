import numpy as np
import pytest

from simpeg import utils
from simpeg import objective_function
from simpeg.objective_function import BaseObjectiveFunction, ComboObjectiveFunction
from simpeg.objective_function import _validate_multiplier, _need_to_pass_fields
from simpeg.utils import Zero
from simpeg.fields import Fields


class MockObjectiveFunction(BaseObjectiveFunction):
    """Mock objective function class to run tests."""

    def __init__(self, nP=None, result=None):
        self._nP = nP
        self._result = result

    def __call__(self, model, f=None):
        if self._result is None:
            return 1.0
        return self._result

    def deriv(self, model):
        raise NotImplementedError()

    def deriv2(self, model, v=None):
        raise NotImplementedError()

    @property
    def nP(self):
        return self._nP


class MockL2ObjectiveFunction(BaseObjectiveFunction):
    """
    Mock L2 objective function to use in tests.
    """

    def __init__(self):
        self._nP = 3

    @property
    def nP(self):
        return self._nP

    def __call__(self, model, f=None):
        residual = self.G @ model
        return residual.T @ residual

    def deriv(self, model):
        return 2 * self.G.T @ self.G @ model

    def deriv2(self, model, v=None):
        return 2 * self.G.T @ self.G

    @property
    def G(self):
        if not hasattr(self, "_G"):
            matrix = [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
            self._G = np.array(matrix, dtype=np.float64)
        return self._G


def test_invalid_objfcts_in_combo():
    """Test invalid objective function class in ComboObjectiveFunction."""

    class Dummy:
        pass

    phi = MockObjectiveFunction()
    invalid_phi = Dummy()
    msg = "Unrecognized objective function type Dummy in 'objfcts'."
    with pytest.raises(TypeError, match=msg):
        objective_function.ComboObjectiveFunction(objfcts=[phi, invalid_phi])


def test_test_derivatives():
    """
    Check the `.test_derivatives` method in `BaseObjectiveFunction`
    """
    objfct = MockL2ObjectiveFunction()
    assert objfct.test_derivatives(eps=1e-9, random_seed=42)


@pytest.mark.parametrize(
    "objfcts, multipliers",
    (
        (None, None),
        ([MockObjectiveFunction()], None),
        ([MockObjectiveFunction()], [2.5]),
    ),
)
def test_empty_combo(objfcts, multipliers):
    """Test defining an empty ComboObjectiveFunction."""
    combo = objective_function.ComboObjectiveFunction(
        objfcts=objfcts, multipliers=multipliers
    )
    if objfcts is None and multipliers is None:
        assert combo.objfcts == []
        assert combo.multipliers == []
    if objfcts is not None:
        assert combo.objfcts == objfcts
        if multipliers is None:
            assert combo.multipliers == [1]
        else:
            assert combo.multipliers == [2.5]


class TestOperationsObjectiveFunctions:
    """Test arithmetic operations involving BaseObjectiveFunction"""

    @pytest.fixture
    def dummy_class(self):
        class Dummy:
            pass

        return Dummy

    @pytest.mark.parametrize("left", (True, False))
    def test_mul(self, left):
        """
        Test scalar multiplication for BaseObjectiveFunction
        """
        phi = MockObjectiveFunction(nP=3, result=2.0)
        if left:
            new_phi = 3.0 * phi
        else:
            new_phi = phi * 3.0
        assert new_phi.multipliers == [3.0]
        assert new_phi.objfcts == [phi]
        model = np.array([1.0])
        np.testing.assert_allclose(
            3.0 * phi(model),
            new_phi(model),
        )

    def test_div(self):
        """
        Test div of BaseObjectiveFunctions over a scalar
        """
        phi = MockObjectiveFunction(nP=3, result=2.0)
        new_phi = phi / 3.0
        assert new_phi.multipliers == [1.0 / 3.0]
        assert new_phi.objfcts == [phi]
        model = np.array([1.0])
        np.testing.assert_allclose(
            phi(model) / 3.0,
            new_phi(model),
        )

    def test_add(self):
        """
        Test add between two BaseObjectiveFunctions
        """
        phi1 = MockObjectiveFunction(nP=3, result=1.0)
        phi2 = MockObjectiveFunction(nP=3, result=2.0)
        new_phi = phi1 + phi2
        assert new_phi.multipliers == [1.0, 1.0]
        assert new_phi.objfcts == [phi1, phi2]
        model = np.array([1.0])
        np.testing.assert_allclose(
            phi1(model) + phi2(model),
            new_phi(model),
        )

    def test_add_and_mul(self):
        """
        Test add between several BaseObjectiveFunctions with scalar multiplications
        """
        phi1 = MockObjectiveFunction(nP=3, result=1.0)
        phi2 = MockObjectiveFunction(nP=3, result=2.0)
        phi3 = MockObjectiveFunction(nP=3, result=3.0)
        new_phi = 1.3 * phi1 + 3.3 * phi2 + 4.2 * phi3
        assert new_phi.multipliers == [1.3, 3.3, 4.2]
        assert new_phi.objfcts == [phi1, phi2, phi3]
        model = np.array([1.0])
        np.testing.assert_allclose(
            1.3 * phi1(model) + 3.3 * phi2(model) + 4.2 * phi3(model),
            new_phi(model),
        )

    @pytest.mark.parametrize("mul_scalar", (True, False))
    def test_mul_by_zero_object(self, mul_scalar):
        """
        Test forming a ComboObjectiveFunction adding two objective functions
        but multiplying one by Zero.
        """
        n_params = 20
        phi1 = MockObjectiveFunction(nP=n_params, result=1.0)
        phi2 = MockObjectiveFunction(nP=n_params, result=2.0)
        if mul_scalar:
            combo = 3.3 * phi1 + utils.Zero() * phi2
            assert combo.objfcts == [phi1]
            assert combo.multipliers == [3.3]
        else:
            combo = phi1 + utils.Zero() * phi2
            # The combo should be the objective function phi1
            assert combo == phi1

    def test_mul_by_zero_float(self):
        """
        Test forming a ComboObjectiveFunction adding two objective functions
        but multiplying one by zero (float).
        """
        n_params = 20
        phi1 = MockObjectiveFunction(nP=n_params, result=1.0)
        phi2 = MockObjectiveFunction(nP=n_params, result=2.0)
        combo = 2.3 * phi1 + 0.0 * phi2
        assert combo.objfcts == [phi1, phi2]
        assert combo.multipliers == [2.3, 0.0]
        model = np.array([1.0])
        np.testing.assert_allclose(combo(model), 2.3 * phi1(model))

    @pytest.mark.parametrize("radd", (False, True), ids=("add", "radd"))
    def test_error_add_not_objective_functions(self, dummy_class, radd):
        """
        Test if error is raised when trying to add a non-objective function object.
        """
        phi = MockObjectiveFunction(nP=1)
        dummy = dummy_class()
        msg = (
            "Cannot add type 'Dummy' to an objective function. "
            "Only 'BaseObjectiveFunction's can be added together."
        )
        with pytest.raises(TypeError, match=msg):
            if radd:
                dummy + phi
            else:
                phi + dummy

    def test_error_different_np(self):
        """
        Test if error is raised after trying to add objective functions with
        different nP
        """
        phi1 = MockObjectiveFunction(nP=1)
        phi2 = MockObjectiveFunction(nP=3)
        msg = "Invalid number of parameters"
        with pytest.raises(ValueError, match=msg):
            phi1 + phi2

    @pytest.mark.parametrize("two_combos", (True, False))
    def test_error_different_np_combo(self, two_combos):
        """
        Test if error is raised after trying to add combo objective functions
        with different nP
        """
        phi1 = 2.0 * MockObjectiveFunction(nP=1) + 3.0 * MockObjectiveFunction(nP=1)
        if two_combos:
            phi2 = 5.0 * MockObjectiveFunction(nP=3) + 6.0 * MockObjectiveFunction(nP=3)
        else:
            phi2 = MockObjectiveFunction(nP=3)
        msg = "Invalid number of parameters"
        with pytest.raises(ValueError, match=msg):
            phi1 + phi2


class TestOperationsComboObjectiveFunctions:
    """Test arithmetic operations involving ComboObjectiveFunction"""

    @pytest.mark.parametrize("unpack_on_add", (True, False))
    def test_mul(self, unpack_on_add):
        """Test if ComboObjectiveFunction multiplication works as expected"""
        n_params = 10
        phi1 = MockObjectiveFunction(nP=n_params, result=1.0)
        phi2 = MockObjectiveFunction(nP=n_params, result=2.0)
        combo = objective_function.ComboObjectiveFunction(
            [phi1, phi2], [2, 3], unpack_on_add=unpack_on_add
        )
        combo_mul = 3.5 * combo
        assert len(combo_mul) == 1
        assert combo_mul.multipliers == [3.5]
        assert combo_mul.objfcts == [combo]
        model = np.array([1.0])
        np.testing.assert_allclose(
            3.5 * combo(model),
            combo_mul(model),
        )

    @pytest.mark.parametrize("unpack_on_add", (True, False))
    def test_add(self, unpack_on_add):
        """Test if ComboObjectiveFunction addition works as expected"""
        n_params = 10
        phi1 = MockObjectiveFunction(nP=n_params, result=1.0)
        phi2 = MockObjectiveFunction(nP=n_params, result=2.0)
        phi3 = MockObjectiveFunction(nP=n_params, result=3.0)
        combo_1 = objective_function.ComboObjectiveFunction(
            [phi1, phi2], [2, 3], unpack_on_add=unpack_on_add
        )
        combo_2 = phi3 + combo_1
        if unpack_on_add:
            assert len(combo_2) == 3
            assert combo_2.multipliers == [1, 2, 3]
            assert combo_2.objfcts == [phi3, phi1, phi2]
        else:
            assert len(combo_2) == 2
            assert combo_2.multipliers == [1, 1]
            assert combo_2.objfcts == [phi3, combo_1]
            combo_1 = combo_2.objfcts[1]
            assert combo_1.multipliers == [2, 3]
        model = np.array([1.0])
        np.testing.assert_allclose(
            phi3(model) + 2 * phi1(model) + 3 * phi2(model),
            combo_2(model),
        )

    def test_add_multiple_terms(self):
        """Test addition of multiple BaseObjectiveFunctions"""
        n_params = 10
        phi1 = MockObjectiveFunction(nP=n_params, result=1.0)
        phi2 = MockObjectiveFunction(nP=n_params, result=2.0)
        phi3 = MockObjectiveFunction(nP=n_params, result=3.0)
        combo = 1.1 * phi1 + 1.2 * phi2 + 1.3 * phi3
        assert len(combo) == 3
        assert combo.multipliers == [1.1, 1.2, 1.3]
        assert combo.objfcts == [phi1, phi2, phi3]
        model = np.array([1.0])
        np.testing.assert_allclose(
            1.1 * phi1(model) + 1.2 * phi2(model) + 1.3 * phi3(model),
            combo(model),
        )

    @pytest.mark.parametrize("unpack_on_add", (True, False))
    def test_add_and_mul(self, unpack_on_add):
        """
        Test ComboObjectiveFunction addition with multiplication

        After multiplying a Combo with a scalar, the `__mul__` method creates
        another Combo for it.
        """
        n_params = 10
        phi1 = MockObjectiveFunction(nP=n_params)
        phi2 = MockObjectiveFunction(nP=n_params)
        phi3 = MockObjectiveFunction(nP=n_params)
        combo_1 = objective_function.ComboObjectiveFunction(
            [phi1, phi2], [2, 3], unpack_on_add=unpack_on_add
        )
        combo_2 = 5 * phi3 + 1.2 * combo_1
        assert len(combo_2) == 2
        assert combo_2.multipliers == [5, 1.2]
        assert combo_2.objfcts == [phi3, combo_1]
        model = np.array([1.0])
        np.testing.assert_allclose(
            5 * phi3(model) + 1.2 * (2 * phi1(model) + 3 * phi2(model)),
            combo_2(model),
        )


class TestMultiplierValidation:
    """
    Test the _validate_multiplier private function.
    """

    valid_multipliers = (
        3.14,
        1,
        np.float64(-15.3),
        np.float32(-10.2),
        np.int64(10),
        np.int32(33),
        Zero(),
    )
    invalid_multipliers = (
        np.array([1, 3.14]),
        np.array(3),
        [1, 2, 3],
        "string",
        True,
        None,
    )

    @pytest.mark.parametrize("multiplier", valid_multipliers)
    def test_valid_multipliers(self, multiplier):
        """
        Test function against valid multipliers
        """
        _validate_multiplier(multiplier)

    @pytest.mark.parametrize("multiplier", invalid_multipliers)
    def test_invalid_multipliers(self, multiplier):
        """
        Test function against invalid multipliers
        """
        with pytest.raises(TypeError, match="Invalid multiplier"):
            _validate_multiplier(multiplier)

    def test_multipliers_setter(self):
        """
        Test multipliers setter against valid multipliers
        """
        objective_functions = [MockObjectiveFunction() for _ in range(3)]
        combo = objective_function.ComboObjectiveFunction(objfcts=objective_functions)
        multipliers = [i + 3 for i in range(len(objective_functions))]
        combo.multipliers = multipliers

    @pytest.mark.parametrize("multiplier", invalid_multipliers)
    def test_multipliers_setter_invalid(self, multiplier):
        """
        Test multipliers setter against invalid multipliers
        """
        phi = MockObjectiveFunction()
        combo = objective_function.ComboObjectiveFunction(objfcts=[phi])
        with pytest.raises(TypeError, match="Invalid multiplier"):
            combo.multipliers = [multiplier]

    def test_multipliers_setter_invalid_length(self):
        """
        Test error when setting multipliers of invalid length.
        """
        objective_functions = [MockObjectiveFunction() for _ in range(3)]
        multipliers = [1, 2, 3, 4, 5, 6]
        combo = objective_function.ComboObjectiveFunction(objfcts=objective_functions)
        msg = "Inconsistent number of elements between objective functions "
        with pytest.raises(ValueError, match=msg):
            combo.multipliers = multipliers


class TestGetFunctionsOfType:
    """
    Test the ``get_functions_of_type`` method in ``ComboObjectiveFunction``.
    """

    @pytest.fixture
    def mock_class_type_a(self):
        class MockTypeA(MockObjectiveFunction):
            pass

        return MockTypeA

    @pytest.fixture
    def mock_class_type_b(self):
        class MockTypeB(MockObjectiveFunction):
            pass

        return MockTypeB

    def test_same_class(self, mock_class_type_a, mock_class_type_b):
        """
        Test combo with two objective functions of the same class.
        """
        phi_1, phi_2 = mock_class_type_a(), mock_class_type_a()
        combo = 2 * phi_1 + 3 * phi_2
        result = combo.get_functions_of_type(mock_class_type_a)
        assert result == [phi_1, phi_2]

    @pytest.mark.parametrize("class_type", ("a", "b"))
    def test_different_classes(self, class_type, mock_class_type_a, mock_class_type_b):
        """
        Test combo with two objective functions of different classes.
        """
        phi_1, phi_2 = mock_class_type_a(), mock_class_type_b()
        combo = 2 * phi_1 + 3 * phi_2
        if class_type == "a":
            result = combo.get_functions_of_type(mock_class_type_a)
            assert result == [phi_1]
        else:
            result = combo.get_functions_of_type(mock_class_type_b)
            assert result == [phi_2]

    def test_combo_class(self, mock_class_type_a, mock_class_type_b):
        """
        Test if the required class is a ComboObjectiveFunction.
        """
        phi_1, phi_2 = mock_class_type_a(), mock_class_type_b()
        combo = 2 * phi_1 + 3 * phi_2
        result = combo.get_functions_of_type(ComboObjectiveFunction)
        assert result == [combo]

    @pytest.mark.parametrize("class_type", ("a", "b"))
    def test_nested_combos(self, class_type, mock_class_type_a, mock_class_type_b):
        """
        Test when the combo has nested combos.
        """
        phi_1, phi_2 = mock_class_type_a(), mock_class_type_b()
        phi_3, phi_4 = mock_class_type_a(), mock_class_type_b()
        sub_combo_1 = 2 * phi_1 + 3 * phi_2
        sub_combo_2 = 5 * phi_3 + 4 * phi_4
        combo = ComboObjectiveFunction(
            objfcts=[sub_combo_1, sub_combo_2], multipliers=[1, 1]
        )
        # Just check the nested structure before running the tests
        assert combo.objfcts == [sub_combo_1, sub_combo_2]
        # Tests the get_functions_of_type method
        if class_type == "a":
            result = combo.get_functions_of_type(mock_class_type_a)
            assert result == [[phi_1], [phi_3]]
        else:
            result = combo.get_functions_of_type(mock_class_type_b)
            assert result == [[phi_2], [phi_4]]


class TestNeedsFields:
    """
    Test the private ``_need_to_pass_fields`` function.
    """

    @pytest.fixture
    def mock_with_has_fields(self):
        class MockWithHasField(MockObjectiveFunction):

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.has_fields = True

        return MockWithHasField

    @pytest.fixture
    def combo_with_fields(self):
        class ComboWithHasField(ComboObjectiveFunction):

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.has_fields = True

        return ComboWithHasField

    def test_with_regular_class(self):
        """
        Test ``_need_to_pass_fields`` with a regular objective function class.
        """
        phi = MockObjectiveFunction()
        assert not _need_to_pass_fields(phi)

    def test_with_class_that_has_fields(self, mock_with_has_fields):
        """
        Test ``_need_to_pass_fields`` with a class that has ``has_fields``.
        """
        assert _need_to_pass_fields(mock_with_has_fields())

    def test_with_combo(self):
        """
        Test ``_need_to_pass_fields`` with a combo class.
        """
        phi = MockObjectiveFunction()
        assert _need_to_pass_fields(ComboObjectiveFunction(objfcts=[phi]))

    def test_with_combo_that_has_fields(self, combo_with_fields):
        """
        Test ``_need_to_pass_fields`` with a combo class that has ``has_fields``.
        """
        phi = MockObjectiveFunction()
        assert _need_to_pass_fields(combo_with_fields(objfcts=[phi]))


class TestCallComboObjectiveFunction:
    """
    Test calling a ``ComboObjectiveFunction`` with has_fields.
    """

    @pytest.fixture
    def mock_simulation(self):
        from simpeg.simulation import BaseSimulation

        class MockSimulation(BaseSimulation):
            pass

        return MockSimulation

    @pytest.fixture
    def mock_with_has_fields(self):
        """Mock objective function class with ``has_fields``."""

        class MockWithHasField(MockObjectiveFunction):

            def __init__(self, **kwargs):
                super().__init__(nP=1, **kwargs)  # the model must have a single value
                self.has_fields = True

            def __call__(self, model, f=None):
                """
                Evaluate as the model value times the field "foo" in ``f``.
                """
                (value,) = model
                return value * f.knownFields["foo"]

        return MockWithHasField

    @pytest.fixture
    def mock_without_fields(self):
        """Mock objective function class without ``has_fields``."""

        class MockWithoutField(MockObjectiveFunction):

            def __init__(self, **kwargs):
                super().__init__(nP=1, **kwargs)  # the model must have a single value

            def __call__(self, model, f=None):
                """
                Evaluate as the model value times the field "foo" in ``f``.
                """
                (value,) = model
                return value

        return MockWithoutField

    def test_mock_classes(self, mock_simulation, mock_with_has_fields):
        """Simple test on mock classes."""
        simulation = mock_simulation()
        phi = mock_with_has_fields()
        model = np.array([3.5])
        fields = Fields(simulation, knownFields={"foo": 3.0})
        np.testing.assert_allclose(phi(model, fields), 3.5 * 3.0)

    def test_call_combo_with_fields(self, mock_simulation, mock_with_has_fields):
        """Test calling a combo with a single objective function with fields."""
        simulation = mock_simulation()
        phi = mock_with_has_fields()
        combo = ComboObjectiveFunction(objfcts=[phi], multipliers=[2.0])
        model = np.array([3.5])
        fields = Fields(simulation, knownFields={"foo": 3.0})
        np.testing.assert_allclose(combo(model, [fields]), 2.0 * 3.5 * 3.0)

    def test_call_combo_multiple_objective_with_fields(
        self, mock_simulation, mock_with_has_fields, mock_without_fields
    ):
        """Test calling a combo with a multiple objective functions with fields."""
        simulation = mock_simulation()
        phi_1 = mock_with_has_fields()
        phi_2 = mock_without_fields()
        combo = ComboObjectiveFunction(objfcts=[phi_1, phi_2], multipliers=[2.0, 5.0])
        model = np.array([3.5])
        fields = Fields(simulation, knownFields={"foo": 3.0})
        np.testing.assert_allclose(
            combo(model, [fields, None]), 2.0 * 3.5 * 3.0 + 5.0 * 3.5
        )

    def test_call_zero_multiplier(
        self, mock_simulation, mock_with_has_fields, mock_without_fields
    ):
        """Test calling a combo with a function with multiplier equal to zero."""
        simulation = mock_simulation()
        phi_1 = mock_with_has_fields()
        phi_2 = mock_with_has_fields()
        phi_3 = mock_without_fields()
        combo = ComboObjectiveFunction(
            objfcts=[phi_1, phi_2, phi_3], multipliers=[2.0, 0.0, 5.0]
        )
        model = np.array([3.5])
        fields_1 = Fields(simulation, knownFields={"foo": 3.0})
        fields_2 = Fields(simulation, knownFields={"foo": -2.0})
        expected_result = 2.0 * 3.5 * 3.0 + 5.0 * 3.5
        np.testing.assert_allclose(
            combo(model, [fields_1, fields_2, None]), expected_result
        )
