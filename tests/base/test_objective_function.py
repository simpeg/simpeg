import numpy as np
import scipy.sparse as sp
import pytest
import unittest

from simpeg import utils, maps
from simpeg import objective_function
from simpeg.objective_function import _validate_multiplier
from simpeg.utils import Zero

np.random.seed(130)

EPS = 1e-9


class Empty_ObjFct(objective_function.BaseObjectiveFunction):
    def __init__(self):
        super(Empty_ObjFct, self).__init__()


class Error_if_Hit_ObjFct(objective_function.BaseObjectiveFunction):
    def __init__(self):
        super(Error_if_Hit_ObjFct, self).__init__()

    def __call__(self, m):
        raise Exception("entered __call__")

    def deriv(self, m):
        raise Exception("entered deriv")

    def deriv2(self, m, v=None):
        raise Exception("entered deriv2")


class TestBaseObjFct(unittest.TestCase):
    def test_derivs(self):
        objfct = objective_function.L2ObjectiveFunction()
        self.assertTrue(objfct.test(eps=1e-9))

    def test_deriv2(self):
        nP = 100
        mapping = maps.ExpMap(nP=nP)
        m = np.random.rand(nP)
        v = np.random.rand(nP)
        objfct = objective_function.L2ObjectiveFunction(nP=nP, mapping=mapping)
        self.assertTrue(np.allclose(objfct.deriv2(m=m, v=v), objfct.deriv2(m=m) * v))

    def test_scalarmul(self):
        scalar = 10.0
        nP = 100
        objfct_a = objective_function.L2ObjectiveFunction(
            W=utils.sdiag(np.random.randn(nP))
        )
        objfct_b = scalar * objfct_a
        m = np.random.rand(nP)

        objfct_c = objfct_a + objfct_b

        self.assertTrue(scalar * objfct_a(m) == objfct_b(m))
        self.assertTrue(objfct_b.test())
        self.assertTrue(objfct_c(m) == objfct_a(m) + objfct_b(m))

        self.assertTrue(len(objfct_c.objfcts) == 2)
        self.assertTrue(len(objfct_c.multipliers) == 2)
        self.assertTrue(len(objfct_c) == 2)

    def test_sum(self):
        scalar = 10.0
        nP = 100.0
        objfct = objective_function.L2ObjectiveFunction(
            W=sp.eye(nP)
        ) + scalar * objective_function.L2ObjectiveFunction(W=sp.eye(nP))
        self.assertTrue(objfct.test(eps=1e-9))

        self.assertTrue(np.all(objfct.multipliers == np.r_[1.0, scalar]))

    def test_2sum(self):
        nP = 80
        alpha1 = 100
        alpha2 = 200

        phi1 = (
            objective_function.L2ObjectiveFunction(W=utils.sdiag(np.random.rand(nP)))
            + alpha1 * objective_function.L2ObjectiveFunction()
        )
        phi2 = objective_function.L2ObjectiveFunction() + alpha2 * phi1
        self.assertTrue(phi2.test(eps=EPS))

        self.assertTrue(len(phi1.multipliers) == 2)
        self.assertTrue(len(phi2.multipliers) == 2)

        self.assertTrue(len(phi1.objfcts) == 2)
        self.assertTrue(len(phi2.objfcts) == 2)
        self.assertTrue(len(phi2) == 2)

        self.assertTrue(len(phi1) == 2)
        self.assertTrue(len(phi2) == 2)

        self.assertTrue(np.all(phi1.multipliers == np.r_[1.0, alpha1]))
        self.assertTrue(np.all(phi2.multipliers == np.r_[1.0, alpha2]))

    def test_3sum(self):
        nP = 90

        alpha1 = 0.3
        alpha2 = 0.6
        alpha3inv = 9

        phi1 = objective_function.L2ObjectiveFunction(W=sp.eye(nP))
        phi2 = objective_function.L2ObjectiveFunction(W=sp.eye(nP))
        phi3 = objective_function.L2ObjectiveFunction(W=sp.eye(nP))

        phi = alpha1 * phi1 + alpha2 * phi2 + phi3 / alpha3inv

        m = np.random.rand(nP)

        self.assertTrue(
            np.all(phi.multipliers == np.r_[alpha1, alpha2, 1.0 / alpha3inv])
        )

        self.assertTrue(
            np.allclose(
                (alpha1 * phi1(m) + alpha2 * phi2(m) + phi3(m) / alpha3inv), phi(m)
            )
        )

        self.assertTrue(len(phi.objfcts) == 3)

        self.assertTrue(phi.test())

    def test_sum_fail(self):
        nP1 = 10
        nP2 = 30

        phi1 = objective_function.L2ObjectiveFunction(
            W=utils.sdiag(np.random.rand(nP1))
        )

        phi2 = objective_function.L2ObjectiveFunction(
            W=utils.sdiag(np.random.rand(nP2))
        )

        with self.assertRaises(Exception):
            phi1 + phi2

        with self.assertRaises(Exception):
            phi1 + 100 * phi2

    def test_emptyObjFct(self):
        phi = Empty_ObjFct()
        x = np.random.rand(20)

        with self.assertRaises(NotImplementedError):
            phi(x)
            phi.deriv(x)
            phi.deriv2(x)

    def test_ZeroObjFct(self):
        # This is not a combo objective function, it will just give back an
        # L2 objective function. That might be ok? or should this be a combo
        # objective function?
        nP = 20
        alpha = 2.0
        phi = alpha * (
            objective_function.L2ObjectiveFunction(W=sp.eye(nP))
            + utils.Zero() * objective_function.L2ObjectiveFunction()
        )
        self.assertTrue(len(phi.objfcts) == 1)
        self.assertTrue(phi.test())

    def test_updateMultipliers(self):
        nP = 10

        m = np.random.rand(nP)

        W1 = utils.sdiag(np.random.rand(nP))
        W2 = utils.sdiag(np.random.rand(nP))

        phi1 = objective_function.L2ObjectiveFunction(W=W1)
        phi2 = objective_function.L2ObjectiveFunction(W=W2)

        phi = phi1 + phi2

        self.assertTrue(phi(m) == phi1(m) + phi2(m))

        phi.multipliers[0] = utils.Zero()
        self.assertTrue(phi(m) == phi2(m))

        phi.multipliers[0] = 1.0
        phi.multipliers[1] = utils.Zero()

        self.assertTrue(len(phi.objfcts) == 2)
        self.assertTrue(len(phi.multipliers) == 2)
        self.assertTrue(len(phi) == 2)

        self.assertTrue(phi(m) == phi1(m))

    def test_invalid_mapping(self):
        """Test if setting mapping of wrong type raises errors."""

        class Dummy:
            pass

        phi = objective_function.L2ObjectiveFunction()
        invalid_mapping = Dummy()
        msg = "Invalid mapping of class 'Dummy'."
        with pytest.raises(TypeError, match=msg):
            phi.mapping = invalid_mapping

    def test_early_exits(self):
        nP = 10

        m = np.random.rand(nP)
        v = np.random.rand(nP)

        W1 = utils.sdiag(np.random.rand(nP))
        phi1 = objective_function.L2ObjectiveFunction(W=W1)

        phi2 = Error_if_Hit_ObjFct()

        objfct = phi1 + 0 * phi2

        self.assertTrue(len(objfct) == 2)
        self.assertTrue(np.all(objfct.multipliers == np.r_[1, 0]))
        self.assertTrue(objfct(m) == phi1(m))
        self.assertTrue(np.all(objfct.deriv(m) == phi1.deriv(m)))
        self.assertTrue(np.all(objfct.deriv2(m, v) == phi1.deriv2(m, v)))

        objfct.multipliers[1] = utils.Zero()

        self.assertTrue(len(objfct) == 2)
        self.assertTrue(np.all(objfct.multipliers == np.r_[1, 0]))
        self.assertTrue(objfct(m) == phi1(m))
        self.assertTrue(np.all(objfct.deriv(m) == phi1.deriv(m)))
        self.assertTrue(np.all(objfct.deriv2(m, v) == phi1.deriv2(m, v)))

    def test_Maps(self):
        nP = 10
        m = np.random.rand(2 * nP)

        wires = maps.Wires(("sigma", nP), ("mu", nP))

        objfct1 = objective_function.L2ObjectiveFunction(mapping=wires.sigma)
        objfct2 = objective_function.L2ObjectiveFunction(mapping=wires.mu)

        objfct3 = objfct1 + objfct2

        objfct4 = objective_function.L2ObjectiveFunction(nP=nP)

        self.assertTrue(objfct1.nP == 2 * nP)
        self.assertTrue(objfct2.nP == 2 * nP)
        self.assertTrue(objfct3.nP == 2 * nP)

        # print(objfct1.nP, objfct4.nP, objfct4.W.shape, objfct1.W.shape, m[:nP].shape)
        self.assertTrue(objfct1(m) == objfct4(m[:nP]))
        self.assertTrue(objfct2(m) == objfct4(m[nP:]))

        self.assertTrue(objfct3(m) == objfct1(m) + objfct2(m))

        objfct1.test()
        objfct2.test()
        objfct3.test()

    def test_ComboW(self):
        nP = 15
        m = np.random.rand(nP)

        phi1 = objective_function.L2ObjectiveFunction(nP=nP)
        phi2 = objective_function.L2ObjectiveFunction(nP=nP)

        alpha1 = 2.0
        alpha2 = 0.5

        phi = alpha1 * phi1 + alpha2 * phi2

        r = phi.W * m

        r1 = phi1.W * m
        r2 = phi2.W * m

        print(phi(m), np.inner(r, r))

        self.assertTrue(np.allclose(phi(m), np.inner(r, r)))
        self.assertTrue(
            np.allclose(phi(m), (alpha1 * np.inner(r1, r1) + alpha2 * np.inner(r2, r2)))
        )

    def test_ComboConstruction(self):
        nP = 10
        m = np.random.rand(nP)
        v = np.random.rand(nP)

        phi1 = objective_function.L2ObjectiveFunction(nP=nP)
        phi2 = objective_function.L2ObjectiveFunction(nP=nP)

        phi3 = 2 * phi1 + 3 * phi2

        phi4 = objective_function.ComboObjectiveFunction([phi1, phi2], [2, 3])

        self.assertTrue(phi3(m) == phi4(m))
        self.assertTrue(np.all(phi3.deriv(m) == phi4.deriv(m)))
        self.assertTrue(np.all(phi3.deriv2(m, v) == phi4.deriv2(m, v)))

    def test_updating_multipliers(self):
        nP = 20

        phi1 = objective_function.L2ObjectiveFunction(nP=nP)
        phi2 = objective_function.L2ObjectiveFunction(nP=nP)

        phi3 = 2 * phi1 + 4 * phi2

        self.assertTrue(all(phi3.multipliers == np.r_[2, 4]))

        phi3.multipliers[1] = 3
        self.assertTrue(all(phi3.multipliers == np.r_[2, 3]))

        phi3.multipliers = np.r_[1.0, 5.0]
        self.assertTrue(all(phi3.multipliers == np.r_[1.0, 5.0]))

        with self.assertRaises(Exception):
            phi3.multipliers[0] = "a"

        with self.assertRaises(Exception):
            phi3.multipliers = np.r_[0.0, 3.0, 4.0]

        with self.assertRaises(Exception):
            phi3.multipliers = ["a", "b"]

    def test_inconsistent_nparams_and_weights(self):
        """
        Test if L2ObjectiveFunction raises error after nP != columns in W
        """
        n_params = 9
        weights = np.zeros((5, n_params + 1))
        with pytest.raises(ValueError, match="Number of parameters nP"):
            objective_function.L2ObjectiveFunction(nP=n_params, W=weights)


class TestOperationsComboObjectiveFunctions:
    """Test arithmetic operations involving ComboObjectiveFunction"""

    @pytest.mark.parametrize("unpack_on_add", (True, False))
    def test_mul(self, unpack_on_add):
        """Test if ComboObjectiveFunction multiplication works as expected"""
        n_params = 10
        phi1 = objective_function.L2ObjectiveFunction(nP=n_params)
        phi2 = objective_function.L2ObjectiveFunction(nP=n_params)
        combo = objective_function.ComboObjectiveFunction(
            [phi1, phi2], [2, 3], unpack_on_add=unpack_on_add
        )
        combo_mul = 3.5 * combo
        assert len(combo_mul) == 1
        assert combo_mul.multipliers == [3.5]
        assert combo_mul.objfcts == [combo]

    @pytest.mark.parametrize("unpack_on_add", (True, False))
    def test_add(self, unpack_on_add):
        """Test if ComboObjectiveFunction addition works as expected"""
        n_params = 10
        phi1 = objective_function.L2ObjectiveFunction(nP=n_params)
        phi2 = objective_function.L2ObjectiveFunction(nP=n_params)
        phi3 = objective_function.L2ObjectiveFunction(nP=n_params)
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

    def test_add_multiple_terms(self):
        """Test addition of multiple BaseObjectiveFunctions"""
        n_params = 10
        phi1 = objective_function.L2ObjectiveFunction(nP=n_params)
        phi2 = objective_function.L2ObjectiveFunction(nP=n_params)
        phi3 = objective_function.L2ObjectiveFunction(nP=n_params)
        combo = 1.1 * phi1 + 1.2 * phi2 + 1.3 * phi3
        assert len(combo) == 3
        assert combo.multipliers == [1.1, 1.2, 1.3]
        assert combo.objfcts == [phi1, phi2, phi3]

    @pytest.mark.parametrize("unpack_on_add", (True, False))
    def test_add_and_mul(self, unpack_on_add):
        """
        Test ComboObjectiveFunction addition with multiplication

        After multiplying a Combo with a scalar, the `__mul__` method creates
        another Combo for it.
        """
        n_params = 10
        phi1 = objective_function.L2ObjectiveFunction(nP=n_params)
        phi2 = objective_function.L2ObjectiveFunction(nP=n_params)
        phi3 = objective_function.L2ObjectiveFunction(nP=n_params)
        combo_1 = objective_function.ComboObjectiveFunction(
            [phi1, phi2], [2, 3], unpack_on_add=unpack_on_add
        )
        combo_2 = 5 * phi3 + 1.2 * combo_1
        assert len(combo_2) == 2
        assert combo_2.multipliers == [5, 1.2]
        assert combo_2.objfcts == [phi3, combo_1]


@pytest.mark.parametrize(
    "objfcts, multipliers",
    (
        (None, None),
        ([objective_function.L2ObjectiveFunction()], None),
        ([objective_function.L2ObjectiveFunction()], [2.5]),
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


def test_invalid_objfcts_in_combo():
    """Test invalid objective function class in ComboObjectiveFunction."""

    class Dummy:
        pass

    phi = objective_function.L2ObjectiveFunction()
    invalid_phi = Dummy()
    msg = "Unrecognized objective function type Dummy in 'objfcts'."
    with pytest.raises(TypeError, match=msg):
        objective_function.ComboObjectiveFunction(objfcts=[phi, invalid_phi])


class TestMultiplierValidation:
    """
    Test the _validate_multiplier private function.
    """

    @pytest.mark.parametrize(
        "multiplier",
        (
            3.14,
            1,
            np.float64(-15.3),
            np.float32(-10.2),
            np.int64(10),
            np.int32(33),
            Zero(),
        ),
    )
    def test_valid_multipliers(self, multiplier):
        """
        Test function against valid multipliers
        """
        _validate_multiplier(multiplier)

    @pytest.mark.parametrize(
        "multiplier",
        (np.array([1, 3.14]), np.array(3), [1, 2, 3], "string", True, None),
    )
    def test_invalid_multipliers(self, multiplier):
        """
        Test function against invalid multipliers
        """
        with pytest.raises(TypeError, match="Invalid multiplier"):
            _validate_multiplier(multiplier)


if __name__ == "__main__":
    unittest.main()
