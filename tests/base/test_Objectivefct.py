from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest

from SimPEG import ObjectiveFunction
from SimPEG import Utils


class Empty_ObjFct(ObjectiveFunction.BaseObjectiveFunction):

    def __init__(self):
        super(Empty_ObjFct, self).__init__()


class TestBaseObjFct(unittest.TestCase):

    def test_derivs(self):
        objfct = ObjectiveFunction.L2ObjectiveFunction()
        self.assertTrue(objfct.test())

    def test_scalarmul(self):
        scalar = 10.
        nP = 100
        objfct_a = ObjectiveFunction.L2ObjectiveFunction(
            W=Utils.sdiag(np.random.randn(nP))
        )
        objfct_b = scalar * objfct_a
        m = np.random.rand(nP)

        self.assertTrue(scalar * objfct_a(m) == objfct_b(m))
        self.assertTrue(objfct_b.test())

    def test_sum(self):
        scalar = 10.
        objfct = (
            ObjectiveFunction.L2ObjectiveFunction() +
            scalar * ObjectiveFunction.L2ObjectiveFunction()
        )
        self.assertTrue(objfct.test())

    def test_3sum(self):
        nP = 80
        phi1 = (
            ObjectiveFunction.L2ObjectiveFunction(W=Utils.sdiag(np.random.rand(nP))) +
            100 * ObjectiveFunction.L2ObjectiveFunction()
        )
        phi2 = ObjectiveFunction.L2ObjectiveFunction() + 200 * phi1
        self.assertTrue(phi2.test())

    def test_2sum(self):
        nP = 90
        phi1 = 0.3 * ObjectiveFunction.L2ObjectiveFunction()
        phi2 = 0.6 * ObjectiveFunction.L2ObjectiveFunction()
        phi3 = ObjectiveFunction.L2ObjectiveFunction() / 9.

        phi = phi1 + phi2 + phi3

        self.assertTrue(phi.test())

    def test_sum_fail(self):
        nP1 = 10
        nP2 = 30

        phi1 = ObjectiveFunction.L2ObjectiveFunction(
                    W=Utils.sdiag(np.random.rand(nP1))
        )

        phi2 = ObjectiveFunction.L2ObjectiveFunction(
                    W=Utils.sdiag(np.random.rand(nP2))
                )

        with self.assertRaises(Exception):
            phi = phi1 + phi2

        with self.assertRaises(Exception):
            phi = phi1 + 100 * phi2

    def test_emptyObjFct(self):
        phi = Empty_ObjFct()
        x = np.random.rand(20)

        with self.assertRaises(NotImplementedError):
            phi(x)
            phi.deriv(x)
            phi.deriv2(x)

    def test_ZeroObjFct(self):
        phi = (
            ObjectiveFunction.L2ObjectiveFunction() +
            Utils.Zero()*ObjectiveFunction.L2ObjectiveFunction()
        )
        x = np.random.rand(20)

        self.assertTrue(phi.test())


if __name__ == '__main__':
    unittest.main()

