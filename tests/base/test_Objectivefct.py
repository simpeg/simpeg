from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest

from SimPEG import ObjectiveFunction
from SimPEG import Utils


class Empty_ObjFct(ObjectiveFunction.ObjectiveFunction):

    def __init__(self):
        super(Empty_ObjFct, self).__init__()


class TestBaseObjFct(unittest.TestCase):

    def test_derivs(self):
        objfct = ObjectiveFunction.l2ObjFct()
        self.assertTrue(objfct.test())

    def test_scalarmul(self):
        scalar = 10.
        nP = 100
        objfct_a = ObjectiveFunction.l2ObjFct(
            W=Utils.sdiag(np.random.randn(nP))
        )
        objfct_b = scalar * objfct_a
        m = np.random.rand(nP)

        self.assertTrue(scalar * objfct_a(m) == objfct_b(m))
        self.assertTrue(objfct_b.test())

    def test_sum(self):
        scalar = 10.
        objfct = (
            ObjectiveFunction.l2ObjFct() +
            scalar * ObjectiveFunction.l2ObjFct()
        )
        self.assertTrue(objfct.test())

    def test_3sum(self):
        nP = 80
        phi1 = (
            ObjectiveFunction.l2ObjFct(W=Utils.sdiag(np.random.rand(nP))) +
            100 * ObjectiveFunction.l2ObjFct()
        )
        phi2 = ObjectiveFunction.l2ObjFct() + 200 * phi1
        self.assertTrue(phi2.test())

    def test_sum_fail(self):
        nP1 = 10
        nP2 = 30

        with self.assertRaises(Exception):
            phi = (
                ObjectiveFunction.l2ObjFct(
                    W=Utils.sdiag(np.random.rand(nP1))
                ) +
                ObjectiveFunction.l2ObjFct(
                    W=Utils.sdiag(np.random.rand(nP2))
                )
            )

        with self.assertRaises(Exception):
            phi = (
                ObjectiveFunction.l2ObjFct(
                    W=Utils.sdiag(np.random.rand(nP1))
                ) +
                100 * ObjectiveFunction.l2ObjFct(
                    W=Utils.sdiag(np.random.rand(nP2))
                )
            )

    def test_emptyObjFct(self):
        phi = Empty_ObjFct()
        x = np.random.rand(20)

        with self.assertRaises(NotImplementedError):
            phi(x)
            phi.deriv(x)
            phi.deriv2(x)


if __name__ == '__main__':
    unittest.main()

