from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest

from SimPEG.ObjectiveFunction import ObjectiveFunction
from SimPEG import Utils


class L2_ObjFct(ObjectiveFunction):

    def __init__(self):
        super(L2_ObjFct, self).__init__()

    def _eval(self, m):
        return 0.5 * m.dot(m)

    def deriv(self, m):
        return m

    def deriv2(self, m):
        return Utils.Identity()


class Empty_ObjFct(ObjectiveFunction):

    def __init__(self):
        super(Empty_ObjFct, self).__init__()


class TestBaseObjFct(unittest.TestCase):

    def test_derivs(self):
        objfct = L2_ObjFct()
        self.assertTrue(objfct.test())

    def test_scalarmul(self):
        scalar = 10.
        objfct_a = L2_ObjFct()
        objfct_b = scalar * objfct_a
        m = np.random.rand(100)

        self.assertTrue(scalar * objfct_a(m) == objfct_b(m))
        self.assertTrue(objfct_b.test())

    def test_sum(self):
        scalar = 10.
        objfct = L2_ObjFct() + scalar * L2_ObjFct()
        self.assertTrue(objfct.test())

    def test_3sum(self):
        phi1 = L2_ObjFct() + 100 * L2_ObjFct()
        phi2 = L2_ObjFct() + 200 * phi1
        self.assertTrue(phi2.test())

    def test_emptyObjFct(self):
        phi = Empty_ObjFct()
        x = np.random.rand(20)

        with self.assertRaises(NotImplementedError):
            phi(x)
            phi.deriv(x)
            phi.deriv2(x)


if __name__ == '__main__':
    unittest.main()

