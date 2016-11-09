from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest

from SimPEG import ObjectiveFunction
from SimPEG import Utils


class L2_ObjFct(ObjectiveFunction.ObjectiveFunction):

    def __init__(self):
        super(L2_ObjFct, self).__init__()

    def _eval(self, m):
        return 0.5 * m.dot(m)

    def _deriv(self, m):
        return m

    def _deriv2(self, m):
        return Utils.Identity()


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




if __name__ == '__main__':
    unittest.main()

