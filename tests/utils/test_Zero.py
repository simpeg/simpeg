from __future__ import print_function
import unittest
from SimPEG.Utils import Zero, Identity, sdiag, mkvc
import numpy as np


class Tests(unittest.TestCase):

    def test_zero(self):
        z = Zero()
        assert z == 0
        assert not (z < 0)
        assert z <= 0
        assert not (z > 0)
        assert z >= 0
        assert +z == z
        assert -z == z
        assert z + 1 == 1
        assert z + 3 + z == 3
        assert z - 3 == -3
        assert z - 3 - z == -3
        assert 3*z == 0
        assert z*3 == 0
        assert z/3 == 0

        a = 1
        a += z
        assert a == 1
        a = 1
        a += z
        assert a == 1
        self.assertRaises(ZeroDivisionError, lambda: 3/z)

        assert mkvc(z) == 0
        assert sdiag(z)*a == 0
        assert z.T == 0
        assert z.transpose == 0

    def test_mat_zero(self):
        z = Zero()
        S = sdiag(np.r_[2, 3])
        assert S*z == 0

    def test_numpy_multiply(self):
        z = Zero()
        x = np.r_[1, 2, 3]
        a = x * z
        assert isinstance(a, Zero)

        z = Zero()
        x = np.r_[1, 2, 3]
        a = z * x
        assert isinstance(a, Zero)

    def test_one(self):
        o = Identity()
        assert o == 1
        assert not (o < 1)
        assert o <= 1
        assert not (o > 1)
        assert o >= 1
        o = -o
        assert o == -1
        assert not (o < -1)
        assert o <= -1
        assert not (o > -1)
        assert o >= -1
        assert -1.*(-o)*o == -o
        o = Identity()
        assert +o == o
        assert -o == -o
        assert o*3 == 3
        assert -o*3 == -3
        assert -o*o == -1
        assert -o*o*-o == 1
        assert -o + 3 == 2
        assert 3 + -o == 2

        assert -o - 3 == -4
        assert o - 3 == -2
        assert 3 - -o == 4
        assert 3 - o == 2

        assert o//2 == 0
        assert o/2. == 0.5
        assert -o//2 == -1
        assert -o/2. == -0.5
        assert 2/o == 2
        assert 2/-o == -2

    def test_mat_one(self):

        o = Identity()
        S = sdiag(np.r_[2, 3])

        def check(exp, ans):
            assert np.all((exp).todense() == ans)

        check(S * o, [[2, 0], [0, 3]])
        check(o * S, [[2, 0], [0, 3]])
        check(S * -o, [[-2, 0], [0, -3]])
        check(-o * S, [[-2, 0], [0, -3]])
        check(S/o, [[2, 0], [0, 3]])
        check(S/-o, [[-2, 0], [0, -3]])
        self.assertRaises(NotImplementedError, lambda: o/S)

        check(S + o, [[3, 0], [0, 4]])
        check(o + S, [[3, 0], [0, 4]])
        check(S - o, [[1, 0], [0, 2]])

        check(S + - o, [[1, 0], [0, 2]])
        check(- o + S, [[1, 0], [0, 2]])

    def test_mat_shape(self):
        o = Identity()
        S = sdiag(np.r_[2, 3])[:1, :]
        self.assertRaises(ValueError, lambda: S + o)

        def check(exp, ans):
            assert np.all((exp).todense() == ans)

        check(S * o, [[2, 0]])
        check(S * -o, [[-2, 0]])

    def test_numpy_one(self):
        o = Identity()
        n = np.r_[2., 3]

        assert np.all(n+1 == n+o)
        assert np.all(1+n == o+n)
        assert np.all(n-1 == n-o)
        assert np.all(1-n == o-n)
        assert np.all(n/1 == n/o)
        assert np.all(n/-1 == n/-o)
        assert np.all(1/n == o/n)
        assert np.all(-1/n == -o/n)
        assert np.all(n*1 == n*o)
        assert np.all(n*-1 == n*-o)
        assert np.all(1*n == o*n)
        assert np.all(-1*n == -o*n)

    def test_both(self):
        z = Zero()
        o = Identity()
        assert o*z == 0
        assert o*z + o == 1
        assert o-z == 1

if __name__ == '__main__':
    unittest.main()
