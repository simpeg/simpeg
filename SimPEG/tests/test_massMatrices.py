import numpy as np
import unittest
from OrderTest import OrderTest


class TestEdgeInnerProduct(OrderTest):
    """Integrate an edge function over a unit cube domain using edgeInnerProducts."""

    name = "Edge Inner Product"

    def getError(self):

        call = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])

        ex = lambda x, y, z: x**2+y*z
        ey = lambda x, y, z: (z**2)*x+y*z
        ez = lambda x, y, z: y**2+x*z

        sigma1 = lambda x, y, z: x*y+1
        sigma2 = lambda x, y, z: x*z+2
        sigma3 = lambda x, y, z: 3+z*y
        sigma4 = lambda x, y, z: 0.1*x*y*z
        sigma5 = lambda x, y, z: 0.2*x*y
        sigma6 = lambda x, y, z: 0.1*z

        Ex = call(ex, self.M.gridEx)
        Ey = call(ey, self.M.gridEy)
        Ez = call(ez, self.M.gridEz)

        E = np.matrix(np.r_[Ex, Ey, Ez]).T
        Gc = self.M.gridCC
        sigma = np.c_[call(sigma1, Gc), call(sigma2, Gc), call(sigma3, Gc),
                      call(sigma4, Gc), call(sigma5, Gc), call(sigma6, Gc)]

        A = self.M.getEdgeInnerProduct(sigma)
        numeric = E.T*A*E
        analytic = 69881./21600  # Found using matlab symbolic toolbox.
        err = np.abs(numeric - analytic)
        return err

    def test_order(self):
        self.orderTest()


if __name__ == '__main__':
    unittest.main()
