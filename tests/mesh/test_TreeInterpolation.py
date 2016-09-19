from __future__ import print_function
import numpy as np
import unittest
from SimPEG import Utils, Tests

MESHTYPES = ['uniformTree'] #['randomTree', 'uniformTree']
call2 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1])
call3 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])
cart_row2 = lambda g, xfun, yfun: np.c_[call2(xfun, g), call2(yfun, g)]
cart_row3 = lambda g, xfun, yfun, zfun: np.c_[call3(xfun, g), call3(yfun, g), call3(zfun, g)]
cartF2 = lambda M, fx, fy: np.vstack((cart_row2(M.gridFx, fx, fy), cart_row2(M.gridFy, fx, fy)))
cartE2 = lambda M, ex, ey: np.vstack((cart_row2(M.gridEx, ex, ey), cart_row2(M.gridEy, ex, ey)))
cartF3 = lambda M, fx, fy, fz: np.vstack((cart_row3(M.gridFx, fx, fy, fz), cart_row3(M.gridFy, fx, fy, fz), cart_row3(M.gridFz, fx, fy, fz)))
cartE3 = lambda M, ex, ey, ez: np.vstack((cart_row3(M.gridEx, ex, ey, ez), cart_row3(M.gridEy, ex, ey, ez), cart_row3(M.gridEz, ex, ey, ez)))


plotIt = False


MESHTYPES = ['uniformTree','notatreeTree']


"""

    Face interpolation is O(h)
    Edge interpolation is O(h^2)

"""

class TestInterpolation2d(Tests.OrderTest):
    name = "Interpolation 2D"
    np.random.seed(1)
    LOCS = np.random.rand(50,2)*0.6+0.2
    # LOCS = np.c_[np.ones(100)*0.51, np.linspace(0.3,0.7,100)]
    meshTypes = MESHTYPES
    # tolerance = TOLERANCES
    meshDimension = 2
    meshSizes = [8, 16, 32]
    expectedOrders = 1

    def getError(self):
        funX = lambda x, y: np.cos(2.*np.pi*y)*np.cos(2.*np.pi*x) + x
        funY = lambda x, y: np.cos(2.*np.pi*x)*np.cos(2.*np.pi*y) + y

        # self.LOCS = self.M.gridCC

        if 'x' in self.type:
            ana = call2(funX, self.LOCS)
        elif 'y' in self.type:
            ana = call2(funY, self.LOCS)
        else:
            ana = call2(funX, self.LOCS)

        if 'F' in self.type:
            Fc = cartF2(self.M, funX, funY)
            grid = self.M.projectFaceVector(Fc)
        elif 'E' in self.type:
            Ec = cartE2(self.M, funX, funY)
            grid = self.M.projectEdgeVector(Ec)
        elif 'CC' == self.type:
            grid = call2(funX, self.M.gridCC)
        elif 'N' == self.type:
            grid = call2(funX, self.M.gridN)

        P = self.M.getInterpolationMat(self.LOCS, self.type)
        # print(P)
        comp = P*grid

        err = np.linalg.norm((comp - ana), np.inf)
        if plotIt:
            import matplotlib.pyplot as plt
            ax = plt.subplot(211)
            self.M.plotGrid(ax=ax)
            plt.plot(self.LOCS[:,0],self.LOCS[:,1], 'mx')
            # ax = plt.subplot(111)
            # self.M.plotImage(call2(funX, self.M.gridCC),ax=ax)
            ax = plt.subplot(212)
            plt.plot(self.LOCS[:,1],comp, 'bx')
            plt.plot(self.LOCS[:,1],ana, 'ro')
            plt.show()
        return err

    def test_orderFx(self):
        self.type = 'Fx'
        self.name = 'TreeMesh Interpolation 2D: Fx'
        self.orderTest()

    def test_orderFy(self):
        self.type = 'Fy'
        self.name = 'TreeMesh Interpolation 2D: Fy'
        self.orderTest()



class TestInterpolation3D(Tests.OrderTest):
    name = "Interpolation"
    LOCS = np.random.rand(50,3)*0.6+0.2
    meshTypes = MESHTYPES
    # tolerance = TOLERANCES
    meshDimension = 3
    meshSizes = [8, 16]

    def getError(self):
        funX = lambda x, y, z: np.cos(2*np.pi*y)
        funY = lambda x, y, z: np.cos(2*np.pi*z)
        funZ = lambda x, y, z: np.cos(2*np.pi*x)

        if 'x' in self.type:
            ana = call3(funX, self.LOCS)
        elif 'y' in self.type:
            ana = call3(funY, self.LOCS)
        elif 'z' in self.type:
            ana = call3(funZ, self.LOCS)
        else:
            ana = call3(funX, self.LOCS)

        if 'F' in self.type:
            Fc = cartF3(self.M, funX, funY, funZ)
            grid = self.M.projectFaceVector(Fc)
        elif 'E' in self.type:
            Ec = cartE3(self.M, funX, funY, funZ)
            grid = self.M.projectEdgeVector(Ec)
        elif 'CC' == self.type:
            grid = call3(funX, self.M.gridCC)
        elif 'N' == self.type:
            grid = call3(funX, self.M.gridN)

        comp = self.M.getInterpolationMat(self.LOCS, self.type)*grid

        err = np.linalg.norm((comp - ana), np.inf)
        return err

    def test_orderCC(self):
        self.type = 'CC'
        self.name = 'Interpolation 3D: CC'
        self.expectedOrders = 1
        self.orderTest()
        self.expectedOrders = 2

    def test_orderN(self):
        self.type = 'N'
        self.name = 'Interpolation 3D: N'
        self.orderTest()

    def test_orderFx(self):
        self.type = 'Fx'
        self.name = 'Interpolation 3D: Fx'
        self.expectedOrders = 1
        self.orderTest()
        self.expectedOrders = 2

    def test_orderFy(self):
        self.type = 'Fy'
        self.name = 'Interpolation 3D: Fy'
        self.expectedOrders = 1
        self.orderTest()
        self.expectedOrders = 2

    def test_orderFz(self):
        self.type = 'Fz'
        self.name = 'Interpolation 3D: Fz'
        self.expectedOrders = 1
        self.orderTest()
        self.expectedOrders = 2

    def test_orderEx(self):
        self.type = 'Ex'
        self.name = 'Interpolation 3D: Ex'
        self.orderTest()

    def test_orderEy(self):
        self.type = 'Ey'
        self.name = 'Interpolation 3D: Ey'
        self.orderTest()

    def test_orderEz(self):
        self.type = 'Ez'
        self.name = 'Interpolation 3D: Ez'
        self.orderTest()


if __name__ == '__main__':
    unittest.main()
