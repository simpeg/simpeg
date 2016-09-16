from __future__ import print_function
import numpy as np
import unittest
from SimPEG.Utils import mkvc
from SimPEG import Mesh, Tests

MESHTYPES = ['uniformTensorMesh', 'randomTensorMesh']
TOLERANCES = [0.9, 0.5, 0.5]
call1 = lambda fun, xyz: fun(xyz)
call2 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, -1])
call3 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])
cart_row2 = lambda g, xfun, yfun: np.c_[call2(xfun, g), call2(yfun, g)]
cart_row3 = lambda g, xfun, yfun, zfun: np.c_[call3(xfun, g), call3(yfun, g), call3(zfun, g)]
cartF2 = lambda M, fx, fy: np.vstack((cart_row2(M.gridFx, fx, fy), cart_row2(M.gridFy, fx, fy)))
cartF2Cyl = lambda M, fx, fy: np.vstack((cart_row2(M.gridFx, fx, fy), cart_row2(M.gridFz, fx, fy)))
cartE2 = lambda M, ex, ey: np.vstack((cart_row2(M.gridEx, ex, ey), cart_row2(M.gridEy, ex, ey)))
cartE2Cyl = lambda M, ex, ey: cart_row2(M.gridEy, ex, ey)
cartF3 = lambda M, fx, fy, fz: np.vstack((cart_row3(M.gridFx, fx, fy, fz), cart_row3(M.gridFy, fx, fy, fz), cart_row3(M.gridFz, fx, fy, fz)))
cartE3 = lambda M, ex, ey, ez: np.vstack((cart_row3(M.gridEx, ex, ey, ez), cart_row3(M.gridEy, ex, ey, ez), cart_row3(M.gridEz, ex, ey, ez)))

TOL = 1e-7


class TestInterpolation1D(Tests.OrderTest):
    LOCS = np.random.rand(50)*0.6+0.2
    name = "Interpolation 1D"
    meshTypes = MESHTYPES
    tolerance = TOLERANCES
    meshDimension = 1
    meshSizes = [8, 16, 32, 64, 128]

    def getError(self):
        funX = lambda x: np.cos(2*np.pi*x)

        ana = call1(funX, self.LOCS)

        if 'CC' == self.type:
            grid = call1(funX, self.M.gridCC)
        elif 'N' == self.type:
            grid = call1(funX, self.M.gridN)

        comp = self.M.getInterpolationMat(self.LOCS, self.type)*grid

        err = np.linalg.norm((comp - ana), 2)
        return err

    def test_orderCC(self):
        self.type = 'CC'
        self.name = 'Interpolation 1D: CC'
        self.orderTest()

    def test_orderN(self):
        self.type = 'N'
        self.name = 'Interpolation 1D: N'
        self.orderTest()

class TestOutliersInterp1D(unittest.TestCase):

    def setUp(self):
        pass

    def test_outliers(self):
        M = Mesh.TensorMesh([4])
        Q = M.getInterpolationMat(np.array([[0],[0.126],[0.127]]),'CC',zerosOutside=True)
        x = np.arange(4)+1
        self.assertTrue(np.linalg.norm(Q*x - np.r_[1,1.004,1.008]) < TOL)
        Q = M.getInterpolationMat(np.array([[-1],[0.126],[0.127]]),'CC',zerosOutside=True)
        self.assertTrue(np.linalg.norm(Q*x - np.r_[0,1.004,1.008]) < TOL)

class TestInterpolation2d(Tests.OrderTest):
    name = "Interpolation 2D"
    LOCS = np.random.rand(50,2)*0.6+0.2
    meshTypes = MESHTYPES
    tolerance = TOLERANCES
    meshDimension = 2
    meshSizes = [8, 16, 32, 64]

    def getError(self):
        funX = lambda x, y: np.cos(2*np.pi*y)
        funY = lambda x, y: np.cos(2*np.pi*x)

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

        comp = self.M.getInterpolationMat(self.LOCS, self.type)*grid

        err = np.linalg.norm((comp - ana), np.inf)
        return err

    def test_orderCC(self):
        self.type = 'CC'
        self.name = 'Interpolation 2D: CC'
        self.orderTest()

    def test_orderN(self):
        self.type = 'N'
        self.name = 'Interpolation 2D: N'
        self.orderTest()

    def test_orderFx(self):
        self.type = 'Fx'
        self.name = 'Interpolation 2D: Fx'
        self.orderTest()

    def test_orderFy(self):
        self.type = 'Fy'
        self.name = 'Interpolation 2D: Fy'
        self.orderTest()

    def test_orderEx(self):
        self.type = 'Ex'
        self.name = 'Interpolation 2D: Ex'
        self.orderTest()

    def test_orderEy(self):
        self.type = 'Ey'
        self.name = 'Interpolation 2D: Ey'
        self.orderTest()


class TestInterpolation2dCyl_Simple(unittest.TestCase):
    def test_simpleInter(self):
        M = Mesh.CylMesh([4,1,1])
        locs = np.r_[0,0,0.5]
        fx = np.array([[ 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        self.assertTrue( np.all(fx == M.getInterpolationMat(locs, 'Fx').todense()) )
        fz = np.array([[ 0., 0., 0., 0., 0.5, 0., 0., 0., 0.5, 0., 0., 0.]])
        self.assertTrue( np.all(fz == M.getInterpolationMat(locs, 'Fz').todense()) )

    def test_exceptions(self):
        M = Mesh.CylMesh([4,1,1])
        locs = np.r_[0,0,0.5]
        self.assertRaises(Exception,lambda:M.getInterpolationMat(locs, 'Fy'))
        self.assertRaises(Exception,lambda:M.getInterpolationMat(locs, 'Ex'))
        self.assertRaises(Exception,lambda:M.getInterpolationMat(locs, 'Ez'))


class TestInterpolation2dCyl(Tests.OrderTest):
    name = "Interpolation 2D"
    LOCS = np.c_[np.random.rand(4)*0.6+0.2, np.zeros(4), np.random.rand(4)*0.6+0.2]
    meshTypes = ['uniformCylMesh'] # MESHTYPES +
    tolerance = 0.6
    meshDimension = 2
    meshSizes = [32, 64, 128, 256]

    def getError(self):
        funX = lambda x, y: np.cos(2*np.pi*y)
        funY = lambda x, y: np.cos(2*np.pi*x)

        if 'x' in self.type:
            ana = call2(funX, self.LOCS)
        elif 'y' in self.type:
            ana = call2(funY, self.LOCS)
        elif 'z' in self.type:
            ana = call2(funY, self.LOCS)
        else:
            ana = call2(funX, self.LOCS)

        if 'Fx' == self.type:
            Fc = cartF2Cyl(self.M, funX, funY)
            Fc = np.c_[Fc[:,0],np.zeros(self.M.nF),Fc[:,1]]
            grid = self.M.projectFaceVector(Fc)
        elif 'Fz' == self.type:
            Fc = cartF2Cyl(self.M, funX, funY)
            Fc = np.c_[Fc[:,0],np.zeros(self.M.nF),Fc[:,1]]

            grid = self.M.projectFaceVector(Fc)
        elif 'E' in self.type:
            Ec = cartE2Cyl(self.M, funX, funY)
            grid = Ec[:,1]
        elif 'CC' == self.type:
            grid = call2(funX, self.M.gridCC)
        elif 'N' == self.type:
            grid = call2(funX, self.M.gridN)

        comp = self.M.getInterpolationMat(self.LOCS, self.type)*grid

        err = np.linalg.norm((comp - ana), np.inf)
        return err

    def test_orderCC(self):
        self.type = 'CC'
        self.name = 'Interpolation 2D CYLMESH: CC'
        self.orderTest()

    def test_orderN(self):
        self.type = 'N'
        self.name = 'Interpolation 2D CYLMESH: N'
        self.orderTest()

    def test_orderFx(self):
        self.type = 'Fx'
        self.name = 'Interpolation 2D CYLMESH: Fx'
        self.orderTest()

    def test_orderFz(self):
        self.type = 'Fz'
        self.name = 'Interpolation 2D CYLMESH: Fz'
        self.orderTest()

    def test_orderEy(self):
        self.type = 'Ey'
        self.name = 'Interpolation 2D CYLMESH: Ey'
        self.orderTest()

class TestInterpolation3D(Tests.OrderTest):
    name = "Interpolation"
    LOCS = np.random.rand(50,3)*0.6+0.2
    meshTypes = MESHTYPES
    tolerance = TOLERANCES
    meshDimension = 3
    meshSizes = [8, 16, 32, 64]

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
        self.orderTest()

    def test_orderN(self):
        self.type = 'N'
        self.name = 'Interpolation 3D: N'
        self.orderTest()

    def test_orderFx(self):
        self.type = 'Fx'
        self.name = 'Interpolation 3D: Fx'
        self.orderTest()

    def test_orderFy(self):
        self.type = 'Fy'
        self.name = 'Interpolation 3D: Fy'
        self.orderTest()

    def test_orderFz(self):
        self.type = 'Fz'
        self.name = 'Interpolation 3D: Fz'
        self.orderTest()

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
