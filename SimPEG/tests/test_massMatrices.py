import numpy as np
import unittest
from OrderTest import OrderTest


# MATLAB code:

# syms x y z

# ex = x.^2+y.*z;
# ey = (z.^2).*x+y.*z;
# ez = y.^2+x.*z;

# e = [ex;ey;ez];

# sigma1 = x.*y+1;
# sigma2 = x.*z+2;
# sigma3 = 3+z.*y;
# sigma4 = 0.1.*x.*y.*z;
# sigma5 = 0.2.*x.*y;
# sigma6 = 0.1.*z;

# S1 = [sigma1,0,0;0,sigma1,0;0,0,sigma1];
# S2 = [sigma1,0,0;0,sigma2,0;0,0,sigma3];
# S3 = [sigma1,sigma4,sigma5;sigma4,sigma2,sigma6;sigma5,sigma6,sigma3];

# i1 = int(int(int(e.'*S1*e,x,0,1),y,0,1),z,0,1);
# i2 = int(int(int(e.'*S2*e,x,0,1),y,0,1),z,0,1);
# i3 = int(int(int(e.'*S3*e,x,0,1),y,0,1),z,0,1);


class TestInnerProducts(OrderTest):
    """Integrate an function over a unit cube domain using edgeInnerProducts and faceInnerProducts."""

    meshTypes = ['uniformTensorMesh', 'uniformLOM', 'rotateLOM']
    meshDimension = 3
    meshSizes = [16, 32]

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

        Gc = self.M.gridCC
        if self.sigmaTest == 1:
            sigma = np.c_[call(sigma1, Gc)]
            analytic = 647./360  # Found using matlab symbolic toolbox.
        elif self.sigmaTest == 3:
            sigma = np.c_[call(sigma1, Gc), call(sigma2, Gc), call(sigma3, Gc)]
            analytic = 37./12  # Found using matlab symbolic toolbox.
        elif self.sigmaTest == 6:
            sigma = np.c_[call(sigma1, Gc), call(sigma2, Gc), call(sigma3, Gc),
                          call(sigma4, Gc), call(sigma5, Gc), call(sigma6, Gc)]
            analytic = 69881./21600  # Found using matlab symbolic toolbox.

        if self.location == 'edges':
            if self.M._meshType == 'TENSOR':
                Ex = call(ex, self.M.gridEx)
                Ey = call(ey, self.M.gridEy)
                Ez = call(ez, self.M.gridEz)
                E = np.matrix(np.r_[Ex, Ey, Ez]).T
            elif self.M._meshType == 'LOM':
                Tx = self.M.r(self.M.tangents, 'E', 'Ex', 'V')
                Ty = self.M.r(self.M.tangents, 'E', 'Ey', 'V')
                Tz = self.M.r(self.M.tangents, 'E', 'Ez', 'V')

                EX_x = call(ex, self.M.gridEx)
                EY_x = call(ey, self.M.gridEx)
                EZ_x = call(ez, self.M.gridEx)
                Ex = np.sum(np.c_[EX_x, EY_x, EZ_x]*np.c_[Tx[0], Tx[1], Tx[2]], 1)

                EX_y = call(ex, self.M.gridEy)
                EY_y = call(ey, self.M.gridEy)
                EZ_y = call(ez, self.M.gridEy)
                Ey = np.sum(np.c_[EX_y, EY_y, EZ_y]*np.c_[Ty[0], Ty[1], Ty[2]], 1)

                EX_z = call(ex, self.M.gridEz)
                EY_z = call(ey, self.M.gridEz)
                EZ_z = call(ez, self.M.gridEz)
                Ez = np.sum(np.c_[EX_z, EY_z, EZ_z]*np.c_[Tz[0], Tz[1], Tz[2]], 1)

                E = np.matrix(np.r_[Ex, Ey, Ez]).T
            A = self.M.getEdgeInnerProduct(sigma)
            numeric = E.T*A*E
        elif self.location == 'faces':
            if self.M._meshType == 'TENSOR':
                Fx = call(ex, self.M.gridFx)
                Fy = call(ey, self.M.gridFy)
                Fz = call(ez, self.M.gridFz)
                F = np.matrix(np.r_[Fx, Fy, Fz]).T
            elif self.M._meshType == 'LOM':
                Nx = self.M.r(self.M.normals, 'F', 'Fx', 'V')
                Ny = self.M.r(self.M.normals, 'F', 'Fy', 'V')
                Nz = self.M.r(self.M.normals, 'F', 'Fz', 'V')

                FX_x = call(ex, self.M.gridFx)
                FY_x = call(ey, self.M.gridFx)
                FZ_x = call(ez, self.M.gridFx)
                Fx = np.sum(np.c_[FX_x, FY_x, FZ_x]*np.c_[Nx[0], Nx[1], Nx[2]], 1)

                FX_y = call(ex, self.M.gridFy)
                FY_y = call(ey, self.M.gridFy)
                FZ_y = call(ez, self.M.gridFy)
                Fy = np.sum(np.c_[FX_y, FY_y, FZ_y]*np.c_[Ny[0], Ny[1], Ny[2]], 1)

                FX_z = call(ex, self.M.gridFz)
                FY_z = call(ey, self.M.gridFz)
                FZ_z = call(ez, self.M.gridFz)
                Fz = np.sum(np.c_[FX_z, FY_z, FZ_z]*np.c_[Nz[0], Nz[1], Nz[2]], 1)

                F = np.matrix(np.r_[Fx, Fy, Fz]).T
            A = self.M.getFaceInnerProduct(sigma)
            numeric = F.T*A*F

        err = np.abs(numeric - analytic)
        return err

    def test_order1_edges(self):
        self.name = "Edge Inner Product - Isotropic"
        self.location = 'edges'
        self.sigmaTest = 1
        self.orderTest()

    def test_order3_edges(self):
        self.name = "Edge Inner Product - Anisotropic"
        self.location = 'edges'
        self.sigmaTest = 3
        self.orderTest()

    def test_order6_edges(self):
        self.name = "Edge Inner Product - Full Tensor"
        self.location = 'edges'
        self.sigmaTest = 6
        self.orderTest()

    def test_order1_faces(self):
        self.name = "Face Inner Product - Isotropic"
        self.location = 'faces'
        self.sigmaTest = 1
        self.orderTest()

    def test_order3_faces(self):
        self.name = "Face Inner Product - Anisotropic"
        self.location = 'faces'
        self.sigmaTest = 3
        self.orderTest()

    def test_order6_faces(self):
        self.name = "Face Inner Product - Full Tensor"
        self.location = 'faces'
        self.sigmaTest = 6
        self.orderTest()


class TestInnerProducts2D(OrderTest):
    """Integrate an function over a unit cube domain using edgeInnerProducts and faceInnerProducts."""

    meshTypes = ['uniformTensorMesh', 'uniformLOM', 'rotateLOM']
    meshDimension = 2
    meshSizes = [4, 8, 16, 32, 64, 128]

    def getError(self):

        z = 5  # Because 5 is just such a great number.

        call = lambda fun, xy: fun(xy[:, 0], xy[:, 1])

        ex = lambda x, y: x**2+y*z
        ey = lambda x, y: (z**2)*x+y*z

        sigma1 = lambda x, y: x*y+1
        sigma2 = lambda x, y: x*z+2
        sigma3 = lambda x, y: 3+z*y

        Gc = self.M.gridCC
        if self.sigmaTest == 1:
            sigma = np.c_[call(sigma1, Gc)]
            analytic = 144877./360  # Found using matlab symbolic toolbox. z=5
        elif self.sigmaTest == 2:
            sigma = np.c_[call(sigma1, Gc), call(sigma2, Gc)]
            analytic = 189959./120  # Found using matlab symbolic toolbox. z=5
        elif self.sigmaTest == 3:
            sigma = np.c_[call(sigma1, Gc), call(sigma2, Gc), call(sigma3, Gc)]
            analytic = 781427./360  # Found using matlab symbolic toolbox. z=5

        if self.location == 'edges':
            if self.M._meshType == 'TENSOR':
                Ex = call(ex, self.M.gridEx)
                Ey = call(ey, self.M.gridEy)
                E = np.matrix(np.r_[Ex, Ey]).T
            elif self.M._meshType == 'LOM':
                Tx = self.M.r(self.M.tangents, 'E', 'Ex', 'V')
                Ty = self.M.r(self.M.tangents, 'E', 'Ey', 'V')

                EX_x = call(ex, self.M.gridEx)
                EY_x = call(ey, self.M.gridEx)
                Ex = np.sum(np.c_[EX_x, EY_x]*np.c_[Tx[0], Tx[1]], 1)

                EX_y = call(ex, self.M.gridEy)
                EY_y = call(ey, self.M.gridEy)
                Ey = np.sum(np.c_[EX_y, EY_y]*np.c_[Ty[0], Ty[1]], 1)

                E = np.matrix(np.r_[Ex, Ey]).T
            A = self.M.getEdgeInnerProduct(sigma)
            numeric = E.T*A*E
        elif self.location == 'faces':
            if self.M._meshType == 'TENSOR':
                Fx = call(ex, self.M.gridFx)
                Fy = call(ey, self.M.gridFy)
                F = np.matrix(np.r_[Fx, Fy]).T
            elif self.M._meshType == 'LOM':
                Nx = self.M.r(self.M.normals, 'F', 'Fx', 'V')
                Ny = self.M.r(self.M.normals, 'F', 'Fy', 'V')

                FX_x = call(ex, self.M.gridFx)
                FY_x = call(ey, self.M.gridFx)
                Fx = np.sum(np.c_[FX_x, FY_x]*np.c_[Nx[0], Nx[1]], 1)

                FX_y = call(ex, self.M.gridFy)
                FY_y = call(ey, self.M.gridFy)
                Fy = np.sum(np.c_[FX_y, FY_y]*np.c_[Ny[0], Ny[1]], 1)

                F = np.matrix(np.r_[Fx, Fy]).T
            A = self.M.getFaceInnerProduct(sigma)
            numeric = F.T*A*F

        err = np.abs(numeric - analytic)
        return err

    def test_order1_edges(self):
        self.name = "2D Edge Inner Product - Isotropic"
        self.location = 'edges'
        self.sigmaTest = 1
        self.orderTest()

    def test_order3_edges(self):
        self.name = "2D Edge Inner Product - Anisotropic"
        self.location = 'edges'
        self.sigmaTest = 2
        self.orderTest()

    def test_order6_edges(self):
        self.name = "2D Edge Inner Product - Full Tensor"
        self.location = 'edges'
        self.sigmaTest = 3
        self.orderTest()

    def test_order1_faces(self):
        self.name = "2D Face Inner Product - Isotropic"
        self.location = 'faces'
        self.sigmaTest = 1
        self.orderTest()

    def test_order2_faces(self):
        self.name = "2D Face Inner Product - Anisotropic"
        self.location = 'faces'
        self.sigmaTest = 2
        self.orderTest()

    def test_order3_faces(self):
        self.name = "2D Face Inner Product - Full Tensor"
        self.location = 'faces'
        self.sigmaTest = 3
        self.orderTest()


if __name__ == '__main__':
    unittest.main()
