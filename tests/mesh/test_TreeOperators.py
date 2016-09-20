from __future__ import print_function
import numpy as np
import unittest
from SimPEG import Utils, Tests
import matplotlib.pyplot as plt

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

class TestFaceDiv2D(Tests.OrderTest):
    name = "Face Divergence 2D"
    meshTypes = MESHTYPES
    meshDimension = 2
    meshSizes = [16, 32]

    def getError(self):
        #Test function
        fx = lambda x, y: np.sin(2*np.pi*x)
        fy = lambda x, y: np.sin(2*np.pi*y)
        sol = lambda x, y: 2*np.pi*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))

        Fc = cartF2(self.M, fx, fy)
        F = self.M.projectFaceVector(Fc)

        divF = self.M.faceDiv.dot(F)
        divF_ana = call2(sol, self.M.gridCC)

        err = np.linalg.norm((divF-divF_ana), np.inf)

        # self.M.plotImage(divF-divF_ana, showIt=True)

        return err

    def test_order(self):
        self.orderTest()

class TestFaceDiv3D(Tests.OrderTest):
    name = "Face Divergence 3D"
    meshTypes = MESHTYPES
    meshSizes = [8, 16]

    def getError(self):
        fx = lambda x, y, z: np.sin(2*np.pi*x)
        fy = lambda x, y, z: np.sin(2*np.pi*y)
        fz = lambda x, y, z: np.sin(2*np.pi*z)
        sol = lambda x, y, z: (2*np.pi*np.cos(2*np.pi*x)+2*np.pi*np.cos(2*np.pi*y)+2*np.pi*np.cos(2*np.pi*z))

        Fc = cartF3(self.M, fx, fy, fz)
        F = self.M.projectFaceVector(Fc)

        divF = self.M.faceDiv.dot(F)
        divF_ana = call3(sol, self.M.gridCC)

        return np.linalg.norm((divF-divF_ana), np.inf)


    def test_order(self):
        self.orderTest()


class TestCurl(Tests.OrderTest):
    name = "Curl"
    meshTypes = ['notatreeTree', 'uniformTree'] #, 'randomTree']#, 'uniformTree']
    meshSizes = [8, 16]#, 32]
    expectedOrders = [2,1] # This is due to linear interpolation in the Re projection

    def getError(self):
        # fun: i (cos(y)) + j (cos(z)) + k (cos(x))
        # sol: i (sin(z)) + j (sin(x)) + k (sin(y))

        funX = lambda x, y, z: np.cos(2*np.pi*y)
        funY = lambda x, y, z: np.cos(2*np.pi*z)
        funZ = lambda x, y, z: np.cos(2*np.pi*x)

        solX = lambda x, y, z: 2*np.pi*np.sin(2*np.pi*z)
        solY = lambda x, y, z: 2*np.pi*np.sin(2*np.pi*x)
        solZ = lambda x, y, z: 2*np.pi*np.sin(2*np.pi*y)

        Ec = cartE3(self.M, funX, funY, funZ)
        E = self.M.projectEdgeVector(Ec)

        Fc = cartF3(self.M, solX, solY, solZ)
        curlE_ana = self.M.projectFaceVector(Fc)

        curlE = self.M.edgeCurl.dot(E)

        err = np.linalg.norm((curlE - curlE_ana), np.inf)
        # err = np.linalg.norm((curlE - curlE_ana)*self.M.area, 2)

        return err

    def test_order(self):
        self.orderTest()


class TestNodalGrad(Tests.OrderTest):
    name = "Nodal Gradient"
    meshTypes = ['notatreeTree', 'uniformTree'] #['randomTree', 'uniformTree']
    meshSizes = [8, 16]#, 32]
    expectedOrders = [2,1]

    def getError(self):
        #Test function
        fun = lambda x, y, z: (np.cos(x)+np.cos(y)+np.cos(z))
        # i (sin(x)) + j (sin(y)) + k (sin(z))
        solX = lambda x, y, z: -np.sin(x)
        solY = lambda x, y, z: -np.sin(y)
        solZ = lambda x, y, z: -np.sin(z)

        phi = call3(fun, self.M.gridN)
        gradE = self.M.nodalGrad.dot(phi)

        Ec = cartE3(self.M, solX, solY, solZ)
        gradE_ana = self.M.projectEdgeVector(Ec)

        err = np.linalg.norm((gradE-gradE_ana), np.inf)

        return err

    def test_order(self):
        self.orderTest()


class TestNodalGrad2D(Tests.OrderTest):
    name = "Nodal Gradient 2D"
    meshTypes = ['notatreeTree', 'uniformTree'] #['randomTree', 'uniformTree']
    meshSizes = [8, 16]#, 32]
    expectedOrders = [2,1]
    meshDimension = 2

    def getError(self):
        #Test function
        fun = lambda x, y: (np.cos(x)+np.cos(y))
        # i (sin(x)) + j (sin(y)) + k (sin(z))
        solX = lambda x, y: -np.sin(x)
        solY = lambda x, y: -np.sin(y)

        phi = call2(fun, self.M.gridN)
        gradE = self.M.nodalGrad.dot(phi)

        Ec = cartE2(self.M, solX, solY)
        gradE_ana = self.M.projectEdgeVector(Ec)

        err = np.linalg.norm((gradE-gradE_ana), np.inf)

        return err

    def test_order(self):
        self.orderTest()


class TestTreeInnerProducts(Tests.OrderTest):
    """Integrate an function over a unit cube domain using edgeInnerProducts and faceInnerProducts."""

    meshTypes = ['uniformTree', 'notatreeTree'] #['uniformTensorMesh', 'uniformCurv', 'rotateCurv']
    meshDimension = 3
    meshSizes = [4, 8]

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
            analytic = 647./360  # Found using sympy.
        elif self.sigmaTest == 3:
            sigma = np.r_[call(sigma1, Gc), call(sigma2, Gc), call(sigma3, Gc)]
            analytic = 37./12  # Found using sympy.
        elif self.sigmaTest == 6:
            sigma = np.c_[call(sigma1, Gc), call(sigma2, Gc), call(sigma3, Gc),
                          call(sigma4, Gc), call(sigma5, Gc), call(sigma6, Gc)]
            analytic = 69881./21600  # Found using sympy.

        if self.location == 'edges':
            cart = lambda g: np.c_[call(ex, g), call(ey, g), call(ez, g)]
            Ec = np.vstack((cart(self.M.gridEx),
                            cart(self.M.gridEy),
                            cart(self.M.gridEz)))
            E = self.M.projectEdgeVector(Ec)

            if self.invProp:
                A = self.M.getEdgeInnerProduct(Utils.invPropertyTensor(self.M, sigma), invProp=True)
            else:
                A = self.M.getEdgeInnerProduct(sigma)
            numeric = E.T.dot(A.dot(E))
        elif self.location == 'faces':
            cart = lambda g: np.c_[call(ex, g), call(ey, g), call(ez, g)]
            Fc = np.vstack((cart(self.M.gridFx),
                            cart(self.M.gridFy),
                            cart(self.M.gridFz)))
            F = self.M.projectFaceVector(Fc)

            if self.invProp:
                A = self.M.getFaceInnerProduct(Utils.invPropertyTensor(self.M, sigma), invProp=True)
            else:
                A = self.M.getFaceInnerProduct(sigma)
            numeric = F.T.dot(A.dot(F))

        err = np.abs(numeric - analytic)
        return err

    def test_order1_edges(self):
        self.name = "Edge Inner Product - Isotropic"
        self.location = 'edges'
        self.sigmaTest = 1
        self.invProp = False
        self.orderTest()

    def test_order1_edges_invProp(self):
        self.name = "Edge Inner Product - Isotropic - invProp"
        self.location = 'edges'
        self.sigmaTest = 1
        self.invProp = True
        self.orderTest()

    def test_order3_edges(self):
        self.name = "Edge Inner Product - Anisotropic"
        self.location = 'edges'
        self.sigmaTest = 3
        self.invProp = False
        self.orderTest()

    def test_order3_edges_invProp(self):
        self.name = "Edge Inner Product - Anisotropic - invProp"
        self.location = 'edges'
        self.sigmaTest = 3
        self.invProp = True
        self.orderTest()

    def test_order6_edges(self):
        self.name = "Edge Inner Product - Full Tensor"
        self.location = 'edges'
        self.sigmaTest = 6
        self.invProp = False
        self.orderTest()

    def test_order6_edges_invProp(self):
        self.name = "Edge Inner Product - Full Tensor - invProp"
        self.location = 'edges'
        self.sigmaTest = 6
        self.invProp = True
        self.orderTest()

    def test_order1_faces(self):
        self.name = "Face Inner Product - Isotropic"
        self.location = 'faces'
        self.sigmaTest = 1
        self.invProp = False
        self.orderTest()

    def test_order1_faces_invProp(self):
        self.name = "Face Inner Product - Isotropic - invProp"
        self.location = 'faces'
        self.sigmaTest = 1
        self.invProp = True
        self.orderTest()

    def test_order3_faces(self):
        self.name = "Face Inner Product - Anisotropic"
        self.location = 'faces'
        self.sigmaTest = 3
        self.invProp = False
        self.orderTest()

    def test_order3_faces_invProp(self):
        self.name = "Face Inner Product - Anisotropic - invProp"
        self.location = 'faces'
        self.sigmaTest = 3
        self.invProp = True
        self.orderTest()

    def test_order6_faces(self):
        self.name = "Face Inner Product - Full Tensor"
        self.location = 'faces'
        self.sigmaTest = 6
        self.invProp = False
        self.orderTest()

    def test_order6_faces_invProp(self):
        self.name = "Face Inner Product - Full Tensor - invProp"
        self.location = 'faces'
        self.sigmaTest = 6
        self.invProp = True
        self.orderTest()


class TestTreeInnerProducts2D(Tests.OrderTest):
    """Integrate an function over a unit cube domain using edgeInnerProducts and faceInnerProducts."""

    meshTypes = ['uniformTree']
    meshDimension = 2
    meshSizes = [4, 8]

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
            analytic = 144877./360  # Found using sympy. z=5
        elif self.sigmaTest == 2:
            sigma = np.c_[call(sigma1, Gc), call(sigma2, Gc)]
            analytic = 189959./120  # Found using sympy. z=5
        elif self.sigmaTest == 3:
            sigma = np.r_[call(sigma1, Gc), call(sigma2, Gc), call(sigma3, Gc)]
            analytic = 781427./360  # Found using sympy. z=5

        if self.location == 'edges':
            cart = lambda g: np.c_[call(ex, g), call(ey, g)]
            Ec = np.vstack((cart(self.M.gridEx),
                            cart(self.M.gridEy)))
            E = self.M.projectEdgeVector(Ec)
            if self.invProp:
                A = self.M.getEdgeInnerProduct(Utils.invPropertyTensor(self.M, sigma), invProp=True)
            else:
                A = self.M.getEdgeInnerProduct(sigma)
            numeric = E.T.dot(A.dot(E))
        elif self.location == 'faces':
            cart = lambda g: np.c_[call(ex, g), call(ey, g)]
            Fc = np.vstack((cart(self.M.gridFx),
                            cart(self.M.gridFy)))
            F = self.M.projectFaceVector(Fc)

            if self.invProp:
                A = self.M.getFaceInnerProduct(Utils.invPropertyTensor(self.M, sigma), invProp=True)
            else:
                A = self.M.getFaceInnerProduct(sigma)
            numeric = F.T.dot(A.dot(F))

        err = np.abs(numeric - analytic)
        return err

    # def test_order1_edges(self):
    #     self.name = "2D Edge Inner Product - Isotropic"
    #     self.location = 'edges'
    #     self.sigmaTest = 1
    #     self.invProp = False
    #     self.orderTest()

    # def test_order1_edges_invProp(self):
    #     self.name = "2D Edge Inner Product - Isotropic - invProp"
    #     self.location = 'edges'
    #     self.sigmaTest = 1
    #     self.invProp = True
    #     self.orderTest()

    # def test_order3_edges(self):
    #     self.name = "2D Edge Inner Product - Anisotropic"
    #     self.location = 'edges'
    #     self.sigmaTest = 2
    #     self.invProp = False
    #     self.orderTest()

    # def test_order3_edges_invProp(self):
    #     self.name = "2D Edge Inner Product - Anisotropic - invProp"
    #     self.location = 'edges'
    #     self.sigmaTest = 2
    #     self.invProp = True
    #     self.orderTest()

    # def test_order6_edges(self):
    #     self.name = "2D Edge Inner Product - Full Tensor"
    #     self.location = 'edges'
    #     self.sigmaTest = 3
    #     self.invProp = False
    #     self.orderTest()

    # def test_order6_edges_invProp(self):
    #     self.name = "2D Edge Inner Product - Full Tensor - invProp"
    #     self.location = 'edges'
    #     self.sigmaTest = 3
    #     self.invProp = True
    #     self.orderTest()

    def test_order1_faces(self):
        self.name = "2D Face Inner Product - Isotropic"
        self.location = 'faces'
        self.sigmaTest = 1
        self.invProp = False
        self.orderTest()

    def test_order1_faces_invProp(self):
        self.name = "2D Face Inner Product - Isotropic - invProp"
        self.location = 'faces'
        self.sigmaTest = 1
        self.invProp = True
        self.orderTest()

    def test_order2_faces(self):
        self.name = "2D Face Inner Product - Anisotropic"
        self.location = 'faces'
        self.sigmaTest = 2
        self.invProp = False
        self.orderTest()

    def test_order2_faces_invProp(self):
        self.name = "2D Face Inner Product - Anisotropic - invProp"
        self.location = 'faces'
        self.sigmaTest = 2
        self.invProp = True
        self.orderTest()

    def test_order3_faces(self):
        self.name = "2D Face Inner Product - Full Tensor"
        self.location = 'faces'
        self.sigmaTest = 3
        self.invProp = False
        self.orderTest()

    def test_order3_faces_invProp(self):
        self.name = "2D Face Inner Product - Full Tensor - invProp"
        self.location = 'faces'
        self.sigmaTest = 3
        self.invProp = True
        self.orderTest()


class TestTreeAveraging2D(Tests.OrderTest):
    """Integrate an function over a unit cube domain using edgeInnerProducts and faceInnerProducts."""

    meshTypes = ['notatreeTree', 'uniformTree']#, 'randomTree']
    meshDimension = 2
    meshSizes = [4,8,16]
    expectedOrders = [2,1]

    def getError(self):
        if plotIt:
            plt.spy(self.getAve(self.M))
            plt.show()

        num = self.getAve(self.M) * self.getHere(self.M)
        err = np.linalg.norm((self.getThere(self.M)-num), np.inf)

        if plotIt:
            self.M.plotImage(self.getThere(self.M)-num)
            plt.show()
            plt.tight_layout

        return err

    def test_orderN2CC(self):
        self.name = "Averaging 2D: N2CC"
        fun = lambda x, y: (np.cos(x)+np.sin(y))
        self.getHere = lambda M: call2(fun, M.gridN)
        self.getThere = lambda M: call2(fun, M.gridCC)
        self.getAve = lambda M: M.aveN2CC
        self.orderTest()

    # def test_orderN2F(self):
    #     self.name = "Averaging 2D: N2F"
    #     fun = lambda x, y: (np.cos(x)+np.sin(y))
    #     self.getHere = lambda M: call2(fun, M.gridN)
    #     self.getThere = lambda M: np.r_[call2(fun, M.gridFx), call2(fun, M.gridFy)]
    #     self.getAve = lambda M: M.aveN2F
    #     self.orderTest()

    # def test_orderN2E(self):
    #     self.name = "Averaging 2D: N2E"
    #     fun = lambda x, y: (np.cos(x)+np.sin(y))
    #     self.getHere = lambda M: call2(fun, M.gridN)
    #     self.getThere = lambda M: np.r_[call2(fun, M.gridEx), call2(fun, M.gridEy)]
    #     self.getAve = lambda M: M.aveN2E
    #     self.orderTest()

    def test_orderF2CC(self):
        self.name = "Averaging 2D: F2CC"
        fun = lambda x, y: (np.cos(x)+np.sin(y))
        self.getHere = lambda M: np.r_[call2(fun, np.r_[M.gridFx, M.gridFy])]
        self.getThere = lambda M: call2(fun, M.gridCC)
        self.getAve = lambda M: M.aveF2CC
        self.orderTest()

    def test_orderFx2CC(self):
        self.name = "Averaging 2D: Fx2CC"
        funX = lambda x, y: (np.cos(x)+np.sin(y))
        self.getHere = lambda M: np.r_[call2(funX, M.gridFx)]
        self.getThere = lambda M: np.r_[call2(funX, M.gridCC)]
        self.getAve = lambda M: M.aveFx2CC
        self.orderTest()

    def test_orderFy2CC(self):
        self.name = "Averaging 2D: Fy2CC"
        funY = lambda x, y: (np.cos(y)*np.sin(x))
        self.getHere = lambda M: np.r_[call2(funY, M.gridFy)]
        self.getThere = lambda M: np.r_[call2(funY, M.gridCC)]
        self.getAve = lambda M: M.aveFy2CC
        self.orderTest()

    def test_orderF2CCV(self):
        self.name = "Averaging 2D: F2CCV"
        funX = lambda x, y: (np.cos(x)+np.sin(y))
        funY = lambda x, y: (np.cos(y)*np.sin(x))
        self.getHere = lambda M: np.r_[call2(funX, M.gridFx), call2(funY, M.gridFy)]
        self.getThere = lambda M: np.r_[call2(funX, M.gridCC), call2(funY, M.gridCC)]
        self.getAve = lambda M: M.aveF2CCV
        self.orderTest()

    # def test_orderCC2F(self):
    #     self.name = "Averaging 2D: CC2F"
    #     fun = lambda x, y: (np.cos(x)+np.sin(y))
    #     self.getHere = lambda M: call2(fun, M.gridCC)
    #     self.getThere = lambda M: np.r_[call2(fun, M.gridFx), call2(fun, M.gridFy)]
    #     self.getAve = lambda M: M.aveCC2F
    #     self.expectedOrders = 1
    #     self.orderTest()
    #     self.expectedOrders = 2

class TestAveraging3D(Tests.OrderTest):
    name = "Averaging 3D"
    meshTypes = ['notatreeTree', 'uniformTree']#, 'randomTree']
    meshDimension = 3
    meshSizes = [8,16]
    expectedOrders = [2,1]

    def getError(self):
        if plotIt:
            plt.spy(self.getAve(self.M))
            plt.show()

        num = self.getAve(self.M) * self.getHere(self.M)
        err = np.linalg.norm((self.getThere(self.M)-num), np.inf)
        return err

    def test_orderN2CC(self):
        self.name = "Averaging 3D: N2CC"
        fun = lambda x, y, z: (np.cos(x)+np.sin(y)+np.exp(z))
        self.getHere = lambda M: call3(fun, M.gridN)
        self.getThere = lambda M: call3(fun, M.gridCC)
        self.getAve = lambda M: M.aveN2CC
        self.orderTest()

#     def test_orderN2F(self):
#         self.name = "Averaging 3D: N2F"
#         fun = lambda x, y, z: (np.cos(x)+np.sin(y)+np.exp(z))
#         self.getHere = lambda M: call3(fun, M.gridN)
#         self.getThere = lambda M: np.r_[call3(fun, M.gridFx), call3(fun, M.gridFy), call3(fun, M.gridFz)]
#         self.getAve = lambda M: M.aveN2F
#         self.orderTest()

#     def test_orderN2E(self):
#         self.name = "Averaging 3D: N2E"
#         fun = lambda x, y, z: (np.cos(x)+np.sin(y)+np.exp(z))
#         self.getHere = lambda M: call3(fun, M.gridN)
#         self.getThere = lambda M: np.r_[call3(fun, M.gridEx), call3(fun, M.gridEy), call3(fun, M.gridEz)]
#         self.getAve = lambda M: M.aveN2E
#         self.orderTest()

    def test_orderF2CC(self):
        self.name = "Averaging 3D: F2CC"
        fun = lambda x, y, z: (np.cos(x)+np.sin(y)+np.exp(z))
        self.getHere = lambda M: np.r_[call3(fun, M.gridFx), call3(fun, M.gridFy), call3(fun, M.gridFz)]
        self.getThere = lambda M: call3(fun, M.gridCC)
        self.getAve = lambda M: M.aveF2CC
        self.orderTest()

    def test_orderFx2CC(self):
        self.name = "Averaging 3D: Fx2CC"
        funX = lambda x, y, z: (np.cos(x)+np.sin(y)+np.exp(z))
        self.getHere = lambda M: np.r_[call3(funX, M.gridFx)]
        self.getThere = lambda M: np.r_[call3(funX, M.gridCC)]
        self.getAve = lambda M: M.aveFx2CC
        self.orderTest()

    def test_orderFy2CC(self):
        self.name = "Averaging 3D: Fy2CC"
        funY = lambda x, y, z: (np.cos(x)+np.sin(y)*np.exp(z))
        self.getHere = lambda M: np.r_[call3(funY, M.gridFy)]
        self.getThere = lambda M: np.r_[call3(funY, M.gridCC)]
        self.getAve = lambda M: M.aveFy2CC
        self.orderTest()

    def test_orderFz2CC(self):
        self.name = "Averaging 3D: Fz2CC"
        funZ = lambda x, y, z: (np.cos(x)+np.sin(y)*np.exp(z))
        self.getHere = lambda M: np.r_[call3(funZ, M.gridFz)]
        self.getThere = lambda M: np.r_[call3(funZ, M.gridCC)]
        self.getAve = lambda M: M.aveFz2CC
        self.orderTest()

    def test_orderF2CCV(self):
        self.name = "Averaging 3D: F2CCV"
        funX = lambda x, y, z: (np.cos(x)+np.sin(y)+np.exp(z))
        funY = lambda x, y, z: (np.cos(x)+np.sin(y)*np.exp(z))
        funZ = lambda x, y, z: (np.cos(x)*np.sin(y)+np.exp(z))
        self.getHere = lambda M: np.r_[call3(funX, M.gridFx), call3(funY, M.gridFy), call3(funZ, M.gridFz)]
        self.getThere = lambda M: np.r_[call3(funX, M.gridCC), call3(funY, M.gridCC), call3(funZ, M.gridCC)]
        self.getAve = lambda M: M.aveF2CCV
        self.orderTest()

    def test_orderEx2CC(self):
        self.name = "Averaging 3D: Ex2CC"
        funX = lambda x, y, z: (np.cos(x)+np.sin(y)+np.exp(z))
        self.getHere = lambda M: np.r_[call3(funX, M.gridEx)]
        self.getThere = lambda M: np.r_[call3(funX, M.gridCC)]
        self.getAve = lambda M: M.aveEx2CC
        self.orderTest()

    def test_orderEy2CC(self):
        self.name = "Averaging 3D: Ey2CC"
        funY = lambda x, y, z: (np.cos(x)+np.sin(y)+np.exp(z))
        self.getHere = lambda M: np.r_[call3(funY, M.gridEy)]
        self.getThere = lambda M: np.r_[call3(funY, M.gridCC)]
        self.getAve = lambda M: M.aveEy2CC
        self.orderTest()

    def test_orderEz2CC(self):
        self.name = "Averaging 3D: Ez2CC"
        funZ = lambda x, y, z: (np.cos(x)+np.sin(y)+np.exp(z))
        self.getHere = lambda M: np.r_[call3(funZ, M.gridEz)]
        self.getThere = lambda M: np.r_[call3(funZ, M.gridCC)]
        self.getAve = lambda M: M.aveEz2CC
        self.orderTest()

    def test_orderE2CC(self):
        self.name = "Averaging 3D: E2CC"
        fun = lambda x, y, z: (np.cos(x)+np.sin(y)+np.exp(z))
        self.getHere = lambda M: np.r_[call3(fun, M.gridEx), call3(fun, M.gridEy), call3(fun, M.gridEz)]
        self.getThere = lambda M: call3(fun, M.gridCC)
        self.getAve = lambda M: M.aveE2CC
        self.orderTest()

    def test_orderE2CCV(self):
        self.name = "Averaging 3D: E2CCV"
        funX = lambda x, y, z: (np.cos(x)+np.sin(y)+np.exp(z))
        funY = lambda x, y, z: (np.cos(x)+np.sin(y)*np.exp(z))
        funZ = lambda x, y, z: (np.cos(x)*np.sin(y)+np.exp(z))
        self.getHere = lambda M: np.r_[call3(funX, M.gridEx), call3(funY, M.gridEy), call3(funZ, M.gridEz)]
        self.getThere = lambda M: np.r_[call3(funX, M.gridCC), call3(funY, M.gridCC), call3(funZ, M.gridCC)]
        self.getAve = lambda M: M.aveE2CCV
        self.orderTest()

#     def test_orderCC2F(self):
#         self.name = "Averaging 3D: CC2F"
#         fun = lambda x, y, z: (np.cos(x)+np.sin(y)+np.exp(z))
#         self.getHere = lambda M: call3(fun, M.gridCC)
#         self.getThere = lambda M: np.r_[call3(fun, M.gridFx), call3(fun, M.gridFy), call3(fun, M.gridFz)]
#         self.getAve = lambda M: M.aveCC2F
#         self.expectedOrders = 1
#         self.orderTest()
#         self.expectedOrders = 2


if __name__ == '__main__':
    unittest.main()
