import unittest
import sys
from SimPEG import *
from TestUtils import OrderTest


class TestCyl2DMesh(unittest.TestCase):

    def setUp(self):
        hx = np.r_[1,1,0.5]
        hz = np.r_[2,1]
        self.mesh = Mesh.CylMesh([hx, 1,hz])

    def test_dim(self):
        self.assertTrue(self.mesh.dim == 3)

    def test_nC(self):
        self.assertTrue(self.mesh.nC == 6)
        self.assertTrue(self.mesh.nCx == 3)
        self.assertTrue(self.mesh.nCy == 1)
        self.assertTrue(self.mesh.nCz == 2)
        self.assertTrue(np.all(self.mesh.vnC == [3, 1, 2]))

    def test_nN(self):
        self.assertTrue(self.mesh.nN == 0)
        self.assertTrue(self.mesh.nNx == 3)
        self.assertTrue(self.mesh.nNy == 0)
        self.assertTrue(self.mesh.nNz == 3)
        self.assertTrue(np.all(self.mesh.vnN == [3, 0, 3]))

    def test_nF(self):
        self.assertTrue(self.mesh.nFx == 6)
        self.assertTrue(np.all(self.mesh.vnFx == [3, 1, 2]))
        self.assertTrue(self.mesh.nFy == 0)
        self.assertTrue(np.all(self.mesh.vnFy == [3, 0, 2]))
        self.assertTrue(self.mesh.nFz == 9)
        self.assertTrue(np.all(self.mesh.vnFz == [3, 1, 3]))
        self.assertTrue(self.mesh.nF == 15)
        self.assertTrue(np.all(self.mesh.vnF == [6, 0, 9]))

    def test_nE(self):
        self.assertTrue(self.mesh.nEx == 0)
        self.assertTrue(np.all(self.mesh.vnEx == [3, 0, 3]))
        self.assertTrue(self.mesh.nEy == 9)
        self.assertTrue(np.all(self.mesh.vnEy == [3, 1, 3]))
        self.assertTrue(self.mesh.nEz == 0)
        self.assertTrue(np.all(self.mesh.vnEz == [3, 0, 2]))
        self.assertTrue(self.mesh.nE == 9)
        self.assertTrue(np.all(self.mesh.vnE == [0, 9, 0]))

    def test_vectorsCC(self):
        v = np.r_[0.5, 1.5, 2.25]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorCCx)) == 0)
        v = np.r_[0]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorCCy)) == 0)
        v = np.r_[1, 2.5]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorCCz)) == 0)

    def test_vectorsN(self):
        v = np.r_[1, 2, 2.5]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorNx)) == 0)
        v = np.r_[0]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorNy)) == 0)
        v = np.r_[0, 2, 3.]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorNz)) == 0)

    def test_edge(self):
        edge = np.r_[1, 2, 2.5, 1, 2, 2.5, 1, 2, 2.5] * 2 * np.pi
        self.assertTrue(np.linalg.norm((edge-self.mesh.edge)) == 0)

    def test_area(self):
        r = np.r_[0, 1, 2, 2.5]
        a = r[1:]*2*np.pi
        areaX = np.r_[2*a,a]
        a = (r[1:]**2 - r[:-1]**2)*np.pi
        areaZ = np.r_[a,a,a]
        area = np.r_[areaX, areaZ]
        self.assertTrue(np.linalg.norm((area-self.mesh.area)) == 0)

    def test_vol(self):
        r = np.r_[0, 1, 2, 2.5]
        a = (r[1:]**2 - r[:-1]**2)*np.pi
        vol = np.r_[2*a,a]
        self.assertTrue(np.linalg.norm((vol-self.mesh.vol)) == 0)

    def test_gridSizes(self):
        self.assertTrue(self.mesh.gridCC.shape == (self.mesh.nC, 3))
        self.assertTrue(self.mesh.gridN.shape == (9, 3))

        self.assertTrue(self.mesh.gridFx.shape == (self.mesh.nFx, 3))
        self.assertTrue(self.mesh.gridFy is None)
        self.assertTrue(self.mesh.gridFz.shape == (self.mesh.nFz, 3))

        self.assertTrue(self.mesh.gridEx is None)
        self.assertTrue(self.mesh.gridEy.shape == (self.mesh.nEy, 3))
        self.assertTrue(self.mesh.gridEz is None)

    def test_gridCC(self):
        x = np.r_[0.5,1.5,2.25,0.5,1.5,2.25]
        y = np.zeros(6)
        z = np.r_[1,1,1,2.5,2.5,2.5]
        G = np.c_[x,y,z]
        self.assertTrue(np.linalg.norm((G-self.mesh.gridCC).ravel()) == 0)

    def test_gridN(self):
        x = np.r_[1,2,2.5,1,2,2.5,1,2,2.5]
        y = np.zeros(9)
        z = np.r_[0,0,0,2,2,2,3,3,3.]
        G = np.c_[x,y,z]
        self.assertTrue(np.linalg.norm((G-self.mesh.gridN).ravel()) == 0)

    def test_gridFx(self):
        x = np.r_[1,2,2.5,1,2,2.5]
        y = np.zeros(6)
        z = np.r_[1,1,1,2.5,2.5,2.5]
        G = np.c_[x,y,z]
        self.assertTrue(np.linalg.norm((G-self.mesh.gridFx).ravel()) == 0)

    def test_gridFz(self):
        x = np.r_[0.5,1.5,2.25,0.5,1.5,2.25,0.5,1.5,2.25]
        y = np.zeros(9)
        z = np.r_[0,0,0,2,2,2,3,3,3.]
        G = np.c_[x,y,z]
        self.assertTrue(np.linalg.norm((G-self.mesh.gridFz).ravel()) == 0)

    def test_gridEy(self):
        x = np.r_[1,2,2.5,1,2,2.5,1,2,2.5]
        y = np.zeros(9)
        z = np.r_[0,0,0,2,2,2,3,3,3.]
        G = np.c_[x,y,z]
        self.assertTrue(np.linalg.norm((G-self.mesh.gridEy).ravel()) == 0)

    def test_lightOperators(self):
        self.assertTrue(self.mesh.nodalGrad is None)



MESHTYPES = ['uniformCylMesh']
call2 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 2])
call3 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])
cyl_row2 = lambda g, xfun, yfun: np.c_[call2(xfun, g), call2(yfun, g)]
cyl_row3 = lambda g, xfun, yfun, zfun: np.c_[call3(xfun, g), call3(yfun, g), call3(zfun, g)]
cylF2 = lambda M, fx, fy: np.vstack((cyl_row2(M.gridFx, fx, fy), cyl_row2(M.gridFz, fx, fy)))


class TestFaceDiv2D(OrderTest):
    name = "FaceDiv"
    meshTypes = MESHTYPES
    meshDimension = 2

    def getError(self):

        funR = lambda r, z: np.sin(2.*np.pi*r)
        funZ = lambda r, z: np.sin(2.*np.pi*z)

        sol = lambda r, t, z: (2*np.pi*r*np.cos(2*np.pi*r) + np.sin(2*np.pi*r))/r + 2*np.pi*np.cos(2*np.pi*z)

        Fc = cylF2(self.M, funR, funZ)
        Fc = np.c_[Fc[:,0],np.zeros(self.M.nF),Fc[:,1]]
        F = self.M.projectFaceVector(Fc)

        divF = self.M.faceDiv.dot(F)
        divF_anal = call3(sol, self.M.gridCC)

        err = np.linalg.norm((divF-divF_anal), np.inf)
        return err

    def test_order(self):
        self.orderTest()

class TestEdgeCurl2D(OrderTest):
    name = "EdgeCurl"
    meshTypes = MESHTYPES
    meshDimension = 2

    def getError(self):
        # To Recreate or change the functions:

        # import sympy
        # r,t,z = sympy.symbols('r,t,z')

        # fR = 0
        # fZ = 0
        # fT = sympy.sin(2.*sympy.pi*z)

        # print 1/r*sympy.diff(fZ,t) - sympy.diff(fT,z)
        # print sympy.diff(fR,z) - sympy.diff(fZ,r)
        # print 1/r*(sympy.diff(r*fT,r) - sympy.diff(fR,t))

        funT = lambda r, t, z: np.sin(2.*np.pi*z)

        solR = lambda r, z: -2.0*np.pi*np.cos(2.0*np.pi*z)
        solZ = lambda r, z: np.sin(2.0*np.pi*z)/r

        E = call3(funT, self.M.gridEy)

        curlE = self.M.edgeCurl.dot(E)

        Fc = cylF2(self.M, solR, solZ)
        Fc = np.c_[Fc[:,0],np.zeros(self.M.nF),Fc[:,1]]
        curlE_anal = self.M.projectFaceVector(Fc)

        err = np.linalg.norm((curlE-curlE_anal), np.inf)
        return err

    def test_order(self):
        self.orderTest()


# class TestInnerProducts2D(OrderTest):
#     """Integrate an function over a unit cube domain using edgeInnerProducts and faceInnerProducts."""

#     meshTypes = MESHTYPES
#     meshDimension = 2
#     meshSizes = [4, 8, 16, 32, 64, 128]

#     def getError(self):

#         funR = lambda r, t, z: np.cos(2.0*np.pi*z)
#         funT = lambda r, t, z: 0*t
#         funZ = lambda r, t, z: np.sin(2.0*np.pi*r)

#         call = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])

#         sigma1 = lambda r, t, z: z+1
#         sigma2 = lambda r, t, z: r*z+50
#         sigma3 = lambda r, t, z: 3+t*r
#         sigma4 = lambda r, t, z: 0.1*r*t*z
#         sigma5 = lambda r, t, z: 0.2*z*r*t
#         sigma6 = lambda r, t, z: 0.1*t

#         Gc = self.M.gridCC
#         if self.sigmaTest == 1:
#             sigma = np.c_[call(sigma1, Gc)]
#             analytic = 144877./360  # Found using sympy. z=5
#         elif self.sigmaTest == 2:
#             sigma = np.c_[call(sigma1, Gc), call(sigma2, Gc)]
#             analytic = 189959./120  # Found using sympy. z=5
#         elif self.sigmaTest == 3:
#             sigma = np.r_[call(sigma1, Gc), call(sigma2, Gc), call(sigma3, Gc)]
#             analytic = 781427./360  # Found using sympy. z=5

#         if self.location == 'edges':
#             E = call(funT, self.M.gridEy)
#             A = self.M.getEdgeInnerProduct(sigma)
#             numeric = E.T.dot(A.dot(E))
#         elif self.location == 'faces':
#             Fr = call(funR, self.M.gridFx)
#             Fz = call(funZ, self.M.gridFz)
#             A = self.M.getFaceInnerProduct(sigma)
#             F = np.r_[Fr,Fz]
#             numeric = F.T.dot(A.dot(F))

#         print numeric
#         err = np.abs(numeric - analytic)
#         return err

#     def test_order1_faces(self):
#         self.name = "2D Face Inner Product - Isotropic"
#         self.location = 'faces'
#         self.sigmaTest = 1
#         self.orderTest()


class TestCyl3DMesh(unittest.TestCase):

    def setUp(self):
        hx = np.r_[1,1,0.5]
        hy = np.r_[np.pi, np.pi]
        hz = np.r_[2,1]
        self.mesh = Mesh.CylMesh([hx, hy,hz])

    def test_dim(self):
        self.assertTrue(self.mesh.dim == 3)

    def test_nC(self):
        self.assertTrue(self.mesh.nCx == 3)
        self.assertTrue(self.mesh.nCy == 2)
        self.assertTrue(self.mesh.nCz == 2)
        self.assertTrue(np.all(self.mesh.vnC == [3, 2, 2]))

    def test_nN(self):
        self.assertTrue(self.mesh.nN == 24)
        self.assertTrue(self.mesh.nNx == 4)
        self.assertTrue(self.mesh.nNy == 2)
        self.assertTrue(self.mesh.nNz == 3)
        self.assertTrue(np.all(self.mesh.vnN == [4, 2, 3]))

    def test_nF(self):
        self.assertTrue(self.mesh.nFx == 12)
        self.assertTrue(np.all(self.mesh.vnFx == [3, 2, 2]))
        self.assertTrue(self.mesh.nFy == 12)
        self.assertTrue(np.all(self.mesh.vnFy == [3, 2, 2]))
        self.assertTrue(self.mesh.nFz == 18)
        self.assertTrue(np.all(self.mesh.vnFz == [3, 2, 3]))
        self.assertTrue(self.mesh.nF == 42)
        self.assertTrue(np.all(self.mesh.vnF == [12, 12, 18]))

    def test_nE(self):
        self.assertTrue(self.mesh.nEx == 18)
        self.assertTrue(np.all(self.mesh.vnEx == [3, 2, 3]))
        self.assertTrue(self.mesh.nEy == 18)
        self.assertTrue(np.all(self.mesh.vnEy == [3, 2, 3]))
        self.assertTrue(self.mesh.nEz == 12 + 2)
        self.assertTrue(self.mesh.vnEz is None)
        self.assertTrue(self.mesh.nE == 50)
        self.assertTrue(np.all(self.mesh.vnE == [18, 18, 14]))

    def test_vectorsCC(self):
        v = np.r_[0.5, 1.5, 2.25]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorCCx)) == 0)
        v = np.r_[0, np.pi]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorCCy)) == 0)
        v = np.r_[1, 2.5]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorCCz)) == 0)

    def test_vectorsN(self):
        v = np.r_[0, 1, 2, 2.5]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorNx)) == 0)
        v = np.r_[np.pi/2, 1.5*np.pi]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorNy)) == 0)
        v = np.r_[0, 2, 3]
        self.assertTrue(np.linalg.norm((v-self.mesh.vectorNz)) == 0)


if __name__ == '__main__':
    unittest.main()
