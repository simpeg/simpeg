from SimPEG.Mesh import TensorMesh
from SimPEG.Mesh.NewTreeMesh import TreeMesh
import numpy as np
import unittest
import matplotlib.pyplot as plt

TOL = 1e-10

class TestQuadTreeMesh(unittest.TestCase):

    def setUp(self):
        M = TreeMesh([np.ones(x) for x in [3,2]])
        M.refineFace(0)
        self.M = M
        M.number()
        # M.plotGrid(showIt=True)

    def test_MeshSizes(self):
        self.assertTrue(self.M.nC==9)
        self.assertTrue(self.M.nF==25)
        self.assertTrue(self.M.nFx==12)
        self.assertTrue(self.M.nFy==13)
        self.assertTrue(self.M.nE==25)
        self.assertTrue(self.M.nEx==13)
        self.assertTrue(self.M.nEy==12)

    def test_gridCC(self):
        x = np.r_[0.25,0.75,1.5,2.5,0.25,0.75,0.5,1.5,2.5]
        y = np.r_[0.25,0.25,0.5,0.5,0.75,0.75,1.5,1.5,1.5]
        self.assertTrue(np.linalg.norm((np.c_[x,y]-self.M.gridCC).flatten()) == 0)

    def test_gridN(self):
        x = np.r_[0,0.5,1,2,3,0,0.5,1,0,0.5,1,2,3,0,1,2,3]
        y = np.r_[0,0,0,0,0,.5,.5,.5,1,1,1,1,1,2,2,2,2]
        self.assertTrue(np.linalg.norm((np.c_[x,y]-self.M.gridN).flatten()) == 0)

    def test_gridFx(self):
        x = np.r_[0.0,0.5,1.0,2.0,3.0,0.0,0.5,1.0,0.0,1.0,2.0,3.0]
        y = np.r_[0.25,0.25,0.25,0.5,0.5,0.75,0.75,0.75,1.5,1.5,1.5,1.5]
        self.assertTrue(np.linalg.norm((np.c_[x,y]-self.M.gridFx).flatten()) == 0)

    def test_gridFy(self):
        x = np.r_[0.25,0.75,1.5,2.5,0.25,0.75,0.25,0.75,1.5,2.5,0.5,1.5,2.5]
        y = np.r_[0,0,0,0,0.5,0.5,1,1,1,1,2,2,2]
        self.assertTrue(np.linalg.norm((np.c_[x,y]-self.M.gridFy).flatten()) == 0)

    def test_gridEx(self):
        x = np.r_[0.25,0.75,1.5,2.5,0.25,0.75,0.25,0.75,1.5,2.5,0.5,1.5,2.5]
        y = np.r_[0,0,0,0,0.5,0.5,1,1,1,1,2,2,2]
        self.assertTrue(np.linalg.norm((np.c_[x,y]-self.M.gridEx).flatten()) == 0)

    def test_gridEy(self):
        x = np.r_[0.0,0.5,1.0,2.0,3.0,0.0,0.5,1.0,0.0,1.0,2.0,3.0]
        y = np.r_[0.25,0.25,0.25,0.5,0.5,0.75,0.75,0.75,1.5,1.5,1.5,1.5]
        self.assertTrue(np.linalg.norm((np.c_[x,y]-self.M.gridEy).flatten()) == 0)

    def test_vol(self):
        v = np.r_[0.25,0.25,1,1,0.25,0.25,1,1,1]
        self.assertTrue(np.linalg.norm((v-self.M.vol)) < TOL)

    def test_edge(self):
        ex = np.r_[0.5,0.5,1,1,0.5,0.5,0.5,0.5,1,1,1,1,1]
        ey = np.r_[0.5,0.5,0.5,1,1,0.5,0.5,0.5,1,1,1,1]
        self.assertTrue(np.linalg.norm((np.r_[ex,ey]-self.M.edge)) < TOL)

    def test_area(self):
        ax = np.r_[0.5,0.5,0.5,1,1,0.5,0.5,0.5,1,1,1,1]
        ay = np.r_[0.5,0.5,1,1,0.5,0.5,0.5,0.5,1,1,1,1,1]
        self.assertTrue(np.linalg.norm((np.r_[ax,ay]-self.M.area)) < TOL)



class SimpleOctreeOperatorTests(unittest.TestCase):

    def setUp(self):
        h1 = np.random.rand(5)
        h2 = np.random.rand(7)
        h3 = np.random.rand(3)
        # self.tM = TensorMesh([h1,h2,h3])
        # self.oM = TreeMesh([h1,h2,h3])
        self.tM2 = TensorMesh([h1,h2])
        self.oM2 = TreeMesh([h1,h2])
        # self.oM2.plotGrid(showIt=True)

    def test_faceDiv(self):
        # self.assertAlmostEqual((self.tM.faceDiv - self.oM.faceDiv).toarray().sum(), 0)
        self.assertAlmostEqual((self.tM2.faceDiv - self.oM2.faceDiv).toarray().sum(), 0)

    # def test_nodalGrad(self):
    #     self.assertAlmostEqual((self.tM.nodalGrad - self.oM.nodalGrad).toarray().sum(), 0)
    #     self.assertAlmostEqual((self.tM2.nodalGrad - self.oM2.nodalGrad).toarray().sum(), 0)

    # def test_edgeCurl(self):
    #     self.assertAlmostEqual((self.tM.edgeCurl - self.oM.edgeCurl).toarray().sum(), 0)
    #     # self.assertAlmostEqual((self.tM2.edgeCurl - self.oM2.edgeCurl).toarray().sum(), 0)

    # def test_InnerProducts(self):
    #     self.assertAlmostEqual((self.tM.getFaceInnerProduct() - self.oM.getFaceInnerProduct()).toarray().sum(), 0)
    #     self.assertAlmostEqual((self.tM2.getFaceInnerProduct() - self.oM2.getFaceInnerProduct()).toarray().sum(), 0)
    #     self.assertAlmostEqual((self.tM2.getEdgeInnerProduct() - self.oM2.getEdgeInnerProduct()).toarray().sum(), 0)
    #     self.assertAlmostEqual((self.tM.getEdgeInnerProduct() - self.oM.getEdgeInnerProduct()).toarray().sum(), 0)



class TestOcTreeObjects(unittest.TestCase):

    def setUp(self):
        self.M  = TreeMesh([2,1,1])
        self.M.number()

        # self.Mr = TreeMesh([2,1,1])
        # self.Mr.children[0,0,0].refine()
        # self.Mr.number()

    def test_counts(self):
        self.assertTrue(self.M.nC == 2)
        self.assertTrue(self.M.nFx == 3)
        self.assertTrue(self.M.nFy == 4)
        self.assertTrue(self.M.nFz == 4)
        self.assertTrue(self.M.nF == 11)
        self.assertTrue(self.M.nEx == 8)
        self.assertTrue(self.M.nEy == 6)
        self.assertTrue(self.M.nEz == 6)
        self.assertTrue(self.M.nE == 20)
        self.assertTrue(self.M.nN == 12)

        # self.assertTrue(self.Mr.nC == 9)
        # self.assertTrue(self.Mr.nFx == 13)
        # self.assertTrue(self.Mr.nFy == 14)
        # self.assertTrue(self.Mr.nFz == 14)
        # self.assertTrue(self.Mr.nF == 41)


        # for cell in self.Mr.sortedCells:
        #     for e in cell.edgeDict:
        #         self.assertTrue(cell.edgeDict[e].edgeType==e[1].lower())

        # self.assertTrue(self.Mr.nN == 31)
        # self.assertTrue(self.Mr.nEx == 22)
        # self.assertTrue(self.Mr.nEy == 20)
        # self.assertTrue(self.Mr.nEz == 20)


if __name__ == '__main__':
    unittest.main()
