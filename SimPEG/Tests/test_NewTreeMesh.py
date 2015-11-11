from SimPEG.Mesh import TensorMesh
from SimPEG.Mesh.NewTreeMesh import TreeMesh, TreeCell
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

    def test_numbering(self):
        M = TreeMesh([2,2,2])
        M.number()
        M.refineCell(0)
        M.refineCell(3)
        assert M.isNumbered is False

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

class TestOcTreeConnectivity(unittest.TestCase):

    def setUp(self):
        self.oM = TreeMesh([1,1,1])
        self.oM.refine(lambda c: 1)

    def test_setup(self):
        C = TreeCell(self.oM, 0)
        children = C.children
        assert not C.isleaf
        assert len(children.index) == 8
        # assert not TreeCell(self.oM, 0).isleaf
        c0, c1, c2, c3, c4, c5, c6, c7 = [TreeCell(self.oM, i) for i in range(1,9)]


        #                                                                      .----------------.----------------.
        #                                                                     /|               /|               /|
        #                                                                    / |              / |              / |
        #                                                                   /  |     c6      /  |    c7       /  |
        #                                                                  /   |            /   |            /   |
        #                                                                 .----------------.----+-----------.    |
        #                                                                /|    . ---------/|----.----------/|----.
        #                      fZp                                      / |   /|         / |   /|         / |   /|
        #                       |                                      /  |  / | c4     /  |  / |  c5    /  |  X |
        #                 6 ------eX3------ 7                         /   | /  |       /   | /  |       /   | /  |
        #                /|     |         / |                        . -------------- .----------------.    |/   |
        #               /eZ2    .        / eZ3                       |    . ---+------|----.----+------|----.    |
        #             eY2 |        fYp eY3  |                        |   /|    .______|___/|____.______|___/|____.
        #             /   |            / fXp|                        |  / |   /   c2  |  / |   /    c3 |  / |   /
        #            4 ------eX2----- 5     |                        | /  |  /        | /  |  /        | /  |  /
        #            |fXm 2 -----eX1--|---- 3          z             . ---+---------- . ---+---------- .    | /
        #           eZ0  /            |  eY1           ^   y         |    |/          |    |/          |    |/
        #            | eY0   .  fYm  eZ1 /             |  /          |    . ----------|----.-----------|----.
        #            | /     |        | /              | /           |   /    c0      |   /     c1     |   /
        #            0 ------eX0------1                o----> x      |  /             |  /             |  /
        #                    |                                       | /              | /              | /
        #                   fZm                                      . -------------- . -------------- .
        #
        #
        #            fX                                  fY                                 fZ
        #      2___________3                       2___________3                      2___________3
        #      |     e1    |                       |     e1    |                      |     e1    |
        #      |           |                       |           |                      |           |
        #   e2 |     x     | e3      z          e2 |     x     | e3      z         e2 |     x     | e3      y
        #      |           |         ^             |           |         ^            |           |         ^
        #      |___________|         |___> y       |___________|         |___> x      |___________|         |___> x
        #      0    e0     1                       0    e0     1                      0    e0     1
        #


        # there are two faces for each edge
        for ii, c in enumerate([c0, c1, c2, c3, c4, c5, c6, c7]):
            assert c.fZm.e0.index == c.fYm.e0.index, "Cell %d: fZm.e0 and fYm.e0"%ii
            assert c.fZm.e1.index == c.fYp.e0.index, "Cell %d: fZm.e1 and fYp.e0"%ii
            assert c.fZp.e0.index == c.fYm.e1.index, "Cell %d: fZp.e0 and fYm.e1"%ii
            assert c.fZp.e1.index == c.fYp.e1.index, "Cell %d: fZp.e1 and fYp.e1"%ii
            assert c.fZm.e2.index == c.fXm.e0.index, "Cell %d: fZm.e2 and fXm.e0"%ii
            assert c.fZm.e3.index == c.fXp.e0.index, "Cell %d: fZm.e3 and fXp.e0"%ii
            assert c.fZp.e2.index == c.fXm.e1.index, "Cell %d: fZp.e2 and fXm.e1"%ii
            assert c.fZp.e3.index == c.fXp.e1.index, "Cell %d: fZp.e3 and fXp.e1"%ii
            assert c.fYm.e2.index == c.fXm.e2.index, "Cell %d: fYm.e2 and fXm.e2"%ii
            assert c.fYm.e3.index == c.fXp.e2.index, "Cell %d: fYm.e3 and fXp.e2"%ii
            assert c.fYp.e2.index == c.fXm.e3.index, "Cell %d: fYp.e2 and fXm.e3"%ii
            assert c.fYp.e3.index == c.fXp.e3.index, "Cell %d: fYp.e3 and fXp.e3"%ii

        assert c0.eZ1.index == c1.eZ0.index
        assert c0.eZ3.index == c1.eZ2.index
        assert c2.eZ1.index == c3.eZ0.index
        assert c2.eZ3.index == c3.eZ2.index

        assert c4.eZ1.index == c5.eZ0.index
        assert c4.eZ3.index == c5.eZ2.index
        assert c6.eZ1.index == c7.eZ0.index
        assert c6.eZ3.index == c7.eZ2.index

        assert c0.n7.index == c7.n0.index





class SimpleOctreeOperatorTests(unittest.TestCase):

    def setUp(self):
        h1 = np.random.rand(5)
        h2 = np.random.rand(7)
        h3 = np.random.rand(3)
        self.tM = TensorMesh([h1,h2,h3])
        self.oM = TreeMesh([h1,h2,h3])
        self.tM2 = TensorMesh([h1,h2])
        self.oM2 = TreeMesh([h1,h2])
        # self.oM2.plotGrid(showIt=True)

    def test_faceDiv(self):
        self.assertAlmostEqual((self.tM.faceDiv - self.oM.faceDiv).toarray().sum(), 0)
        self.assertAlmostEqual((self.tM2.faceDiv - self.oM2.faceDiv).toarray().sum(), 0)

    def test_nodalGrad(self):
        self.assertAlmostEqual((self.tM.nodalGrad - self.oM.nodalGrad).toarray().sum(), 0)
        self.assertAlmostEqual((self.tM2.nodalGrad - self.oM2.nodalGrad).toarray().sum(), 0)

    def test_edgeCurl(self):
        self.assertAlmostEqual((self.tM.edgeCurl - self.oM.edgeCurl).toarray().sum(), 0)
        # self.assertAlmostEqual((self.tM2.edgeCurl - self.oM2.edgeCurl).toarray().sum(), 0)

    def test_InnerProducts(self):
        self.assertAlmostEqual((self.tM.getFaceInnerProduct() - self.oM.getFaceInnerProduct()).toarray().sum(), 0)
        self.assertAlmostEqual((self.tM.getEdgeInnerProduct() - self.oM.getEdgeInnerProduct()).toarray().sum(), 0)
    #     self.assertAlmostEqual((self.tM2.getFaceInnerProduct() - self.oM2.getFaceInnerProduct()).toarray().sum(), 0)
    #     self.assertAlmostEqual((self.tM2.getEdgeInnerProduct() - self.oM2.getEdgeInnerProduct()).toarray().sum(), 0)


class SimpleOctreeOperatorTestsRefined(unittest.TestCase):
    def setUp(self):
        self.tM = TreeMesh([1,1,1])
        self.tM.refine(lambda c:2)
        self.M = TensorMesh([4,4,4])

        self.tM2 = TreeMesh([1,1])
        self.tM2.refine(lambda c:2)
        self.M2 = TensorMesh([4,4])

    def test_grids(self):
        self.assertAlmostEqual((self.tM2.gridN - self.M2.gridN).sum(), 0)
        self.assertAlmostEqual((self.tM2.gridCC - self.M2.gridCC).sum(), 0)
        self.assertAlmostEqual((self.tM2.gridFx - self.M2.gridFx).sum(), 0)
        self.assertAlmostEqual((self.tM2.gridFy - self.M2.gridFy).sum(), 0)
        self.assertAlmostEqual((self.tM2.gridEx - self.M2.gridEx).sum(), 0)
        self.assertAlmostEqual((self.tM2.gridEy - self.M2.gridEy).sum(), 0)

        self.assertAlmostEqual((self.tM.gridN - self.M.gridN).sum(), 0)
        self.assertAlmostEqual((self.tM.gridCC - self.M.gridCC).sum(), 0)
        self.assertAlmostEqual((self.tM.gridFx - self.M.gridFx).sum(), 0)
        self.assertAlmostEqual((self.tM.gridFy - self.M.gridFy).sum(), 0)
        self.assertAlmostEqual((self.tM.gridFz - self.M.gridFz).sum(), 0)
        self.assertAlmostEqual((self.tM.gridEx - self.M.gridEx).sum(), 0)
        self.assertAlmostEqual((self.tM.gridEy - self.M.gridEy).sum(), 0)
        self.assertAlmostEqual((self.tM.gridEz - self.M.gridEz).sum(), 0)

    def test_InnerProducts(self):
        self.assertAlmostEqual((self.tM.getFaceInnerProduct() - self.M.getFaceInnerProduct()).toarray().sum(), 0)
        self.assertAlmostEqual((self.tM.getEdgeInnerProduct() - self.M.getEdgeInnerProduct()).toarray().sum(), 0)

        # self.assertAlmostEqual((self.tM2.getFaceInnerProduct() - self.M2.getFaceInnerProduct()).toarray().sum(), 0)
        # self.assertAlmostEqual((self.tM2.getEdgeInnerProduct() - self.M2.getEdgeInnerProduct()).toarray().sum(), 0)

    def test_faceDiv(self):
        self.assertAlmostEqual((self.tM2.faceDiv - self.M2.faceDiv).toarray().sum(), 0)
        self.assertAlmostEqual((self.tM.faceDiv - self.M.faceDiv).toarray().sum(), 0)

    def test_nodalGrad(self):
        self.assertAlmostEqual((self.tM2.nodalGrad - self.M2.nodalGrad).toarray().sum(), 0)
        self.assertAlmostEqual((self.tM.nodalGrad - self.M.nodalGrad).toarray().sum(), 0)

    def test_edgeCurl(self):
        self.assertAlmostEqual((self.tM.edgeCurl - self.M.edgeCurl).toarray().sum(), 0)
        # self.assertAlmostEqual((self.tM2.edgeCurl - self.M2.edgeCurl).toarray().sum(), 0)

    def test_InnerProducts(self):
        self.assertAlmostEqual((self.tM.getFaceInnerProduct() - self.M.getFaceInnerProduct()).toarray().sum(), 0)
        self.assertAlmostEqual((self.tM.getEdgeInnerProduct() - self.M.getEdgeInnerProduct()).toarray().sum(), 0)
        # self.assertAlmostEqual((self.tM2.getFaceInnerProduct() - self.M2.getFaceInnerProduct()).toarray().sum(), 0)
        # self.assertAlmostEqual((self.tM2.getEdgeInnerProduct() - self.M2.getEdgeInnerProduct()).toarray().sum(), 0)


class TestOcTreeObjects(unittest.TestCase):

    def setUp(self):
        self.M  = TreeMesh([2,1,1])
        self.M.number()

        self.Mr = TreeMesh([2,1,1])
        self.Mr.refineCell(0)
        self.Mr.number()

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

        self.assertTrue(self.Mr.nC == 9)
        self.assertTrue(self.Mr.nFx == 13)
        self.assertTrue(self.Mr.nFy == 14)
        self.assertTrue(self.Mr.nFz == 14)
        self.assertTrue(self.Mr.nF == 41)

        self.assertTrue(self.Mr.nN == 31)
        self.assertTrue(self.Mr.nEx == 22)
        self.assertTrue(self.Mr.nEy == 20)
        self.assertTrue(self.Mr.nEz == 20)


    def test_gridCC(self):
        x = np.r_[0.25,0.75]
        y = np.r_[0.5,0.5]
        z = np.r_[0.5,0.5]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.M.gridCC).flatten()) == 0)

        x = np.r_[0.125,0.375,0.75,0.125,0.375,0.125,0.375,0.125,0.375]
        y = np.r_[0.25,0.25,0.5,0.75,0.75,0.25,0.25,0.75,0.75]
        z = np.r_[0.25,0.25,0.5,0.25,0.25,0.75,0.75,0.75,0.75]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.Mr.gridCC).flatten()) == 0)

    def test_gridN(self):
        x = np.r_[0,0.5,1,0,0.5,1,0,0.5,1,0,0.5,1]
        y = np.r_[0,0,0,1,1,1,0,0,0,1,1,1.]
        z = np.r_[0,0,0,0,0,0,1,1,1,1,1,1.]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.M.gridN).flatten()) == 0)

        x = np.r_[0,0.25,0.5,1,0,0.25,0.5,0,0.25,0.5,1,0,0.25,0.5,0,0.25,0.5,0,0.25,0.5,0,0.25,0.5,1,0,0.25,0.5,0,0.25,0.5,1]
        y = np.r_[0,0,0,0,0.5,0.5,0.5,1,1,1,1,0,0,0,0.5,0.5,0.5,1,1,1,0,0,0,0,0.5,0.5,0.5,1,1,1,1]
        z = np.r_[0,0,0,0,0,0,0,0,0,0,0,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,1,1,1,1,1,1]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.Mr.gridN).flatten()) == 0)

    def test_gridFx(self):
        x = np.r_[0.0,0.5,1.0]
        y = np.r_[0.5,0.5,0.5]
        z = np.r_[0.5,0.5,0.5]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.M.gridFx).flatten()) == 0)

        x = np.r_[0.0,0.25,0.5,1.0,0.0,0.25,0.5,0.0,0.25,0.5,0.0,0.25,0.5]
        y = np.r_[0.25,0.25,0.25,0.5,0.75,0.75,0.75,0.25,0.25,0.25,0.75,0.75,0.75]
        z = np.r_[0.25,0.25,0.25,0.5,0.25,0.25,0.25,0.75,0.75,0.75,0.75,0.75,0.75]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.Mr.gridFx).flatten()) == 0)

    def test_gridFy(self):
        x = np.r_[0.25,0.75,0.25,0.75]
        y = np.r_[0,0,1.,1.]
        z = np.r_[0.5,0.5,0.5,0.5]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.M.gridFy).flatten()) == 0)

        x = np.r_[0.125,0.375,0.75,0.125,0.375,0.125,0.375,0.75,0.125,0.375,0.125,0.375,0.125,0.375]
        y = np.r_[0,0,0,0.5,0.5,1,1,1,0,0,0.5,0.5,1,1]
        z = np.r_[0.25,0.25,0.5,0.25,0.25,0.25,0.25,0.5,0.75,0.75,0.75,0.75,0.75,0.75]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.Mr.gridFy).flatten()) == 0)

    def test_gridFz(self):
        x = np.r_[0.25,0.75,0.25,0.75]
        y = np.r_[0.5,0.5,0.5,0.5]
        z = np.r_[0,0,1.,1.]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.M.gridFz).flatten()) == 0)

        x = np.r_[0.125,0.375,0.75,0.125,0.375,0.125,0.375,0.125,0.375,0.125,0.375,0.75,0.125,0.375]
        y = np.r_[0.25,0.25,0.5,0.75,0.75,0.25,0.25,0.75,0.75,0.25,0.25,0.5,0.75,0.75]
        z = np.r_[0,0,0,0,0,0.5,0.5,0.5,0.5,1,1,1,1,1]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.Mr.gridFz).flatten()) == 0)


    def test_gridEx(self):
        x = np.r_[0.25,0.75,0.25,0.75,0.25,0.75,0.25,0.75]
        y = np.r_[0,0,1.,1.,0,0,1.,1.]
        z = np.r_[0,0,0,0,1.,1.,1.,1.]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.M.gridEx).flatten()) == 0)

        x = np.r_[0.125,0.375,0.75,0.125,0.375,0.125,0.375,0.75,0.125,0.375,0.125,0.375,0.125,0.375,0.125,0.375,0.75,0.125,0.375,0.125,0.375,0.75]
        y = np.r_[0,0,0,0.5,0.5,1,1,1,0,0,0.5,0.5,1,1,0,0,0,0.5,0.5,1,1,1]
        z = np.r_[0,0,0,0,0,0,0,0,0.5,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,1,1,1]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.Mr.gridEx).flatten()) == 0)

    def test_gridEy(self):
        x = np.r_[0,0.5,1,0,0.5,1]
        y = np.r_[0.5,0.5,0.5,0.5,0.5,0.5]
        z = np.r_[0,0,0,1.,1.,1.]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.M.gridEy).flatten()) == 0)

        x = np.r_[0,0.25,0.5,1,0,0.25,0.5,0,0.25,0.5,0,0.25,0.5,0,0.25,0.5,1,0,0.25,0.5]
        y = np.r_[0.25,0.25,0.25,0.5,0.75,0.75,0.75,0.25,0.25,0.25,0.75,0.75,0.75,0.25,0.25,0.25,0.5,0.75,0.75,0.75]
        z = np.r_[0,0,0,0,0,0,0,0.5,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,1,1]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.Mr.gridEy).flatten()) == 0)

    def test_gridEz(self):
        x = np.r_[0,0.5,1,0,0.5,1]
        y = np.r_[0,0,0,1.,1.,1.]
        z = np.r_[0.5,0.5,0.5,0.5,0.5,0.5]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.M.gridEz).flatten()) == 0)

        x = np.r_[0,0.25,0.5,1,0  ,0.25,0.5,0,0.25,0.5,1,0,0.25,0.5,0  ,0.25,0.5,0  ,0.25,0.5]
        y = np.r_[0,0   ,0  ,0,0.5,0.5 ,0.5,1,1   ,1  ,1,0,0   ,0  ,0.5,0.5 ,0.5,1  ,1   ,1  ]
        z = np.r_[0.25,0.25,0.25,0.5,0.25,0.25,0.25,0.25,0.25,0.25,0.5,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.Mr.gridEz).flatten()) == 0)

if __name__ == '__main__':
    unittest.main()
