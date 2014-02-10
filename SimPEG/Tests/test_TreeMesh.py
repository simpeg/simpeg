from SimPEG.Mesh.TreeMesh import TreeMesh, TreeFace, TreeCell
import numpy as np
import unittest
import matplotlib.pyplot as plt

class TestOcTreeObjects(unittest.TestCase):

    def setUp(self):
        self.M  = TreeMesh([2,1,1])
        self.Mr = TreeMesh([2,1,1])
        self.Mr.children[0,0,0].refine()
        self.Mr.number()

    def test_counts(self):
        ax = plt.subplot(111,projection='3d')
        # self.Mr.plotGrid(showIt=False,plotC=True,plotEy=True)

        cell = self.Mr.sortedCells[1]
        [cell.edges[e].plotGrid(ax) for e in cell.edges]
        cell.plotGrid(ax)
        plt.show()

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

        cell = self.Mr.sortedCells[1]
        self.assertTrue(cell.edges['eX0'].edgeType=='x')
        self.assertTrue(cell.edges['eX1'].edgeType=='x')
        self.assertTrue(cell.edges['eX2'].edgeType=='x')
        self.assertTrue(cell.edges['eX3'].edgeType=='x')
        self.assertTrue(cell.edges['eY0'].edgeType=='y')
        self.assertTrue(cell.edges['eY1'].edgeType=='y')
        self.assertTrue(cell.edges['eY2'].edgeType=='y')
        self.assertTrue(cell.edges['eY3'].edgeType=='y')
        self.assertTrue(cell.edges['eZ0'].edgeType=='z')
        self.assertTrue(cell.edges['eZ1'].edgeType=='z')
        self.assertTrue(cell.edges['eZ2'].edgeType=='z')
        self.assertTrue(cell.edges['eZ3'].edgeType=='z')

        print self.Mr.nN
        # self.assertTrue(self.Mr.nN == 22)

    def test_pointersM(self):
        c0    = self.M.children[0,0,0]
        c0fXm = c0.faces['fXm']
        c0fXp = c0.faces['fXp']
        c0fYm = c0.faces['fYm']
        c0fYp = c0.faces['fYp']

        c1    = self.M.children[1,0,0]
        c1fXm = c1.faces['fXm']
        c1fXp = c1.faces['fXp']
        c1fYm = c1.faces['fYm']
        c1fYp = c1.faces['fYp']

        self.assertTrue(c0fXp is c1fXm)
        self.assertTrue(c0fXp.edges['e0'] is c1fXm.edges['e0'])
        self.assertTrue(c0fXp.edges['e1'] is c1fXm.edges['e1'])
        self.assertTrue(c0fXp.edges['e2'] is c1fXm.edges['e2'])
        self.assertTrue(c0fXp.edges['e3'] is c1fXm.edges['e3'])
        self.assertTrue(c0fYp is not c1fYm)
        self.assertTrue(c0fXm is not c1fXm)


    def test_pointersMr(self):
        c0    = self.Mr.sortedCells[0]
        c0fXm = c0.faces['fXm']
        c0fXp = c0.faces['fXp']
        c0fYm = c0.faces['fYm']
        c0fYp = c0.faces['fYp']
        c0fZm = c0.faces['fZm']
        c0fZp = c0.faces['fZp']
        self.assertTrue(np.all(c0.center==np.r_[0.125,0.25,0.25]))

        c1    = self.Mr.sortedCells[1]
        c1fXm = c1.faces['fXm']
        c1fXp = c1.faces['fXp']
        c1fYm = c1.faces['fYm']
        c1fYp = c1.faces['fYp']
        c1fZm = c1.faces['fZm']
        c1fZp = c1.faces['fZp']
        self.assertTrue(np.all(c1.center==np.r_[0.375,0.25,0.25]))

        c2    = self.Mr.sortedCells[2]
        c2fXm = c2.faces['fXm']
        c2fXp = c2.faces['fXp']
        c2fYm = c2.faces['fYm']
        c2fYp = c2.faces['fYp']
        c2fZm = c2.faces['fZm']
        c2fZp = c2.faces['fZp']
        self.assertTrue(np.all(c2.center==np.r_[0.75,0.5,0.5]))

        c4    = self.Mr.sortedCells[4]
        c4fXm = c4.faces['fXm']
        c4fXp = c4.faces['fXp']
        c4fYm = c4.faces['fYm']
        c4fYp = c4.faces['fYp']
        c4fZm = c4.faces['fZm']
        c4fZp = c4.faces['fZp']
        self.assertTrue(np.all(c4.center==np.r_[0.375,0.75,0.25]))

        c6    = self.Mr.sortedCells[6]
        c6fXm = c6.faces['fXm']
        c6fXp = c6.faces['fXp']
        c6fYm = c6.faces['fYm']
        c6fYp = c6.faces['fYp']
        c6fZm = c6.faces['fZm']
        c6fZp = c6.faces['fZp']
        self.assertTrue(np.all(c6.center==np.r_[0.375,0.25,0.75]))

        self.assertTrue(c0fXp is c1fXm)
        self.assertTrue(c0fYp is not c1fYm)
        self.assertTrue(c0fXm is not c1fXm)

        self.assertTrue(c1fXp is c2fXm.children[0,0])
        self.assertTrue(c1fXp.parent is c2fXm)

        self.assertTrue(c1fYp is c4fYm)
        self.assertTrue(c1fZp is c6fZm)

        self.assertTrue(c6fXp is c2fXm.children[0,1])
        self.assertTrue(c6fXp.parent is c2fXm)

        self.assertTrue(c4fXp is c2fXm.children[1,0])
        self.assertTrue(c4fXp.parent is c2fXm)

    def test_gridCC(self):
        x = np.r_[0.25,0.75]
        y = np.r_[0.5,0.5]
        z = np.r_[0.5,0.5]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.M.gridCC).flatten()) == 0)

        x = np.r_[0.125,0.375,0.75,0.125,0.375,0.125,0.375,0.125,0.375]
        y = np.r_[0.25,0.25,0.5,0.75,0.75,0.25,0.25,0.75,0.75]
        z = np.r_[0.25,0.25,0.5,0.25,0.25,0.75,0.75,0.75,0.75]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.Mr.gridCC).flatten()) == 0)

    def test_gridFx(self):
        x = np.r_[0.0,0.5,1.0]
        y = np.r_[0.5,0.5,0.5]
        z = np.r_[0.5,0.5,0.5]
        self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.M.gridFx).flatten()) == 0)

        x = np.r_[0.0,0.25,0.5,1.0,0.0,0.25,0.5,0.0,0.25,0.5,0.0,0.25,0.5]
        y = np.r_[0.25,0.25,0.25,0.5,0.75,0.75,0.75,0.25,0.25,0.25,0.75,0.75,0.75]
        z = np.r_[0.25,0.25,0.25,0.5,0.25,0.25,0.25,0.75,0.75,0.75,0.75,0.75,0.75]
        # print self.Mr.gridFx - np.c_[x,y,z]
        # self.assertTrue(np.linalg.norm((np.c_[x,y,z]-self.Mr.gridFx).flatten()) == 0)




class TestQuadTreeObjects(unittest.TestCase):

    def setUp(self):
        self.M  = TreeMesh([2,1])
        self.Mr = TreeMesh([2,1])
        self.Mr.children[0,0].refine()
        self.Mr.number()
        # self.Mr.plotGrid(showIt=True)

    def test_pointersM(self):
        c0    = self.M.children[0,0]
        c0fXm = c0.faces['fXm']
        c0fXp = c0.faces['fXp']
        c0fYm = c0.faces['fYm']
        c0fYp = c0.faces['fYp']

        c1    = self.M.children[1,0]
        c1fXm = c1.faces['fXm']
        c1fXp = c1.faces['fXp']
        c1fYm = c1.faces['fYm']
        c1fYp = c1.faces['fYp']

        self.assertTrue(c0fXp is c1fXm)
        self.assertTrue(c0fYp is not c1fYm)
        self.assertTrue(c0fXm is not c1fXm)

        self.assertTrue(c0fXm.area == 1)
        self.assertTrue(c0fYm.area == 0.5)

        self.assertTrue(c0.nodes['n1'] is c1.nodes['n0'])
        self.assertTrue(c0.nodes['n3'] is c1.nodes['n2'])
        self.assertTrue(self.M.nN == 6)


    def test_pointersMr(self):
        c0    = self.Mr.sortedCells[0]
        c0fXm = c0.faces['fXm']
        c0fXp = c0.faces['fXp']
        c0fYm = c0.faces['fYm']
        c0fYp = c0.faces['fYp']

        c1    = self.Mr.sortedCells[1]
        c1fXm = c1.faces['fXm']
        c1fXp = c1.faces['fXp']
        c1fYm = c1.faces['fYm']
        c1fYp = c1.faces['fYp']

        c2    = self.Mr.sortedCells[2]
        c2fXm = c2.faces['fXm']
        c2fXp = c2.faces['fXp']
        c2fYm = c2.faces['fYm']
        c2fYp = c2.faces['fYp']

        c4    = self.Mr.sortedCells[4]
        c4fXm = c4.faces['fXm']
        c4fXp = c4.faces['fXp']
        c4fYm = c4.faces['fYm']
        c4fYp = c4.faces['fYp']

        self.assertTrue(c0fXp is c1fXm)
        self.assertTrue(c1fXp.parent is c2fXm)
        self.assertTrue(c1fXp.node0 is c2fXm.node0)
        self.assertTrue(c1fXp.node0 is c2fXm.node0)
        self.assertTrue(c4fYm is c1fYp)
        self.assertTrue(c4fXp.parent is c2fXm)
        self.assertTrue(c4fXp.node1 is c2fXm.node1)
        self.assertTrue(c4fXp.node0 is c1fYp.node1)
        self.assertTrue(c0fXp.node1 is c4fYm.node0)

        self.assertTrue(self.Mr.nN == 11)

        self.assertTrue(np.all(c1fXp.node0.x0 == np.r_[0.5,0]))
        self.assertTrue(np.all(c1fYp.node0.x0 == np.r_[0.25,0.5]))


class TestQuadTreeMesh(unittest.TestCase):

    def setUp(self):
        M = TreeMesh([np.ones(x) for x in [3,2]])
        for ii in range(1):
            M.children[ii,ii].refine()
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



if __name__ == '__main__':
    unittest.main()
