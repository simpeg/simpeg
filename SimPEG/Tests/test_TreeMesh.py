from SimPEG.Mesh.TreeMesh import TreeMesh, TreeFace, TreeCell
import numpy as np
import unittest
import matplotlib.pyplot as plt

class TestOcTreeObjects(unittest.TestCase):

    def setUp(self):
        self.M  = TreeMesh([2,1,1])
        self.M.number()

        self.Mr = TreeMesh([2,1,1])
        self.Mr.children[0,0,0].refine()
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


        for cell in self.Mr.sortedCells:
            for e in cell.edges:
                self.assertTrue(cell.edges[e].edgeType==e[1].lower())

        # print self.Mr.nEx
        # self.assertTrue(self.Mr.nN == 22)
        # self.assertTrue(self.Mr.nEx == 22)

    def test_pointersM(self):
        c0    = self.M.children[0,0,0]
        c0fXm = c0.faces['fXm']
        c0fXp = c0.faces['fXp']
        c0fYm = c0.faces['fYm']
        c0fYp = c0.faces['fYp']
        c0fZm = c0.faces['fZm']
        c0fZp = c0.faces['fZp']

        c1    = self.M.children[1,0,0]
        c1fXm = c1.faces['fXm']
        c1fXp = c1.faces['fXp']
        c1fYm = c1.faces['fYm']
        c1fYp = c1.faces['fYp']
        c1fZm = c1.faces['fZm']
        c1fZp = c1.faces['fZp']

        self.assertTrue(c0fXp is c1fXm)
        self.assertTrue(c0fXp.edges['e0'] is c1fXm.edges['e0'])
        self.assertTrue(c0fXp.edges['e1'] is c1fXm.edges['e1'])
        self.assertTrue(c0fXp.edges['e2'] is c1fXm.edges['e2'])
        self.assertTrue(c0fXp.edges['e3'] is c1fXm.edges['e3'])
        self.assertTrue(c0fYp is not c1fYm)
        self.assertTrue(c0fXm is not c1fXm)

        # Test connectivity of shared edges
        self.assertTrue(c0fZp.edges['e3'] is not c1fZp.edges['e0'])
        self.assertTrue(c0fZp.edges['e3'] is not c1fZp.edges['e1'])
        self.assertTrue(c0fZp.edges['e3'] is c1fZp.edges['e2'])
        self.assertTrue(c0fZp.edges['e3'] is not c1fZp.edges['e3'])

        self.assertTrue(c0fZm.edges['e3'] is not c1fZm.edges['e0'])
        self.assertTrue(c0fZm.edges['e3'] is not c1fZm.edges['e1'])
        self.assertTrue(c0fZm.edges['e3'] is c1fZm.edges['e2'])
        self.assertTrue(c0fZm.edges['e3'] is not c1fZm.edges['e3'])

        self.assertTrue(c0fYp.edges['e3'] is not c1fYp.edges['e0'])
        self.assertTrue(c0fYp.edges['e3'] is not c1fYp.edges['e1'])
        self.assertTrue(c0fYp.edges['e3'] is c1fYp.edges['e2'])
        self.assertTrue(c0fYp.edges['e3'] is not c1fYp.edges['e3'])

        self.assertTrue(c0fYm.edges['e3'] is not c1fYm.edges['e0'])
        self.assertTrue(c0fYm.edges['e3'] is not c1fYm.edges['e1'])
        self.assertTrue(c0fYm.edges['e3'] is c1fYm.edges['e2'])
        self.assertTrue(c0fYm.edges['e3'] is not c1fYm.edges['e3'])

        self.assertTrue(c0fZm.edges['e3'] is c1fXm.edges['e0'])
        self.assertTrue(c0fZp.edges['e3'] is c1fXm.edges['e1'])
        self.assertTrue(c0fYm.edges['e3'] is c1fXm.edges['e2'])
        self.assertTrue(c0fYp.edges['e3'] is c1fXm.edges['e3'])

        self.assertTrue(c0fZm.edges['e3'] is c0fXp.edges['e0'])
        self.assertTrue(c0fZp.edges['e3'] is c0fXp.edges['e1'])
        self.assertTrue(c0fYm.edges['e3'] is c0fXp.edges['e2'])
        self.assertTrue(c0fYp.edges['e3'] is c0fXp.edges['e3'])

        self.assertTrue(c1fZm.edges['e2'] is c1fXm.edges['e0'])
        self.assertTrue(c1fZp.edges['e2'] is c1fXm.edges['e1'])
        self.assertTrue(c1fYm.edges['e2'] is c1fXm.edges['e2'])
        self.assertTrue(c1fYp.edges['e2'] is c1fXm.edges['e3'])

        self.assertTrue(c1fZm.edges['e2'] is c0fXp.edges['e0'])
        self.assertTrue(c1fZp.edges['e2'] is c0fXp.edges['e1'])
        self.assertTrue(c1fYm.edges['e2'] is c0fXp.edges['e2'])
        self.assertTrue(c1fYp.edges['e2'] is c0fXp.edges['e3'])



    def test_pointersMr(self):
        ax = plt.subplot(111, projection='3d')
        self.Mr.plotGrid(ax=ax,showIt=False,plotC=True,plotEy=True, text=False)

        cell = self.Mr.sortedCells[1]
        [cell.edges[e].plotGrid(ax,lineOpts={'color':'b','ls':'-'}) for e in cell.edges]
        cell.plotGrid(ax)
        plt.show()

        def q(s):
            c = self.Mr.sortedCells[int(s[1])]
            if len(s) == 2: return c
            if s[2] == 'f': return c.faces[s[2:]]
            if s[2] == 'e': return c.edges[s[2:]]

        c0    = self.Mr.sortedCells[0]
        c0fXm = c0.faces['fXm']
        c0eX0 = c0.edges['eX0']
        self.assertTrue(c0 is q('c0'))
        self.assertTrue(c0fXm is q('c0fXm'))
        self.assertTrue(c0eX0 is q('c0eX0'))

        self.assertTrue(q('c0').depth == 1)
        self.assertTrue(q('c1').depth == 1)
        self.assertTrue(q('c2').depth == 0)

        # Make sure we know where the center of the cells are.
        self.assertTrue(np.all(q('c0').center == np.r_[0.125,0.25,0.25]))
        self.assertTrue(np.all(q('c1').center == np.r_[0.375,0.25,0.25]))
        self.assertTrue(np.all(q('c2').center == np.r_[0.75,0.5,0.5]))
        self.assertTrue(np.all(q('c3').center == np.r_[0.125,0.75,0.25]))
        self.assertTrue(np.all(q('c4').center == np.r_[0.375,0.75,0.25]))
        self.assertTrue(np.all(q('c5').center == np.r_[0.125,0.25,0.75]))
        self.assertTrue(np.all(q('c6').center == np.r_[0.375,0.25,0.75]))
        self.assertTrue(np.all(q('c7').center == np.r_[0.125,0.75,0.75]))
        self.assertTrue(np.all(q('c8').center == np.r_[0.375,0.75,0.75]))

        self.assertTrue(np.all(q('c0fXm').center == np.r_[0,0.25,0.25]))
        self.assertTrue(np.all(q('c0fXp').center == np.r_[0.25,0.25,0.25]))
        self.assertTrue(q('c0fXp') is q('c1fXm'))
        self.assertTrue(np.all(q('c1fXp').center == np.r_[0.5,0.25,0.25]))
        self.assertTrue(np.all(q('c2fXm').center == np.r_[0.5,0.5,0.5]))
        self.assertTrue(q('c2fXm').branchdepth == 1)
        self.assertTrue(q('c1fXp').parent is q('c2fXm'))
        self.assertTrue(q('c2fXm').children[0,0] is q('c1fXp'))

        self.assertTrue(q('c0fXp') is q('c1fXm'))
        self.assertTrue(q('c0fYp') is not q('c1fYm'))
        self.assertTrue(q('c0fXm') is not q('c1fXm'))

        self.assertTrue(q('c1fXp') is q('c2fXm').children[0,0])
        self.assertTrue(q('c1fXp').parent is q('c2fXm'))

        self.assertTrue(q('c1fYp') is q('c4fYm'))
        self.assertTrue(q('c1fZp') is q('c6fZm'))

        self.assertTrue(q('c6fXp') is q('c2fXm').children[0,1])
        self.assertTrue(q('c6fXp').parent is q('c2fXm'))

        self.assertTrue(q('c4fXp') is q('c2fXm').children[1,0])
        self.assertTrue(q('c4fXp').parent is q('c2fXm'))

        #TEST EDGES!

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
