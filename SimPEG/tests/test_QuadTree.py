from SimPEG import Mesh, np
import unittest



class TestQuadTreeMesh(unittest.TestCase):

    def setUp(self):
        M = Mesh.TreeMesh([np.ones(x) for x in [3,2]])
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

    def test_gridFx(self):
        x = np.r_[0.0,0.5,1.0,2.0,3.0,0.0,0.5,1.0,0.0,1.0,2.0,3.0]
        y = np.r_[0.25,0.25,0.25,0.5,0.5,0.75,0.75,0.75,1.5,1.5,1.5,1.5]
        self.assertTrue(np.linalg.norm((np.c_[x,y]-self.M.gridFx).flatten()) == 0)

    def test_gridFy(self):
        x = np.r_[0.25,0.75,1.5,2.5,0.25,0.75,0.25,0.75,1.5,2.5,0.5,1.5,2.5]
        y = np.r_[0,0,0,0,0.5,0.5,1,1,1,1,2,2,2]
        self.assertTrue(np.linalg.norm((np.c_[x,y]-self.M.gridFy).flatten()) == 0)



if __name__ == '__main__':
    unittest.main()
