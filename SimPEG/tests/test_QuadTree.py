from SimPEG import mesh, np
import unittest



class TestCheckDerivative(unittest.TestCase):

    def setUp(self):
        M = mesh.QuadTreeMesh([3,2],[1,2])
        for ii in range(1):
            M.children[ii,ii].refine()
        self.M = M

    def test_MeshSizes(self):
        self.assertTrue(len(self.M.faces)==25)
        self.assertTrue(len(self.M.cells)==9)



if __name__ == '__main__':
    unittest.main()
