import numpy as np
import unittest, os
import SimPEG as simpeg
from SimPEG.Mesh import TensorMesh, TreeMesh


class TestOcTreeIO(unittest.TestCase):

    def setUp(self):
        h = np.ones(16)
        mesh = simpeg.Mesh.TreeMesh([h,2*h,3*h])
        mesh.refine(3)
        mesh._refineCell([0,0,0,3])
        mesh._refineCell([0,2,0,3])
        self.mesh = mesh

    def test_UBCfiles(self):

        mesh = self.mesh
        # Make a vector
        vec = np.arange(mesh.nC)
        # Write aand read
        simpeg.Utils.meshutils.writeUBCocTreeFiles('temp.msh',mesh,{'arange.txt':vec})
        meshUBC, vecUBC = simpeg.Utils.meshutils.readUBCocTreeFiles('temp.msh',['arange.txt'])

        # The mesh
        assert mesh.__str__() == meshUBC.__str__()
        assert np.sum(mesh.gridCC - meshUBC.gridCC) == 0
        assert np.sum(vec - vecUBC) == 0
        assert np.all(np.array(mesh.h) - np.array(meshUBC.h) == 0)
        print 'IO of UBC octree files is working'
        os.remove('temp.msh')
        os.remove('arange.txt')

    def test_VTUfiles(self):
        mesh = self.mesh
        vec = np.arange(mesh.nC)
        try:
            simpeg.Utils.meshutils.writeVTUFile('temp.vtu',mesh,{'arange':vec})
            run = True
        except:
            run = False
        assert run
        print 'Writing of VTU files is working'
        os.remove('temp.vtu')



if __name__ == '__main__':
    unittest.main()
