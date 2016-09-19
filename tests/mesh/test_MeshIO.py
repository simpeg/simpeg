from __future__ import print_function
import numpy as np
import unittest
import os
import SimPEG as simpeg
from SimPEG.Mesh import TensorMesh, TreeMesh

try:
    import vtk
except ImportError:
    has_vtk=False
else:
    has_vtk=True



class TestTensorMeshIO(unittest.TestCase):

    def setUp(self):
        h = np.ones(16)
        mesh = TensorMesh([h, 2*h, 3*h])
        self.mesh = mesh

    def test_UBCfiles(self):

        mesh = self.mesh
        # Make a vector
        vec = np.arange(mesh.nC)
        # Write and read
        mesh.writeUBC('temp.msh', {'arange.txt': vec})
        meshUBC = TensorMesh.readUBC('temp.msh')
        vecUBC = meshUBC.readModelUBC('arange.txt')

        # The mesh
        assert mesh.__str__() == meshUBC.__str__()
        assert np.sum(mesh.gridCC - meshUBC.gridCC) == 0
        assert np.sum(vec - vecUBC) == 0
        assert np.all(np.array(mesh.h) - np.array(meshUBC.h) == 0)

        vecUBC = mesh.readModelUBC('arange.txt')
        assert np.sum(vec - vecUBC) == 0

        mesh.writeModelUBC('arange2.txt', vec + 1)
        vec2UBC = mesh.readModelUBC('arange2.txt')
        assert np.sum(vec + 1 - vec2UBC) == 0

        print('IO of UBC tensor mesh files is working')
        os.remove('temp.msh')
        os.remove('arange.txt')
        os.remove('arange2.txt')

    if has_vtk:
        def test_VTKfiles(self):
            mesh = self.mesh
            vec = np.arange(mesh.nC)

            mesh.writeVTK('temp.vtr', {'arange.txt': vec})
            meshVTR, models = TensorMesh.readVTK('temp.vtr')

            assert mesh.__str__() == meshVTR.__str__()
            assert np.all(np.array(mesh.h) - np.array(meshVTR.h) == 0)

            assert 'arange.txt' in models
            vecVTK = models['arange.txt']
            assert np.sum(vec - vecVTK) == 0

            print( 'IO of VTR tensor mesh files is working')
            os.remove('temp.vtr')



class TestOcTreeMeshIO(unittest.TestCase):

    def setUp(self):
        h = np.ones(16)
        mesh = TreeMesh([h, 2*h, 3*h])
        mesh.refine(3)
        mesh._refineCell([0, 0, 0, 3])
        mesh._refineCell([0, 2, 0, 3])
        self.mesh = mesh

    def test_UBCfiles(self):

        mesh = self.mesh
        # Make a vector
        vec = np.arange(mesh.nC)
        # Write and read
        mesh.writeUBC('temp.msh', {'arange.txt': vec})
        meshUBC = TreeMesh.readUBC('temp.msh')
        vecUBC = meshUBC.readModelUBC('arange.txt')

        # The mesh
        assert mesh.__str__() == meshUBC.__str__()
        assert np.sum(mesh.gridCC - meshUBC.gridCC) == 0
        assert np.sum(vec - vecUBC) == 0
        assert np.all(np.array(mesh.h) - np.array(meshUBC.h) == 0)
        print('IO of UBC octree files is working')
        os.remove('temp.msh')
        os.remove('arange.txt')

    if has_vtk:
        def test_VTUfiles(self):
            mesh = self.mesh
            vec = np.arange(mesh.nC)
            mesh.writeVTK('temp.vtu', {'arange': vec})
            print('Writing of VTU files is working')
            os.remove('temp.vtu')


if __name__ == '__main__':
    unittest.main()
