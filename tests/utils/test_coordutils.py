import unittest, os
import numpy as np
from SimPEG import Utils

tol = 1e-15

class coorUtilsTest(unittest.TestCase):

    def test_crossProd(self):
        self.assertTrue(np.linalg.norm(Utils.coordutils.crossProd(np.r_[1.,0.,0.],np.r_[0.,1.,0.]) - np.r_[0.,0.,1.]) < tol)
        self.assertTrue(np.linalg.norm(Utils.coordutils.crossProd(np.r_[0.,1.,0.],np.r_[0.,0.,1.]) - np.r_[1.,0.,0.]) < tol)
        self.assertTrue(np.linalg.norm(Utils.coordutils.crossProd(np.r_[0.,0.,1.],np.r_[1.,0.,0.]) - np.r_[0.,1.,0.]) < tol)

    def test_rotationMatrixFromNormals(self):
        v0 = np.random.rand(3)
        v0 *= 1./np.linalg.norm(v0)
        v1 = np.random.rand(3)
        v1 *= 1./np.linalg.norm(v1)
        Rf = Utils.coordutils.rotationMatrixFromNormals(v0,v1)
        Ri = Utils.coordutils.rotationMatrixFromNormals(v1,v0)

        self.assertTrue(np.linalg.norm(Utils.mkvc(Rf.dot(v0) - v1)) < tol)
        self.assertTrue(np.linalg.norm(Utils.mkvc(Ri.dot(v1) - v0)) < tol)

    def test_rotatePointsFromNormals(self):
        v0 = np.random.rand(3)
        v0*= 1./np.linalg.norm(v0)
        v1 = np.random.rand(3) 
        v1*= 1./np.linalg.norm(v1)   

        self.assertTrue(np.linalg.norm(Utils.mkvc(Utils.coordutils.rotatePointsFromNormals(Utils.mkvc(v0,2).T,v0,v1))-v1) < tol)

if __name__ == '__main__':
    unittest.main()

