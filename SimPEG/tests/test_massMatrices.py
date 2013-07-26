import numpy as np
import unittest
import sys
sys.path.append('../')
from TensorMesh import TensorMesh
from OrderTest import OrderTest
from scipy.sparse.linalg import dsolve
from getEdgeInnerProducts import getEdgeInnerProducts


class TestNodalGrad(OrderTest):
    name = "Nodal Gradient"
    
    meshSizes = [4, 8, 16, 32]

    def getError(self):
        ex = lambda x, y, z: x**2+y*z
        ey = lambda x, y, z: (z**2)*x+y*z
        ez = lambda x, y, z: y**2+x*z

        sigma1 = lambda x, y, z: x*y+1
        sigma2 = lambda x, y, z: x*z+2
        sigma3 = lambda x, y, z: 3+z*y
        sigma4 = lambda x, y, z: 0.1*x*y*z
        sigma5 = lambda x, y, z: 0.2*x*y
        sigma6 = lambda x, y, z: 0.1*z
        
        Ex = ex(self.M.gridEx[:, 0],self.M.gridEx[:, 1],self.M.gridEx[:, 2])
        Ey = ey(self.M.gridEy[:, 0],self.M.gridEy[:, 1],self.M.gridEy[:, 2])
        Ez = ez(self.M.gridEz[:, 0],self.M.gridEz[:, 1],self.M.gridEz[:, 2])
        
        E = np.r_[Ex,Ey,Ez]
        Gc = self.M.gridCC
        sigma = np.c_[sigma1(Gc[:,0],Gc[:,1],Gc[:,2]),
                      sigma2(Gc[:,0],Gc[:,1],Gc[:,2]),
                      sigma3(Gc[:,0],Gc[:,1],Gc[:,2]),
                      sigma4(Gc[:,0],Gc[:,1],Gc[:,2]),
                      sigma5(Gc[:,0],Gc[:,1],Gc[:,2]),
                      sigma6(Gc[:,0],Gc[:,1],Gc[:,2])]

        A = getEdgeInnerProducts(self.M, sigma)

        err = np.abs(E.T*A*E - 69881./21600)

        return err

    def test_order(self):
        self.orderTest()
