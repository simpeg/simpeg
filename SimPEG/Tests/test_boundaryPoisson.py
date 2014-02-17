import numpy as np
import scipy.sparse as sp
import unittest
from TestUtils import OrderTest
import matplotlib.pyplot as plt
from SimPEG import Utils, Solver

MESHTYPES = ['uniformTensorMesh']

class Test1D_InhomogeneousDirichlet(OrderTest):
    name = "1D - Dirichlet"
    meshTypes = MESHTYPES
    meshDimension = 1
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32, 64, 128]

    def getError(self):
        #Test function
        phi = lambda x: np.cos(np.pi*x)
        j_fun = lambda x: -np.pi*np.sin(np.pi*x)
        q_fun = lambda x: -(np.pi**2)*np.cos(np.pi*x)

        xc_anal = phi(self.M.gridCC)
        q_anal = q_fun(self.M.gridCC)
        j_anal = j_fun(self.M.gridFx)

        #TODO: Check where our boundary conditions are CCx or Nx
        # vec = self.M.vectorNx
        vec = self.M.vectorCCx
        bc = phi(vec[[0,-1]])

        P, Pin, Pout = self.M.getBCProjWF([['dirichlet', 'dirichlet']])

        Mc = self.M.getFaceInnerProduct()
        McI = Utils.sdInv(self.M.getFaceInnerProduct())
        G = -self.M.faceDiv.T * Utils.sdiag(self.M.vol)
        D = self.M.faceDiv
        j = McI*(G*xc_anal + P*bc)
        q = D*j

        # Rearrange if we know q to solve for x
        A = D*McI*G
        rhs = q_anal - D*McI*P*bc


        if self.myTest == 'j':
            err = np.linalg.norm((j-j_anal), np.inf)
        elif self.myTest == 'q':
            err = np.linalg.norm((q-q_anal), np.inf)
        elif self.myTest == 'xc':
            xc = Solver(A).solve(rhs)
            err = np.linalg.norm((xc-xc_anal), np.inf)
        elif self.myTest == 'xcJ':
            xc = Solver(A).solve(rhs)
            j = McI*(G*xc + P*bc)
            err = np.linalg.norm((j-j_anal), np.inf)

        return err

    def test_orderJ(self):
        self.name = "1D - InhomogeneousDirichlet_Forward j"
        self.myTest = 'j'
        self.orderTest()

    def test_orderQ(self):
        self.name = "1D - InhomogeneousDirichlet_Forward q"
        self.myTest = 'q'
        self.orderTest()

    def test_orderX(self):
        self.name = "1D - InhomogeneousDirichlet_Inverse"
        self.myTest = 'xc'
        self.orderTest()

    def test_orderXJ(self):
        self.name = "1D - InhomogeneousDirichlet_Inverse J"
        self.myTest = 'xcJ'
        self.orderTest()


class Test2D_InhomogeneousDirichlet(OrderTest):
    name = "2D - Dirichlet"
    meshTypes = MESHTYPES
    meshDimension = 2
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32]

    def getError(self):
        #Test function
        phi = lambda x: np.cos(np.pi*x[:,0])*np.cos(np.pi*x[:,1])
        j_funX = lambda x: -np.pi*np.sin(np.pi*x[:,0])*np.cos(np.pi*x[:,1])
        j_funY = lambda x: -np.pi*np.cos(np.pi*x[:,0])*np.sin(np.pi*x[:,1])
        q_fun = lambda x: -2*(np.pi**2)*phi(x)

        xc_anal = phi(self.M.gridCC)
        q_anal = q_fun(self.M.gridCC)
        jX_anal = j_funX(self.M.gridFx)
        jY_anal = j_funY(self.M.gridFy)
        j_anal = np.r_[jX_anal,jY_anal]

        #TODO: Check where our boundary conditions are CCx or Nx
        # fxm,fxp,fym,fyp = self.M.faceBoundaryInd
        # gBFx = self.M.gridFx[(fxm|fxp),:]
        # gBFy = self.M.gridFy[(fym|fyp),:]
        fxm,fxp,fym,fyp = self.M.cellBoundaryInd
        gBFx = self.M.gridCC[(fxm|fxp),:]
        gBFy = self.M.gridCC[(fym|fyp),:]

        bc = phi(np.r_[gBFx,gBFy])

        # P = sp.csr_matrix(([-1,1],([0,self.M.nF-1],[0,1])), shape=(self.M.nF, 2))

        P, Pin, Pout = self.M.getBCProjWF('dirichlet')

        Mc = self.M.getFaceInnerProduct()
        McI = Utils.sdInv(self.M.getFaceInnerProduct())
        G = -self.M.faceDiv.T * Utils.sdiag(self.M.vol)
        D = self.M.faceDiv
        j = McI*(G*xc_anal + P*bc)
        q = D*j

        # self.M.plotImage(j, 'FxFy', showIt=True)

        # Rearrange if we know q to solve for x
        A = D*McI*G
        rhs = q_anal - D*McI*P*bc

        if self.myTest == 'j':
            err = np.linalg.norm((j-j_anal), np.inf)
        elif self.myTest == 'q':
            err = np.linalg.norm((q-q_anal), np.inf)
        elif self.myTest == 'xc':
            xc = Solver(A).solve(rhs)
            err = np.linalg.norm((xc-xc_anal), np.inf)
        elif self.myTest == 'xcJ':
            xc = Solver(A).solve(rhs)
            j = McI*(G*xc + P*bc)
            err = np.linalg.norm((j-j_anal), np.inf)

        return err

    def test_orderJ(self):
        self.name = "2D - InhomogeneousDirichlet_Forward j"
        self.myTest = 'j'
        self.orderTest()

    def test_orderQ(self):
        self.name = "2D - InhomogeneousDirichlet_Forward q"
        self.myTest = 'q'
        self.orderTest()

    def test_orderX(self):
        self.name = "2D - InhomogeneousDirichlet_Inverse"
        self.myTest = 'xc'
        self.orderTest()

    def test_orderXJ(self):
        self.name = "2D - InhomogeneousDirichlet_Inverse J"
        self.myTest = 'xcJ'
        self.orderTest()




if __name__ == '__main__':
    unittest.main()
