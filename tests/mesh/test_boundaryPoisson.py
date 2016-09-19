from __future__ import print_function
import numpy as np
import scipy.sparse as sp
import unittest
import matplotlib.pyplot as plt
from SimPEG import Mesh, Utils, Tests, Solver, SolverCG

MESHTYPES = ['uniformTensorMesh']

class Test1D_InhomogeneousDirichlet(Tests.OrderTest):
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

        xc_ana = phi(self.M.gridCC)
        q_ana = q_fun(self.M.gridCC)
        j_ana = j_fun(self.M.gridFx)

        #TODO: Check where our boundary conditions are CCx or Nx
        # vec = self.M.vectorNx
        vec = self.M.vectorCCx

        phi_bc = phi(vec[[0,-1]])
        j_bc = j_fun(vec[[0,-1]])

        P, Pin, Pout = self.M.getBCProjWF([['dirichlet', 'dirichlet']])

        Mc = self.M.getFaceInnerProduct()
        McI = Utils.sdInv(self.M.getFaceInnerProduct())
        V = Utils.sdiag(self.M.vol)
        G = -Pin.T*Pin*self.M.faceDiv.T * V
        D = self.M.faceDiv
        j = McI*(G*xc_ana + P*phi_bc)
        q = V*D*Pin.T*Pin*j + V*D*Pout.T*j_bc

        # Rearrange if we know q to solve for x
        A = V*D*Pin.T*Pin*McI*G
        rhs = V*q_ana - V*D*Pin.T*Pin*McI*P*phi_bc - V*D*Pout.T*j_bc
        # A = D*McI*G
        # rhs = q_ana - D*McI*P*phi_bc


        if self.myTest == 'j':
            err = np.linalg.norm((j-j_ana), np.inf)
        elif self.myTest == 'q':
            err = np.linalg.norm((q-V*q_ana), np.inf)
        elif self.myTest == 'xc':
            #TODO: fix the null space
            solver = SolverCG(A, maxiter=1000)
            xc = solver * (rhs)
            print('ACCURACY', np.linalg.norm(Utils.mkvc(A*xc) - rhs))
            err = np.linalg.norm((xc-xc_ana), np.inf)
        elif self.myTest == 'xcJ':
            #TODO: fix the null space
            xc = Solver(A) * (rhs)
            print(np.linalg.norm(Utils.mkvc(A*xc) - rhs))
            j = McI*(G*xc + P*phi_bc)
            err = np.linalg.norm((j-j_ana), np.inf)

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


class Test2D_InhomogeneousDirichlet(Tests.OrderTest):
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

        xc_ana = phi(self.M.gridCC)
        q_ana = q_fun(self.M.gridCC)
        jX_ana = j_funX(self.M.gridFx)
        jY_ana = j_funY(self.M.gridFy)
        j_ana = np.r_[jX_ana,jY_ana]

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
        j = McI*(G*xc_ana + P*bc)
        q = D*j

        # self.M.plotImage(j, 'FxFy', showIt=True)

        # Rearrange if we know q to solve for x
        A = D*McI*G
        rhs = q_ana - D*McI*P*bc

        if self.myTest == 'j':
            err = np.linalg.norm((j-j_ana), np.inf)
        elif self.myTest == 'q':
            err = np.linalg.norm((q-q_ana), np.inf)
        elif self.myTest == 'xc':
            xc = Solver(A) * (rhs)
            err = np.linalg.norm((xc-xc_ana), np.inf)
        elif self.myTest == 'xcJ':
            xc = Solver(A) * (rhs)
            j = McI*(G*xc + P*bc)
            err = np.linalg.norm((j-j_ana), np.inf)

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

class Test1D_InhomogeneousNeumann(Tests.OrderTest):
    name = "1D - Neumann"
    meshTypes = MESHTYPES
    meshDimension = 1
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32, 64, 128]

    def getError(self):
        #Test function
        phi = lambda x: np.sin(np.pi*x)
        j_fun = lambda x: np.pi*np.cos(np.pi*x)
        q_fun = lambda x: -(np.pi**2)*np.sin(np.pi*x)

        xc_ana = phi(self.M.gridCC)
        q_ana = q_fun(self.M.gridCC)
        j_ana = j_fun(self.M.gridFx)

        #TODO: Check where our boundary conditions are CCx or Nx
        vecN = self.M.vectorNx
        vecC = self.M.vectorCCx

        phi_bc = phi(vecC[[0,-1]])
        j_bc = j_fun(vecN[[0,-1]])

        P, Pin, Pout = self.M.getBCProjWF([['neumann', 'neumann']])

        Mc = self.M.getFaceInnerProduct()
        McI = Utils.sdInv(self.M.getFaceInnerProduct())
        V = Utils.sdiag(self.M.vol)
        G = -Pin.T*Pin*self.M.faceDiv.T * V
        D = self.M.faceDiv
        j = McI*(G*xc_ana + P*phi_bc)
        q = V*D*Pin.T*Pin*j + V*D*Pout.T*j_bc

        # Rearrange if we know q to solve for x
        A = V*D*Pin.T*Pin*McI*G
        rhs = V*q_ana - V*D*Pin.T*Pin*McI*P*phi_bc - V*D*Pout.T*j_bc
        # A = D*McI*G
        # rhs = q_ana - D*McI*P*phi_bc


        if self.myTest == 'j':
            err = np.linalg.norm((Pin*j-Pin*j_ana), np.inf)
        elif self.myTest == 'q':
            err = np.linalg.norm((q-V*q_ana), np.inf)
        elif self.myTest == 'xc':
            #TODO: fix the null space
            xc, info = sp.linalg.minres(A, rhs, tol = 1e-6)
            err = np.linalg.norm((xc-xc_ana), np.inf)
            if info > 0:
                print('Solve does not work well')
                print('ACCURACY', np.linalg.norm(Utils.mkvc(A*xc) - rhs))
        elif self.myTest == 'xcJ':
            #TODO: fix the null space
            xc, info = sp.linalg.minres(A, rhs, tol = 1e-6)
            j = McI*(G*xc + P*phi_bc)
            err = np.linalg.norm((Pin*j-Pin*j_ana), np.inf)
            if info > 0:
                print('Solve does not work well')
                print('ACCURACY', np.linalg.norm(Utils.mkvc(A*xc) - rhs))
        return err

    def test_orderJ(self):
        self.name = "1D - InhomogeneousNeumann_Forward j"
        self.myTest = 'j'
        self.orderTest()

    def test_orderQ(self):
        self.name = "1D - InhomogeneousNeumann_Forward q"
        self.myTest = 'q'
        self.orderTest()

    def test_orderXJ(self):
        self.name = "1D - InhomogeneousNeumann_Inverse J"
        self.myTest = 'xcJ'
        self.orderTest()

class Test2D_InhomogeneousNeumann(Tests.OrderTest):
    name = "2D - Neumann"
    meshTypes = MESHTYPES
    meshDimension = 2
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32]
    # meshSizes = [4]

    def getError(self):
        #Test function
        phi = lambda x: np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1])
        j_funX = lambda x: np.pi*np.cos(np.pi*x[:,0])*np.sin(np.pi*x[:,1])
        j_funY = lambda x: np.pi*np.sin(np.pi*x[:,0])*np.cos(np.pi*x[:,1])
        q_fun = lambda x: -2*(np.pi**2)*phi(x)

        xc_ana = phi(self.M.gridCC)
        q_ana = q_fun(self.M.gridCC)
        jX_ana = j_funX(self.M.gridFx)
        jY_ana = j_funY(self.M.gridFy)
        j_ana = np.r_[jX_ana,jY_ana]

        #TODO: Check where our boundary conditions are CCx or Nx

        cxm,cxp,cym,cyp = self.M.cellBoundaryInd
        fxm,fxp,fym,fyp = self.M.faceBoundaryInd

        gBFx = self.M.gridFx[(fxm|fxp),:]
        gBFy = self.M.gridFy[(fym|fyp),:]

        gBCx = self.M.gridCC[(cxm|cxp),:]
        gBCy = self.M.gridCC[(cym|cyp),:]

        phi_bc = phi(np.r_[gBFx,gBFy])
        j_bc = np.r_[j_funX(gBFx), j_funY(gBFy)]

        # P = sp.csr_matrix(([-1,1],([0,self.M.nF-1],[0,1])), shape=(self.M.nF, 2))

        P, Pin, Pout = self.M.getBCProjWF('neumann')

        Mc = self.M.getFaceInnerProduct()
        McI = Utils.sdInv(self.M.getFaceInnerProduct())
        V = Utils.sdiag(self.M.vol)
        G = -Pin.T*Pin*self.M.faceDiv.T * V
        D = self.M.faceDiv
        j = McI*(G*xc_ana + P*phi_bc)
        q = V*D*Pin.T*Pin*j + V*D*Pout.T*j_bc

        # Rearrange if we know q to solve for x
        A = V*D*Pin.T*Pin*McI*G
        rhs = V*q_ana - V*D*Pin.T*Pin*McI*P*phi_bc - V*D*Pout.T*j_bc

        if self.myTest == 'j':
            err = np.linalg.norm((Pin*j-Pin*j_ana), np.inf)
        elif self.myTest == 'q':
            err = np.linalg.norm((q-V*q_ana), np.inf)
        elif self.myTest == 'xc':
            #TODO: fix the null space
            xc, info = sp.linalg.minres(A, rhs, tol = 1e-6)
            err = np.linalg.norm((xc-xc_ana), np.inf)
            if info > 0:
                print('Solve does not work well')
                print('ACCURACY', np.linalg.norm(Utils.mkvc(A*xc) - rhs))
        elif self.myTest == 'xcJ':
            #TODO: fix the null space
            xc, info = sp.linalg.minres(A, rhs, tol = 1e-6)
            j = McI*(G*xc + P*phi_bc)
            err = np.linalg.norm((Pin*j-Pin*j_ana), np.inf)
            if info > 0:
                print('Solve does not work well')
                print('ACCURACY', np.linalg.norm(Utils.mkvc(A*xc) - rhs))
        return err

    def test_orderJ(self):
        self.name = "2D - InhomogeneousNeumann_Forward j"
        self.myTest = 'j'
        self.orderTest()

    def test_orderQ(self):
        self.name = "2D - InhomogeneousNeumann_Forward q"
        self.myTest = 'q'
        self.orderTest()

    def test_orderXJ(self):
        self.name = "2D - InhomogeneousNeumann_Inverse J"
        self.myTest = 'xcJ'
        self.orderTest()

class Test1D_InhomogeneousMixed(Tests.OrderTest):
    name = "1D - Mixed"
    meshTypes = MESHTYPES
    meshDimension = 1
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32, 64, 128]

    def getError(self):
        #Test function
        phi = lambda x: np.cos(0.5*np.pi*x)
        j_fun = lambda x: -0.5*np.pi*np.sin(0.5*np.pi*x)
        q_fun = lambda x: -0.25*(np.pi**2)*np.cos(0.5*np.pi*x)

        xc_ana = phi(self.M.gridCC)
        q_ana = q_fun(self.M.gridCC)
        j_ana = j_fun(self.M.gridFx)

        #TODO: Check where our boundary conditions are CCx or Nx
        vecN = self.M.vectorNx
        vecC = self.M.vectorCCx

        phi_bc = phi(vecC[[0,-1]])
        j_bc = j_fun(vecN[[0,-1]])

        P, Pin, Pout = self.M.getBCProjWF([['dirichlet', 'neumann']])

        Mc = self.M.getFaceInnerProduct()
        McI = Utils.sdInv(self.M.getFaceInnerProduct())
        V = Utils.sdiag(self.M.vol)
        G = -Pin.T*Pin*self.M.faceDiv.T * V
        D = self.M.faceDiv
        j = McI*(G*xc_ana + P*phi_bc)
        q = V*D*Pin.T*Pin*j + V*D*Pout.T*j_bc

        # Rearrange if we know q to solve for x
        A = V*D*Pin.T*Pin*McI*G
        rhs = V*q_ana - V*D*Pin.T*Pin*McI*P*phi_bc - V*D*Pout.T*j_bc
        # A = D*McI*G
        # rhs = q_ana - D*McI*P*phi_bc


        if self.myTest == 'j':
            err = np.linalg.norm((Pin*j-Pin*j_ana), np.inf)
        elif self.myTest == 'q':
            err = np.linalg.norm((q-V*q_ana), np.inf)
        elif self.myTest == 'xc':
            #TODO: fix the null space
            xc, info = sp.linalg.minres(A, rhs, tol = 1e-6)
            err = np.linalg.norm((xc-xc_ana), np.inf)
            if info > 0:
                print('Solve does not work well')
                print('ACCURACY', np.linalg.norm(Utils.mkvc(A*xc) - rhs))
        elif self.myTest == 'xcJ':
            #TODO: fix the null space
            xc, info = sp.linalg.minres(A, rhs, tol = 1e-6)
            j = McI*(G*xc + P*phi_bc)
            err = np.linalg.norm((Pin*j-Pin*j_ana), np.inf)
            if info > 0:
                print('Solve does not work well')
                print('ACCURACY', np.linalg.norm(Utils.mkvc(A*xc) - rhs))
        return err

    def test_orderJ(self):
        self.name = "1D - InhomogeneousMixed_Forward j"
        self.myTest = 'j'
        self.orderTest()

    def test_orderQ(self):
        self.name = "1D - InhomogeneousMixed_Forward q"
        self.myTest = 'q'
        self.orderTest()

    def test_orderXJ(self):
        self.name = "1D - InhomogeneousMixed_Inverse J"
        self.myTest = 'xcJ'
        self.orderTest()

class Test2D_InhomogeneousMixed(Tests.OrderTest):
    name = "2D - Mixed"
    meshTypes = MESHTYPES
    meshDimension = 2
    expectedOrders = 2
    meshSizes = [2, 4, 8, 16]
    # meshSizes = [4]

    def getError(self):
        #Test function
        phi = lambda x: np.cos(0.5*np.pi*x[:,0])*np.cos(0.5*np.pi*x[:,1])
        j_funX = lambda x: -0.5*np.pi*np.sin(0.5*np.pi*x[:,0])*np.cos(0.5*np.pi*x[:,1])
        j_funY = lambda x: -0.5*np.pi*np.cos(0.5*np.pi*x[:,0])*np.sin(0.5*np.pi*x[:,1])
        q_fun = lambda x: -2*((0.5*np.pi)**2)*phi(x)

        xc_ana = phi(self.M.gridCC)
        q_ana = q_fun(self.M.gridCC)
        jX_ana = j_funX(self.M.gridFx)
        jY_ana = j_funY(self.M.gridFy)
        j_ana = np.r_[jX_ana,jY_ana]

        #TODO: Check where our boundary conditions are CCx or Nx

        cxm,cxp,cym,cyp = self.M.cellBoundaryInd
        fxm,fxp,fym,fyp = self.M.faceBoundaryInd

        gBFx = self.M.gridFx[(fxm|fxp),:]
        gBFy = self.M.gridFy[(fym|fyp),:]

        gBCx = self.M.gridCC[(cxm|cxp),:]
        gBCy = self.M.gridCC[(cym|cyp),:]

        phi_bc = phi(np.r_[gBCx,gBCy])
        j_bc = np.r_[j_funX(gBFx), j_funY(gBFy)]

        # P = sp.csr_matrix(([-1,1],([0,self.M.nF-1],[0,1])), shape=(self.M.nF, 2))

        P, Pin, Pout = self.M.getBCProjWF([['dirichlet', 'neumann'], ['dirichlet', 'neumann']])

        Mc = self.M.getFaceInnerProduct()
        McI = Utils.sdInv(self.M.getFaceInnerProduct())
        V = Utils.sdiag(self.M.vol)
        G = -Pin.T*Pin*self.M.faceDiv.T * V
        D = self.M.faceDiv
        j = McI*(G*xc_ana + P*phi_bc)
        q = V*D*Pin.T*Pin*j + V*D*Pout.T*j_bc

        # Rearrange if we know q to solve for x
        A = V*D*Pin.T*Pin*McI*G
        rhs = V*q_ana - V*D*Pin.T*Pin*McI*P*phi_bc - V*D*Pout.T*j_bc

        if self.myTest == 'j':
            err = np.linalg.norm((Pin*j-Pin*j_ana), np.inf)
        elif self.myTest == 'q':
            err = np.linalg.norm((q-V*q_ana), np.inf)
        elif self.myTest == 'xc':
            #TODO: fix the null space
            xc, info = sp.linalg.minres(A, rhs, tol = 1e-6)
            err = np.linalg.norm((xc-xc_ana), np.inf)
            if info > 0:
                print('Solve does not work well')
                print('ACCURACY', np.linalg.norm(Utils.mkvc(A*xc) - rhs))
        elif self.myTest == 'xcJ':
            #TODO: fix the null space
            xc, info = sp.linalg.minres(A, rhs, tol = 1e-6)
            j = McI*(G*xc + P*phi_bc)
            err = np.linalg.norm((Pin*j-Pin*j_ana), np.inf)
            if info > 0:
                print('Solve does not work well')
                print('ACCURACY', np.linalg.norm(Utils.mkvc(A*xc) - rhs))
        return err

    def test_orderJ(self):
        self.name = "2D - InhomogeneousMixed_Forward j"
        self.myTest = 'j'
        self.orderTest()

    def test_orderQ(self):
        self.name = "2D - InhomogeneousMixed_Forward q"
        self.myTest = 'q'
        self.orderTest()

    def test_orderXJ(self):
        self.name = "2D - InhomogeneousMixed_Inverse J"
        self.myTest = 'xcJ'
        self.orderTest()

if __name__ == '__main__':
    unittest.main()
