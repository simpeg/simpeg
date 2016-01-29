import numpy as np
import unittest
from SimPEG import *
from scipy.sparse.linalg import dsolve
import inspect

TOL = 1e-20

class RegularizationTests(unittest.TestCase):

    def setUp(self):
        hx, hy, hz = np.random.rand(10), np.random.rand(9), np.random.rand(8)
        hx, hy, hz = hx/hx.sum(), hy/hy.sum(), hz/hz.sum()
        mesh1 = Mesh.TensorMesh([hx])
        mesh2 = Mesh.TensorMesh([hx, hy])
        mesh3 = Mesh.TensorMesh([hx, hy, hz])
        self.meshlist = [mesh1,mesh2, mesh3]

    def test_regularization(self):
        for R in dir(Regularization):
            r = getattr(Regularization, R)
            if not inspect.isclass(r): continue
            if not issubclass(r, Regularization.BaseRegularization):
                continue

            for i, mesh in enumerate(self.meshlist):

                print 'Testing %iD'%mesh.dim

                mapping = r.mapPair(mesh)
                reg = r(mesh, mapping=mapping)
                m = np.random.rand(mapping.nP)
                reg.mref = np.ones_like(m)*np.mean(m)

                print 'Check: phi_m (mref) = %f' %reg.eval(reg.mref)
                passed = reg.eval(reg.mref) < TOL
                self.assertTrue(passed)

                print 'Check:', R
                passed = Tests.checkDerivative(lambda m : [reg.eval(m), reg.evalDeriv(m)], m, plotIt=False)
                self.assertTrue(passed)

                print 'Check 2 Deriv:', R
                passed = Tests.checkDerivative(lambda m : [reg.evalDeriv(m), reg.eval2Deriv(m)], m, plotIt=False)
                self.assertTrue(passed)

    def test_regularization_ActiveCells(self):
        for R in dir(Regularization):
            r = getattr(Regularization, R)
            if not inspect.isclass(r): continue
            if not issubclass(r, Regularization.BaseRegularization):
                continue

            for i, mesh in enumerate(self.meshlist):

                print 'Testing Active Cells %iD'%(mesh.dim)

                if mesh.dim == 1:
                    indAct = Utils.mkvc(mesh.gridCC <= 0.8)
                elif mesh.dim == 2:
                    indAct = Utils.mkvc(mesh.gridCC[:,-1] <= 2*np.sin(2*np.pi*mesh.gridCC[:,0])+0.5)
                elif mesh.dim == 3:
                    indAct = Utils.mkvc(mesh.gridCC[:,-1] <= 2*np.sin(2*np.pi*mesh.gridCC[:,0])+0.5 * 2*np.sin(2*np.pi*mesh.gridCC[:,1])+0.5)

                mapping = Maps.IdentityMap(nP=indAct.nonzero()[0].size)

                reg = r(mesh, mapping=mapping, indActive=indAct)
                m = np.random.rand(mesh.nC)[indAct]
                reg.mref = np.ones_like(m)*np.mean(m)

                print 'Check: phi_m (mref) = %f' %reg.eval(reg.mref)
                passed = reg.eval(reg.mref) < TOL
                self.assertTrue(passed)

                print 'Check:', R
                passed = Tests.checkDerivative(lambda m : [reg.eval(m), reg.evalDeriv(m)], m, plotIt=False)
                self.assertTrue(passed)

                print 'Check 2 Deriv:', R
                passed = Tests.checkDerivative(lambda m : [reg.evalDeriv(m), reg.eval2Deriv(m)], m, plotIt=False)
                self.assertTrue(passed)


if __name__ == '__main__':
    unittest.main()
