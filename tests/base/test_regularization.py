from __future__ import print_function
import numpy as np
import unittest
from SimPEG import Mesh, Regularization, Utils, Tests
from scipy.sparse.linalg import dsolve
import inspect

TOL = 1e-20
testReg = True
testRegMesh = True

testMeshes = ['Tensor', 'Tree']

np.random.seed(653)


class RegularizationTests(unittest.TestCase):

    def setUp(self):
        hx, hy, hz = np.random.rand(10), np.random.rand(9), np.random.rand(8)
        hx, hy, hz = hx/hx.sum(), hy/hy.sum(), hz/hz.sum()

        self.meshlist = []
        if 'Tensor' in testMeshes:
            mesh1 = Mesh.TensorMesh([hx])
            mesh2 = Mesh.TensorMesh([hx, hy])
            mesh3 = Mesh.TensorMesh([hx, hy, hz])
            self.meshlist += [mesh1, mesh2, mesh3]

        if 'Tree' in testMeshes:
            def refine(cell):
                xyz = cell.center
                for i in range(3):
                    if np.abs(np.sin(xyz[0]*np.pi*2)*0.5 + 0.5 - xyz[1]) < 0.2*i:
                        return 5-i
                return 0

            mesh2tree = Mesh.TreeMesh([16, 16])
            mesh2tree.refine(2)
            mesh2tree.refine(refine)

            mesh3tree = Mesh.TreeMesh([16, 16, 16])
            mesh3tree.refine(2)
            mesh3tree.refine(refine)

            self.meshlist += [mesh2tree, mesh3tree]

    if testReg:
        def test_regularization(self):
            for R in dir(Regularization):
                r = getattr(Regularization, R)
                if not inspect.isclass(r):
                    continue
                if not issubclass(r, Regularization.BaseRegularization):
                    continue

                for i, mesh in enumerate(self.meshlist):

                    print('Testing {0:d}D'.format(mesh.dim))

                    mapping = r.mapPair(mesh)
                    reg = r(mesh, mapping=mapping)
                    m = np.random.rand(mapping.nP)
                    reg.mref = np.ones_like(m)*np.mean(m)

                    print('Check: phi_m (mref) = {0:f}'.format(reg.eval(reg.mref)))
                    passed = reg.eval(reg.mref) < TOL
                    self.assertTrue(passed)

                    print('Check: {}'.format(R))
                    passed = Tests.checkDerivative(
                        lambda m: [reg.eval(m), reg.evalDeriv(m)], m,
                        plotIt=False
                    )
                    self.assertTrue(passed)

                    print('Check 2 Deriv: {}'.format(R))
                    passed = Tests.checkDerivative(
                        lambda m: [reg.evalDeriv(m), reg.eval2Deriv(m)], m,
                        plotIt=False
                    )
                    self.assertTrue(passed)

        def test_regularization_ActiveCells(self):
            for R in dir(Regularization):
                r = getattr(Regularization, R)
                if not inspect.isclass(r):
                    continue
                if not issubclass(r, Regularization.BaseRegularization):
                    continue

                for i, mesh in enumerate(self.meshlist):

                    print('Testing Active Cells {0:d}D'.format((mesh.dim)))

                    if mesh.dim == 1:
                        indActive = Utils.mkvc(mesh.gridCC <= 0.8)
                    elif mesh.dim == 2:
                        indActive = Utils.mkvc(
                            mesh.gridCC[:, -1] <=
                            2*np.sin(2*np.pi*mesh.gridCC[:, 0])+0.5
                        )
                    elif mesh.dim == 3:
                        indActive = Utils.mkvc(
                            mesh.gridCC[:, -1] <=
                            2*np.sin(2*np.pi*mesh.gridCC[:, 0])+0.5 *
                            2*np.sin(2*np.pi*mesh.gridCC[:, 1])+0.5
                        )

                    # test both bool and integers
                    for indAct in [indActive, indActive.nonzero()[0]]:
                        reg = r(mesh, indActive=indAct)
                        m = np.random.rand(mesh.nC)[indAct]
                        reg.mref = np.ones_like(m)*np.mean(m)

                    print(
                        'Check: phi_m (mref) = {0:f}'.format(
                            reg.eval(reg.mref)
                        )
                    )
                    passed = reg.eval(reg.mref) < TOL
                    self.assertTrue(passed)

                    print('Check:', R)
                    passed = Tests.checkDerivative(
                        lambda m : [reg.eval(m), reg.evalDeriv(m)], m,
                        plotIt=False
                    )
                    self.assertTrue(passed)

                    print('Check 2 Deriv:', R)
                    passed = Tests.checkDerivative(
                        lambda m : [reg.evalDeriv(m), reg.eval2Deriv(m)], m,
                        plotIt=False
                    )
                    self.assertTrue(passed)

    if testRegMesh:
        def test_regularizationMesh(self):

            for i, mesh in enumerate(self.meshlist):

                print('Testing {0:d}D'.format(mesh.dim))

                # mapping = r.mapPair(mesh)
                # reg = r(mesh, mapping=mapping)
                # m = np.random.rand(mapping.nP)

                if mesh.dim == 1:
                    indAct = Utils.mkvc(mesh.gridCC <= 0.8)
                elif mesh.dim == 2:
                    indAct = Utils.mkvc(
                        mesh.gridCC[:, -1] <=
                        2*np.sin(2*np.pi*mesh.gridCC[:, 0])+0.5
                    )
                elif mesh.dim == 3:
                    indAct = Utils.mkvc(
                        mesh.gridCC[:, -1] <=
                        2*np.sin(2*np.pi*mesh.gridCC[:, 0])+0.5 *
                        2*np.sin(2*np.pi*mesh.gridCC[:, 1])+0.5
                    )

                regmesh = Regularization.RegularizationMesh(
                    mesh, indActive=indAct
                )

                assert (regmesh.vol == mesh.vol[indAct]).all()


if __name__ == '__main__':
    unittest.main()
