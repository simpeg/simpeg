from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest
from SimPEG import Mesh, Maps, Regularization, Utils, Tests, ObjectiveFunction
from scipy.sparse.linalg import dsolve
import inspect

TOL = 1e-7
testReg = True
testRegMesh = True

np.random.seed(639)

IGNORE_ME = [
    'BaseRegularization',
    'BaseComboRegularization',
    'BaseSparse'
]


class RegularizationTests(unittest.TestCase):

    def setUp(self):
        hx, hy, hz = np.random.rand(10), np.random.rand(9), np.random.rand(8)
        hx, hy, hz = hx/hx.sum(), hy/hy.sum(), hz/hz.sum()
        mesh1 = Mesh.TensorMesh([hx])
        mesh2 = Mesh.TensorMesh([hx, hy])
        mesh3 = Mesh.TensorMesh([hx, hy, hz])
        self.meshlist = [mesh1, mesh2, mesh3]

    if testReg:
        def test_regularization(self):
            for R in dir(Regularization):
                r = getattr(Regularization, R)
                if not inspect.isclass(r):
                    continue
                if not issubclass(r, ObjectiveFunction.BaseObjectiveFunction):
                    continue
                if r.__name__ in IGNORE_ME:
                    continue

                for i, mesh in enumerate(self.meshlist):

                    if mesh.dim < 3 and r.__name__[-1] == 'z':
                        continue
                    if mesh.dim < 2 and r.__name__[-1] == 'y':
                        continue

                    print('Testing {0:d}D'.format(mesh.dim))

                    mapping = Maps.IdentityMap(mesh)
                    reg = r(mesh=mesh, mapping=mapping)

                    print(
                        '--- Checking {} --- \n'.format(reg.__class__.__name__)
                    )

                    if mapping.nP != '*':
                        m = np.random.rand(mapping.nP)
                    else:
                        m = np.random.rand(mesh.nC)
                    mref = np.ones_like(m)*np.mean(m)
                    reg.mref = mref

                    # test derivs
                    passed = reg.test(m, eps=TOL)
                    self.assertTrue(passed)


        def test_regularization_ActiveCells(self):
            for R in dir(Regularization):
                r = getattr(Regularization, R)
                if not inspect.isclass(r):
                    continue
                if not issubclass(r, ObjectiveFunction.BaseObjectiveFunction):
                    continue
                if r.__name__ in IGNORE_ME:
                    continue

                for i, mesh in enumerate(self.meshlist):

                    print('Testing Active Cells {0:d}D'.format((mesh.dim)))

                    if mesh.dim == 1:
                        indActive = Utils.mkvc(mesh.gridCC <= 0.8)
                    elif mesh.dim == 2:
                        indActive = Utils.mkvc(mesh.gridCC[:, -1] <= (
                            2*np.sin(2*np.pi*mesh.gridCC[:, 0])+0.5)
                        )
                    elif mesh.dim == 3:
                        indActive = Utils.mkvc(mesh.gridCC[:, -1] <= (
                                2 * np.sin(2*np.pi*mesh.gridCC[:, 0])+0.5 *
                                2 * np.sin(2*np.pi*mesh.gridCC[:, 1])+0.5)
                            )

                    if mesh.dim < 3 and r.__name__[-1] == 'z':
                        continue
                    if mesh.dim < 2 and r.__name__[-1] == 'y':
                        continue

                    for indAct in [indActive, indActive.nonzero()[0]]: # test both bool and integers
                        reg = r(mesh, indActive=indAct)
                        m = np.random.rand(mesh.nC)[indAct]
                        mref = np.ones_like(m)*np.mean(m)
                        reg.mref = mref

                        print(
                                '--- Checking {} ---\n'.format(
                                    reg.__class__.__name__
                                )
                            )

                        passed = reg.test(m, eps=TOL)
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
                    indAct = (
                        Utils.mkvc(
                            mesh.gridCC[:,-1] <=
                            2*np.sin(2*np.pi*mesh.gridCC[:, 0]) + 0.5
                        )
                    )
                elif mesh.dim == 3:
                    indAct = (
                        Utils.mkvc(
                            mesh.gridCC[:, -1] <=
                            2*np.sin(2*np.pi*mesh.gridCC[:, 0]) +
                            0.5 * 2*np.sin(2*np.pi*mesh.gridCC[:, 1]) + 0.5
                        )
                    )

                regmesh = Regularization.RegularizationMesh(
                    mesh, indActive=indAct
                )

                assert (regmesh.vol == mesh.vol[indAct]).all()

    def test_property_mirroring(self):
        mesh = Mesh.TensorMesh([8, 7, 6])

        for regType in ['Tikhonov', 'Sparse', 'Simple']:
            reg = getattr(Regularization, regType)(mesh)

            self.assertTrue(reg.nP == mesh.nC)

            # Test assignment of active indices
            indActive = mesh.gridCC[:, 2] < 0.6
            reg.indActive = indActive

            self.assertTrue(reg.nP == int(indActive.sum()))

            [
                self.assertTrue(np.all(fct.indActive == indActive))
                for fct in reg.objfcts
            ]

            # test assignment of cell weights
            cell_weights = np.random.rand(indActive.sum())
            reg.cell_weights = cell_weights
            [
                self.assertTrue(np.all(fct.cell_weights == cell_weights))
                for fct in reg.objfcts
            ]

            # test updated mappings
            mapping = Maps.ExpMap(nP=indActive.sum())
            reg.mapping = mapping
            m = np.random.rand(mapping.nP)
            [
                self.assertTrue(np.all(fct.mapping * m == mapping * m))
                for fct in reg.objfcts
            ]

            # test alphas
            m = np.random.rand(reg.nP)
            a = reg(m)
            [
                setattr(
                    reg, '{}'.format(objfct._multiplier_pair),
                    0.5*getattr(reg, '{}'.format(objfct._multiplier_pair))
                )
                for objfct in reg.objfcts
            ]
            b = reg(m)
            self.assertTrue(0.5*a == b)

    def test_addition(self):
        mesh = Mesh.TensorMesh([8, 7, 6])
        m = np.random.rand(mesh.nC)

        reg1 = Regularization.Tikhonov(mesh)
        reg2 = Regularization.Simple(mesh)

        reg_a = reg1 + reg2
        self.assertTrue(len(reg_a)==2)
        self.assertTrue(reg1(m) + reg2(m) == reg_a(m))
        reg_a.test(eps=TOL)

        reg_b = 2*reg1 + reg2
        self.assertTrue(len(reg_b)==2)
        self.assertTrue(2*reg1(m) + reg2(m) == reg_b(m))
        reg_b.test(eps=TOL)

        reg_c = reg1 + reg2/2
        self.assertTrue(len(reg_c)==2)
        self.assertTrue(reg1(m) + 0.5*reg2(m) == reg_c(m))
        reg_c.test(eps=TOL)

    def test_mappings(self):
        mesh = Mesh.TensorMesh([8, 7, 6])
        m = np.random.rand(2*mesh.nC)

        wires = Maps.Wires(('sigma', mesh.nC), ('mu', mesh.nC))

        for regType in ['Tikhonov', 'Sparse', 'Simple']:
            reg1 = getattr(Regularization, regType)(mesh, mapping=wires.sigma)
            reg2 = getattr(Regularization, regType)(mesh, mapping=wires.mu)

            reg3 = reg1 + reg2

            self.assertTrue(reg1.nP == 2*mesh.nC)
            self.assertTrue(reg2.nP == 2*mesh.nC)
            self.assertTrue(reg3.nP == 2*mesh.nC)

            print(reg3(m), reg1(m), reg2(m))
            self.assertTrue(reg3(m) == reg1(m) + reg2(m))

            reg1.test(eps=TOL)
            reg2.test(eps=TOL)
            reg3.test(eps=TOL)

    def test_mref_is_zero(self):

        mesh = Mesh.TensorMesh([10, 5, 8])
        mref = np.ones(mesh.nC)

        for regType in ['Tikhonov', 'Sparse', 'Simple']:
            reg = getattr(Regularization, regType)(mesh, mref=mref)

            print('Check: phi_m (mref) = {0:f}'.format(reg(mref)))
            passed = reg(mref) < TOL
            self.assertTrue(passed)

    def test_mappings_and_cell_weights(self):
        mesh = Mesh.TensorMesh([8, 7, 6])
        m = np.random.rand(2*mesh.nC)
        v = np.random.rand(2*mesh.nC)

        cell_weights = np.random.rand(mesh.nC)

        wires = Maps.Wires(('sigma', mesh.nC), ('mu', mesh.nC))

        reg = Regularization.SimpleSmall(
            mesh, mapping=wires.sigma, cell_weights=cell_weights
        )

        objfct = ObjectiveFunction.L2ObjectiveFunction(
            W=Utils.sdiag(np.sqrt(cell_weights)), mapping=wires.sigma
        )

        self.assertTrue(reg(m) == objfct(m))
        self.assertTrue(np.all(reg.deriv(m) == objfct.deriv(m)))
        self.assertTrue(np.all(reg.deriv2(m, v=v) == objfct.deriv2(m, v=v)))

    def test_update_of_sparse_norms(self):
        mesh = Mesh.TensorMesh([8, 7, 6])
        m = np.random.rand(mesh.nC)
        v = np.random.rand(mesh.nC)

        cell_weights = np.random.rand(mesh.nC)

        reg = Regularization.Sparse(
            mesh, cell_weights=cell_weights
        )
        self.assertTrue(np.all(reg.norms == [2., 2., 2., 2.]))
        self.assertTrue(reg.objfcts[0].norm == 2.)
        self.assertTrue(reg.objfcts[1].norm == 2.)
        self.assertTrue(reg.objfcts[2].norm == 2.)
        self.assertTrue(reg.objfcts[3].norm == 2.)

        reg.norms = [0., 1., 1., 1.]
        self.assertTrue(np.all(reg.norms == [0., 1., 1., 1.]))
        self.assertTrue(reg.objfcts[0].norm == 0.)
        self.assertTrue(reg.objfcts[1].norm == 1.)
        self.assertTrue(reg.objfcts[2].norm == 1.)
        self.assertTrue(reg.objfcts[3].norm == 1.)

if __name__ == '__main__':
    unittest.main()
