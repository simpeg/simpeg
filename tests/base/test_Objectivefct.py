from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import numpy as np
import scipy.sparse as sp
import unittest

from SimPEG import ObjectiveFunction, Utils, Maps

np.random.seed(130)

EPS = 1e-9


class Empty_ObjFct(ObjectiveFunction.BaseObjectiveFunction):

    def __init__(self):
        super(Empty_ObjFct, self).__init__()


class Props_ObjFct(ObjectiveFunction.BaseObjectiveFunction):

    def __init__(self, **kwargs):
        super(Props_ObjFct, self).__init__(**kwargs)

    @property
    def x(self):
        return getattr(self, '_x', None)

    @x.setter
    def x(self, val):
        self._x = val

    @property
    def y(self):
        return getattr(self, '_y', None)

    @y.setter
    def y(self, val):
        self._y = val

    @property
    def z(self):
        return getattr(self, '_z', None)

    @z.setter
    def z(self, val):
        self._z = val

class Error_if_Hit_ObjFct(ObjectiveFunction.BaseObjectiveFunction):

    def __init__(self):
        super(Error_if_Hit_ObjFct, self).__init__()

    def __call__(self, m):
        raise Exception('entered __call__')

    def deriv(self, m):
        raise Exception('entered deriv')

    def deriv2(self, m, v=None):
        raise Exception('entered deriv2')


class TestBaseObjFct(unittest.TestCase):

    def test_derivs(self):
        objfct = ObjectiveFunction.L2ObjectiveFunction()
        self.assertTrue(objfct.test(eps=1e-9))

    def test_scalarmul(self):
        scalar = 10.
        nP = 100
        objfct_a = ObjectiveFunction.L2ObjectiveFunction(
            W=Utils.sdiag(np.random.randn(nP))
        )
        objfct_b = scalar * objfct_a
        m = np.random.rand(nP)

        objfct_c = objfct_a + objfct_b

        self.assertTrue(scalar * objfct_a(m) == objfct_b(m))
        self.assertTrue(objfct_b.test())
        self.assertTrue(objfct_c(m) == objfct_a(m) + objfct_b(m))

        self.assertTrue(len(objfct_c.objfcts) == 2)
        self.assertTrue(len(objfct_c._multipliers) == 2)
        self.assertTrue(len(objfct_c) == 2)

    def test_sum(self):
        scalar = 10.
        nP = 100.
        objfct = (
            ObjectiveFunction.L2ObjectiveFunction(W=sp.eye(nP)) +
            scalar * ObjectiveFunction.L2ObjectiveFunction(W=sp.eye(nP))
        )
        self.assertTrue(objfct.test(eps=1e-9))

        self.assertTrue(np.all(objfct._multipliers == np.r_[1., scalar]))

    def test_2sum(self):
        nP = 80
        alpha1 = 100
        alpha2 = 200

        phi1 = (
            ObjectiveFunction.L2ObjectiveFunction(W=Utils.sdiag(np.random.rand(nP))) +
            alpha1 * ObjectiveFunction.L2ObjectiveFunction()
        )
        phi2 = ObjectiveFunction.L2ObjectiveFunction() + alpha2 * phi1
        self.assertTrue(phi2.test(eps=EPS))

        self.assertTrue(len(phi1._multipliers) == 2)
        self.assertTrue(len(phi2._multipliers) == 2)

        self.assertTrue(len(phi1.objfcts) == 2)
        self.assertTrue(len(phi2.objfcts) == 2)
        self.assertTrue(len(phi2) == 2)

        self.assertTrue(len(phi1) == 2)
        self.assertTrue(len(phi2) == 2)

        self.assertTrue(np.all(phi1._multipliers == np.r_[1., alpha1]))
        self.assertTrue(np.all(phi2._multipliers == np.r_[1., alpha2]))


    def test_3sum(self):
        nP = 90

        alpha1 = 0.3
        alpha2 = 0.6
        alpha3inv = 9

        phi1 = ObjectiveFunction.L2ObjectiveFunction(W=sp.eye(nP))
        phi2 = ObjectiveFunction.L2ObjectiveFunction(W=sp.eye(nP))
        phi3 = ObjectiveFunction.L2ObjectiveFunction(W=sp.eye(nP))

        phi = alpha1 * phi1 + alpha2 * phi2 + phi3 / alpha3inv

        m = np.random.rand(nP)

        self.assertTrue(
            np.all(phi._multipliers == np.r_[alpha1, alpha2, 1./alpha3inv])
        )

        self.assertTrue(
            alpha1*phi1(m) + alpha2*phi2(m) + phi3(m)/alpha3inv == phi(m)
        )

        self.assertTrue(len(phi.objfcts) == 3)

        self.assertTrue(phi.test())

    def test_sum_fail(self):
        nP1 = 10
        nP2 = 30

        phi1 = ObjectiveFunction.L2ObjectiveFunction(
                    W=Utils.sdiag(np.random.rand(nP1))
        )

        phi2 = ObjectiveFunction.L2ObjectiveFunction(
                    W=Utils.sdiag(np.random.rand(nP2))
                )

        with self.assertRaises(Exception):
            phi = phi1 + phi2

        with self.assertRaises(Exception):
            phi = phi1 + 100 * phi2

    def test_emptyObjFct(self):
        phi = Empty_ObjFct()
        x = np.random.rand(20)

        with self.assertRaises(NotImplementedError):
            phi(x)
            phi.deriv(x)
            phi.deriv2(x)

    def test_ZeroObjFct(self):
        # This is not a combo objective function, it will just give back an
        # L2 objective function. That might be ok? or should this be a combo
        # objective function?
        nP = 20
        alpha = 2.
        phi = alpha*(
            ObjectiveFunction.L2ObjectiveFunction(W = sp.eye(nP)) +
            Utils.Zero()*ObjectiveFunction.L2ObjectiveFunction()
        )
        self.assertTrue(len(phi.objfcts) == 1)
        self.assertTrue(phi.test())

    def test_updateMultipliers(self):
        nP = 10

        m = np.random.rand(nP)

        W1 = Utils.sdiag(np.random.rand(nP))
        W2 = Utils.sdiag(np.random.rand(nP))

        phi1 = ObjectiveFunction.L2ObjectiveFunction(W=W1)
        phi2 = ObjectiveFunction.L2ObjectiveFunction(W=W2)

        phi = phi1 + phi2

        self.assertTrue(phi(m) == phi1(m) + phi2(m))

        phi._multipliers[0] = Utils.Zero()
        self.assertTrue(phi(m) == phi2(m))

        phi._multipliers[0] = 1.
        phi._multipliers[1] = Utils.Zero()

        self.assertTrue(len(phi.objfcts) == 2)
        self.assertTrue(len(phi._multipliers) == 2)
        self.assertTrue(len(phi) == 2)

        self.assertTrue(phi(m) == phi1(m))

    def test_early_exits(self):
        nP = 10

        m = np.random.rand(nP)
        v = np.random.rand(nP)

        W1 = Utils.sdiag(np.random.rand(nP))
        phi1 = ObjectiveFunction.L2ObjectiveFunction(W=W1)

        phi2 = Error_if_Hit_ObjFct()

        objfct = phi1 + 0*phi2

        self.assertTrue(len(objfct) == 2)
        self.assertTrue(np.all(objfct._multipliers == np.r_[1, 0]))
        self.assertTrue(objfct(m) == phi1(m))
        self.assertTrue(np.all(objfct.deriv(m) == phi1.deriv(m)))
        self.assertTrue(np.all(objfct.deriv2(m, v) == phi1.deriv2(m, v)))

        objfct._multipliers[1] = Utils.Zero()

        self.assertTrue(len(objfct) == 2)
        self.assertTrue(np.all(objfct._multipliers == np.r_[1, 0]))
        self.assertTrue(objfct(m) == phi1(m))
        self.assertTrue(np.all(objfct.deriv(m) == phi1.deriv(m)))
        self.assertTrue(np.all(objfct.deriv2(m, v) == phi1.deriv2(m, v)))


    def test_Maps(self):
        nP = 10
        m = np.random.rand(2*nP)

        wires = Maps.Wires(('sigma', nP), ('mu', nP))

        objfct1 = ObjectiveFunction.L2ObjectiveFunction(mapping=wires.sigma)
        objfct2 = ObjectiveFunction.L2ObjectiveFunction(mapping=wires.mu)

        objfct3 = objfct1 + objfct2

        objfct4 = ObjectiveFunction.L2ObjectiveFunction(nP=nP)

        self.assertTrue(objfct1.nP == 2*nP)
        self.assertTrue(objfct2.nP == 2*nP)
        self.assertTrue(objfct3.nP == 2*nP)

        # print(objfct1.nP, objfct4.nP, objfct4.W.shape, objfct1.W.shape, m[:nP].shape)
        self.assertTrue(objfct1(m) == objfct4(m[:nP]))
        self.assertTrue(objfct2(m) == objfct4(m[nP:]))

        self.assertTrue(objfct3(m) == objfct1(m) + objfct2(m))

        objfct1.test()
        objfct2.test()
        objfct3.test()

    def test_ComboW(self):
        nP = 15
        m = np.random.rand(nP)

        phi1 = ObjectiveFunction.L2ObjectiveFunction(nP=nP)
        phi2 = ObjectiveFunction.L2ObjectiveFunction(nP=nP)

        alpha1 = 2.
        alpha2 = 0.5

        phi = alpha1*phi1 + alpha2*phi2

        r = phi.W * m

        r1 = phi1.W * m
        r2 = phi2.W * m

        print(phi(m), 0.5*np.inner(r, r))

        self.assertTrue(np.allclose(phi(m), 0.5*np.inner(r, r)))
        self.assertTrue(np.allclose(phi(m), 0.5*(
            alpha1*np.inner(r1, r1) + alpha2*np.inner(r2, r2))
        ))

    def test_ComboConstruction(self):
        nP = 10
        m = np.random.rand(nP)
        v = np.random.rand(nP)

        phi1 = ObjectiveFunction.L2ObjectiveFunction(nP=nP)
        phi2 = ObjectiveFunction.L2ObjectiveFunction(nP=nP)

        phi3 = 2*phi1 + 3*phi2

        phi4 = ObjectiveFunction.ComboObjectiveFunction(
            [phi1, phi2], [2, 3]
        )

        self.assertTrue(phi3(m) == phi4(m))
        self.assertTrue(np.all(phi3.deriv(m) == phi4.deriv(m)))
        self.assertTrue(np.all(phi3.deriv2(m, v) == phi4.deriv2(m, v)))


class ExposeTest(unittest.TestCase):

    def test_expose_mapping_string(self):
        nP = 10
        m = np.random.rand(nP)

        phi1 = ObjectiveFunction.L2ObjectiveFunction(nP=nP)
        phi2 = ObjectiveFunction.L2ObjectiveFunction(nP=nP)

        phi3 = 2*phi1 + 3*phi2

        phi3.expose('mapping')

        phi3.mapping = Maps.ExpMap(nP=10)

        # check that it is being propagated
        self.assertTrue(all([
            isinstance(objfct.mapping, Maps.ExpMap) for objfct in phi3.objfcts
        ]))

        phi1.mapping = Maps.IdentityMap(nP=10)
        self.assertTrue(
            phi3.objfcts[0].mapping.__class__.__name__ == 'IdentityMap'
        )
        self.assertTrue(
            phi3.objfcts[1].mapping.__class__.__name__ == 'ExpMap'
        )

    def test_expose_mapping_list(self):
        nP = 10
        m = np.random.rand(nP)

        phi1 = ObjectiveFunction.L2ObjectiveFunction(nP=nP)
        phi2 = ObjectiveFunction.L2ObjectiveFunction(nP=nP)

        phi3 = 2*phi1 + 3*phi2

        phi3.expose(['mapping'])

        phi3.mapping = Maps.LogMap(nP=10)

        # check that it is being propagated
        self.assertTrue(all([
            isinstance(objfct.mapping, Maps.LogMap) for objfct in phi3.objfcts
        ]))

        phi1.mapping = Maps.IdentityMap(nP=10)
        self.assertTrue(
            phi3.objfcts[0].mapping.__class__.__name__ == 'IdentityMap'
        )
        self.assertTrue(
            phi3.objfcts[1].mapping.__class__.__name__ == 'LogMap'
        )

    def test_expose_mapping_dict(self):
        nP = 10
        m = np.random.rand(nP)

        phi1 = ObjectiveFunction.L2ObjectiveFunction(nP=nP)
        phi2 = ObjectiveFunction.L2ObjectiveFunction(nP=nP)

        phi3 = 2*phi1 + 3*phi2

        phi3.expose({'mapping': Maps.ExpMap(nP=10)})

        # check that it is being propagated
        self.assertTrue(all([
            isinstance(objfct.mapping, Maps.ExpMap) for objfct in phi3.objfcts
        ]))

        phi1.mapping = Maps.IdentityMap(nP=10)
        self.assertTrue(
            phi3.objfcts[0].mapping.__class__.__name__ == 'IdentityMap'
        )
        self.assertTrue(
            phi3.objfcts[1].mapping.__class__.__name__ == 'ExpMap'
        )

    def test_not_allowed(self):
        nP = 10
        m = np.random.rand(nP)

        phi1 = Props_ObjFct(nP=nP)
        phi2 = ObjectiveFunction.L2ObjectiveFunction(nP=nP)

        phi1.x = 10.

        phi3 = 2*phi1 + 3*phi2

        with self.assertRaises(Exception):
            phi3.expose('W')
            phi3.expose('nP')

    def test_expose_all(self):
        nP = 10
        m = np.random.rand(nP)

        phi1 = Props_ObjFct(nP=nP)
        phi2 = ObjectiveFunction.L2ObjectiveFunction(nP=nP)

        phi1.x = 10.

        phi3 = 2*phi1 + 3*phi2

        phi3.expose('all')
        self.assertTrue(
            len(
                set(phi3._exposed).difference(
                    set(['x', 'y', 'z', 'mapping'])
                )
            ) == 0
        )

        # check that it is being propagated
        phi3.x = 40.
        self.assertTrue(phi1.x == 40.)

    def test_nested_objfcts(self):
        nP = 10
        m = np.random.rand(nP)

        phi1 = Props_ObjFct(nP=nP)
        phi2 = Props_ObjFct(nP=nP)

        phi1.x = 10.

        phi3 = 2*phi1 + 3*phi2
        phi3.x=10
        phi3.expose('all')

        phi4 = phi1 + 2*phi3
        phi4.expose('all')
        phi4.x = 20
        print(phi2.x)

        getattr(phi4, 'x')
        setattr(phi4, 'x', 10)

    def test_reg(self):
        from SimPEG import Regularization, Mesh

        mesh = Mesh.TensorMesh([10, 10, 10])
        reg1 = Regularization.Tikhonov(mesh=mesh)
        reg2 = Regularization.Tikhonov(mesh=mesh)

        reg3 = reg1 + reg2

        reg3.expose('all')
        reg3.mapping = Maps.ExpMap(mesh)

        for objfct in reg3.objfcts:
            assert isinstance(objfct.mapping, Maps.ExpMap)

if __name__ == '__main__':
    unittest.main()

