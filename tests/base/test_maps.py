import numpy as np
import unittest
from SimPEG import Mesh, Maps, Models, Utils
import inspect

TOL = 1e-14

np.random.seed(121)


MAPS_TO_EXCLUDE_2D = ["ComboMap", "ActiveCells", "InjectActiveCells",
                      "LogMap", "ReciprocalMap",
                      "Surject2Dto3D", "Map2Dto3D", "Mesh2Mesh",
                      "ParametricPolyMap", "PolyMap", "ParametricSplineMap",
                      "SplineMap", "ParametrizedCasingAndLayer",
                      "ParametrizedLayer", "ParametrizedBlockInLayer",
                      "Projection", "SelfConsistentEffectiveMedium"]
MAPS_TO_EXCLUDE_3D = ["ComboMap", "ActiveCells", "InjectActiveCells",
                      "LogMap", "ReciprocalMap",
                      "CircleMap", "ParametricCircleMap", "Mesh2Mesh",
                      "ParametricPolyMap", "PolyMap", "ParametricSplineMap",
                      "SplineMap", "ParametrizedCasingAndLayer",
                      "ParametrizedLayer", "ParametrizedBlockInLayer",
                      "Projection", "SelfConsistentEffectiveMedium"]


class MapTests(unittest.TestCase):

    def setUp(self):

        maps2test2D = [M for M in dir(Maps) if M not in MAPS_TO_EXCLUDE_2D]
        maps2test3D = [M for M in dir(Maps) if M not in MAPS_TO_EXCLUDE_3D]

        self.maps2test2D = [getattr(Maps, M) for M in maps2test2D if
                            (inspect.isclass(getattr(Maps, M)) and
                             issubclass(getattr(Maps, M), Maps.IdentityMap))]

        self.maps2test3D = [getattr(Maps, M) for M in maps2test3D if
                            inspect.isclass(getattr(Maps, M)) and
                            issubclass(getattr(Maps, M), Maps.IdentityMap)]

        a = np.array([1, 1, 1])
        b = np.array([1, 2])

        self.mesh2 = Mesh.TensorMesh([a, b], x0=np.array([3, 5]))
        self.mesh3 = Mesh.TensorMesh([a, b, [3, 4]], x0=np.array([3, 5, 2]))
        self.mesh22 = Mesh.TensorMesh([b, a], x0=np.array([3, 5]))
        self.meshCyl = Mesh.CylMesh([10., 1., 10.], x0='00C')

    def test_transforms2D(self):
        for M in self.maps2test2D:
            self.assertTrue(M(self.mesh2).test())

    def test_transforms2Dvec(self):
        for M in self.maps2test2D:
            self.assertTrue(M(self.mesh2).testVec())

    def test_transforms3D(self):
        for M in self.maps2test3D:
            self.assertTrue(M(self.mesh3).test())

    def test_transforms3Dvec(self):
        for M in self.maps2test3D:
            self.assertTrue(M(self.mesh3).testVec())

    def test_invtransforms2D(self):
        for M in self.maps2test2D:
            print('Testing Inverse {0}'.format(str(M.__name__)))
            mapping = M(self.mesh2)
            d = np.random.rand(mapping.shape[0])
            try:
                m = mapping.inverse(d)
                print('Testing Inverse {0}'.format(str(M.__name__)))
                self.assertLess(np.linalg.norm(d - mapping._transform(m)), TOL)
                print('  ... ok\n')
            except NotImplementedError:
                pass

    def test_invtransforms3D(self):
        for M in self.maps2test3D:
            print('Testing Inverse {0}'.format(str(M.__name__)))
            mapping = M(self.mesh3)
            d = np.random.rand(mapping.shape[0])
            try:
                m = mapping.inverse(d)
                print('Testing Inverse {0}'.format(str(M.__name__)))
                self.assertLess(np.linalg.norm(d - mapping._transform(m)), TOL)
                print('  ... ok\n')
            except NotImplementedError:
                pass

    def test_ParametricCasingAndLayer(self):
        mapping = Maps.ParametrizedCasingAndLayer(self.meshCyl)
        m = np.r_[-2., 1., 6., 2., -0.1, 0.2, 0.5, 0.2, -0.2, 0.2]
        self.assertTrue(mapping.test(m))

    def test_transforms_logMap_reciprocalMap(self):

        # Note that log/reciprocal maps can be kinda finicky, so we are being
        # explicit about the random seed.

        v2 = np.r_[0.40077291, 0.1441044, 0.58452314, 0.96323738, 0.01198519,
                   0.79754415]
        dv2 = np.r_[0.80653921, 0.13132446, 0.4901117, 0.03358737, 0.65473762,
                    0.44252488]
        v3 = np.r_[0.96084865, 0.34385186, 0.39430044, 0.81671285, 0.65929109,
                   0.2235217, 0.87897526, 0.5784033, 0.96876393, 0.63535864,
                   0.84130763, 0.22123854]
        dv3 = np.r_[0.96827838, 0.26072111, 0.45090749, 0.10573893,
                    0.65276365, 0.15646586, 0.51679682, 0.23071984,
                    0.95106218, 0.14201845, 0.25093564, 0.3732866]

        maps = Maps.LogMap(self.mesh2)
        self.assertTrue(maps.test(v2, dx=dv2))
        maps = Maps.LogMap(self.mesh3)
        self.assertTrue(maps.test(v3, dx=dv3))

        maps = Maps.ReciprocalMap(self.mesh2)
        self.assertTrue(maps.test(v2, dx=dv2))
        maps = Maps.ReciprocalMap(self.mesh3)
        self.assertTrue(maps.test(v3, dx=dv3))

    def test_Mesh2MeshMap(self):
        maps = Maps.Mesh2Mesh([self.mesh22, self.mesh2])
        self.assertTrue(maps.test())

    def test_Mesh2MeshMapVec(self):
        maps = Maps.Mesh2Mesh([self.mesh22, self.mesh2])
        self.assertTrue(maps.testVec())

    def test_mapMultiplication(self):
        M = Mesh.TensorMesh([2, 3])
        expMap = Maps.ExpMap(M)
        vertMap = Maps.SurjectVertical1D(M)
        combo = expMap*vertMap
        m = np.arange(3.0)
        t_true = np.exp(np.r_[0, 0, 1, 1, 2, 2.])
        self.assertLess(np.linalg.norm((combo * m) - t_true, np.inf), TOL)
        self.assertLess(np.linalg.norm((expMap * vertMap * m)-t_true, np.inf),
                        TOL)
        self.assertLess(np.linalg.norm(expMap * (vertMap * m)-t_true, np.inf),
                        TOL)
        self.assertLess(np.linalg.norm((expMap * vertMap) * m-t_true, np.inf),
                        TOL)
        # Try making a model
        mod = Models.Model(m, mapping=combo)
        # print mod.transform
        # import matplotlib.pyplot as plt
        # plt.colorbar(M.plotImage(mod.transform)[0])
        # plt.show()
        self.assertLess(np.linalg.norm(mod.transform - t_true, np.inf), TOL)

        self.assertRaises(Exception, Models.Model, np.r_[1.0], mapping=combo)

        self.assertRaises(ValueError, lambda: combo * (vertMap * expMap))
        self.assertRaises(ValueError, lambda: (combo * vertMap) * expMap)
        self.assertRaises(ValueError, lambda: vertMap * expMap)
        self.assertRaises(ValueError, lambda: expMap * np.ones(100))
        self.assertRaises(ValueError, lambda: expMap * np.ones((100, 1)))
        self.assertRaises(ValueError, lambda: expMap * np.ones((100, 5)))
        self.assertRaises(ValueError, lambda: combo * np.ones(100))
        self.assertRaises(ValueError, lambda: combo * np.ones((100, 1)))
        self.assertRaises(ValueError, lambda: combo * np.ones((100, 5)))

    def test_activeCells(self):
        M = Mesh.TensorMesh([2, 4], '0C')
        for actMap in [Maps.InjectActiveCells(M, M.vectorCCy <= 0, 10,
                       nC=M.nCy), Maps.ActiveCells(M, M.vectorCCy <= 0, 10,
                       nC=M.nCy)]:

            vertMap = Maps.SurjectVertical1D(M)
            combo = vertMap * actMap
            m = np.r_[1., 2.]
            mod = Models.Model(m, combo)

            self.assertLess(np.linalg.norm(mod.transform -
                            np.r_[1, 1, 2, 2, 10, 10, 10, 10.]), TOL)
            self.assertLess((mod.transformDeriv -
                            combo.deriv(m)).toarray().sum(), TOL)

    def test_tripleMultiply(self):
        M = Mesh.TensorMesh([2, 4], '0C')
        expMap = Maps.ExpMap(M)
        vertMap = Maps.SurjectVertical1D(M)
        actMap = Maps.InjectActiveCells(M, M.vectorCCy <= 0, 10, nC=M.nCy)
        m = np.r_[1., 2.]
        t_true = np.exp(np.r_[1, 1, 2, 2, 10, 10, 10, 10.])

        self.assertLess(np.linalg.norm((expMap * vertMap * actMap * m) -
                        t_true, np.inf), TOL)
        self.assertLess(np.linalg.norm(((expMap * vertMap * actMap) * m) -
                        t_true, np.inf), TOL)
        self.assertLess(np.linalg.norm((expMap * vertMap * (actMap * m)) -
                        t_true, np.inf), TOL)
        self.assertLess(np.linalg.norm((expMap * (vertMap * actMap) * m) -
                        t_true, np.inf), TOL)
        self.assertLess(np.linalg.norm(((expMap * vertMap) * actMap * m) -
                        t_true, np.inf), TOL)

        self.assertRaises(ValueError, lambda: expMap * actMap * vertMap)
        self.assertRaises(ValueError, lambda: actMap * vertMap * expMap)

    def test_map2Dto3D_x(self):
        M2 = Mesh.TensorMesh([2, 4])
        M3 = Mesh.TensorMesh([3, 2, 4])
        m = np.random.rand(int(M2.nC))

        for m2to3 in [Maps.Surject2Dto3D(M3, normal='X'),
                      Maps.Map2Dto3D(M3, normal='X')]:

            # m2to3 = Maps.Surject2Dto3D(M3, normal='X')
            m = np.arange(m2to3.nP)
            self.assertTrue(m2to3.test())
            self.assertTrue(m2to3.testVec())
            self.assertTrue(np.all(Utils.mkvc((m2to3 * m).reshape(M3.vnC,
                            order='F')[0, :, :]) == m))

    def test_map2Dto3D_y(self):
        M2 = Mesh.TensorMesh([3, 4])
        M3 = Mesh.TensorMesh([3, 2, 4])
        m = np.random.rand(M2.nC)

        for m2to3 in [Maps.Surject2Dto3D(M3, normal='Y'), Maps.Map2Dto3D(M3,
                      normal='Y')]:
            # m2to3 = Maps.Surject2Dto3D(M3, normal='Y')
            m = np.arange(m2to3.nP)
            self.assertTrue(m2to3.test())
            self.assertTrue(m2to3.testVec())
            self.assertTrue(np.all(Utils.mkvc((m2to3 * m).reshape(M3.vnC,
                            order='F')[:, 0, :]) == m))

    def test_map2Dto3D_z(self):
        M2 = Mesh.TensorMesh([3, 2])
        M3 = Mesh.TensorMesh([3, 2, 4])
        m = np.random.rand(M2.nC)

        for m2to3 in [Maps.Surject2Dto3D(M3, normal='Z'), Maps.Map2Dto3D(M3,
                      normal='Z')]:

            # m2to3 = Maps.Surject2Dto3D(M3, normal='Z')
            m = np.arange(m2to3.nP)
            self.assertTrue(m2to3.test())
            self.assertTrue(m2to3.testVec())
            self.assertTrue(np.all(Utils.mkvc((m2to3 * m).reshape(M3.vnC,
                            order='F')[:, :, 0]) == m))

    def test_ParametricPolyMap(self):
        M2 = Mesh.TensorMesh([np.ones(10), np.ones(10)], "CN")
        mParamPoly = Maps.ParametricPolyMap(M2, 2, logSigma=True, normal='Y')
        self.assertTrue(mParamPoly.test(m=np.r_[1., 1., 0., 0., 0.]))
        self.assertTrue(mParamPoly.testVec(m=np.r_[1., 1., 0., 0., 0.]))

    def test_ParametricSplineMap(self):
        M2 = Mesh.TensorMesh([np.ones(10), np.ones(10)], "CN")
        x = M2.vectorCCx
        mParamSpline = Maps.ParametricSplineMap(M2, x, normal='Y', order=1)
        self.assertTrue(mParamSpline.test())
        self.assertTrue(mParamSpline.testVec())

    def test_Projection(self):
        nP = 10
        m = np.arange(nP)
        self.assertTrue(
            np.all(Maps.Projection(nP, slice(5)) * m == m[:5])
        )
        self.assertTrue(
            np.all(Maps.Projection(nP, slice(5, None)) * m == m[5:])
        )
        self.assertTrue(np.all(
            Maps.Projection(nP, np.r_[1, 5, 3, 2, 9, 9]) * m ==
            np.r_[1, 5, 3, 2, 9, 9]
        ))
        self.assertTrue(np.all(
            Maps.Projection(nP, [1, 5, 3, 2, 9, 9]) * m ==
            np.r_[1, 5, 3, 2, 9, 9]
        ))
        with self.assertRaises(AssertionError):
            Maps.Projection(nP, np.r_[10]) * m

        mapping = Maps.Projection(nP, np.r_[1, 2, 6, 1, 3, 5, 4, 9, 9, 8, 0])
        mapping.test()


class TestWires(unittest.TestCase):

    def test_basic(self):
        mesh = Mesh.TensorMesh([10, 10, 10])

        wires = Maps.Wires(
            ('sigma', mesh.nCz),
            ('mu_casing', 1),
        )

        model = np.arange(mesh.nCz + 1)

        assert isinstance(wires.sigma, Maps.Projection)
        assert wires.nP == mesh.nCz + 1

        named_model = wires * model

        named_model.sigma == model[:mesh.nCz]
        assert named_model.mu_casing == 10


class TestSCEMT(unittest.TestCase):
    def test_sphericalInclusions(self):
        mesh = Mesh.TensorMesh([4,  5, 3])
        mapping = Maps.SelfConsistentEffectiveMedium(
            mesh, sigma0=1e-1, sigma1=1.
        )
        m = np.abs(np.random.rand(mesh.nC))
        mapping.test(m=m, dx=0.05, num=3)

    def test_spheroidalInclusions(self):
        mesh = Mesh.TensorMesh([4,  3, 2])
        mapping = Maps.SelfConsistentEffectiveMedium(
            mesh, sigma0=1e-1, sigma1=1., alpha0=0.8, alpha1=0.9, rel_tol=1e-8
        )
        m = np.abs(np.random.rand(mesh.nC))
        mapping.test(m=m, dx=0.05, num=3)

if __name__ == '__main__':
    unittest.main()
