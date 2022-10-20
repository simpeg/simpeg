import numpy as np
import unittest
import discretize
from SimPEG import maps, models, utils
from discretize.utils import mesh_builder_xyz, refine_tree_xyz
import inspect

TOL = 1e-14

np.random.seed(121)


REMOVED_IGNORE = [
    "FullMap",
    "CircleMap",
    "Map2Dto3D",
    "Vertical1DMap",
    "ActiveCells",
]

MAPS_TO_EXCLUDE_2D = [
    "ComboMap",
    "ActiveCells",
    "InjectActiveCells",
    "LogMap",
    "ReciprocalMap",
    "PolynomialPetroClusterMap",
    "Surject2Dto3D",
    "Map2Dto3D",
    "Mesh2Mesh",
    "ParametricPolyMap",
    "PolyMap",
    "ParametricSplineMap",
    "SplineMap",
    "BaseParametric",
    "ParametricBlock",
    "ParametricEllipsoid",
    "ParametricCasingAndLayer",
    "ParametricLayer",
    "ParametricBlockInLayer",
    "Projection",
    "SelfConsistentEffectiveMedium",
    "SumMap",
    "SurjectUnits",
    "TileMap",
] + REMOVED_IGNORE
MAPS_TO_EXCLUDE_3D = [
    "ComboMap",
    "ActiveCells",
    "InjectActiveCells",
    "LogMap",
    "ReciprocalMap",
    "PolynomialPetroClusterMap",
    "CircleMap",
    "ParametricCircleMap",
    "Mesh2Mesh",
    "BaseParametric",
    "ParametricBlock",
    "ParametricEllipsoid",
    "ParametricPolyMap",
    "PolyMap",
    "ParametricSplineMap",
    "SplineMap",
    "ParametricCasingAndLayer",
    "ParametricLayer",
    "ParametricBlockInLayer",
    "Projection",
    "SelfConsistentEffectiveMedium",
    "SumMap",
    "SurjectUnits",
    "TileMap",
] + REMOVED_IGNORE


class MapTests(unittest.TestCase):
    def setUp(self):

        maps2test2D = [M for M in dir(maps) if M not in MAPS_TO_EXCLUDE_2D]
        maps2test3D = [M for M in dir(maps) if M not in MAPS_TO_EXCLUDE_3D]

        self.maps2test2D = [
            getattr(maps, M)
            for M in maps2test2D
            if (
                inspect.isclass(getattr(maps, M))
                and issubclass(getattr(maps, M), maps.IdentityMap)
            )
        ]

        self.maps2test3D = [
            getattr(maps, M)
            for M in maps2test3D
            if inspect.isclass(getattr(maps, M))
            and issubclass(getattr(maps, M), maps.IdentityMap)
        ]

        a = np.array([1, 1, 1])
        b = np.array([1, 2])

        self.mesh2 = discretize.TensorMesh([a, b], x0=np.array([3, 5]))
        self.mesh3 = discretize.TensorMesh([a, b, [3, 4]], x0=np.array([3, 5, 2]))
        self.mesh22 = discretize.TensorMesh([b, a], x0=np.array([3, 5]))
        self.meshCyl = discretize.CylMesh([10.0, 1.0, 10.0], x0="00C")

    def test_transforms2D(self):
        for M in self.maps2test2D:
            self.assertTrue(M(self.mesh2).test())

    def test_transforms2Dvec(self):
        for M in self.maps2test2D:
            self.assertTrue(M(self.mesh2).test())

    def test_transforms3D(self):
        for M in self.maps2test3D:
            self.assertTrue(M(self.mesh3).test())

    def test_transforms3Dvec(self):
        for M in self.maps2test3D:
            self.assertTrue(M(self.mesh3).test())

    def test_invtransforms2D(self):
        for M in self.maps2test2D:
            print("Testing Inverse {0}".format(str(M.__name__)))
            mapping = M(self.mesh2)
            d = np.random.rand(mapping.shape[0])
            try:
                m = mapping.inverse(d)
                test_val = np.linalg.norm(d - mapping._transform(m))
                if M.__name__ == "SphericalSystem":
                    self.assertLess(
                        test_val, 1e-7
                    )  # This mapping is much less accurate
                else:
                    self.assertLess(test_val, TOL)
                print("  ... ok\n")
            except NotImplementedError:
                pass

    def test_invtransforms3D(self):
        for M in self.maps2test3D:
            print("Testing Inverse {0}".format(str(M.__name__)))

            mapping = M(self.mesh3)
            d = np.random.rand(mapping.shape[0])
            try:
                m = mapping.inverse(d)

                test_val = np.linalg.norm(d - mapping._transform(m))
                if M.__name__ == "SphericalSystem":
                    self.assertLess(
                        test_val, 1e-7
                    )  # This mapping is much less accurate
                else:
                    self.assertLess(test_val, TOL)
                print("  ... ok\n")
            except NotImplementedError:
                pass

    def test_ParametricCasingAndLayer(self):
        mapping = maps.ParametricCasingAndLayer(self.meshCyl)
        m = np.r_[-2.0, 1.0, 6.0, 2.0, -0.1, 0.2, 0.5, 0.2, -0.2, 0.2]
        self.assertTrue(mapping.test(m))

    def test_ParametricBlock2D(self):
        mesh = discretize.TensorMesh([np.ones(30), np.ones(20)], x0=np.array([-15, -5]))
        mapping = maps.ParametricBlock(mesh)
        # val_background,val_block, block_x0, block_dx, block_y0, block_dy
        m = np.r_[-2.0, 1.0, -5, 10, 5, 4]
        self.assertTrue(mapping.test(m))

    def test_transforms_logMap_reciprocalMap(self):

        # Note that log/reciprocal maps can be kinda finicky, so we are being
        # explicit about the random seed.

        v2 = np.r_[
            0.40077291, 0.1441044, 0.58452314, 0.96323738, 0.01198519, 0.79754415
        ]
        dv2 = np.r_[
            0.80653921, 0.13132446, 0.4901117, 0.03358737, 0.65473762, 0.44252488
        ]
        v3 = np.r_[
            0.96084865,
            0.34385186,
            0.39430044,
            0.81671285,
            0.65929109,
            0.2235217,
            0.87897526,
            0.5784033,
            0.96876393,
            0.63535864,
            0.84130763,
            0.22123854,
        ]
        dv3 = np.r_[
            0.96827838,
            0.26072111,
            0.45090749,
            0.10573893,
            0.65276365,
            0.15646586,
            0.51679682,
            0.23071984,
            0.95106218,
            0.14201845,
            0.25093564,
            0.3732866,
        ]

        mapping = maps.LogMap(self.mesh2)
        self.assertTrue(mapping.test(v2, dx=dv2))
        mapping = maps.LogMap(self.mesh3)
        self.assertTrue(mapping.test(v3, dx=dv3))

        mapping = maps.ReciprocalMap(self.mesh2)
        self.assertTrue(mapping.test(v2, dx=dv2))
        mapping = maps.ReciprocalMap(self.mesh3)
        self.assertTrue(mapping.test(v3, dx=dv3))

    def test_Mesh2MeshMap(self):
        mapping = maps.Mesh2Mesh([self.mesh22, self.mesh2])
        self.assertTrue(mapping.test())

    def test_Mesh2MeshMapVec(self):
        mapping = maps.Mesh2Mesh([self.mesh22, self.mesh2])
        self.assertTrue(mapping.test())

    def test_mapMultiplication(self):
        M = discretize.TensorMesh([2, 3])
        expMap = maps.ExpMap(M)
        vertMap = maps.SurjectVertical1D(M)
        combo = expMap * vertMap
        m = np.arange(3.0)
        t_true = np.exp(np.r_[0, 0, 1, 1, 2, 2.0])
        self.assertLess(np.linalg.norm((combo * m) - t_true, np.inf), TOL)
        self.assertLess(np.linalg.norm((expMap * vertMap * m) - t_true, np.inf), TOL)
        self.assertLess(np.linalg.norm(expMap * (vertMap * m) - t_true, np.inf), TOL)
        self.assertLess(np.linalg.norm((expMap * vertMap) * m - t_true, np.inf), TOL)
        # Try making a model
        mod = models.Model(m, mapping=combo)
        # print mod.transform
        # import matplotlib.pyplot as plt
        # plt.colorbar(M.plotImage(mod.transform)[0])
        # plt.show()
        self.assertLess(np.linalg.norm(mod.transform - t_true, np.inf), TOL)

        self.assertRaises(Exception, models.Model, np.r_[1.0], mapping=combo)

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
        M = discretize.TensorMesh([2, 4], "0C")
        for actMap in [
            maps.InjectActiveCells(M, M.vectorCCy <= 0, 10, nC=M.nCy),
        ]:

            vertMap = maps.SurjectVertical1D(M)
            combo = vertMap * actMap
            m = np.r_[1.0, 2.0]
            mod = models.Model(m, combo)

            self.assertLess(
                np.linalg.norm(mod.transform - np.r_[1, 1, 2, 2, 10, 10, 10, 10.0]), TOL
            )
            self.assertLess((mod.transformDeriv - combo.deriv(m)).toarray().sum(), TOL)

    def test_tripleMultiply(self):
        M = discretize.TensorMesh([2, 4], "0C")
        expMap = maps.ExpMap(M)
        vertMap = maps.SurjectVertical1D(M)
        actMap = maps.InjectActiveCells(M, M.vectorCCy <= 0, 10, nC=M.nCy)
        m = np.r_[1.0, 2.0]
        t_true = np.exp(np.r_[1, 1, 2, 2, 10, 10, 10, 10.0])

        self.assertLess(
            np.linalg.norm((expMap * vertMap * actMap * m) - t_true, np.inf), TOL
        )
        self.assertLess(
            np.linalg.norm(((expMap * vertMap * actMap) * m) - t_true, np.inf), TOL
        )
        self.assertLess(
            np.linalg.norm((expMap * vertMap * (actMap * m)) - t_true, np.inf), TOL
        )
        self.assertLess(
            np.linalg.norm((expMap * (vertMap * actMap) * m) - t_true, np.inf), TOL
        )
        self.assertLess(
            np.linalg.norm(((expMap * vertMap) * actMap * m) - t_true, np.inf), TOL
        )

        self.assertRaises(ValueError, lambda: expMap * actMap * vertMap)
        self.assertRaises(ValueError, lambda: actMap * vertMap * expMap)

    def test_map2Dto3D_x(self):
        M2 = discretize.TensorMesh([2, 4])
        M3 = discretize.TensorMesh([3, 2, 4])
        m = np.random.rand(int(M2.nC))

        for m2to3 in [
            maps.Surject2Dto3D(M3, normal="X"),
        ]:

            # m2to3 = maps.Surject2Dto3D(M3, normal='X')
            m = np.arange(m2to3.nP)
            self.assertTrue(m2to3.test())
            self.assertTrue(
                np.all(utils.mkvc((m2to3 * m).reshape(M3.vnC, order="F")[0, :, :]) == m)
            )

    def test_map2Dto3D_y(self):
        M2 = discretize.TensorMesh([3, 4])
        M3 = discretize.TensorMesh([3, 2, 4])
        m = np.random.rand(M2.nC)

        for m2to3 in [
            maps.Surject2Dto3D(M3, normal="Y"),
        ]:
            # m2to3 = maps.Surject2Dto3D(M3, normal='Y')
            m = np.arange(m2to3.nP)
            self.assertTrue(m2to3.test())
            self.assertTrue(
                np.all(utils.mkvc((m2to3 * m).reshape(M3.vnC, order="F")[:, 0, :]) == m)
            )

    def test_map2Dto3D_z(self):
        M2 = discretize.TensorMesh([3, 2])
        M3 = discretize.TensorMesh([3, 2, 4])
        m = np.random.rand(M2.nC)

        for m2to3 in [
            maps.Surject2Dto3D(M3, normal="Z"),
        ]:

            # m2to3 = maps.Surject2Dto3D(M3, normal='Z')
            m = np.arange(m2to3.nP)
            self.assertTrue(m2to3.test())
            self.assertTrue(
                np.all(utils.mkvc((m2to3 * m).reshape(M3.vnC, order="F")[:, :, 0]) == m)
            )

    def test_ParametricPolyMap(self):
        M2 = discretize.TensorMesh([np.ones(10), np.ones(10)], "CN")
        mParamPoly = maps.ParametricPolyMap(M2, 2, logSigma=True, normal="Y")
        self.assertTrue(mParamPoly.test(m=np.r_[1.0, 1.0, 0.0, 0.0, 0.0]))

    def test_ParametricSplineMap(self):
        M2 = discretize.TensorMesh([np.ones(10), np.ones(10)], "CN")
        x = M2.vectorCCx
        mParamSpline = maps.ParametricSplineMap(M2, x, normal="Y", order=1)
        self.assertTrue(mParamSpline.test())

    def test_parametric_block(self):
        M1 = discretize.TensorMesh([np.ones(10)], "C")
        block = maps.ParametricBlock(M1)
        self.assertTrue(
            block.test(m=np.hstack([np.random.rand(2), np.r_[M1.x0, 2 * M1.hx.min()]]))
        )

        M2 = discretize.TensorMesh([np.ones(10), np.ones(20)], "CC")
        block = maps.ParametricBlock(M2)
        self.assertTrue(
            block.test(
                m=np.hstack(
                    [
                        np.random.rand(2),
                        np.r_[M2.x0[0], 2 * M2.hx.min()],
                        np.r_[M2.x0[1], 4 * M2.hy.min()],
                    ]
                )
            )
        )

        M3 = discretize.TensorMesh([np.ones(10), np.ones(20), np.ones(30)], "CCC")
        block = maps.ParametricBlock(M3)
        self.assertTrue(
            block.test(
                m=np.hstack(
                    [
                        np.random.rand(2),
                        np.r_[M3.x0[0], 2 * M3.hx.min()],
                        np.r_[M3.x0[1], 4 * M3.hy.min()],
                        np.r_[M3.x0[2], 5 * M3.hz.min()],
                    ]
                )
            )
        )

    def test_parametric_ellipsoid(self):
        M2 = discretize.TensorMesh([np.ones(10), np.ones(20)], "CC")
        block = maps.ParametricEllipsoid(M2)
        self.assertTrue(
            block.test(
                m=np.hstack(
                    [
                        np.random.rand(2),
                        np.r_[M2.x0[0], 2 * M2.hx.min()],
                        np.r_[M2.x0[1], 4 * M2.hy.min()],
                    ]
                )
            )
        )

        M3 = discretize.TensorMesh([np.ones(10), np.ones(20), np.ones(30)], "CCC")
        block = maps.ParametricEllipsoid(M3)
        self.assertTrue(
            block.test(
                m=np.hstack(
                    [
                        np.random.rand(2),
                        np.r_[M3.x0[0], 2 * M3.hx.min()],
                        np.r_[M3.x0[1], 4 * M3.hy.min()],
                        np.r_[M3.x0[2], 5 * M3.hz.min()],
                    ]
                )
            )
        )

    def test_sum(self):
        M2 = discretize.TensorMesh([np.ones(10), np.ones(20)], "CC")
        block = maps.ParametricEllipsoid(M2) * maps.Projection(
            7, np.r_[1, 2, 3, 4, 5, 6]
        )
        background = (
            maps.ExpMap(M2) * maps.SurjectFull(M2) * maps.Projection(7, np.r_[0])
        )

        summap0 = maps.SumMap([block, background])
        summap1 = block + background

        m0 = np.hstack(
            [
                np.random.rand(3),
                np.r_[M2.x0[0], 2 * M2.hx.min()],
                np.r_[M2.x0[1], 4 * M2.hy.min()],
            ]
        )

        self.assertTrue(np.all(summap0 * m0 == summap1 * m0))

        self.assertTrue(summap0.test(m0))
        self.assertTrue(summap1.test(m0))

    def test_surject_units(self):
        M2 = discretize.TensorMesh([np.ones(10), np.ones(20)], "CC")
        unit1 = M2.gridCC[:, 0] < 0
        unit2 = M2.gridCC[:, 0] >= 0

        surject_units = maps.SurjectUnits([unit1, unit2])

        m0 = np.r_[0, 1]
        m1 = surject_units * m0

        self.assertTrue(np.all(m1[unit1] == 0))
        self.assertTrue(np.all(m1[unit2] == 1))
        self.assertTrue(surject_units.test(m0))

    def test_Projection(self):
        nP = 10
        m = np.arange(nP)
        self.assertTrue(np.all(maps.Projection(nP, slice(5)) * m == m[:5]))
        self.assertTrue(np.all(maps.Projection(nP, slice(5, None)) * m == m[5:]))
        self.assertTrue(
            np.all(
                maps.Projection(nP, np.r_[1, 5, 3, 2, 9, 9]) * m
                == np.r_[1, 5, 3, 2, 9, 9]
            )
        )
        self.assertTrue(
            np.all(
                maps.Projection(nP, [1, 5, 3, 2, 9, 9]) * m == np.r_[1, 5, 3, 2, 9, 9]
            )
        )
        with self.assertRaises(AssertionError):
            maps.Projection(nP, np.r_[10]) * m

        mapping = maps.Projection(nP, np.r_[1, 2, 6, 1, 3, 5, 4, 9, 9, 8, 0])
        mapping.test()

    def test_Tile(self):
        """
        Test for TileMap
        """
        rxLocs = np.random.randn(3, 3) * 20
        h = [5, 5, 5]
        padDist = np.ones((3, 2)) * 100

        local_meshes = []

        for ii in range(rxLocs.shape[0]):

            local_mesh = mesh_builder_xyz(
                rxLocs, h, padding_distance=padDist, mesh_type="tree"
            )
            local_mesh = refine_tree_xyz(
                local_mesh,
                rxLocs[ii, :].reshape((1, -1)),
                method="radial",
                octree_levels=[1],
                finalize=True,
            )

            local_meshes.append(local_mesh)

        mesh = mesh_builder_xyz(rxLocs, h, padding_distance=padDist, mesh_type="tree")

        # This garantees that the local meshes are always coarser or equal
        for local_mesh in local_meshes:
            mesh.insert_cells(
                local_mesh.gridCC,
                local_mesh.cell_levels_by_index(np.arange(local_mesh.nC)),
                finalize=False,
            )
        mesh.finalize()

        # Define an active cells from topo
        activeCells = utils.surface2ind_topo(mesh, rxLocs)

        model = np.random.randn(int(activeCells.sum()))
        total_mass = (model * mesh.vol[activeCells]).sum()

        for local_mesh in local_meshes:

            tile_map = maps.TileMap(
                mesh,
                activeCells,
                local_mesh,
            )

            local_mass = (
                (tile_map * model) * local_mesh.vol[tile_map.local_active]
            ).sum()

            self.assertTrue((local_mass - total_mass) / total_mass < 1e-8)


class TestWires(unittest.TestCase):
    def test_basic(self):
        mesh = discretize.TensorMesh([10, 10, 10])

        wires = maps.Wires(
            ("sigma", mesh.nCz),
            ("mu_casing", 1),
        )

        model = np.arange(mesh.nCz + 1)

        assert isinstance(wires.sigma, maps.Projection)
        assert wires.nP == mesh.nCz + 1

        named_model = wires * model

        named_model.sigma == model[: mesh.nCz]
        assert named_model.mu_casing == 10


class TestSCEMT(unittest.TestCase):
    def test_sphericalInclusions(self):
        mesh = discretize.TensorMesh([4, 5, 3])
        mapping = maps.SelfConsistentEffectiveMedium(mesh, sigma0=1e-1, sigma1=1.0)
        m = np.abs(np.random.rand(mesh.nC))
        mapping.test(m=m, dx=0.05 * np.ones(mesh.n_cells), num=3)

    def test_spheroidalInclusions(self):
        mesh = discretize.TensorMesh([4, 3, 2])
        mapping = maps.SelfConsistentEffectiveMedium(
            mesh, sigma0=1e-1, sigma1=1.0, alpha0=0.8, alpha1=0.9, rel_tol=1e-8
        )
        m = np.abs(np.random.rand(mesh.nC))
        mapping.test(m=m, dx=0.05 * np.ones(mesh.n_cells), num=3)


if __name__ == "__main__":
    unittest.main()
