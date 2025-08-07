from copy import deepcopy
import numpy as np
import unittest
import discretize
import pytest
import scipy.sparse as sp

from simpeg import maps, models, utils
from discretize.utils import mesh_builder_xyz, refine_tree_xyz, active_from_xyz
import inspect

from simpeg.maps._parametric import (
    BaseParametric,
    ParametricLayer,
    ParametricBlock,
    ParametricEllipsoid,
)

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
    "LinearMap",
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
    "LinearMap",
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


def test_IdentityMap_init():
    m = maps.IdentityMap()
    assert m.nP == "*"

    m = maps.IdentityMap(nP="*")
    assert m.nP == "*"

    m = maps.IdentityMap(nP=3)
    assert m.nP == 3

    mesh = discretize.TensorMesh([3])
    m = maps.IdentityMap(mesh)
    assert m.nP == 3

    m = maps.IdentityMap(mesh, nP="*")
    assert m.nP == 3

    with pytest.raises(TypeError):
        maps.IdentityMap(nP="x")


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
        self.meshCyl = discretize.CylindricalMesh([10.0, 1.0, 10.0], x0="00C")

    def test_transforms2D(self):
        for M in self.maps2test2D:
            self.assertTrue(M(self.mesh2).test(random_seed=42))

    def test_transforms2Dvec(self):
        for M in self.maps2test2D:
            self.assertTrue(M(self.mesh2).test(random_seed=42))

    def test_transforms3D(self):
        for M in self.maps2test3D:
            self.assertTrue(M(self.mesh3).test(random_seed=42))

    def test_transforms3Dvec(self):
        for M in self.maps2test3D:
            self.assertTrue(M(self.mesh3).test(random_seed=42))

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
        self.assertTrue(mapping.test(m=m, random_seed=42))

    def test_ParametricBlock2D(self):
        mesh = discretize.TensorMesh([np.ones(30), np.ones(20)], x0=np.array([-15, -5]))
        mapping = maps.ParametricBlock(mesh)
        # val_background,val_block, block_x0, block_dx, block_y0, block_dy
        m = np.r_[-2.0, 1.0, -5, 10, 5, 4]
        self.assertTrue(mapping.test(m=m, random_seed=42))

    def test_transforms_logMap_reciprocalMap(self):

        mapping = maps.LogMap(self.mesh2)
        self.assertTrue(mapping.test(random_seed=42))
        mapping = maps.LogMap(self.mesh3)
        self.assertTrue(mapping.test(random_seed=42))

        mapping = maps.ReciprocalMap(self.mesh2)
        self.assertTrue(mapping.test(random_seed=42))
        mapping = maps.ReciprocalMap(self.mesh3)
        self.assertTrue(mapping.test(random_seed=42))

    def test_Mesh2MeshMap(self):
        mapping = maps.Mesh2Mesh([self.mesh22, self.mesh2])
        self.assertTrue(mapping.test(random_seed=42))

    def test_Mesh2MeshMapVec(self):
        mapping = maps.Mesh2Mesh([self.mesh22, self.mesh2])
        self.assertTrue(mapping.test(random_seed=42))

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
        # plt.colorbar(M.plot_image(mod.transform)[0])
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
            maps.InjectActiveCells(M, M.cell_centers_y <= 0, 10, nC=M.shape_cells[1]),
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
        actMap = maps.InjectActiveCells(
            M, M.cell_centers_y <= 0, 10, nC=M.shape_cells[1]
        )
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
            self.assertTrue(m2to3.test(random_seed=42))
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
            self.assertTrue(m2to3.test(random_seed=42))
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
            self.assertTrue(m2to3.test(random_seed=42))
            self.assertTrue(
                np.all(utils.mkvc((m2to3 * m).reshape(M3.vnC, order="F")[:, :, 0]) == m)
            )

    def test_ParametricPolyMap(self):
        M2 = discretize.TensorMesh([np.ones(10), np.ones(10)], "CN")
        mParamPoly = maps.ParametricPolyMap(M2, 2, logSigma=True, normal="Y")
        self.assertTrue(
            mParamPoly.test(m=np.r_[1.0, 1.0, 0.0, 0.0, 0.0], random_seed=42)
        )

    def test_ParametricSplineMap(self):
        M2 = discretize.TensorMesh([np.ones(10), np.ones(10)], "CN")
        x = M2.cell_centers_x
        mParamSpline = maps.ParametricSplineMap(M2, x, normal="Y", order=1)
        self.assertTrue(mParamSpline.test(random_seed=42))

    def test_parametric_block(self):
        M1 = discretize.TensorMesh([np.ones(10)], "C")
        block = maps.ParametricBlock(M1)
        self.assertTrue(
            block.test(
                m=np.hstack([np.random.rand(2), np.r_[M1.x0, 2 * M1.h[0].min()]]),
                random_seed=42,
            )
        )

        M2 = discretize.TensorMesh([np.ones(10), np.ones(20)], "CC")
        block = maps.ParametricBlock(M2)
        self.assertTrue(
            block.test(
                m=np.hstack(
                    [
                        np.random.rand(2),
                        np.r_[M2.x0[0], 2 * M2.h[0].min()],
                        np.r_[M2.x0[1], 4 * M2.h[1].min()],
                    ]
                ),
                random_seed=42,
            )
        )

        M3 = discretize.TensorMesh([np.ones(10), np.ones(20), np.ones(30)], "CCC")
        block = maps.ParametricBlock(M3)
        self.assertTrue(
            block.test(
                m=np.hstack(
                    [
                        np.random.rand(2),
                        np.r_[M3.x0[0], 2 * M3.h[0].min()],
                        np.r_[M3.x0[1], 4 * M3.h[1].min()],
                        np.r_[M3.x0[2], 5 * M3.h[2].min()],
                    ]
                ),
                random_seed=42,
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
                        np.r_[M2.x0[0], 2 * M2.h[0].min()],
                        np.r_[M2.x0[1], 4 * M2.h[1].min()],
                    ]
                ),
                random_seed=42,
            )
        )

        M3 = discretize.TensorMesh([np.ones(10), np.ones(20), np.ones(30)], "CCC")
        block = maps.ParametricEllipsoid(M3)
        self.assertTrue(
            block.test(
                m=np.hstack(
                    [
                        np.random.rand(2),
                        np.r_[M3.x0[0], 2 * M3.h[0].min()],
                        np.r_[M3.x0[1], 4 * M3.h[1].min()],
                        np.r_[M3.x0[2], 5 * M3.h[2].min()],
                    ]
                ),
                random_seed=42,
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
                np.r_[M2.x0[0], 2 * M2.h[0].min()],
                np.r_[M2.x0[1], 4 * M2.h[1].min()],
            ]
        )

        self.assertTrue(np.all(summap0 * m0 == summap1 * m0))

        self.assertTrue(summap0.test(m=m0, random_seed=42))
        self.assertTrue(summap1.test(m=m0, random_seed=42))

    def test_surject_units(self):
        M2 = discretize.TensorMesh([np.ones(10), np.ones(20)], "CC")
        unit1 = M2.gridCC[:, 0] < 0
        unit2 = M2.gridCC[:, 0] >= 0

        surject_units = maps.SurjectUnits([unit1, unit2])

        m0 = np.r_[0, 1]
        m1 = surject_units * m0

        self.assertTrue(np.all(m1[unit1] == 0))
        self.assertTrue(np.all(m1[unit2] == 1))
        self.assertTrue(surject_units.test(m=m0, random_seed=42))

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
        mapping.test(random_seed=42)

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
        activeCells = active_from_xyz(mesh, rxLocs)

        model = np.random.randn(int(activeCells.sum()))
        total_mass = (model * mesh.cell_volumes[activeCells]).sum()

        for local_mesh in local_meshes:
            tile_map = maps.TileMap(
                mesh,
                activeCells,
                local_mesh,
            )

            local_mass = (
                (tile_map * model) * local_mesh.cell_volumes[tile_map.local_active]
            ).sum()

            self.assertTrue((local_mass - total_mass) / total_mass < 1e-8)

    def test_logit_errors(self):
        nP = 10
        scalar_lower = -2
        scalar_upper = 2
        good_vector_lower = np.random.rand(nP) - 2
        good_vector_upper = np.random.rand(nP) + 2

        bad_vector_lower = np.random.rand(nP - 2) - 2
        bad_vector_upper = np.random.rand(nP - 2) + 2

        # test that lower is not equal to nP
        with pytest.raises(
            ValueError,
            match="Lower bound does not broadcast to the number of parameters.*",
        ):
            maps.LogisticSigmoidMap(
                nP=10, lower_bound=bad_vector_lower, upper_bound=scalar_upper
            )

        # test that bad is not equal to nP
        with pytest.raises(
            ValueError,
            match="Upper bound does not broadcast to the number of parameters.*",
        ):
            maps.LogisticSigmoidMap(
                nP=10, lower_bound=scalar_lower, upper_bound=bad_vector_upper
            )

        # test that two upper and lower arrays will not broadcast when not specifying the number of parameters
        with pytest.raises(
            ValueError, match="Upper bound does not broadcast to the lower bound.*"
        ):
            maps.LogisticSigmoidMap(
                lower_bound=good_vector_lower, upper_bound=bad_vector_upper
            )

        # test that passing a lower bound higher than an upper bound)
        with pytest.raises(
            ValueError,
            match="A lower bound is greater than or equal to the upper bound.",
        ):
            maps.LogisticSigmoidMap(
                lower_bound=good_vector_upper, upper_bound=good_vector_lower
            )


class TestWires(unittest.TestCase):
    def test_basic(self):
        mesh = discretize.TensorMesh([10, 10, 10])

        wires = maps.Wires(
            ("sigma", mesh.shape_cells[2]),
            ("mu_casing", 1),
        )

        model = np.arange(mesh.shape_cells[2] + 1)

        assert isinstance(wires.sigma, maps.Projection)
        assert wires.nP == mesh.shape_cells[2] + 1

        named_model = wires * model

        np.testing.assert_equal(named_model.sigma, model[: mesh.shape_cells[2]])
        np.testing.assert_equal(named_model.mu_casing, 10)


class TestSCEMT(unittest.TestCase):
    def test_sphericalInclusions(self):
        mesh = discretize.TensorMesh([4, 5, 3])
        mapping = maps.SelfConsistentEffectiveMedium(mesh, sigma0=1e-1, sigma1=1.0)
        mapping.test(num=3, random_seed=42)

    def test_spheroidalInclusions(self):
        mesh = discretize.TensorMesh([4, 3, 2])
        mapping = maps.SelfConsistentEffectiveMedium(
            mesh, sigma0=1e-1, sigma1=1.0, alpha0=0.8, alpha1=0.9, rel_tol=1e-8
        )
        mapping.test(num=3, random_seed=42)


@pytest.mark.parametrize(
    "A, b",
    [
        (np.random.rand(20, 40), np.random.rand(20)),  # test with np.array
        (np.random.rand(20, 40), None),  # Test without B
        (sp.random(20, 40, density=0.1), np.random.rand(20)),  # test with sparse matrix
        (
            sp.csr_array(sp.random(20, 40, density=0.1)),
            np.random.rand(20),
        ),  # test with sparse array
        (sp.linalg.aslinearoperator(np.random.rand(20, 40)), np.random.rand(20)),
    ],
)
def test_LinearMap(A, b):
    map1 = maps.LinearMap(A, b)
    x = np.random.rand(map1.nP)
    y1 = A @ x
    if b is not None:
        y1 += b
    y2 = map1 * x
    np.testing.assert_allclose(y1, y2)


@pytest.mark.parametrize(
    "A, b",
    [
        (np.random.rand(20, 40), np.random.rand(20)),  # test with np.array
        (np.random.rand(20, 40), None),  # Test without B
        (sp.random(20, 40, density=0.1), np.random.rand(20)),  # test with sparse matrix
        (
            sp.csr_array(sp.random(20, 40, density=0.1)),
            np.random.rand(20),
        ),  # test with sparse array
        (sp.linalg.aslinearoperator(np.random.rand(20, 40)), np.random.rand(20)),
    ],
)
def test_LinearMapDerivs(A, b):
    mapping = maps.LinearMap(A, b)
    # check passing v=None
    m = np.random.rand(mapping.nP)
    v = np.random.rand(mapping.nP)
    y1 = mapping.deriv(m) @ v
    y2 = mapping.deriv(m, v=v)
    np.testing.assert_equal(y1, y2)
    mapping.test(random_seed=42)


def test_LinearMap_errors():
    # object with no matmul operator
    with pytest.raises(TypeError):
        maps.LinearMap("No Matmul")

    # object with matmul and no shape
    class IncompleteLinearOperator:
        def __matmul__(self, other):
            return other

    incomplete_A = IncompleteLinearOperator()
    with pytest.raises(TypeError):
        maps.LinearMap(incomplete_A)

    # poorly shaped b vector
    A = np.random.rand(20, 40)
    b = np.random.rand(40)
    with pytest.raises(ValueError):
        maps.LinearMap(A, b=b)


def test_linearity():
    mesh1 = discretize.TensorMesh([3])
    mesh2 = discretize.TensorMesh([3, 4])
    mesh3 = discretize.TensorMesh([3, 4, 5])
    mesh_cyl = discretize.CylindricalMesh([5, 1, 5])
    mesh_tree = discretize.TreeMesh([8, 8, 8])
    mesh_tree.refine(-1)
    # make a list of linear maps
    linear_maps = [
        maps.IdentityMap(mesh3),
        maps.LinearMap(np.eye(3)),
        maps.Projection(2, np.array([1, 0, 1, 0], dtype=int)),
        maps.SurjectUnits([[True, False, True], [False, True, False]], nP=3),
        maps.ChiMap(),
        maps.MuRelative(),
        maps.Weighting(nP=mesh1.n_cells),
        maps.ComplexMap(mesh3),
        maps.SurjectFull(mesh3),
        maps.SurjectVertical1D(mesh2),
        maps.Surject2Dto3D(mesh3),
        maps.Mesh2Mesh((mesh3, mesh3)),
        maps.InjectActiveCells(
            mesh3,
            mesh3.cell_centers[:, -1] < 0.75,
        ),
        maps.TileMap(
            mesh_tree,
            mesh_tree.cell_centers[:, -1] < 0.75,
            mesh_tree,
        ),
        maps.IdentityMap() + maps.IdentityMap(),  # A simple SumMap
        maps.ChiMap() * maps.MuRelative(),  # A simple ComboMap
    ]
    non_linear_maps = [
        maps.SphericalSystem(mesh2),
        maps.SelfConsistentEffectiveMedium(mesh2, sigma0=1, sigma1=2),
        maps.ExpMap(),
        maps.LogisticSigmoidMap(),
        maps.ReciprocalMap(),
        maps.LogMap(),
        maps.ParametricCircleMap(mesh2),
        maps.ParametricPolyMap(mesh2, 4),
        maps.ParametricSplineMap(mesh2, np.r_[0.25, 0.35]),
        maps.BaseParametric(mesh3),
        maps.ParametricLayer(mesh3),
        maps.ParametricBlock(mesh3),
        maps.ParametricEllipsoid(mesh3),
        maps.ParametricCasingAndLayer(mesh_cyl),
        maps.ParametricBlockInLayer(mesh3),
        maps.PolynomialPetroClusterMap(),
        maps.IdentityMap() + maps.ExpMap(),  # A simple SumMap
        maps.ChiMap() * maps.ExpMap(),  # A simple ComboMap
    ]

    assert all(m.is_linear for m in linear_maps)
    assert all(not m.is_linear for m in non_linear_maps)


class DeprecatedIndActive:
    """Base class to test deprecated ``actInd`` and ``indActive`` arguments in maps."""

    @pytest.fixture
    def mesh(self):
        """Sample mesh."""
        return discretize.TensorMesh([np.ones(10), np.ones(10)], "CN")

    @pytest.fixture
    def active_cells(self, mesh):
        """Sample active cells for the mesh."""
        active_cells = np.ones(mesh.n_cells, dtype=bool)
        active_cells[0] = False
        return active_cells

    def get_message_duplicated_error(self, old_name, new_name, version="v0.24.0"):
        msg = (
            f"Cannot pass both '{new_name}' and '{old_name}'."
            f"'{old_name}' has been deprecated and will be removed in "
            f" SimPEG {version}, please use '{new_name}' instead."
        )
        return msg

    def get_message_deprecated_warning(self, old_name, new_name, version="v0.24.0"):
        msg = (
            f"'{old_name}' has been deprecated and will be removed in "
            f" SimPEG {version}, please use '{new_name}' instead."
        )
        return msg


class TestParametricPolyMap(DeprecatedIndActive):
    """Test deprecated ``actInd`` in ParametricPolyMap."""

    def test_warning_argument(self, mesh, active_cells):
        """
        Test if warning is raised after passing ``actInd`` to the constructor.
        """
        msg = self.get_message_deprecated_warning("actInd", "active_cells")
        with pytest.warns(FutureWarning, match=msg):
            map_instance = maps.ParametricPolyMap(mesh, 2, actInd=active_cells)
        np.testing.assert_allclose(map_instance.active_cells, active_cells)

    def test_error_duplicated_argument(self, mesh, active_cells):
        """
        Test error after passing ``actInd`` and ``active_cells`` to the constructor.
        """
        msg = self.get_message_duplicated_error("actInd", "active_cells")
        with pytest.raises(TypeError, match=msg):
            maps.ParametricPolyMap(
                mesh, 2, active_cells=active_cells, actInd=active_cells
            )

    def test_warning_accessing_property(self, mesh, active_cells):
        """
        Test warning when trying to access the ``actInd`` property.
        """
        mapping = maps.ParametricPolyMap(mesh, 2, active_cells=active_cells)
        msg = "actInd has been deprecated, please use active_cells"
        with pytest.warns(FutureWarning, match=msg):
            old_act_ind = mapping.actInd
        np.testing.assert_allclose(mapping.active_cells, old_act_ind)

    def test_warning_setter(self, mesh, active_cells):
        """
        Test warning when trying to set the ``actInd`` property.
        """
        mapping = maps.ParametricPolyMap(mesh, 2, active_cells=active_cells)
        # Define new active cells to pass to the setter
        new_active_cells = active_cells.copy()
        new_active_cells[-4:] = False
        msg = "actInd has been deprecated, please use active_cells"
        with pytest.warns(FutureWarning, match=msg):
            mapping.actInd = new_active_cells
        np.testing.assert_allclose(mapping.active_cells, new_active_cells)


class TestMesh2Mesh(DeprecatedIndActive):
    """Test deprecated ``indActive`` in ``Mesh2Mesh``."""

    @pytest.fixture
    def meshes(self, mesh):
        return [mesh, deepcopy(mesh)]

    def test_warning_argument(self, meshes, active_cells):
        """
        Test if warning is raised after passing ``indActive`` to the constructor.
        """
        msg = self.get_message_deprecated_warning("indActive", "active_cells")
        with pytest.warns(FutureWarning, match=msg):
            mapping_instance = maps.Mesh2Mesh(meshes, indActive=active_cells)
        np.testing.assert_allclose(mapping_instance.active_cells, active_cells)

    def test_error_duplicated_argument(self, meshes, active_cells):
        """
        Test error after passing ``indActive`` and ``active_cells`` to the constructor.
        """
        msg = self.get_message_duplicated_error("indActive", "active_cells")
        with pytest.raises(TypeError, match=msg):
            maps.Mesh2Mesh(meshes, active_cells=active_cells, indActive=active_cells)

    def test_warning_accessing_property(self, meshes, active_cells):
        """
        Test warning when trying to access the ``indActive`` property.
        """
        mapping = maps.Mesh2Mesh(meshes, active_cells=active_cells)
        msg = "indActive has been deprecated, please use active_cells"
        with pytest.warns(FutureWarning, match=msg):
            old_act_ind = mapping.indActive
        np.testing.assert_allclose(mapping.active_cells, old_act_ind)

    def test_warning_setter(self, meshes, active_cells):
        """
        Test warning when trying to set the ``indActive`` property.
        """
        mapping = maps.Mesh2Mesh(meshes, active_cells=active_cells)
        # Define new active cells to pass to the setter
        new_active_cells = active_cells.copy()
        new_active_cells[-4:] = False
        msg = "indActive has been deprecated, please use active_cells"
        with pytest.warns(FutureWarning, match=msg):
            mapping.indActive = new_active_cells
        np.testing.assert_allclose(mapping.active_cells, new_active_cells)


class TestInjectActiveCells(DeprecatedIndActive):
    """Test deprecated ``indActive`` and ``valInactive`` in ``InjectActiveCells``."""

    def test_indactive_warning_argument(self, mesh, active_cells):
        """
        Test if warning is raised after passing ``indActive`` to the constructor.
        """
        msg = self.get_message_deprecated_warning("indActive", "active_cells")
        with pytest.warns(FutureWarning, match=msg):
            mapping_instance = maps.InjectActiveCells(mesh, indActive=active_cells)
        np.testing.assert_allclose(mapping_instance.active_cells, active_cells)

    def test_indactive_error_duplicated_argument(self, mesh, active_cells):
        """
        Test error after passing ``indActive`` and ``active_cells`` to the constructor.
        """
        msg = self.get_message_duplicated_error("indActive", "active_cells")
        with pytest.raises(TypeError, match=msg):
            maps.InjectActiveCells(
                mesh, active_cells=active_cells, indActive=active_cells
            )

    def test_indactive_warning_accessing_property(self, mesh, active_cells):
        """
        Test warning when trying to access the ``indActive`` property.
        """
        mapping = maps.InjectActiveCells(mesh, active_cells=active_cells)
        msg = "indActive has been deprecated, please use active_cells"
        with pytest.warns(FutureWarning, match=msg):
            old_act_ind = mapping.indActive
        np.testing.assert_allclose(mapping.active_cells, old_act_ind)

    def test_indactive_warning_setter(self, mesh, active_cells):
        """
        Test warning when trying to set the ``indActive`` property.
        """
        mapping = maps.InjectActiveCells(mesh, active_cells=active_cells)
        # Define new active cells to pass to the setter
        new_active_cells = active_cells.copy()
        new_active_cells[-4:] = False
        msg = "indActive has been deprecated, please use active_cells"
        with pytest.warns(FutureWarning, match=msg):
            mapping.indActive = new_active_cells
        np.testing.assert_allclose(mapping.active_cells, new_active_cells)

    @pytest.mark.parametrize("value_inactive", (3.14, np.array([1])))
    def test_valinactive_warning_argument(self, mesh, active_cells, value_inactive):
        """
        Test if warning is raised after passing ``valInactive`` to the constructor.
        """
        msg = self.get_message_deprecated_warning("valInactive", "value_inactive")
        with pytest.warns(FutureWarning, match=msg):
            mapping_instance = maps.InjectActiveCells(
                mesh, active_cells=active_cells, valInactive=value_inactive
            )
        # Ensure that the value passed to valInactive was correctly used
        expected = np.zeros_like(active_cells, dtype=np.float64)
        expected[~active_cells] = value_inactive
        np.testing.assert_allclose(mapping_instance.value_inactive, expected)

    @pytest.mark.parametrize("valInactive", (3.14, np.array([3.14])))
    @pytest.mark.parametrize("value_inactive", (3.14, np.array([3.14])))
    def test_valinactive_error_duplicated_argument(
        self, mesh, active_cells, valInactive, value_inactive
    ):
        """
        Test error after passing ``valInactive`` and ``value_inactive`` to the
        constructor.
        """
        msg = self.get_message_duplicated_error("valInactive", "value_inactive")
        with pytest.raises(TypeError, match=msg):
            maps.InjectActiveCells(
                mesh,
                active_cells=active_cells,
                value_inactive=value_inactive,
                valInactive=valInactive,
            )

    def test_valinactive_warning_accessing_property(self, mesh, active_cells):
        """
        Test warning when trying to access the ``valInactive`` property.
        """
        mapping = maps.InjectActiveCells(
            mesh, active_cells=active_cells, value_inactive=3.14
        )
        msg = "valInactive has been deprecated, please use value_inactive"
        with pytest.warns(FutureWarning, match=msg):
            old_value = mapping.valInactive
        np.testing.assert_allclose(mapping.value_inactive, old_value)

    def test_valinactive_warning_setter(self, mesh, active_cells):
        """
        Test warning when trying to set the ``valInactive`` property.
        """
        mapping = maps.InjectActiveCells(
            mesh, active_cells=active_cells, value_inactive=3.14
        )
        msg = "valInactive has been deprecated, please use value_inactive"
        with pytest.warns(FutureWarning, match=msg):
            mapping.valInactive = 4.5
        np.testing.assert_allclose(mapping.value_inactive[~mapping.active_cells], 4.5)


class TestParametric(DeprecatedIndActive):
    """Test deprecated ``indActive`` in parametric mappings."""

    CLASSES = (BaseParametric, ParametricLayer, ParametricBlock, ParametricEllipsoid)

    @pytest.mark.parametrize("map_class", CLASSES)
    def test_indactive_warning_argument(self, mesh, active_cells, map_class):
        """
        Test if warning is raised after passing ``indActive`` to the constructor.
        """
        msg = self.get_message_deprecated_warning("indActive", "active_cells")
        with pytest.warns(FutureWarning, match=msg):
            mapping_instance = map_class(mesh, indActive=active_cells)
        np.testing.assert_allclose(mapping_instance.active_cells, active_cells)

    @pytest.mark.parametrize("map_class", CLASSES)
    def test_indactive_error_duplicated_argument(self, mesh, active_cells, map_class):
        """
        Test error after passing ``indActive`` and ``active_cells`` to the constructor.
        """
        msg = self.get_message_duplicated_error("indActive", "active_cells")
        with pytest.raises(TypeError, match=msg):
            map_class(mesh, active_cells=active_cells, indActive=active_cells)

    @pytest.mark.parametrize("map_class", CLASSES)
    def test_indactive_warning_accessing_property(self, mesh, active_cells, map_class):
        """
        Test warning when trying to access the ``indActive`` property.
        """
        mapping = map_class(mesh, active_cells=active_cells)
        msg = "indActive has been deprecated, please use active_cells"
        with pytest.warns(FutureWarning, match=msg):
            old_act_ind = mapping.indActive
        np.testing.assert_allclose(mapping.active_cells, old_act_ind)

    @pytest.mark.parametrize("map_class", CLASSES)
    def test_indactive_warning_setter(self, mesh, active_cells, map_class):
        """
        Test warning when trying to set the ``indActive`` property.
        """
        mapping = map_class(mesh, active_cells=active_cells)
        # Define new active cells to pass to the setter
        new_active_cells = active_cells.copy()
        new_active_cells[-4:] = False
        msg = "indActive has been deprecated, please use active_cells"
        with pytest.warns(FutureWarning, match=msg):
            mapping.indActive = new_active_cells
        np.testing.assert_allclose(mapping.active_cells, new_active_cells)


class TestParametricDeriv:
    """
    Test the ``deriv`` method of parametric maps.


    Test if ``map.deriv(m) @ v`` is equivalent to ``map.deriv(m, v=v)``.
    """

    @pytest.fixture
    def mesh_2d(self):
        """Sample mesh."""
        h = 10
        return discretize.TensorMesh([h, h], "CC")

    @pytest.fixture
    def mesh_3d(self):
        """Sample mesh."""
        h = 10
        return discretize.TensorMesh([h, h, h], "CCN")

    @pytest.fixture
    def cyl_mesh(self):
        """Sample cylindrical mesh."""
        return discretize.CylindricalMesh([4, 6, 5])

    @pytest.mark.parametrize(
        "map_class",
        [
            maps.ParametricBlock,
            maps.ParametricBlockInLayer,
            maps.ParametricEllipsoid,
            maps.ParametricLayer,
            maps.ParametricPolyMap,
        ],
    )
    def test_deriv_mesh_3d(self, mesh_3d, map_class):
        """
        Test maps on a 3d mesh.
        """
        kwargs = {}
        if map_class is maps.ParametricPolyMap:
            kwargs["order"] = [1, 1]
        mapping = map_class(mesh_3d, **kwargs)
        model_size = mapping.shape[1]
        rng = np.random.default_rng(seed=48)
        model = rng.uniform(size=model_size)
        v = rng.uniform(size=model_size)
        derivative = mapping.deriv(model)
        np.testing.assert_allclose(derivative @ v, mapping.deriv(model, v=v))

    def test_deriv_mesh_2d(self, mesh_2d):
        """
        Test maps on a 2d mesh.
        """
        mapping = maps.ParametricCircleMap(mesh_2d)
        model_size = mapping.shape[1]
        rng = np.random.default_rng(seed=48)
        model = rng.uniform(size=model_size)
        v = rng.uniform(size=model_size)
        derivative = mapping.deriv(model)
        np.testing.assert_allclose(derivative @ v, mapping.deriv(model, v=v))

    def test_deriv_cyl_mesh(self, cyl_mesh):
        """
        Test maps on a cylindrical mesh.
        """
        mapping = maps.ParametricCasingAndLayer(cyl_mesh)
        model_size = mapping.shape[1]
        rng = np.random.default_rng(seed=48)
        model = rng.uniform(size=model_size)
        v = rng.uniform(size=model_size)
        derivative = mapping.deriv(model)
        np.testing.assert_allclose(derivative @ v, mapping.deriv(model, v=v))


def test_deriv_SelfConsistentEffectiveMedium():
    """
    Test deriv method of ``SelfConsistentEffectiveMedium``.
    """
    h = 10
    mesh = discretize.TensorMesh([h, h, h], "CCN")
    mapping = maps.SelfConsistentEffectiveMedium(mesh, sigma0=1, sigma1=2)
    model_size = mapping.shape[1]
    rng = np.random.default_rng(seed=48)
    model = rng.uniform(size=model_size)
    v = rng.uniform(size=model_size)
    derivative = mapping.deriv(model)
    np.testing.assert_allclose(derivative @ v, mapping.deriv(model, v=v), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
