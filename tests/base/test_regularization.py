import numpy as np
import unittest

import pytest
import inspect

import discretize
from SimPEG import maps, objective_function, regularization, utils


TOL = 1e-7
testReg = True
testRegMesh = True

np.random.seed(639)

IGNORE_ME = [
    "BaseRegularization",
    "BaseComboRegularization",
    "BaseSimilarityMeasure",
    "SimpleComboRegularization",
    "BaseSparse",
    "PGI",
    "PGIwithRelationships",
    "PGIwithNonlinearRelationshipsSmallness",
    "PGIsmallness",
    "CrossGradient",
    "LinearCorrespondence",
    "JointTotalVariation",
    "VectorAmplitude",
]


class RegularizationTests(unittest.TestCase):
    def setUp(self):
        hx, hy, hz = np.random.rand(10), np.random.rand(9), np.random.rand(8)
        hx, hy, hz = hx / hx.sum(), hy / hy.sum(), hz / hz.sum()
        mesh1 = discretize.TensorMesh([hx])
        mesh2 = discretize.TensorMesh([hx, hy])
        mesh3 = discretize.TensorMesh([hx, hy, hz])
        self.meshlist = [mesh1, mesh2, mesh3]

    if testReg:

        def test_regularization(self):
            for R in dir(regularization):
                r = getattr(regularization, R)
                if not inspect.isclass(r):
                    continue
                if not issubclass(r, objective_function.BaseObjectiveFunction):
                    continue
                if r.__name__ in IGNORE_ME:
                    continue

                for mesh in self.meshlist:
                    if mesh.dim < 3 and r.__name__[-1] == "z":
                        continue
                    if mesh.dim < 2 and r.__name__[-1] == "y":
                        continue

                    print("Testing {0:d}D".format(mesh.dim))

                    mapping = maps.IdentityMap(mesh)
                    reg = r(mesh=mesh, mapping=mapping)

                    print("--- Checking {} --- \n".format(reg.__class__.__name__))

                    if mapping.nP != "*":
                        m = np.random.rand(mapping.nP)
                    else:
                        m = np.random.rand(mesh.nC)
                    mref = np.ones_like(m) * np.mean(m)
                    reg.mref = mref

                    # test derivs
                    passed = reg.test(m, eps=TOL)
                    self.assertTrue(passed)

        def test_regularization_ActiveCells(self):
            for R in dir(regularization):
                r = getattr(regularization, R)
                if not inspect.isclass(r):
                    continue
                if not issubclass(r, objective_function.BaseObjectiveFunction):
                    continue
                if r.__name__ in IGNORE_ME:
                    continue

                for mesh in self.meshlist[:1]:
                    print("Testing Active Cells {0:d}D".format((mesh.dim)))

                    if mesh.dim == 1:
                        active_cells = utils.mkvc(mesh.gridCC <= 0.8)
                    elif mesh.dim == 2:
                        active_cells = utils.mkvc(
                            mesh.gridCC[:, -1]
                            <= (2 * np.sin(2 * np.pi * mesh.gridCC[:, 0]) + 0.5)
                        )
                    elif mesh.dim == 3:
                        active_cells = utils.mkvc(
                            mesh.gridCC[:, -1]
                            <= (
                                2 * np.sin(2 * np.pi * mesh.gridCC[:, 0])
                                + 0.5 * 2 * np.sin(2 * np.pi * mesh.gridCC[:, 1])
                                + 0.5
                            )
                        )

                    if mesh.dim < 3 and r.__name__[-1] == "z":
                        continue
                    if mesh.dim < 2 and r.__name__[-1] == "y":
                        continue

                    nP = int(active_cells.sum())
                    reg = r(
                        mesh, active_cells=active_cells, mapping=maps.IdentityMap(nP=nP)
                    )
                    m = np.random.rand(mesh.nC)[active_cells]
                    mref = np.ones_like(m) * np.mean(m)
                    reg.reference_model = mref

                    print("--- Checking {} ---\n".format(reg.__class__.__name__))

                    passed = reg.test(m, eps=TOL)
                    self.assertTrue(passed)

    if testRegMesh:

        def test_regularizationMesh(self):
            for mesh in self.meshlist:
                print("Testing {0:d}D".format(mesh.dim))

                if mesh.dim == 1:
                    indAct = utils.mkvc(mesh.gridCC <= 0.8)
                elif mesh.dim == 2:
                    indAct = utils.mkvc(
                        mesh.gridCC[:, -1]
                        <= 2 * np.sin(2 * np.pi * mesh.gridCC[:, 0]) + 0.5
                    )
                elif mesh.dim == 3:
                    indAct = utils.mkvc(
                        mesh.gridCC[:, -1]
                        <= 2 * np.sin(2 * np.pi * mesh.gridCC[:, 0])
                        + 0.5 * 2 * np.sin(2 * np.pi * mesh.gridCC[:, 1])
                        + 0.5
                    )

                regularization_mesh = regularization.RegularizationMesh(
                    mesh, active_cells=indAct
                )

                assert (regularization_mesh.vol == mesh.cell_volumes[indAct]).all()

    def test_property_mirroring(self):
        mesh = discretize.TensorMesh([8, 7, 6])

        for regType in ["Sparse"]:
            active_cells = mesh.gridCC[:, 2] < 0.6
            reg = getattr(regularization, regType)(mesh, active_cells=active_cells)

            self.assertTrue(reg.nP == reg.regularization_mesh.nC)

            [
                self.assertTrue(np.all(fct.active_cells == active_cells))
                for fct in reg.objfcts
            ]

            # test assignment of cell weights
            cell_weights = np.random.rand(active_cells.sum())
            reg.set_weights(user_weights=cell_weights)
            [
                self.assertTrue(np.all(fct.get_weights("user_weights") == cell_weights))
                for fct in reg.objfcts
            ]

            # test removing cell weights
            reg.remove_weights("user_weights")
            [
                self.assertTrue("user_weights" not in reg.objfcts[0]._weights)
                for fct in reg.objfcts
            ]

            # test updated mappings
            mapping = maps.ExpMap(nP=int(active_cells.sum()))
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
                    reg,
                    "{}".format(objfct._multiplier_pair),
                    0.5 * getattr(reg, "{}".format(objfct._multiplier_pair)),
                )
                for objfct in reg.objfcts
            ]
            b = reg(m)
            self.assertTrue(0.5 * a == b)

            # Change units
            with pytest.raises(TypeError) as error:
                reg.units = -1

            assert "'units' must be None or type str." in str(error)

            reg.units = "radian"

            [self.assertTrue(fct.units == "radian") for fct in reg.objfcts]

    def test_addition(self):
        mesh = discretize.TensorMesh([8, 7, 6])
        m = np.random.rand(mesh.nC)

        reg1 = regularization.WeightedLeastSquares(mesh)
        reg2 = regularization.WeightedLeastSquares(mesh)

        reg_a = reg1 + reg2
        self.assertTrue(len(reg_a) == 2)
        self.assertTrue(reg1(m) + reg2(m) == reg_a(m))
        reg_a.test(eps=TOL)

        reg_b = 2 * reg1 + reg2
        self.assertTrue(len(reg_b) == 2)
        self.assertTrue(2 * reg1(m) + reg2(m) == reg_b(m))
        reg_b.test(eps=TOL)

        reg_c = reg1 + reg2 / 2
        self.assertTrue(len(reg_c) == 2)
        self.assertTrue(reg1(m) + 0.5 * reg2(m) == reg_c(m))
        reg_c.test(eps=TOL)

    def test_mappings(self):
        mesh = discretize.TensorMesh([8, 7, 6])
        m = np.random.rand(2 * mesh.nC)

        wires = maps.Wires(("sigma", mesh.nC), ("mu", mesh.nC))

        for regType in ["WeightedLeastSquares", "Sparse"]:
            reg1 = getattr(regularization, regType)(mesh, mapping=wires.sigma)
            reg2 = getattr(regularization, regType)(mesh, mapping=wires.mu)

            reg3 = reg1 + reg2

            self.assertTrue(reg1.nP == 2 * mesh.nC)
            self.assertTrue(reg2.nP == 2 * mesh.nC)
            self.assertTrue(reg3.nP == 2 * mesh.nC)
            self.assertTrue(reg3(m) == reg1(m) + reg2(m))

            reg1.test(eps=TOL)
            reg2.test(eps=TOL)
            reg3.test(eps=TOL)

    def test_mref_is_zero(self):
        mesh = discretize.TensorMesh([10, 5, 8])
        mref = np.ones(mesh.nC)

        for regType in ["WeightedLeastSquares", "Sparse"]:
            reg = getattr(regularization, regType)(
                mesh, reference_model=mref, mapping=maps.IdentityMap(mesh)
            )

            print("Check: phi_m (mref) = {0:f}".format(reg(mref)))
            passed = reg(mref) < TOL
            self.assertTrue(passed)

    def test_mappings_and_cell_weights(self):
        mesh = discretize.TensorMesh([8, 7, 6])
        m = np.random.rand(2 * mesh.nC)
        v = np.random.rand(2 * mesh.nC)

        cell_weights = np.random.rand(mesh.nC)

        wires = maps.Wires(("sigma", mesh.nC), ("mu", mesh.nC))

        reg = regularization.Smallness(mesh, mapping=wires.sigma, weights=cell_weights)

        objfct = objective_function.L2ObjectiveFunction(
            W=utils.sdiag(np.sqrt(cell_weights * mesh.cell_volumes)),
            mapping=wires.sigma,
        )

        self.assertTrue(reg(m) == objfct(m))
        self.assertTrue(np.all(reg.deriv(m) == objfct.deriv(m)))
        self.assertTrue(np.all(reg.deriv2(m, v=v) == objfct.deriv2(m, v=v)))

        reg.set_weights(user_weights=cell_weights)

        # test removing the weigths
        reg.remove_weights("user_weights")

        assert "user_weights" not in reg._weights, "Issue removing the weights"

        with pytest.raises(KeyError) as error:
            reg.remove_weights("user_weights")

        assert "user_weights is not in the weights dictionary" in str(error)

        # test adding weights of bad type or shape
        with pytest.raises(TypeError) as error:
            reg.set_weights(user_weights="abc")

        with pytest.raises(ValueError) as error:
            reg.set_weights(user_weights=cell_weights[1:])

    def test_update_of_sparse_norms(self):
        mesh = discretize.TensorMesh([8, 7, 6])
        v = np.random.rand(mesh.nC)

        cell_weights = np.random.rand(mesh.nC)

        reg = regularization.Sparse(mesh, weights=cell_weights)

        np.testing.assert_equal(reg.norms, [1, 1, 1, 1])

        with pytest.raises(ValueError):
            reg.norms = [1, 1]

        reg.norms = [2.0, 2.0, 2.0, 2.0]
        np.testing.assert_equal(reg.objfcts[0].norm, 2.0 * np.ones(mesh.nC))
        np.testing.assert_equal(reg.objfcts[1].norm, 2.0 * np.ones(mesh.nFx))
        np.testing.assert_equal(reg.objfcts[2].norm, 2.0 * np.ones(mesh.nFy))
        np.testing.assert_equal(reg.objfcts[3].norm, 2.0 * np.ones(mesh.nFz))
        for norm, objfct in zip(reg.norms, reg.objfcts):
            np.testing.assert_equal(norm, objfct.norm)

        reg.norms = np.r_[0, 1, 1, 1]
        np.testing.assert_equal(reg.objfcts[0].norm, 0.0 * np.ones(mesh.nC))
        np.testing.assert_equal(reg.objfcts[1].norm, 1.0 * np.ones(mesh.nFx))
        np.testing.assert_equal(reg.objfcts[2].norm, 1.0 * np.ones(mesh.nFy))
        np.testing.assert_equal(reg.objfcts[3].norm, 1.0 * np.ones(mesh.nFz))
        for norm, objfct in zip(reg.norms, reg.objfcts):
            np.testing.assert_equal(norm, objfct.norm)

        reg.norms = None
        for obj in reg.objfcts:
            np.testing.assert_equal(obj.norm, 2.0 * np.ones(obj._weights_shapes[0]))

        # test with setting as multiple arrays
        reg.norms = [
            np.random.rand(mesh.n_cells),
            np.random.rand(mesh.nFx),
            np.random.rand(mesh.nFy),
            np.random.rand(mesh.nFz),
        ]
        for norm, objfct in zip(reg.norms, reg.objfcts):
            np.testing.assert_equal(norm, objfct.norm)

        # test with not setting all as an array
        v = [0, np.random.rand(mesh.nFx), 2, 2]
        reg.norms = v
        np.testing.assert_equal(reg.objfcts[0].norm, 0.0 * np.ones(mesh.nC))
        np.testing.assert_equal(reg.objfcts[1].norm, v[1])
        np.testing.assert_equal(reg.objfcts[2].norm, 2 * np.ones(mesh.nFy))
        np.testing.assert_equal(reg.objfcts[3].norm, 2 * np.ones(mesh.nFz))

        # test resetting if not all work...
        reg.norms = [1, 1, 1, 1]
        with pytest.raises(ValueError):
            reg.norms = [1, 2, 3, 2]
        assert reg.norms == [1, 1, 1, 1]

    def test_linked_properties(self):
        mesh = discretize.TensorMesh([8, 7, 6])
        reg = regularization.WeightedLeastSquares(mesh)

        [
            self.assertTrue(reg.regularization_mesh is fct.regularization_mesh)
            for fct in reg.objfcts
        ]
        [self.assertTrue(reg.mapping is fct.mapping) for fct in reg.objfcts]

        D = reg.regularization_mesh.cellDiffx
        reg.regularization_mesh._cell_gradient_x = 4 * D
        v = np.random.rand(D.shape[1])
        [
            self.assertTrue(
                np.all(
                    reg.regularization_mesh._cell_gradient_x * v
                    == fct.regularization_mesh.cellDiffx * v
                )
            )
            for fct in reg.objfcts
        ]

        active_cells = mesh.gridCC[:, 2] < 0.4
        reg.active_cells = active_cells
        self.assertTrue(np.all(reg.regularization_mesh.active_cells == active_cells))
        [
            self.assertTrue(np.all(reg.active_cells == fct.active_cells))
            for fct in reg.objfcts
        ]

        [
            self.assertTrue(
                np.all(reg.active_cells == fct.regularization_mesh.active_cells)
            )
            for fct in reg.objfcts
        ]

    def test_weighted_least_squares(self):
        mesh = discretize.TensorMesh([8, 7, 6])
        reg = regularization.WeightedLeastSquares(mesh)
        for comp in ["s", "x", "y", "z", "xx", "yy", "zz"]:
            with pytest.raises(TypeError):
                setattr(reg, f"alpha_{comp}", "abc")

            with pytest.raises(ValueError):
                setattr(reg, f"alpha_{comp}", -1)

            if comp in ["x", "y", "z"]:
                with pytest.raises(TypeError):
                    setattr(reg, f"length_scale_{comp}", "abc")

        with pytest.raises(ValueError):
            reg = regularization.WeightedLeastSquares(mesh, alpha_x=1, length_scale_x=1)

        with pytest.raises(ValueError):
            reg = regularization.WeightedLeastSquares(mesh, alpha_y=1, length_scale_y=1)

        with pytest.raises(ValueError):
            reg = regularization.WeightedLeastSquares(mesh, alpha_z=1, length_scale_z=1)

    def test_nC_residual(self):
        # x-direction
        cs, ncx, npad = 1.0, 10.0, 20
        hx = [(cs, ncx), (cs, npad, 1.3)]

        # z direction
        npad = 12
        temp = np.logspace(np.log10(1.0), np.log10(12.0), 19)
        temp_pad = temp[-1] * 1.3 ** np.arange(npad)
        hz = np.r_[temp_pad[::-1], temp[::-1], temp, temp_pad]
        mesh = discretize.CylindricalMesh([hx, 1, hz], "00C")
        active = mesh.cell_centers_z < 0.0

        active = mesh.cell_centers_z < 0.0
        actMap = maps.InjectActiveCells(
            mesh, active, np.log(1e-8), nC=mesh.shape_cells[2]
        )
        mapping = maps.ExpMap(mesh) * maps.SurjectVertical1D(mesh) * actMap

        regMesh = discretize.TensorMesh([mesh.h[2][mapping.maps[-1].indActive]])
        reg = regularization.Simple(regMesh)

        self.assertTrue(reg._nC_residual == regMesh.nC)
        self.assertTrue(all([fct._nC_residual == regMesh.nC for fct in reg.objfcts]))

    def test_active_cells_nc_residual(self):
        # x-direction
        cs, ncx, npad = 1.0, 10.0, 20
        hx = [(cs, ncx), (cs, npad, 1.3)]

        # z direction
        npad = 12
        temp = np.logspace(np.log10(1.0), np.log10(12.0), 19)
        temp_pad = temp[-1] * 1.3 ** np.arange(npad)
        hz = np.r_[temp_pad[::-1], temp[::-1], temp, temp_pad]
        mesh = discretize.CylindricalMesh([hx, 3, hz], "00C")
        active = mesh.cell_centers[:, 2] < 0.0

        reg = regularization.WeightedLeastSquares(mesh, active_cells=active)
        self.assertTrue(reg._nC_residual == len(active.nonzero()[0]))

    def test_base_regularization(self):
        mesh = discretize.TensorMesh([8, 7, 6])

        with pytest.raises(TypeError) as error:
            regularization.BaseRegularization(np.ones(1))

        assert "'regularization_mesh' must be of type " in str(error)

        reg = regularization.BaseRegularization(mesh)
        with pytest.raises(TypeError) as error:
            reg.mapping = np.ones(1)

        assert "'mapping' must be of type " in str(error)

        with pytest.raises(TypeError) as error:
            reg.units = 1

        assert "'units' must be None or type str." in str(error)

        reg.model = 1.0

        assert reg.model.shape[0] == mesh.nC, "Issue setting a model from float."

        with pytest.raises(AttributeError) as error:
            print(reg.f_m(reg.model))

        assert "Regularization class must have a 'f_m' implementation." in str(error)

        with pytest.raises(AttributeError) as error:
            print(reg.f_m_deriv(reg.model))

        assert "Regularization class must have a 'f_m_deriv' implementation." in str(
            error
        )

    def test_smooth_deriv(self):
        mesh = discretize.TensorMesh([8, 7])

        with pytest.raises(ValueError) as error:
            regularization.SmoothnessFirstOrder(mesh, orientation="w")

        assert "Orientation must be 'x', 'y' or 'z'" in str(error)

        with pytest.raises(ValueError) as error:
            regularization.SmoothnessFirstOrder(mesh, orientation="z")

        assert "Mesh must have at least 3 dimensions" in str(error)

        mesh = discretize.TensorMesh([2])

        with pytest.raises(ValueError) as error:
            regularization.SmoothnessFirstOrder(mesh, orientation="y")

        assert "Mesh must have at least 2 dimensions" in str(error)

        smooth_deriv = regularization.SmoothnessFirstOrder(mesh, units="radian")

        with pytest.raises(TypeError) as error:
            smooth_deriv.reference_model_in_smooth = "abc"

        assert "'reference_model_in_smooth must be of type 'bool'." in str(error)

        deriv_angle = smooth_deriv.f_m(np.r_[-np.pi, np.pi])
        np.testing.assert_almost_equal(
            deriv_angle, 0.0, err_msg="Error computing coterminal angle"
        )

    def test_sparse_properties(self):
        mesh = discretize.TensorMesh([8, 7])
        for reg_fun in [regularization.Sparse, regularization.SparseSmoothness]:
            reg = reg_fun(mesh)
            assert reg.irls_threshold == 1e-8  # Default

            with pytest.raises(ValueError):
                reg.irls_threshold = -1

            assert reg.irls_scaled  # Default

            with pytest.raises(TypeError):
                reg.irls_scaled = -1

            assert reg.gradient_type == "total"  # Check default

    def test_vector_amplitude(self):
        n_comp = 4
        mesh = discretize.TensorMesh([8, 7])
        model = np.random.randn(mesh.nC, n_comp)

        with pytest.raises(TypeError, match="'regularization_mesh' must be of type"):
            regularization.VectorAmplitude("abc")

        with pytest.raises(TypeError, match="A 'mapping' of type"):
            regularization.VectorAmplitude(mesh, maps.IdentityMap(mesh))

        reg = regularization.VectorAmplitude(mesh)
        assert len(reg.objfcts[0].mapping.maps) == 1

        with pytest.raises(ValueError, match="All models must be the same size!"):
            wires = ((f"wire{ind}", mesh.nC + ind) for ind in range(n_comp))
            regularization.VectorAmplitude(mesh, maps.Wires(*wires))

        wires = ((f"wire{ind}", mesh.nC) for ind in range(n_comp))

        reg = regularization.VectorAmplitude(mesh, maps.Wires(*wires))

        with pytest.raises(
            ValueError, match=f"must be a tuple of len\({n_comp}\)"  # noqa: W605
        ):
            reg.set_weights(abc=(1.0, 1.0))

        np.testing.assert_almost_equal(
            reg.objfcts[0].f_m(model.flatten(order="F")), np.linalg.norm(model, axis=1)
        )


def test_WeightedLeastSquares():
    mesh = discretize.TensorMesh([3, 4, 5])

    reg = regularization.WeightedLeastSquares(mesh)

    reg.length_scale_x = 0.5
    np.testing.assert_allclose(reg.length_scale_x, 0.5)

    reg.length_scale_y = 0.3
    np.testing.assert_allclose(reg.length_scale_y, 0.3)

    reg.length_scale_z = 0.8
    np.testing.assert_allclose(reg.length_scale_z, 0.8)


if __name__ == "__main__":
    unittest.main()
