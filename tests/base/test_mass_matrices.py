from simpeg.base import with_property_mass_matrices, BasePDESimulation
from simpeg import props, maps
import unittest
import discretize
import numpy as np
from scipy.constants import mu_0
from discretize.tests import check_derivative
from discretize.utils import Zero
import scipy.sparse as sp
import pytest


# define a very simple class...
@with_property_mass_matrices("conductivity")
@with_property_mass_matrices("mu")
class SimpleSim(BasePDESimulation):
    conductivity, conductivity_map, _con_deriv = props.Invertible(
        "Electrical conductivity (S/m)"
    )

    mu, muMap, muDeriv = props.Invertible("Magnetic Permeability")

    def __init__(
        self,
        mesh,
        survey=None,
        conductivity=None,
        conductivity_map=None,
        mu=mu_0,
        muMap=None,
    ):
        super().__init__(mesh=mesh, survey=survey)
        self.conductivity = conductivity
        self.mu = mu
        self.conductivity_map = conductivity_map
        self.muMap = muMap

    @property
    def _delete_on_model_change(self):
        """
        matrices to be deleted if the model for conductivity/resistivity is updated
        """
        toDelete = super()._delete_on_model_change
        if self.conductivity_map is not None or self.resistivity_map is not None:
            toDelete = toDelete + self._clear_on_conductivity_update
        return toDelete


class TestSim(unittest.TestCase):
    def setUp(self):
        self.mesh = discretize.TensorMesh([5, 6, 7])

        self.sim = SimpleSim(self.mesh, conductivity_map=maps.ExpMap())
        n_cells = self.mesh.n_cells
        self.start_mod = np.log(np.full(n_cells, 1e-2)) + np.random.randn(n_cells)
        self.start_diag_mod = np.r_[
            np.log(np.full(n_cells, 1e-2)),
            np.log(np.full(n_cells, 2e-2)),
            np.log(np.full(n_cells, 3e-2)),
        ] + np.random.randn(3 * n_cells)

        self.sim_full_aniso = SimpleSim(self.mesh, conductivity_map=maps.IdentityMap())

        self.start_full_mod = np.r_[
            np.full(n_cells, 1),
            np.full(n_cells, 2),
            np.full(n_cells, 3),
            np.full(n_cells, -1),
            np.full(n_cells, 1),
            np.full(n_cells, -2),
        ]

    def test_zero_returns(self):
        n_c = self.mesh.n_cells
        n_n = self.mesh.n_nodes
        n_f = self.mesh.n_faces
        n_e = self.mesh.n_edges
        sim = self.sim

        v = np.random.rand(n_c)
        u_c = np.random.rand(n_c)
        u_n = np.random.rand(n_n)
        u_f = np.random.rand(n_f)
        u_e = np.random.rand(n_e)

        # Test zero return on no map
        assert sim.MccMuDeriv(u_c, v).__class__ == Zero
        assert sim.MnMuDeriv(u_n, v).__class__ == Zero
        assert sim.MfMuDeriv(u_f, v).__class__ == Zero
        assert sim.MeMuDeriv(u_e, v).__class__ == Zero
        assert sim.MccMuIDeriv(u_c, v).__class__ == Zero
        assert sim.MnMuIDeriv(u_n, v).__class__ == Zero
        assert sim.MfMuIDeriv(u_f, v).__class__ == Zero
        assert sim.MeMuIDeriv(u_e, v).__class__ == Zero

        # Test zero return on u passed as Zero
        assert sim._Mcc_conductivity_deriv(Zero(), v).__class__ == Zero
        assert sim._Mn_conductivity_deriv(Zero(), v).__class__ == Zero
        assert sim._Mf_conductivity_deriv(Zero(), v).__class__ == Zero
        assert sim._Me_conductivity_deriv(Zero(), v).__class__ == Zero
        assert sim._inv_Mcc_conductivity_deriv(Zero(), v).__class__ == Zero
        assert sim._inv_Mn_conductivity_deriv(Zero(), v).__class__ == Zero
        assert sim._inv_Mf_conductivity_deriv(Zero(), v).__class__ == Zero
        assert sim._inv_Me_conductivity_deriv(Zero(), v).__class__ == Zero

        # Test zero return on v as Zero
        assert sim._Mcc_conductivity_deriv(u_c, Zero()).__class__ == Zero
        assert sim._Mn_conductivity_deriv(u_n, Zero()).__class__ == Zero
        assert sim._Mf_conductivity_deriv(u_f, Zero()).__class__ == Zero
        assert sim._Me_conductivity_deriv(u_e, Zero()).__class__ == Zero
        assert sim._inv_Mcc_conductivity_deriv(u_c, Zero()).__class__ == Zero
        assert sim._inv_Mn_conductivity_deriv(u_n, Zero()).__class__ == Zero
        assert sim._inv_Mf_conductivity_deriv(u_f, Zero()).__class__ == Zero
        assert sim._inv_Me_conductivity_deriv(u_e, Zero()).__class__ == Zero

    def test_simple_mass(self):
        sim = self.sim
        n_c = self.mesh.n_cells
        n_n = self.mesh.n_nodes
        n_f = self.mesh.n_faces
        n_e = self.mesh.n_edges

        e_c = np.ones(n_c)
        e_n = np.ones(n_n)
        e_f = np.ones(n_f)
        e_e = np.ones(n_e)

        volume = np.sum(self.mesh.cell_volumes)
        dim = self.mesh.dim

        # Test volume sum
        np.testing.assert_allclose(e_c @ sim.Mcc @ e_c, volume)
        np.testing.assert_allclose(e_n @ sim.Mn @ e_n, volume)
        np.testing.assert_allclose((e_f @ sim.Mf @ e_f) / dim, volume)
        np.testing.assert_allclose((e_e @ sim.Me @ e_e) / dim, volume)

        # Test matrix simple inverse
        x_c = np.random.rand(n_c)
        x_n = np.random.rand(n_n)
        x_f = np.random.rand(n_f)
        x_e = np.random.rand(n_e)

        np.testing.assert_allclose(x_c, sim.MccI @ (sim.Mcc @ x_c))
        np.testing.assert_allclose(x_n, sim.MnI @ (sim.Mn @ x_n))
        np.testing.assert_allclose(x_f, sim.MfI @ (sim.Mf @ x_f))
        np.testing.assert_allclose(x_e, sim.MeI @ (sim.Me @ x_e))

    def test_forward_expected_shapes(self):
        sim = self.sim
        sim.model = self.start_mod

        n_f = self.mesh.n_faces
        n_c = self.mesh.n_cells
        # if U.shape (n_f, )
        u = np.random.rand(n_f)
        v = np.random.randn(n_c)
        u2 = np.random.rand(n_f, 2)
        v2 = np.random.randn(n_c, 4)

        # These cases should all return an array of shape (n_f, )
        # if V.shape (n_c, )
        out = sim._Mf_conductivity_deriv(u, v)
        assert out.shape == (n_f,)
        out = sim._Mf_conductivity_deriv(u, v[:, None])
        assert out.shape == (n_f,)
        out = sim._Mf_conductivity_deriv(u[:, None], v)
        assert out.shape == (n_f,)
        out = sim._Mf_conductivity_deriv(u[:, None], v[:, None])
        assert out.shape == (n_f,)

        # now check passing multiple V's
        out = sim._Mf_conductivity_deriv(u, v2)
        assert out.shape == (n_f, 4)
        out = sim._Mf_conductivity_deriv(u[:, None], v2)
        assert out.shape == (n_f, 4)

        # also ensure it properly broadcasted the operation....
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i] = sim._Mf_conductivity_deriv(u[:, None], v2[:, i])
        np.testing.assert_equal(out, out_2)

        # now check for multiple source polarizations
        out = sim._Mf_conductivity_deriv(u2, v)
        assert out.shape == (n_f, 2)
        out = sim._Mf_conductivity_deriv(u2, v[:, None])
        assert out.shape == (n_f, 2)

        # and with multiple RHS
        out = sim._Mf_conductivity_deriv(u2, v2)
        assert out.shape == (n_f, v2.shape[1], 2)

        # and test broadcasting here...
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i, :] = sim._Mf_conductivity_deriv(u2, v2[:, i])
        np.testing.assert_equal(out, out_2)

        # test None as v
        UM = sim._Mf_conductivity_deriv(u)
        np.testing.assert_allclose(UM @ v, sim._Mf_conductivity_deriv(u, v))

        UM = sim._Mf_conductivity_deriv(u2)
        np.testing.assert_allclose(
            UM @ v, sim._Mf_conductivity_deriv(u2, v).reshape(-1, order="F")
        )

    def test_forward_anis_expected_shapes(self):
        sim = self.sim
        sim.model = self.start_full_mod

        n_f = self.mesh.n_faces
        n_p = sim.model.size
        # if U.shape (*, )
        u = np.random.rand(n_f)
        v = np.random.randn(n_p)
        u2 = np.random.rand(n_f, 2)
        v2 = np.random.randn(n_p, 4)

        # These cases should all return an array of shape (n_f, )
        # if V.shape (*, )
        out = sim._Mf_conductivity_deriv(u, v)
        assert out.shape == (n_f,)
        out = sim._Mf_conductivity_deriv(u, v[:, None])
        assert out.shape == (n_f,)
        out = sim._Mf_conductivity_deriv(u[:, None], v)
        assert out.shape == (n_f,)
        out = sim._Mf_conductivity_deriv(u[:, None], v[:, None])
        assert out.shape == (n_f,)

        # now check passing multiple V's
        out = sim._Mf_conductivity_deriv(u, v2)
        assert out.shape == (n_f, 4)
        out = sim._Mf_conductivity_deriv(u[:, None], v2)
        assert out.shape == (n_f, 4)

        # also ensure it properly broadcasted the operation....
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i] = sim._Mf_conductivity_deriv(u[:, None], v2[:, i])
        np.testing.assert_equal(out, out_2)

        # now check for multiple source polarizations
        out = sim._Mf_conductivity_deriv(u2, v)
        assert out.shape == (n_f, 2)
        out = sim._Mf_conductivity_deriv(u2, v[:, None])
        assert out.shape == (n_f, 2)

        # and with multiple RHS
        out = sim._Mf_conductivity_deriv(u2, v2)
        assert out.shape == (n_f, v2.shape[1], 2)

        # and test broadcasting here...
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i, :] = sim._Mf_conductivity_deriv(u2, v2[:, i])
        np.testing.assert_equal(out, out_2)

        # test None as v
        UM = sim._Mf_conductivity_deriv(u)
        np.testing.assert_allclose(UM @ v, sim._Mf_conductivity_deriv(u, v))

        UM = sim._Mf_conductivity_deriv(u2)
        np.testing.assert_allclose(
            UM @ v, sim._Mf_conductivity_deriv(u2, v).reshape(-1, order="F")
        )

    def test_adjoint_expected_shapes(self):
        sim = self.sim
        sim.model = self.start_mod

        n_f = self.mesh.n_faces
        n_c = self.mesh.n_cells

        u = np.random.rand(n_f)
        v = np.random.randn(n_f)
        v2 = np.random.randn(n_f, 4)
        u2 = np.random.rand(n_f, 2)
        v2_2 = np.random.randn(n_f, 2)
        v3 = np.random.rand(n_f, 4, 2)

        # These cases should all return an array of shape (n_c, )
        # if V.shape (n_f, )
        out = sim._Mf_conductivity_deriv(u, v, adjoint=True)
        assert out.shape == (n_c,)
        out = sim._Mf_conductivity_deriv(u, v[:, None], adjoint=True)
        assert out.shape == (n_c,)
        out = sim._Mf_conductivity_deriv(u[:, None], v, adjoint=True)
        assert out.shape == (n_c,)
        out = sim._Mf_conductivity_deriv(u[:, None], v[:, None], adjoint=True)
        assert out.shape == (n_c,)

        # now check passing multiple V's
        out = sim._Mf_conductivity_deriv(u, v2, adjoint=True)
        assert out.shape == (n_c, 4)
        out = sim._Mf_conductivity_deriv(u[:, None], v2, adjoint=True)
        assert out.shape == (n_c, 4)

        # also ensure it properly broadcasted the operation....
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i] = sim._Mf_conductivity_deriv(u, v2[:, i], adjoint=True)
        np.testing.assert_equal(out, out_2)

        # now check for multiple source polarizations
        out = sim._Mf_conductivity_deriv(u2, v2_2, adjoint=True)
        assert out.shape == (n_c,)
        out = sim._Mf_conductivity_deriv(u2, v2_2, adjoint=True)
        assert out.shape == (n_c,)

        # and with multiple RHS
        out = sim._Mf_conductivity_deriv(u2, v3, adjoint=True)
        assert out.shape == (n_c, v3.shape[1])

        # and test broadcasting here...
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i] = sim._Mf_conductivity_deriv(u2, v3[:, i, :], adjoint=True)
        np.testing.assert_equal(out, out_2)

        # test None as v
        UMT = sim._Mf_conductivity_deriv(u, adjoint=True)
        np.testing.assert_allclose(
            UMT @ v, sim._Mf_conductivity_deriv(u, v, adjoint=True)
        )

        UMT = sim._Mf_conductivity_deriv(u2, adjoint=True)
        np.testing.assert_allclose(
            UMT @ v2_2.reshape(-1, order="F"),
            sim._Mf_conductivity_deriv(u2, v2_2, adjoint=True),
        )

    def test_adjoint_anis_expected_shapes(self):
        sim = self.sim
        sim.model = self.start_full_mod

        n_f = self.mesh.n_faces
        n_p = sim.model.size

        u = np.random.rand(n_f)
        v = np.random.randn(n_f)
        v2 = np.random.randn(n_f, 4)
        u2 = np.random.rand(n_f, 2)
        v2_2 = np.random.randn(n_f, 2)
        v3 = np.random.rand(n_f, 4, 2)

        # These cases should all return an array of shape (n_c, )
        # if V.shape (n_f, )
        out = sim._Mf_conductivity_deriv(u, v, adjoint=True)
        assert out.shape == (n_p,)
        out = sim._Mf_conductivity_deriv(u, v[:, None], adjoint=True)
        assert out.shape == (n_p,)
        out = sim._Mf_conductivity_deriv(u[:, None], v, adjoint=True)
        assert out.shape == (n_p,)
        out = sim._Mf_conductivity_deriv(u[:, None], v[:, None], adjoint=True)
        assert out.shape == (n_p,)

        # now check passing multiple V's
        out = sim._Mf_conductivity_deriv(u, v2, adjoint=True)
        assert out.shape == (n_p, 4)
        out = sim._Mf_conductivity_deriv(u[:, None], v2, adjoint=True)
        assert out.shape == (n_p, 4)

        # also ensure it properly broadcasted the operation....
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i] = sim._Mf_conductivity_deriv(u, v2[:, i], adjoint=True)
        np.testing.assert_equal(out, out_2)

        # now check for multiple source polarizations
        out = sim._Mf_conductivity_deriv(u2, v2_2, adjoint=True)
        assert out.shape == (n_p,)
        out = sim._Mf_conductivity_deriv(u2, v2_2, adjoint=True)
        assert out.shape == (n_p,)

        # and with multiple RHS
        out = sim._Mf_conductivity_deriv(u2, v3, adjoint=True)
        assert out.shape == (n_p, v3.shape[1])

        # and test broadcasting here...
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i] = sim._Mf_conductivity_deriv(u2, v3[:, i, :], adjoint=True)
        np.testing.assert_equal(out, out_2)

        # test None as v
        UMT = sim._Mf_conductivity_deriv(u, adjoint=True)
        np.testing.assert_allclose(
            UMT @ v, sim._Mf_conductivity_deriv(u, v, adjoint=True)
        )

        UMT = sim._Mf_conductivity_deriv(u2, adjoint=True)
        np.testing.assert_allclose(
            UMT @ v2_2.reshape(-1, order="F"),
            sim._Mf_conductivity_deriv(u2, v2_2, adjoint=True),
        )

    def test_adjoint_opp(self):
        sim = self.sim
        sim.model = self.start_mod

        n_f = self.mesh.n_faces
        n_c = self.mesh.n_cells

        u = np.random.rand(n_f)
        u2 = np.random.rand(n_f, 2)

        y = np.random.rand(n_c)
        y2 = np.random.rand(n_c, 4)

        v = np.random.randn(n_f)
        v2 = np.random.randn(n_f, 4)
        v2_2 = np.random.randn(n_f, 2)
        v3 = np.random.rand(n_f, 4, 2)

        # u1, y1 -> v1
        vJy = v @ sim._Mf_conductivity_deriv(u, y)
        yJtv = y @ sim._Mf_conductivity_deriv(u, v, adjoint=True)
        np.testing.assert_allclose(vJy, yJtv)

        # u1, y2 -> v2
        vJy = np.sum(v2 * sim._Mf_conductivity_deriv(u, y2))
        yJtv = np.sum(y2 * sim._Mf_conductivity_deriv(u, v2, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # u2, y1 -> v2_2
        vJy = np.sum(v2_2 * sim._Mf_conductivity_deriv(u2, y))
        yJtv = np.sum(y * sim._Mf_conductivity_deriv(u2, v2_2, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # u2, y2 -> v3
        vJy = np.sum(v3 * sim._Mf_conductivity_deriv(u2, y2))
        yJtv = np.sum(y2 * sim._Mf_conductivity_deriv(u2, v3, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # Also test Inverse opp, just to be sure...
        # u1, y1 -> v1
        vJy = v @ sim._inv_Mf_conductivity_deriv(u, y)
        yJtv = y @ sim._inv_Mf_conductivity_deriv(u, v, adjoint=True)
        np.testing.assert_allclose(vJy, yJtv)

        # u1, y2 -> v2
        vJy = np.sum(v2 * sim._inv_Mf_conductivity_deriv(u, y2))
        yJtv = np.sum(y2 * sim._inv_Mf_conductivity_deriv(u, v2, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # u2, y1 -> v2_2
        vJy = np.sum(v2_2 * sim._inv_Mf_conductivity_deriv(u2, y))
        yJtv = np.sum(y * sim._inv_Mf_conductivity_deriv(u2, v2_2, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # u2, y2 -> v3
        vJy = np.sum(v3 * sim._inv_Mf_conductivity_deriv(u2, y2))
        yJtv = np.sum(y2 * sim._inv_Mf_conductivity_deriv(u2, v3, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

    def test_anis_adjoint_opp(self):
        sim = self.sim
        sim.model = self.start_full_mod

        n_f = self.mesh.n_faces
        n_p = sim.model.size

        u = np.random.rand(n_f)
        u2 = np.random.rand(n_f, 2)

        y = np.random.rand(n_p)
        y2 = np.random.rand(n_p, 4)

        v = np.random.randn(n_f)
        v2 = np.random.randn(n_f, 4)
        v2_2 = np.random.randn(n_f, 2)
        v3 = np.random.rand(n_f, 4, 2)

        # u1, y1 -> v1
        vJy = v @ sim._Mf_conductivity_deriv(u, y)
        yJtv = y @ sim._Mf_conductivity_deriv(u, v, adjoint=True)
        np.testing.assert_allclose(vJy, yJtv)

        # u1, y2 -> v2
        vJy = np.sum(v2 * sim._Mf_conductivity_deriv(u, y2))
        yJtv = np.sum(y2 * sim._Mf_conductivity_deriv(u, v2, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # u2, y1 -> v2_2
        vJy = np.sum(v2_2 * sim._Mf_conductivity_deriv(u2, y))
        yJtv = np.sum(y * sim._Mf_conductivity_deriv(u2, v2_2, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # u2, y2 -> v3
        vJy = np.sum(v3 * sim._Mf_conductivity_deriv(u2, y2))
        yJtv = np.sum(y2 * sim._Mf_conductivity_deriv(u2, v3, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

    def test_Mcc_deriv(self):
        u = np.random.randn(self.mesh.n_cells)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim._Mcc_conductivity @ u

            def Jvec(v):
                sim.model = x0
                return sim._Mcc_conductivity_deriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False, random_seed=8672354)

    def test_Mn_deriv(self):
        u = np.random.randn(self.mesh.n_nodes)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim._Mn_conductivity @ u

            def Jvec(v):
                sim.model = x0
                return sim._Mn_conductivity_deriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False, random_seed=523876)

    def test_Me_deriv(self):
        u = np.random.randn(self.mesh.n_edges)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim._Me_conductivity @ u

            def Jvec(v):
                sim.model = x0
                return sim._Me_conductivity_deriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False, random_seed=9875163)

    def test_Me_diagonal_anisotropy_deriv(self):
        u = np.random.randn(self.mesh.n_edges)
        sim = self.sim
        x0 = self.start_diag_mod

        def f(x):
            sim.model = x
            d = sim._Me_conductivity @ u

            def Jvec(v):
                sim.model = x0
                return sim._Me_conductivity_deriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False, random_seed=1658372)

    def test_Me_full_anisotropy_deriv(self):
        u = np.random.randn(self.mesh.n_edges)
        sim = self.sim_full_aniso
        x0 = self.start_full_mod

        def f(x):
            sim.model = x
            d = sim._Me_conductivity @ u

            def Jvec(v):
                sim.model = x0
                return sim._Me_conductivity_deriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False, random_seed=9867234)

    def test_Mf_deriv(self):
        u = np.random.randn(self.mesh.n_faces)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim._Mf_conductivity @ u

            def Jvec(v):
                sim.model = x0
                return sim._Mf_conductivity_deriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False, random_seed=10523687)

    def test_Mf_diagonal_anisotropy_deriv(self):
        u = np.random.randn(self.mesh.n_faces)
        sim = self.sim
        x0 = self.start_diag_mod

        def f(x):
            sim.model = x
            d = sim._Mf_conductivity @ u

            def Jvec(v):
                sim.model = x0
                return sim._Mf_conductivity_deriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False, random_seed=19876354)

    def test_Mf_full_anisotropy_deriv(self):
        u = np.random.randn(self.mesh.n_faces)
        sim = self.sim_full_aniso
        x0 = self.start_full_mod

        def f(x):
            sim.model = x
            d = sim._Mf_conductivity @ u

            def Jvec(v):
                sim.model = x0
                return sim._Mf_conductivity_deriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False, random_seed=102309487)

    def test_MccI_deriv(self):
        u = np.random.randn(self.mesh.n_cells)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim._inv_Mcc_conductivity @ u

            def Jvec(v):
                sim.model = x0
                return sim._inv_Mcc_conductivity_deriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False, random_seed=89726354)

    def test_MnI_deriv(self):
        u = np.random.randn(self.mesh.n_nodes)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim._inv_Mn_conductivity @ u

            def Jvec(v):
                sim.model = x0
                return sim._inv_Mn_conductivity_deriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False, random_seed=12503698)

    def test_MeI_deriv(self):
        u = np.random.randn(self.mesh.n_edges)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim._inv_Me_conductivity @ u

            def Jvec(v):
                sim.model = x0
                return sim._inv_Me_conductivity_deriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False, random_seed=5674129834)

    def test_MfI_deriv(self):
        u = np.random.randn(self.mesh.n_faces)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim._inv_Mf_conductivity @ u

            def Jvec(v):
                sim.model = x0
                return sim._inv_Mf_conductivity_deriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False, random_seed=532349)

    def test_Mcc_adjoint(self):
        n_items = self.mesh.n_cells
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_cells)
        y = np.random.randn(n_items)

        yJv = y @ sim._Mcc_conductivity_deriv(u, v)
        vJty = v @ sim._Mcc_conductivity_deriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)

    def test_Mn_adjoint(self):
        n_items = self.mesh.n_nodes
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_cells)
        y = np.random.randn(n_items)

        yJv = y @ sim._Mn_conductivity_deriv(u, v)
        vJty = v @ sim._Mn_conductivity_deriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)

    def test_Me_adjoint(self):
        n_items = self.mesh.n_edges
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_cells)
        y = np.random.randn(n_items)

        yJv = y @ sim._Me_conductivity_deriv(u, v)
        vJty = v @ sim._Me_conductivity_deriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)

    def test_Mf_adjoint(self):
        n_items = self.mesh.n_faces
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_cells)
        y = np.random.randn(n_items)

        yJv = y @ sim._Mf_conductivity_deriv(u, v)
        vJty = v @ sim._Mf_conductivity_deriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)

    def test_MccI_adjoint(self):
        n_items = self.mesh.n_cells
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_cells)
        y = np.random.randn(n_items)

        yJv = y @ sim._inv_Mcc_conductivity_deriv(u, v)
        vJty = v @ sim._inv_Mcc_conductivity_deriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)

    def test_MnI_adjoint(self):
        n_items = self.mesh.n_nodes
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_cells)
        y = np.random.randn(n_items)

        yJv = y @ sim._inv_Mn_conductivity_deriv(u, v)
        vJty = v @ sim._inv_Mn_conductivity_deriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)

    def test_MeI_adjoint(self):
        n_items = self.mesh.n_edges
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_cells)
        y = np.random.randn(n_items)

        yJv = y @ sim._inv_Me_conductivity_deriv(u, v)
        vJty = v @ sim._inv_Me_conductivity_deriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)

    def test_MfI_adjoint(self):
        n_items = self.mesh.n_faces
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_cells)
        y = np.random.randn(n_items)

        yJv = y @ sim._inv_Mf_conductivity_deriv(u, v)
        vJty = v @ sim._inv_Mf_conductivity_deriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)


def test_bad_derivative_stash():
    mesh = discretize.TensorMesh([5, 6, 7])
    sim = SimpleSim(mesh, conductivity_map=maps.ExpMap())
    sim.model = np.random.rand(mesh.n_cells)

    u = np.random.rand(mesh.n_edges)
    v = np.random.rand(mesh.n_cells)

    # This should work
    sim._Me_conductivity_deriv(u, v)
    # stashed derivative operation is a sparse matrix
    assert sp.issparse(sim._Me_Sigma_deriv)

    # Let's set the stashed item as a bad value which would error
    # The user shouldn't cause this to happen, but a developer might.
    sim._Me_Sigma_deriv = [40, 10, 30]

    with pytest.raises(TypeError):
        sim._Me_conductivity_deriv(u, v)
