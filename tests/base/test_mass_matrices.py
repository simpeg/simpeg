from SimPEG.base import with_property_mass_matrices, BasePDESimulation
from SimPEG import props, maps
import unittest
import discretize
import numpy as np
from scipy.constants import mu_0
from discretize.tests import check_derivative
from discretize.utils import Zero


# define a very simple class...
@with_property_mass_matrices("sigma")
@with_property_mass_matrices("mu")
class SimpleSim(BasePDESimulation):
    sigma, sigmaMap, sigmaDeriv = props.Invertible("Electrical conductivity (S/m)")

    mu, muMap, muDeriv = props.Invertible("Magnetic Permeability", default=mu_0)

    @property
    def deleteTheseOnModelUpdate(self):
        """
        matrices to be deleted if the model for conductivity/resistivity is updated
        """
        toDelete = super().deleteTheseOnModelUpdate
        if self.sigmaMap is not None or self.rhoMap is not None:
            toDelete = toDelete + self._clear_on_sigma_update
        return toDelete


class TestSim(unittest.TestCase):
    def setUp(self):
        self.mesh = discretize.TensorMesh([5, 6, 7])

        self.sim = SimpleSim(self.mesh, sigmaMap=maps.ExpMap())
        self.start_mod = np.log(1e-2 * np.ones(self.mesh.n_cells)) + np.random.randn(
            self.mesh.n_cells
        )

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
        assert sim.MccSigmaDeriv(Zero(), v).__class__ == Zero
        assert sim.MnSigmaDeriv(Zero(), v).__class__ == Zero
        assert sim.MfSigmaDeriv(Zero(), v).__class__ == Zero
        assert sim.MeSigmaDeriv(Zero(), v).__class__ == Zero
        assert sim.MccSigmaIDeriv(Zero(), v).__class__ == Zero
        assert sim.MnSigmaIDeriv(Zero(), v).__class__ == Zero
        assert sim.MfSigmaIDeriv(Zero(), v).__class__ == Zero
        assert sim.MeSigmaIDeriv(Zero(), v).__class__ == Zero

        # Test zero return on v as Zero
        assert sim.MccSigmaDeriv(u_c, Zero()).__class__ == Zero
        assert sim.MnSigmaDeriv(u_n, Zero()).__class__ == Zero
        assert sim.MfSigmaDeriv(u_f, Zero()).__class__ == Zero
        assert sim.MeSigmaDeriv(u_e, Zero()).__class__ == Zero
        assert sim.MccSigmaIDeriv(u_c, Zero()).__class__ == Zero
        assert sim.MnSigmaIDeriv(u_n, Zero()).__class__ == Zero
        assert sim.MfSigmaIDeriv(u_f, Zero()).__class__ == Zero
        assert sim.MeSigmaIDeriv(u_e, Zero()).__class__ == Zero

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
        np.testing.assert_allclose((e_f @ sim.Mf @ e_f)/dim, volume)
        np.testing.assert_allclose((e_e @ sim.Me @ e_e)/dim, volume)

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
        out = sim.MfSigmaDeriv(u, v)
        assert out.shape == (n_f,)
        out = sim.MfSigmaDeriv(u, v[:, None])
        assert out.shape == (n_f,)
        out = sim.MfSigmaDeriv(u[:, None], v)
        assert out.shape == (n_f,)
        out = sim.MfSigmaDeriv(u[:, None], v[:, None])
        assert out.shape == (n_f,)

        # now check passing multiple V's
        out = sim.MfSigmaDeriv(u, v2)
        assert out.shape == (n_f, 4)
        out = sim.MfSigmaDeriv(u[:, None], v2)
        assert out.shape == (n_f, 4)

        # also ensure it properly broadcasted the operation....
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i] = sim.MfSigmaDeriv(u[:, None], v2[:, i])
        np.testing.assert_equal(out, out_2)

        # now check for multiple source polarizations
        out = sim.MfSigmaDeriv(u2, v)
        assert out.shape == (n_f, 2)
        out = sim.MfSigmaDeriv(u2, v[:, None])
        assert out.shape == (n_f, 2)

        # and with multiple RHS
        out = sim.MfSigmaDeriv(u2, v2)
        assert out.shape == (n_f, v2.shape[1], 2)

        # and test broadcasting here...
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i, :] = sim.MfSigmaDeriv(u2, v2[:, i])
        np.testing.assert_equal(out, out_2)

        # test None as v
        UM = sim.MfSigmaDeriv(u)
        np.testing.assert_allclose(UM @ v, sim.MfSigmaDeriv(u, v))

        UM = sim.MfSigmaDeriv(u2)
        np.testing.assert_allclose(
            UM @ v, sim.MfSigmaDeriv(u2, v).reshape(-1, order="F")
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
        out = sim.MfSigmaDeriv(u, v, adjoint=True)
        assert out.shape == (n_c,)
        out = sim.MfSigmaDeriv(u, v[:, None], adjoint=True)
        assert out.shape == (n_c,)
        out = sim.MfSigmaDeriv(u[:, None], v, adjoint=True)
        assert out.shape == (n_c,)
        out = sim.MfSigmaDeriv(u[:, None], v[:, None], adjoint=True)
        assert out.shape == (n_c,)

        # now check passing multiple V's
        out = sim.MfSigmaDeriv(u, v2, adjoint=True)
        assert out.shape == (n_c, 4)
        out = sim.MfSigmaDeriv(u[:, None], v2, adjoint=True)
        assert out.shape == (n_c, 4)

        # also ensure it properly broadcasted the operation....
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i] = sim.MfSigmaDeriv(u, v2[:, i], adjoint=True)
        np.testing.assert_equal(out, out_2)

        # now check for multiple source polarizations
        out = sim.MfSigmaDeriv(u2, v2_2, adjoint=True)
        assert out.shape == (n_c,)
        out = sim.MfSigmaDeriv(u2, v2_2, adjoint=True)
        assert out.shape == (n_c,)

        # and with multiple RHS
        out = sim.MfSigmaDeriv(u2, v3, adjoint=True)
        assert out.shape == (n_c, v3.shape[1])

        # and test broadcasting here...
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i] = sim.MfSigmaDeriv(u2, v3[:, i, :], adjoint=True)
        np.testing.assert_equal(out, out_2)

        # test None as v
        UMT = sim.MfSigmaDeriv(u, adjoint=True)
        np.testing.assert_allclose(UMT @ v, sim.MfSigmaDeriv(u, v, adjoint=True))

        UMT = sim.MfSigmaDeriv(u2, adjoint=True)
        np.testing.assert_allclose(
            UMT @ v2_2.reshape(-1, order="F"), sim.MfSigmaDeriv(u2, v2_2, adjoint=True)
        )

    def test_adjoint_opp_shapes(self):
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
        vJy = v @ sim.MfSigmaDeriv(u, y)
        yJtv = y @ sim.MfSigmaDeriv(u, v, adjoint=True)
        np.testing.assert_allclose(vJy, yJtv)

        # u1, y2 -> v2
        vJy = np.sum(v2 * sim.MfSigmaDeriv(u, y2))
        yJtv = np.sum(y2 * sim.MfSigmaDeriv(u, v2, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # u2, y1 -> v2_2
        vJy = np.sum(v2_2 * sim.MfSigmaDeriv(u2, y))
        yJtv = np.sum(y * sim.MfSigmaDeriv(u2, v2_2, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # u2, y2 -> v3
        vJy = np.sum(v3 * sim.MfSigmaDeriv(u2, y2))
        yJtv = np.sum(y2 * sim.MfSigmaDeriv(u2, v3, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # Also test Inverse opp, just to be sure...
        # u1, y1 -> v1
        vJy = v @ sim.MfSigmaIDeriv(u, y)
        yJtv = y @ sim.MfSigmaIDeriv(u, v, adjoint=True)
        np.testing.assert_allclose(vJy, yJtv)

        # u1, y2 -> v2
        vJy = np.sum(v2 * sim.MfSigmaIDeriv(u, y2))
        yJtv = np.sum(y2 * sim.MfSigmaIDeriv(u, v2, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # u2, y1 -> v2_2
        vJy = np.sum(v2_2 * sim.MfSigmaIDeriv(u2, y))
        yJtv = np.sum(y * sim.MfSigmaIDeriv(u2, v2_2, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # u2, y2 -> v3
        vJy = np.sum(v3 * sim.MfSigmaIDeriv(u2, y2))
        yJtv = np.sum(y2 * sim.MfSigmaIDeriv(u2, v3, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

    def test_Mcc_deriv(self):
        u = np.random.randn(self.mesh.n_cells)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim.MccSigma @ u

            def Jvec(v):
                sim.model = x0
                return sim.MccSigmaDeriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False)

    def test_Mn_deriv(self):
        u = np.random.randn(self.mesh.n_nodes)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim.MnSigma @ u

            def Jvec(v):
                sim.model = x0
                return sim.MnSigmaDeriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False)

    def test_Me_deriv(self):
        u = np.random.randn(self.mesh.n_edges)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim.MeSigma @ u

            def Jvec(v):
                sim.model = x0
                return sim.MeSigmaDeriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False)

    def test_Mf_deriv(self):
        u = np.random.randn(self.mesh.n_faces)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim.MfSigma @ u

            def Jvec(v):
                sim.model = x0
                return sim.MfSigmaDeriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False)

    def test_MccI_deriv(self):
        u = np.random.randn(self.mesh.n_cells)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim.MccSigmaI @ u

            def Jvec(v):
                sim.model = x0
                return sim.MccSigmaIDeriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False)

    def test_MnI_deriv(self):
        u = np.random.randn(self.mesh.n_nodes)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim.MnSigmaI @ u

            def Jvec(v):
                sim.model = x0
                return sim.MnSigmaIDeriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False)

    def test_MeI_deriv(self):
        u = np.random.randn(self.mesh.n_edges)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim.MeSigmaI @ u

            def Jvec(v):
                sim.model = x0
                return sim.MeSigmaIDeriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False)

    def test_MfI_deriv(self):
        u = np.random.randn(self.mesh.n_faces)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim.MfSigmaI @ u

            def Jvec(v):
                sim.model = x0
                return sim.MfSigmaIDeriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False)

    def test_Mcc_adjoint(self):
        n_items = self.mesh.n_cells
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_cells)
        y = np.random.randn(n_items)

        yJv = y @ sim.MccSigmaDeriv(u, v)
        vJty = v @ sim.MccSigmaDeriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)

    def test_Mn_adjoint(self):
        n_items = self.mesh.n_nodes
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_cells)
        y = np.random.randn(n_items)

        yJv = y @ sim.MnSigmaDeriv(u, v)
        vJty = v @ sim.MnSigmaDeriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)

    def test_Me_adjoint(self):
        n_items = self.mesh.n_edges
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_cells)
        y = np.random.randn(n_items)

        yJv = y @ sim.MeSigmaDeriv(u, v)
        vJty = v @ sim.MeSigmaDeriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)

    def test_Mf_adjoint(self):
        n_items = self.mesh.n_faces
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_cells)
        y = np.random.randn(n_items)

        yJv = y @ sim.MfSigmaDeriv(u, v)
        vJty = v @ sim.MfSigmaDeriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)

    def test_MccI_adjoint(self):
        n_items = self.mesh.n_cells
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_cells)
        y = np.random.randn(n_items)

        yJv = y @ sim.MccSigmaIDeriv(u, v)
        vJty = v @ sim.MccSigmaIDeriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)

    def test_MnI_adjoint(self):
        n_items = self.mesh.n_nodes
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_cells)
        y = np.random.randn(n_items)

        yJv = y @ sim.MnSigmaIDeriv(u, v)
        vJty = v @ sim.MnSigmaIDeriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)

    def test_MeI_adjoint(self):
        n_items = self.mesh.n_edges
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_cells)
        y = np.random.randn(n_items)

        yJv = y @ sim.MeSigmaIDeriv(u, v)
        vJty = v @ sim.MeSigmaIDeriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)

    def test_MfI_adjoint(self):
        n_items = self.mesh.n_faces
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_cells)
        y = np.random.randn(n_items)

        yJv = y @ sim.MfSigmaIDeriv(u, v)
        vJty = v @ sim.MfSigmaIDeriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)
