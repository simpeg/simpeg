from SimPEG.base import (
    with_property_mass_matrices,
    with_surface_property_mass_matrices,
    with_line_property_mass_matrices,
    BasePDESimulation,
)
from SimPEG import props, maps
import unittest
import discretize
import numpy as np
from scipy.constants import mu_0
from discretize.tests import check_derivative
from discretize.utils import Zero
import scipy.sparse as sp
import pytest


# define a very simple class...
@with_property_mass_matrices("sigma")
@with_property_mass_matrices("mu")
@with_surface_property_mass_matrices("tau")
@with_line_property_mass_matrices("kappa")
class SimpleSim(BasePDESimulation):
    sigma, sigmaMap, sigmaDeriv = props.Invertible("Electrical conductivity (S/m)")
    rho, rhoMap, rhoDeriv = props.Invertible("Electrical conductivity (S/m)")
    props.Reciprocal(sigma, rho)
    mu, muMap, muDeriv = props.Invertible("Magnetic Permeability")
    tau, tauMap, tauDeriv = props.Invertible("Face conductivity, conductance (S)")
    kappa, kappaMap, kappaDeriv = props.Invertible(
        "Edge conductivity, conductivity times area (Sm)"
    )

    def __init__(
        self,
        mesh,
        survey=None,
        sigma=None,
        sigmaMap=None,
        mu=mu_0,
        muMap=None,
        tau=None,
        tauMap=None,
        kappa=None,
        kappaMap=None,
    ):
        super().__init__(mesh=mesh, survey=survey)
        self.sigma = sigma
        self.mu = mu
        self.tau = tau
        self.kappa = kappa
        self.sigmaMap = sigmaMap
        self.muMap = muMap
        self.tauMap = tauMap
        self.kappaMap = kappaMap

    @property
    def deleteTheseOnModelUpdate(self):
        """
        matrices to be deleted if the model for conductivity/resistivity is updated
        """
        toDelete = super().deleteTheseOnModelUpdate
        if self.sigmaMap is not None or self.rhoMap is not None:
            toDelete = toDelete + self._clear_on_sigma_update
        if self.tauMap is not None:
            toDelete = toDelete + self._clear_on_tau_update
        if self.kappaMap is not None:
            toDelete = toDelete + self._clear_on_kappa_update
        return toDelete


class TestSim(unittest.TestCase):
    def setUp(self):
        self.mesh = discretize.TensorMesh([5, 6, 7])

        self.sim = SimpleSim(self.mesh, sigmaMap=maps.ExpMap())
        n_cells = self.mesh.n_cells
        self.start_mod = np.log(np.full(n_cells, 1e-2)) + np.random.randn(n_cells)
        self.start_diag_mod = np.r_[
            np.log(np.full(n_cells, 1e-2)),
            np.log(np.full(n_cells, 2e-2)),
            np.log(np.full(n_cells, 3e-2)),
        ] + np.random.randn(3 * n_cells)

        self.sim_full_aniso = SimpleSim(self.mesh, sigmaMap=maps.IdentityMap())

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
        out = sim.MfSigmaDeriv(u, v, adjoint=True)
        assert out.shape == (n_p,)
        out = sim.MfSigmaDeriv(u, v[:, None], adjoint=True)
        assert out.shape == (n_p,)
        out = sim.MfSigmaDeriv(u[:, None], v, adjoint=True)
        assert out.shape == (n_p,)
        out = sim.MfSigmaDeriv(u[:, None], v[:, None], adjoint=True)
        assert out.shape == (n_p,)

        # now check passing multiple V's
        out = sim.MfSigmaDeriv(u, v2, adjoint=True)
        assert out.shape == (n_p, 4)
        out = sim.MfSigmaDeriv(u[:, None], v2, adjoint=True)
        assert out.shape == (n_p, 4)

        # also ensure it properly broadcasted the operation....
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i] = sim.MfSigmaDeriv(u, v2[:, i], adjoint=True)
        np.testing.assert_equal(out, out_2)

        # now check for multiple source polarizations
        out = sim.MfSigmaDeriv(u2, v2_2, adjoint=True)
        assert out.shape == (n_p,)
        out = sim.MfSigmaDeriv(u2, v2_2, adjoint=True)
        assert out.shape == (n_p,)

        # and with multiple RHS
        out = sim.MfSigmaDeriv(u2, v3, adjoint=True)
        assert out.shape == (n_p, v3.shape[1])

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

    def test_Me_diagonal_anisotropy_deriv(self):
        u = np.random.randn(self.mesh.n_edges)
        sim = self.sim
        x0 = self.start_diag_mod

        def f(x):
            sim.model = x
            d = sim.MeSigma @ u

            def Jvec(v):
                sim.model = x0
                return sim.MeSigmaDeriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False)

    def test_Me_full_anisotropy_deriv(self):
        u = np.random.randn(self.mesh.n_edges)
        sim = self.sim_full_aniso
        x0 = self.start_full_mod

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

    def test_Mf_diagonal_anisotropy_deriv(self):
        u = np.random.randn(self.mesh.n_faces)
        sim = self.sim
        x0 = self.start_diag_mod

        def f(x):
            sim.model = x
            d = sim.MfSigma @ u

            def Jvec(v):
                sim.model = x0
                return sim.MfSigmaDeriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False)

    def test_Mf_full_anisotropy_deriv(self):
        u = np.random.randn(self.mesh.n_faces)
        sim = self.sim_full_aniso
        x0 = self.start_full_mod

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


class TestSimSurfaceProperties(unittest.TestCase):
    def setUp(self):
        self.mesh = discretize.TensorMesh([5, 6, 7])

        self.sim = SimpleSim(self.mesh, tauMap=maps.ExpMap())
        self.start_mod = np.log(1e-2 * np.ones(self.mesh.n_faces)) + np.random.randn(
            self.mesh.n_faces
        )

    def test_zero_returns(self):
        n_f = self.mesh.n_faces
        n_e = self.mesh.n_edges
        sim = self.sim

        v = np.random.rand(n_f)
        u_f = np.random.rand(n_f)
        u_e = np.random.rand(n_e)

        # Test zero return on u passed as Zero
        assert sim._MfTauDeriv(Zero(), v).__class__ == Zero
        assert sim._MeTauDeriv(Zero(), v).__class__ == Zero
        assert sim._MfTauIDeriv(Zero(), v).__class__ == Zero
        assert sim._MeTauIDeriv(Zero(), v).__class__ == Zero

        # Test zero return on v as Zero
        assert sim._MfTauDeriv(u_f, Zero()).__class__ == Zero
        assert sim._MeTauDeriv(u_e, Zero()).__class__ == Zero
        assert sim._MfTauIDeriv(u_f, Zero()).__class__ == Zero
        assert sim._MeTauIDeriv(u_e, Zero()).__class__ == Zero

    def test_forward_expected_shapes(self):
        sim = self.sim
        sim.model = self.start_mod

        n_f = self.mesh.n_faces
        # n_c = self.mesh.n_cells
        # if U.shape (n_f, )
        u = np.random.rand(n_f)
        v = np.random.randn(n_f)
        u2 = np.random.rand(n_f, 2)
        v2 = np.random.randn(n_f, 4)

        # These cases should all return an array of shape (n_f, )
        # if V.shape (n_c, )
        out = sim._MfTauDeriv(u, v)
        assert out.shape == (n_f,)
        out = sim._MfTauDeriv(u, v[:, None])
        assert out.shape == (n_f,)
        out = sim._MfTauDeriv(u[:, None], v)
        assert out.shape == (n_f,)
        out = sim._MfTauDeriv(u[:, None], v[:, None])
        assert out.shape == (n_f,)

        # now check passing multiple V's
        out = sim._MfTauDeriv(u, v2)
        assert out.shape == (n_f, 4)
        out = sim._MfTauDeriv(u[:, None], v2)
        assert out.shape == (n_f, 4)

        # also ensure it properly broadcasted the operation....
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i] = sim._MfTauDeriv(u[:, None], v2[:, i])
        np.testing.assert_equal(out, out_2)

        # now check for multiple source polarizations
        out = sim._MfTauDeriv(u2, v)
        assert out.shape == (n_f, 2)
        out = sim._MfTauDeriv(u2, v[:, None])
        assert out.shape == (n_f, 2)

        # and with multiple RHS
        out = sim._MfTauDeriv(u2, v2)
        assert out.shape == (n_f, v2.shape[1], 2)

        # and test broadcasting here...
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i, :] = sim._MfTauDeriv(u2, v2[:, i])
        np.testing.assert_equal(out, out_2)

        # test None as v
        UM = sim._MfTauDeriv(u)
        np.testing.assert_allclose(UM @ v, sim._MfTauDeriv(u, v))

        UM = sim._MfTauDeriv(u2)
        np.testing.assert_allclose(
            UM @ v, sim._MfTauDeriv(u2, v).reshape(-1, order="F")
        )

    def test_adjoint_expected_shapes(self):
        sim = self.sim
        sim.model = self.start_mod

        n_f = self.mesh.n_faces
        # n_c = self.mesh.n_cells

        u = np.random.rand(n_f)
        v = np.random.randn(n_f)
        v2 = np.random.randn(n_f, 4)
        u2 = np.random.rand(n_f, 2)
        v2_2 = np.random.randn(n_f, 2)
        v3 = np.random.rand(n_f, 4, 2)

        # These cases should all return an array of shape (n_c, )
        # if V.shape (n_f, )
        out = sim._MfTauDeriv(u, v, adjoint=True)
        assert out.shape == (n_f,)
        out = sim._MfTauDeriv(u, v[:, None], adjoint=True)
        assert out.shape == (n_f,)
        out = sim._MfTauDeriv(u[:, None], v, adjoint=True)
        assert out.shape == (n_f,)
        out = sim._MfTauDeriv(u[:, None], v[:, None], adjoint=True)
        assert out.shape == (n_f,)

        # now check passing multiple V's
        out = sim._MfTauDeriv(u, v2, adjoint=True)
        assert out.shape == (n_f, 4)
        out = sim._MfTauDeriv(u[:, None], v2, adjoint=True)
        assert out.shape == (n_f, 4)

        # also ensure it properly broadcasted the operation....
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i] = sim._MfTauDeriv(u, v2[:, i], adjoint=True)
        np.testing.assert_equal(out, out_2)

        # now check for multiple source polarizations
        out = sim._MfTauDeriv(u2, v2_2, adjoint=True)
        assert out.shape == (n_f,)
        out = sim._MfTauDeriv(u2, v2_2, adjoint=True)
        assert out.shape == (n_f,)

        # and with multiple RHS
        out = sim._MfTauDeriv(u2, v3, adjoint=True)
        assert out.shape == (n_f, v3.shape[1])

        # and test broadcasting here...
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i] = sim._MfTauDeriv(u2, v3[:, i, :], adjoint=True)
        np.testing.assert_equal(out, out_2)

        # test None as v
        UMT = sim._MfTauDeriv(u, adjoint=True)
        np.testing.assert_allclose(UMT @ v, sim._MfTauDeriv(u, v, adjoint=True))

        UMT = sim._MfTauDeriv(u2, adjoint=True)
        np.testing.assert_allclose(
            UMT @ v2_2.reshape(-1, order="F"), sim._MfTauDeriv(u2, v2_2, adjoint=True)
        )

    def test_adjoint_opp_shapes(self):
        sim = self.sim
        sim.model = self.start_mod

        n_f = self.mesh.n_faces
        # n_c = self.mesh.n_cells

        u = np.random.rand(n_f)
        u2 = np.random.rand(n_f, 2)

        y = np.random.rand(n_f)
        y2 = np.random.rand(n_f, 4)

        v = np.random.randn(n_f)
        v2 = np.random.randn(n_f, 4)
        v2_2 = np.random.randn(n_f, 2)
        v3 = np.random.rand(n_f, 4, 2)

        # u1, y1 -> v1
        vJy = v @ sim._MfTauDeriv(u, y)
        yJtv = y @ sim._MfTauDeriv(u, v, adjoint=True)
        np.testing.assert_allclose(vJy, yJtv)

        # u1, y2 -> v2
        vJy = np.sum(v2 * sim._MfTauDeriv(u, y2))
        yJtv = np.sum(y2 * sim._MfTauDeriv(u, v2, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # u2, y1 -> v2_2
        vJy = np.sum(v2_2 * sim._MfTauDeriv(u2, y))
        yJtv = np.sum(y * sim._MfTauDeriv(u2, v2_2, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # u2, y2 -> v3
        vJy = np.sum(v3 * sim._MfTauDeriv(u2, y2))
        yJtv = np.sum(y2 * sim._MfTauDeriv(u2, v3, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # Also test Inverse opp, just to be sure...
        # u1, y1 -> v1
        vJy = v @ sim._MfTauIDeriv(u, y)
        yJtv = y @ sim._MfTauIDeriv(u, v, adjoint=True)
        np.testing.assert_allclose(vJy, yJtv)

        # u1, y2 -> v2
        vJy = np.sum(v2 * sim._MfTauIDeriv(u, y2))
        yJtv = np.sum(y2 * sim._MfTauIDeriv(u, v2, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # u2, y1 -> v2_2
        vJy = np.sum(v2_2 * sim._MfTauIDeriv(u2, y))
        yJtv = np.sum(y * sim._MfTauIDeriv(u2, v2_2, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # u2, y2 -> v3
        vJy = np.sum(v3 * sim._MfTauIDeriv(u2, y2))
        yJtv = np.sum(y2 * sim._MfTauIDeriv(u2, v3, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

    def test_Me_deriv(self):
        u = np.random.randn(self.mesh.n_edges)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim._MeTau @ u

            def Jvec(v):
                sim.model = x0
                return sim._MeTauDeriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False)

    def test_Mf_deriv(self):
        u = np.random.randn(self.mesh.n_faces)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim._MfTau @ u

            def Jvec(v):
                sim.model = x0
                return sim._MfTauDeriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False)

    def test_MeI_deriv(self):
        u = np.random.randn(self.mesh.n_edges)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim._MeTauI @ u

            def Jvec(v):
                sim.model = x0
                return sim._MeTauIDeriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False)

    def test_MfI_deriv(self):
        u = np.random.randn(self.mesh.n_faces)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim._MfTauI @ u

            def Jvec(v):
                sim.model = x0
                return sim._MfTauIDeriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False)

    def test_Me_adjoint(self):
        n_items = self.mesh.n_edges
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_faces)
        y = np.random.randn(n_items)

        yJv = y @ sim._MeTauDeriv(u, v)
        vJty = v @ sim._MeTauDeriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)

    def test_Mf_adjoint(self):
        n_items = self.mesh.n_faces
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_faces)
        y = np.random.randn(n_items)

        yJv = y @ sim._MfTauDeriv(u, v)
        vJty = v @ sim._MfTauDeriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)

    def test_MeI_adjoint(self):
        n_items = self.mesh.n_edges
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_faces)
        y = np.random.randn(n_items)

        yJv = y @ sim._MeTauIDeriv(u, v)
        vJty = v @ sim._MeTauIDeriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)

    def test_MfI_adjoint(self):
        n_items = self.mesh.n_faces
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_faces)
        y = np.random.randn(n_items)

        yJv = y @ sim._MfTauIDeriv(u, v)
        vJty = v @ sim._MfTauIDeriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)


class TestSimEdgeProperties(unittest.TestCase):
    def setUp(self):
        self.mesh = discretize.TensorMesh([5, 6, 7])

        self.sim = SimpleSim(self.mesh, kappaMap=maps.ExpMap())
        self.start_mod = np.log(1e-2 * np.ones(self.mesh.n_edges)) + np.random.randn(
            self.mesh.n_edges
        )

    def test_zero_returns(self):
        n_e = self.mesh.n_edges
        sim = self.sim

        v = np.random.rand(n_e)
        u_e = np.random.rand(n_e)

        # Test zero return on u passed as Zero
        assert sim._MeKappaDeriv(Zero(), v).__class__ == Zero
        assert sim._MeKappaIDeriv(Zero(), v).__class__ == Zero

        # Test zero return on v as Zero
        assert sim._MeKappaDeriv(u_e, Zero()).__class__ == Zero
        assert sim._MeKappaIDeriv(u_e, Zero()).__class__ == Zero

    def test_forward_expected_shapes(self):
        sim = self.sim
        sim.model = self.start_mod

        n_e = self.mesh.n_edges
        # n_c = self.mesh.n_cells
        # if U.shape (n_f, )
        u = np.random.rand(n_e)
        v = np.random.randn(n_e)
        u2 = np.random.rand(n_e, 2)
        v2 = np.random.randn(n_e, 4)

        # These cases should all return an array of shape (n_f, )
        # if V.shape (n_c, )
        out = sim._MeKappaDeriv(u, v)
        assert out.shape == (n_e,)
        out = sim._MeKappaDeriv(u, v[:, None])
        assert out.shape == (n_e,)
        out = sim._MeKappaDeriv(u[:, None], v)
        assert out.shape == (n_e,)
        out = sim._MeKappaDeriv(u[:, None], v[:, None])
        assert out.shape == (n_e,)

        # now check passing multiple V's
        out = sim._MeKappaDeriv(u, v2)
        assert out.shape == (n_e, 4)
        out = sim._MeKappaDeriv(u[:, None], v2)
        assert out.shape == (n_e, 4)

        # also ensure it properly broadcasted the operation....
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i] = sim._MeKappaDeriv(u[:, None], v2[:, i])
        np.testing.assert_equal(out, out_2)

        # now check for multiple source polarizations
        out = sim._MeKappaDeriv(u2, v)
        assert out.shape == (n_e, 2)
        out = sim._MeKappaDeriv(u2, v[:, None])
        assert out.shape == (n_e, 2)

        # and with multiple RHS
        out = sim._MeKappaDeriv(u2, v2)
        assert out.shape == (n_e, v2.shape[1], 2)

        # and test broadcasting here...
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i, :] = sim._MeKappaDeriv(u2, v2[:, i])
        np.testing.assert_equal(out, out_2)

        # test None as v
        UM = sim._MeKappaDeriv(u)
        np.testing.assert_allclose(UM @ v, sim._MeKappaDeriv(u, v))

        UM = sim._MeKappaDeriv(u2)
        np.testing.assert_allclose(
            UM @ v, sim._MeKappaDeriv(u2, v).reshape(-1, order="F")
        )

    def test_adjoint_expected_shapes(self):
        sim = self.sim
        sim.model = self.start_mod

        n_e = self.mesh.n_edges
        # n_c = self.mesh.n_cells

        u = np.random.rand(n_e)
        v = np.random.randn(n_e)
        v2 = np.random.randn(n_e, 4)
        u2 = np.random.rand(n_e, 2)
        v2_2 = np.random.randn(n_e, 2)
        v3 = np.random.rand(n_e, 4, 2)

        # These cases should all return an array of shape (n_c, )
        # if V.shape (n_f, )
        out = sim._MeKappaDeriv(u, v, adjoint=True)
        assert out.shape == (n_e,)
        out = sim._MeKappaDeriv(u, v[:, None], adjoint=True)
        assert out.shape == (n_e,)
        out = sim._MeKappaDeriv(u[:, None], v, adjoint=True)
        assert out.shape == (n_e,)
        out = sim._MeKappaDeriv(u[:, None], v[:, None], adjoint=True)
        assert out.shape == (n_e,)

        # now check passing multiple V's
        out = sim._MeKappaDeriv(u, v2, adjoint=True)
        assert out.shape == (n_e, 4)
        out = sim._MeKappaDeriv(u[:, None], v2, adjoint=True)
        assert out.shape == (n_e, 4)

        # also ensure it properly broadcasted the operation....
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i] = sim._MeKappaDeriv(u, v2[:, i], adjoint=True)
        np.testing.assert_equal(out, out_2)

        # now check for multiple source polarizations
        out = sim._MeKappaDeriv(u2, v2_2, adjoint=True)
        assert out.shape == (n_e,)
        out = sim._MeKappaDeriv(u2, v2_2, adjoint=True)
        assert out.shape == (n_e,)

        # and with multiple RHS
        out = sim._MeKappaDeriv(u2, v3, adjoint=True)
        assert out.shape == (n_e, v3.shape[1])

        # and test broadcasting here...
        out_2 = np.empty_like(out)
        for i in range(v2.shape[1]):
            out_2[:, i] = sim._MeKappaDeriv(u2, v3[:, i, :], adjoint=True)
        np.testing.assert_equal(out, out_2)

        # test None as v
        UMT = sim._MeKappaDeriv(u, adjoint=True)
        np.testing.assert_allclose(UMT @ v, sim._MeKappaDeriv(u, v, adjoint=True))

        UMT = sim._MeKappaDeriv(u2, adjoint=True)
        np.testing.assert_allclose(
            UMT @ v2_2.reshape(-1, order="F"), sim._MeKappaDeriv(u2, v2_2, adjoint=True)
        )

    def test_adjoint_opp_shapes(self):
        sim = self.sim
        sim.model = self.start_mod

        n_e = self.mesh.n_edges
        # n_c = self.mesh.n_cells

        u = np.random.rand(n_e)
        u2 = np.random.rand(n_e, 2)

        y = np.random.rand(n_e)
        y2 = np.random.rand(n_e, 4)

        v = np.random.randn(n_e)
        v2 = np.random.randn(n_e, 4)
        v2_2 = np.random.randn(n_e, 2)
        v3 = np.random.rand(n_e, 4, 2)

        # u1, y1 -> v1
        vJy = v @ sim._MeKappaDeriv(u, y)
        yJtv = y @ sim._MeKappaDeriv(u, v, adjoint=True)
        np.testing.assert_allclose(vJy, yJtv)

        # u1, y2 -> v2
        vJy = np.sum(v2 * sim._MeKappaDeriv(u, y2))
        yJtv = np.sum(y2 * sim._MeKappaDeriv(u, v2, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # u2, y1 -> v2_2
        vJy = np.sum(v2_2 * sim._MeKappaDeriv(u2, y))
        yJtv = np.sum(y * sim._MeKappaDeriv(u2, v2_2, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # u2, y2 -> v3
        vJy = np.sum(v3 * sim._MeKappaDeriv(u2, y2))
        yJtv = np.sum(y2 * sim._MeKappaDeriv(u2, v3, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # Also test Inverse opp, just to be sure...
        # u1, y1 -> v1
        vJy = v @ sim._MeKappaIDeriv(u, y)
        yJtv = y @ sim._MeKappaIDeriv(u, v, adjoint=True)
        np.testing.assert_allclose(vJy, yJtv)

        # u1, y2 -> v2
        vJy = np.sum(v2 * sim._MeKappaIDeriv(u, y2))
        yJtv = np.sum(y2 * sim._MeKappaIDeriv(u, v2, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # u2, y1 -> v2_2
        vJy = np.sum(v2_2 * sim._MeKappaIDeriv(u2, y))
        yJtv = np.sum(y * sim._MeKappaIDeriv(u2, v2_2, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

        # u2, y2 -> v3
        vJy = np.sum(v3 * sim._MeKappaIDeriv(u2, y2))
        yJtv = np.sum(y2 * sim._MeKappaIDeriv(u2, v3, adjoint=True))
        np.testing.assert_allclose(vJy, yJtv)

    def test_Me_deriv(self):
        u = np.random.randn(self.mesh.n_edges)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim._MeKappa @ u

            def Jvec(v):
                sim.model = x0
                return sim._MeKappaDeriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False)

    def test_MeI_deriv(self):
        u = np.random.randn(self.mesh.n_edges)
        sim = self.sim
        x0 = self.start_mod

        def f(x):
            sim.model = x
            d = sim._MeKappaI @ u

            def Jvec(v):
                sim.model = x0
                return sim._MeKappaIDeriv(u, v)

            return d, Jvec

        assert check_derivative(f, x0=x0, num=3, plotIt=False)

    def test_Me_adjoint(self):
        n_items = self.mesh.n_edges
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_edges)
        y = np.random.randn(n_items)

        yJv = y @ sim._MeKappaDeriv(u, v)
        vJty = v @ sim._MeKappaDeriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)

    def test_MeI_adjoint(self):
        n_items = self.mesh.n_edges
        u = np.random.randn(n_items)
        sim = self.sim
        sim.model = self.start_mod

        v = np.random.randn(self.mesh.n_edges)
        y = np.random.randn(n_items)

        yJv = y @ sim._MeKappaIDeriv(u, v)
        vJty = v @ sim._MeKappaIDeriv(u, y, adjoint=True)
        np.testing.assert_allclose(yJv, vJty)


def test_bad_derivative_stash():
    mesh = discretize.TensorMesh([5, 6, 7])
    sim = SimpleSim(mesh, sigmaMap=maps.ExpMap())
    sim.model = np.random.rand(mesh.n_cells)

    u = np.random.rand(mesh.n_edges)
    v = np.random.rand(mesh.n_cells)

    # This should work
    sim.MeSigmaDeriv(u, v)
    # stashed derivative operation is a sparse matrix
    assert sp.issparse(sim._Me_Sigma_deriv)

    # Let's set the stashed item as a bad value which would error
    # The user shouldn't cause this to happen, but a developer might.
    sim._Me_Sigma_deriv = [40, 10, 30]

    with pytest.raises(TypeError):
        sim.MeSigmaDeriv(u, v)


if __name__ == "__main__":
    unittest.main()
