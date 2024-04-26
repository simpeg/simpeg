# Test functions
import pytest
import unittest
import numpy as np
from simpeg import tests, mkvc
from simpeg.electromagnetics import natural_source as nsem
from scipy.constants import mu_0

TOLr = 5e-2
TOL = 1e-4
FLR = 1e-20  # "zero", so if residual below this --> pass regardless of order
CONDUCTIVITY = 1e1
MU = mu_0


@pytest.fixture()
def model_simulation_tuple():
    return nsem.utils.test_utils.setupSimpegNSEM_PrimarySecondary(
        nsem.utils.test_utils.halfSpace(1e-2), [0.1], comp="All", singleFreq=False
    )


# Test the Jvec derivative
@pytest.mark.parametrize("weights", [True, False])
def test_Jtjdiag(model_simulation_tuple, weights):
    model, simulation = model_simulation_tuple
    W = None
    if weights:
        W = np.eye(simulation.survey.nD)

    J = simulation.getJ(model)
    if weights:
        J = W @ J

    Jtjdiag = simulation.getJtJdiag(model, W=W)
    np.testing.assert_allclose(Jtjdiag, np.sum(J * J, axis=0))


def test_Jtjdiag_clearing(model_simulation_tuple):
    model, simulation = model_simulation_tuple
    J1 = simulation.getJ(model)
    Jtjdiag1 = simulation.getJtJdiag(model)

    m2 = model + 2
    J2 = simulation.getJ(m2)
    Jtjdiag2 = simulation.getJtJdiag(m2)

    assert J1 is not J2
    assert Jtjdiag1 is not Jtjdiag2


def test_Jmatrix(model_simulation_tuple):
    model, simulation = model_simulation_tuple
    rng = np.random.default_rng(4421)
    # create random vector
    vec = rng.standard_normal(simulation.survey.nD)

    # create the J matrix
    J1 = simulation.getJ(model)
    Jmatrix_vec = J1.T @ vec

    # compare to JTvec function
    jtvec = simulation.Jtvec(model, v=vec)

    np.testing.assert_allclose(Jmatrix_vec, jtvec)


# Test the Jvec derivative
def DerivJvecTest(inputSetup, comp="All", freq=False, expMap=True):
    m, simulation = nsem.utils.test_utils.setupSimpegNSEM_PrimarySecondary(
        inputSetup, [freq], comp=comp, singleFreq=False
    )
    print("Using {0} solver for the simulation".format(simulation.solver))
    print(
        "Derivative test of Jvec for eForm primary/secondary for {} comp at {}\n".format(
            comp, simulation.survey.frequencies
        )
    )
    # simulation.mapping = Maps.ExpMap(simulation.mesh)
    # simulation.sigmaPrimary = np.log(sigBG)
    # x0 = np.log(simulation.sigmaPrimary)
    # cond = sig[0]
    # x0 = np.log(np.ones(simulation.mesh.nC)*cond)
    # simulation.sigmaPrimary = x0
    # if True:
    #     x0  = x0 + np.random.randn(simulation.mesh.nC)*cond*1e-1

    def fun(x):
        return simulation.dpred(x), lambda x: simulation.Jvec(m, x)

    return tests.check_derivative(fun, m, num=3, plotIt=False, eps=FLR)


def DerivProjfieldsTest(inputSetup, comp="All", freq=False):
    survey, simulation = nsem.utils.test_utils.setupSimpegNSEM_ePrimSec(
        inputSetup, comp, freq
    )
    print("Derivative test of data projection for eFormulation primary/secondary\n")
    # simulation.mapping = Maps.ExpMap(simulation.mesh)
    # Initate things for the derivs Test
    src = survey.source_list[0]
    np.random.seed(1983)
    u0x = np.random.randn(survey.mesh.nE) + np.random.randn(survey.mesh.nE) * 1j
    u0y = np.random.randn(survey.mesh.nE) + np.random.randn(survey.mesh.nE) * 1j
    u0 = np.vstack((mkvc(u0x, 2), mkvc(u0y, 2)))
    f0 = simulation.fieldsPair(survey.mesh, survey)
    # u0 = np.hstack((mkvc(u0_px,2),mkvc(u0_py,2)))
    f0[src, "e_pxSolution"] = u0[: len(u0) / 2]  # u0x
    f0[src, "e_pySolution"] = u0[len(u0) / 2 : :]  # u0y

    def fun(u):
        f = simulation.fieldsPair(survey.mesh, survey)
        f[src, "e_pxSolution"] = u[: len(u) / 2]
        f[src, "e_pySolution"] = u[len(u) / 2 : :]
        return (
            rx.eval(src, survey.mesh, f),
            lambda t: rx.evalDeriv(src, survey.mesh, f0, mkvc(t, 2)),
        )

    return tests.check_derivative(fun, u0, num=3, plotIt=False, eps=FLR)


class NSEM_DerivTests(unittest.TestCase):
    def setUp(self):
        pass

    # Do a derivative test of Jvec
    def test_derivJvec_impedanceAll(self):
        self.assertTrue(
            DerivJvecTest(nsem.utils.test_utils.halfSpace(1e-2), "Imp", 0.1)
        )

    def test_derivJvec_zxxr(self):
        self.assertTrue(DerivJvecTest(nsem.utils.test_utils.halfSpace(1e-2), "xx", 0.1))

    def test_derivJvec_zxyi(self):
        self.assertTrue(DerivJvecTest(nsem.utils.test_utils.halfSpace(1e-2), "xy", 0.1))

    def test_derivJvec_zyxr(self):
        self.assertTrue(DerivJvecTest(nsem.utils.test_utils.halfSpace(1e-2), "yx", 0.1))

    def test_derivJvec_zyyi(self):
        self.assertTrue(DerivJvecTest(nsem.utils.test_utils.halfSpace(1e-2), "yy", 0.1))

    # apparent res and phase
    def test_derivJvec_resAll(self):
        self.assertTrue(
            DerivJvecTest(nsem.utils.test_utils.halfSpace(1e-2), "Res", 0.1)
        )

    # Tipper
    def test_derivJvec_tipperAll(self):
        self.assertTrue(
            DerivJvecTest(nsem.utils.test_utils.halfSpace(1e-2), "Tip", 0.1)
        )

    def test_derivJvec_tzxr(self):
        self.assertTrue(DerivJvecTest(nsem.utils.test_utils.halfSpace(1e-2), "zx", 0.1))

    def test_derivJvec_tzyi(self):
        self.assertTrue(DerivJvecTest(nsem.utils.test_utils.halfSpace(1e-2), "zy", 0.1))


if __name__ == "__main__":
    unittest.main()
