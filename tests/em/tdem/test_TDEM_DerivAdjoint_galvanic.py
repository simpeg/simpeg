import unittest
import numpy as np
import discretize
from SimPEG import maps, SolverLU, tests
from SimPEG.electromagnetics import time_domain as tdem
from pymatsolver import Pardiso as Solver

plotIt = False

testDeriv = True
testAdjoint = True

TOL = 0.5

np.random.seed(10)


def setUp_TDEM(prbtype="ElectricField", rxcomp="ElectricFieldx", src_z=0.0):
    cs = 5.0
    ncx = 8
    ncy = 8
    ncz = 8
    npad = 3
    pf = 1.3

    mesh = discretize.TensorMesh(
        [
            [(cs, npad, -pf), (cs, ncx), (cs, npad, pf)],
            [(cs, npad, -pf), (cs, ncy), (cs, npad, pf)],
            [(cs, npad, -pf), (cs, ncz), (cs, npad, pf)],
        ],
        "CCC",
    )

    active = mesh.vectorCCz < 0.0
    activeMap = maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    mapping = maps.ExpMap(mesh) * maps.SurjectVertical1D(mesh) * activeMap

    rxOffset = 0.0
    rxlocs = discretize.utils.ndgrid(
        [np.r_[-17.5, -15, 15, 17.5], np.r_[10], np.r_[-0.1]]
    )
    rxtimes = np.logspace(-4, -3, 20)
    rx = getattr(tdem.Rx, "Point{}".format(rxcomp[:-1]))(
        locations=rxlocs, times=rxtimes, orientation=rxcomp[-1]
    )
    Aloc = np.r_[-10, 0.0, src_z]
    Bloc = np.r_[10, 0.0, src_z]
    srcloc = np.vstack((Aloc, Bloc))

    src = tdem.Src.LineCurrent(
        [rx], location=srcloc, waveform=tdem.Src.StepOffWaveform()
    )
    survey = tdem.Survey([src])

    time_steps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]

    m = np.log(5e-1) * np.ones(mapping.nP) + 1e-3 * np.random.randn(mapping.nP)

    prb = getattr(tdem, "Simulation3D{}".format(prbtype))(
        mesh, survey=survey, time_steps=time_steps, sigmaMap=mapping
    )
    prb.solver = Solver

    return prb, m, mesh


class TDEM_DerivTests(unittest.TestCase):

    # ====== TEST Jvec ========== #

    if testDeriv:

        def JvecTest(self, prbtype, rxcomp, src_z=0.0):
            prb, m, mesh = setUp_TDEM(prbtype, rxcomp, src_z)

            def derChk(m):
                return [prb.dpred(m), lambda mx: prb.Jvec(m, mx)]

            print("test_Jvec_{prbtype}_{rxcomp}".format(prbtype=prbtype, rxcomp=rxcomp))
            tests.checkDerivative(derChk, m, plotIt=False, num=2, eps=1e-20)

        def test_Jvec_e_dbzdt(self):
            self.JvecTest("ElectricField", "MagneticFluxTimeDerivativez")

        def test_Jvec_e_ex(self):
            self.JvecTest("ElectricField", "ElectricFieldx")

        def test_Jvec_e_ey(self):
            self.JvecTest("ElectricField", "ElectricFieldy")

        def test_Jvec_j_dbzdt(self):
            self.JvecTest("CurrentDensity", "MagneticFluxTimeDerivativez")

        def test_Jvec_j_ex(self):
            self.JvecTest("CurrentDensity", "ElectricFieldx")

        def test_Jvec_j_ey(self):
            self.JvecTest("CurrentDensity", "ElectricFieldy")

    # ====== TEST Jtvec ========== #

    if testAdjoint:

        def JvecVsJtvecTest(
            self, prbtype="MagneticFluxDensity", rxcomp="bz", src_z=0.0
        ):

            print("\nAdjoint Testing Jvec, Jtvec prob {}, {}".format(prbtype, rxcomp))

            prb, m0, mesh = setUp_TDEM(prbtype, rxcomp, src_z)
            m = np.random.rand(prb.sigmaMap.nP)
            d = np.random.randn(prb.survey.nD)

            print(m.shape, d.shape, m0.shape)

            V1 = d.dot(prb.Jvec(m0, m))
            V2 = m.dot(prb.Jtvec(m0, d))
            tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.0
            passed = np.abs(V1 - V2) < tol

            print(
                "AdjointTest {prbtype} {v1} {v2} {passed}".format(
                    prbtype=prbtype, v1=V1, v2=V2, passed=passed
                )
            )
            self.assertTrue(passed)

        def test_Jvec_adjoint_e_dbzdt(self):
            self.JvecVsJtvecTest("ElectricField", "MagneticFluxTimeDerivativez")

        def test_Jvec_adjoint_e_ex(self):
            self.JvecVsJtvecTest("ElectricField", "ElectricFieldx")

        def test_Jvec_adjoint_e_ey(self):
            self.JvecVsJtvecTest("ElectricField", "ElectricFieldy")

        def test_Jvec_adjoint_j_dbzdt(self):
            self.JvecVsJtvecTest(
                "CurrentDensity", "MagneticFluxTimeDerivativez", src_z=-2.5
            )

        def test_Jvec_adjoint_j_ex(self):
            self.JvecVsJtvecTest("CurrentDensity", "ElectricFieldx", src_z=-2.5)

        def test_Jvec_adjoint_j_ey(self):
            self.JvecVsJtvecTest("CurrentDensity", "ElectricFieldy", src_z=-2.5)
