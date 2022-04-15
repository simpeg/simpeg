from __future__ import division, print_function
import unittest
import numpy as np
import time
import discretize
from SimPEG import maps, tests
from SimPEG.electromagnetics import time_domain as tdem
from SimPEG.electromagnetics import utils
from scipy.interpolate import interp1d
from pymatsolver import Pardiso as Solver
from discretize.utils import unpack_widths

plotIt = False

testDeriv = True
testAdjoint = False

TOL = 1e-4
EPS = 1e-20
np.random.seed(4)


def get_mesh():
    cs = 5.0
    ncx = 8
    ncy = 8
    ncz = 8
    npad = 4

    return discretize.TensorMesh(
        [
            [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)],
            [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)],
            [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)],
        ],
        "CCC",
    )


def get_mapping(mesh):
    active = mesh.vectorCCz < 0.0
    activeMap = maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    return maps.ExpMap(mesh) * maps.SurjectVertical1D(mesh) * activeMap


def get_prob(mesh, mapping, formulation, **kwargs):
    prb = getattr(tdem, "Simulation3D{}".format(formulation))(
        mesh, sigmaMap=mapping, **kwargs
    )
    prb.solver = Solver
    return prb


def get_survey(times, t0):

    out = utils.VTEMFun(times, 0.00595, 0.006, 100)
    wavefun = interp1d(times, out)

    waveform = tdem.Src.RawWaveform(offTime=t0, waveFct=wavefun)
    src = tdem.Src.MagDipole([], waveform=waveform, location=np.array([0.0, 0.0, 0.0]))

    return tdem.Survey([src])


class Base_DerivAdjoint_Test(unittest.TestCase):

    t0 = 0.006

    @classmethod
    def setUpClass(self):
        # create a prob where we will store the fields
        mesh = get_mesh()
        mapping = get_mapping(mesh)
        time_steps = [(1e-3, 5), (1e-4, 5), (5e-5, 10), (5e-5, 10), (1e-4, 10)]
        t_mesh = discretize.TensorMesh([time_steps])
        times = t_mesh.nodes_x
        self.survey = get_survey(times, self.t0)

        self.prob = get_prob(
            mesh, mapping, self.formulation, survey=self.survey, time_steps=time_steps
        )
        self.m = np.log(1e-1) * np.ones(self.prob.sigmaMap.nP)

        print("Solving Fields for problem {}".format(self.formulation))
        t = time.time()
        self.fields = self.prob.fields(self.m)
        print("... done. Time: {}\n".format(time.time() - t))

        # create a prob where will be re-computing fields at each jvec
        # iteration
        mesh = get_mesh()
        mapping = get_mapping(mesh)
        self.surveyfwd = get_survey(times, self.t0)
        self.probfwd = get_prob(
            mesh,
            mapping,
            self.formulation,
            survey=self.surveyfwd,
            time_steps=time_steps,
        )

    def get_rx(self, rxcomp):
        rxOffset = 15.0

        timerx = self.t0 + np.logspace(-5, -3, 20)
        return getattr(tdem.Rx, "Point{}".format(rxcomp[:-1]))(
            np.array([[rxOffset, 0.0, 0.0]]), timerx, rxcomp[-1]
        )

    def set_receiver_list(self, rxcomp):
        # append the right rxlist to the surveys
        rx = [self.get_rx(rxcomp)]
        rxfwd = [self.get_rx(rxcomp)]

        # append to srclists
        for source_list, rxlist in zip(
            [self.survey.source_list, self.surveyfwd.source_list], [rx, rxfwd]
        ):
            for src in source_list:
                src.receiver_list = rxlist

    def JvecTest(self, rxcomp):
        self.set_receiver_list(rxcomp)

        def derChk(m):
            return [
                self.probfwd.dpred(m),
                lambda mx: self.prob.Jvec(self.m, mx, f=self.fields),
            ]

        print(
            "test_Jvec_{prbtype}_{rxcomp}".format(
                prbtype=self.formulation, rxcomp=rxcomp
            )
        )
        tests.checkDerivative(derChk, self.m, plotIt=False, num=2, eps=1e-20)

    def JvecVsJtvecTest(self, rxcomp):
        self.set_receiver_list(rxcomp)
        print(
            "\nAdjoint Testing Jvec, Jtvec prob {}, {}".format(self.formulation, rxcomp)
        )

        m = np.random.rand(self.prob.sigmaMap.nP)
        d = np.random.randn(self.prob.survey.nD)
        V1 = d.dot(self.prob.Jvec(self.m, m, f=self.fields))
        V2 = m.dot(self.prob.Jtvec(self.m, d, f=self.fields))
        tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.0
        passed = np.abs(V1 - V2) < tol

        print(
            "    {v1} {v2} {passed}".format(
                prbtype=self.formulation, v1=V1, v2=V2, passed=passed
            )
        )
        self.assertTrue(passed)


class DerivAdjoint_E(Base_DerivAdjoint_Test):

    formulation = "ElectricField"

    if testDeriv:

        def test_Jvec_e_dbxdt(self):
            self.JvecTest("MagneticFluxTimeDerivativex")

        def test_Jvec_e_dbzdt(self):
            self.JvecTest("MagneticFluxTimeDerivativez")

        def test_Jvec_e_ey(self):
            self.JvecTest("ElectricFieldy")

    if testAdjoint:

        def test_Jvec_adjoint_e_ey(self):
            self.JvecVsJtvecTest("MagneticFluxTimeDerivativex")

        def test_Jvec_adjoint_e_ey(self):
            self.JvecVsJtvecTest("MagneticFluxTimeDerivativez")

        def test_Jvec_adjoint_e_ey(self):
            self.JvecVsJtvecTest("ElectricFieldy")


class DerivAdjoint_B(Base_DerivAdjoint_Test):

    formulation = "MagneticFluxDensity"

    if testDeriv:

        def test_Jvec_b_bx(self):
            self.JvecTest("MagneticFluxDensityx")

        def test_Jvec_b_bz(self):
            self.JvecTest("MagneticFluxDensityz")

        def test_Jvec_b_dbdtx(self):
            self.JvecTest("MagneticFluxTimeDerivativex")

        def test_Jvec_b_dbdtz(self):
            self.JvecTest("MagneticFluxTimeDerivativez")

        def test_Jvec_b_ey(self):
            self.JvecTest("ElectricFieldy")

    if testAdjoint:

        def test_Jvec_adjoint_b_bx(self):
            self.JvecVsJtvecTest("MagneticFluxDensityx")

        def test_Jvec_adjoint_b_bz(self):
            self.JvecVsJtvecTest("MagneticFluxDensityz")

        def test_Jvec_adjoint_b_dbdtz(self):
            self.JvecVsJtvecTest("MagneticFluxTimeDerivativex")

        def test_Jvec_adjoint_b_dbdtx(self):
            self.JvecVsJtvecTest("MagneticFluxTimeDerivativez")

        def test_Jvec_adjoint_b_ey(self):
            self.JvecVsJtvecTest("ElectricFieldy")


class DerivAdjoint_H(Base_DerivAdjoint_Test):

    formulation = "MagneticField"

    if testDeriv:

        def test_Jvec_h_hx(self):
            self.JvecTest("MagneticFieldx")

        def test_Jvec_h_hz(self):
            self.JvecTest("MagneticFieldz")

        def test_Jvec_h_dhdtx(self):
            self.JvecTest("MagneticFieldTimeDerivativex")

        def test_Jvec_h_dhdtz(self):
            self.JvecTest("MagneticFieldTimeDerivativez")

    if testAdjoint:

        def test_Jvec_adjoint_h_hx(self):
            self.JvecVsJtvecTest("MagneticFieldx")

        def test_Jvec_adjoint_h_hz(self):
            self.JvecVsJtvecTest("MagneticFieldz")

        def test_Jvec_adjoint_h_dhdtx(self):
            self.JvecVsJtvecTest("MagneticFieldTimeDerivativex")

        def test_Jvec_adjoint_h_dhdtz(self):
            self.JvecVsJtvecTest("MagneticFieldTimeDerivativez")

        def test_Jvec_adjoint_h_jy(self):
            self.JvecVsJtvecTest("CurrentDensityy")


class DerivAdjoint_J(Base_DerivAdjoint_Test):

    formulation = "CurrentDensity"

    if testDeriv:

        def test_Jvec_j_jy(self):
            self.JvecTest("CurrentDensityy")

        def test_Jvec_j_dhdtx(self):
            self.JvecTest("MagneticFieldTimeDerivativex")

        def test_Jvec_j_dhdtz(self):
            self.JvecTest("MagneticFieldTimeDerivativez")

    if testAdjoint:

        def test_Jvec_adjoint_j_jy(self):
            self.JvecVsJtvecTest("CurrentDensityy")

        def test_Jvec_adjoint_j_dhdtx(self):
            self.JvecVsJtvecTest("MagneticFieldTimeDerivativex")

        def test_Jvec_adjoint_j_dhdtz(self):
            self.JvecVsJtvecTest("MagneticFieldTimeDerivativez")


# class TDEM_Derivtests(unittest.TestCase):

# # ====== TEST A ========== #

#     def AderivTest(self, prbtype):
#         prb, m0, mesh = setUp_TDEM(prbtype)
#         tInd = 2
#         if prbtype == 'MagneticFluxDensity':
#             nu = mesh.nF
#         elif prbtype == 'ElectricField':
#             nu = mesh.nE
#         v = np.random.rand(nu)

#         def AderivFun(m):
#             prb.model = m
#             A = prb.getAdiag(tInd)
#             Av = A*v
#             prb.model = m0

#             def ADeriv_dm(dm):
#                 return prb.getAdiagDeriv(tInd, v, dm)

#             return Av, ADeriv_dm

#         print('\n Testing ADeriv {}'.format(prbtype))
#         tests.checkDerivative(AderivFun, m0, plotIt=False, num=4, eps=EPS)

#     def A_adjointTest(self, prbtype):
#         prb, m0, mesh = setUp_TDEM(prbtype)
#         tInd = 2

#         print('\n Testing A_adjoint')
#         m = np.random.rand(prb.sigmaMap.nP)
#         if prbtype == 'MagneticFluxDensity':
#             nu = prb.mesh.nF
#         elif prbtype == 'ElectricField':
#             nu = prb.mesh.nE

#         v = np.random.rand(nu)
#         u = np.random.rand(nu)
#         prb.model = m0

#         tInd = 2  # not actually used
#         V1 = v.dot(prb.getAdiagDeriv(tInd, u, m))
#         V2 = m.dot(prb.getAdiagDeriv(tInd, u, v, adjoint=True))
#         passed = (
#             np.abs(V1-V2) < TOL * (np.abs(V1) + np.abs(V2))/2. or
#             np.abs(V1-V2) < EPS
#         )
#         print('AdjointTest {prbtype} {v1} {v2} {passed}'.format(
#             prbtype=prbtype, v1=V1, v2=V2, passed=passed))
#         self.assertTrue(passed)

#     def test_Aderiv_b(self):
#         self.AderivTest(prbtype='MagneticFluxDensity')

#     def test_Aderiv_e(self):
#         self.AderivTest(prbtype='ElectricField')

#     def test_Aadjoint_b(self):
#         self.A_adjointTest(prbtype='MagneticFluxDensity')

#     def test_Aadjoint_e(self):
#         self.A_adjointTest(prbtype='ElectricField')

# # ====== TEST Fields Deriv Pieces ========== #

#     def test_eDeriv_m_adjoint(self):
#         prb, m0, mesh = setUp_TDEM()
#         tInd = 0

#         v = np.random.rand(mesh.nF)

#         print('\n Testing eDeriv_m Adjoint')

#         prb, m0, mesh = setUp_TDEM()
#         f = prb.fields(m0)

#         m = np.random.rand(prb.sigmaMap.nP)
#         e = np.random.randn(prb.mesh.nE)
#         V1 = e.dot(f._eDeriv_m(1, prb.survey.source_list[0], m))
#         V2 = m.dot(f._eDeriv_m(1, prb.survey.source_list[0], e, adjoint=True))
#         tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.
#         passed = np.abs(V1-V2) < tol

#         print('     {v1}, {v2}, {diff}, {tol}, {passed}'.format(
#               v1=V1, v2=V2, diff=np.abs(V1-V2), tol=tol, passed=passed))
#         self.assertTrue(passed)

#     def test_eDeriv_u_adjoint(self):
#         print('\n Testing eDeriv_u Adjoint')

#         prb, m0, mesh = setUp_TDEM()
#         f = prb.fields(m0)

#         b = np.random.rand(prb.mesh.nF)
#         e = np.random.randn(prb.mesh.nE)
#         V1 = e.dot(f._eDeriv_u(1, prb.survey.source_list[0], b))
#         V2 = b.dot(f._eDeriv_u(1, prb.survey.source_list[0], e, adjoint=True))
#         tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.
#         passed = np.abs(V1-V2) < tol

#         print(
#             '     {v1}, {v2}, {diff}, {tol}, {passed}'.format(
#                 v1=V1, v2=V2, diff=np.abs(V1-V2), tol=tol, passed=passed
#             )
#         )
#         self.assertTrue(passed)
