from __future__ import division, print_function
import unittest
import numpy as np
import time
from SimPEG import Mesh, Maps, Tests
from SimPEG import EM
from scipy.interpolate import interp1d
from pymatsolver import Pardiso as Solver

plotIt = False

testDeriv = True
testAdjoint = False

TOL = 1e-4
EPS = 1e-20
np.random.seed(4)


def get_mesh():
    cs = 5.
    ncx = 8
    ncy = 8
    ncz = 8
    npad = 4

    return Mesh.TensorMesh(
        [
            [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)],
            [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)],
            [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
        ], 'CCC'
    )


def get_mapping(mesh):
    active = mesh.vectorCCz < 0.
    activeMap = Maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    return Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * activeMap


def get_prob(mesh, mapping, formulation):
    prb = getattr(EM.TDEM, 'Problem3D_{}'.format(formulation))(
        mesh, sigmaMap=mapping
    )
    prb.timeSteps = [(1e-3, 5), (1e-4, 5), (5e-5, 10), (5e-5, 10), (1e-4, 10)]
    prb.Solver = Solver
    return prb


def get_survey(prob, t0):

    out = EM.Utils.VTEMFun(prob.times, 0.00595, 0.006, 100)
    wavefun = interp1d(prob.times, out)

    waveform = EM.TDEM.Src.RawWaveform(offTime=t0, waveFct=wavefun)
    src = EM.TDEM.Src.MagDipole(
        [], waveform=waveform, loc=np.array([0., 0., 0.])
    )

    return EM.TDEM.Survey([src])


class Base_DerivAdjoint_Test(unittest.TestCase):

    t0 = 0.006

    @classmethod
    def setUpClass(self):
        # create a prob where we will store the fields
        mesh = get_mesh()
        mapping = get_mapping(mesh)
        self.prob = get_prob(mesh, mapping, self.formulation)
        self.survey = get_survey(self.prob, self.t0)
        self.m = np.log(1e-1)*np.ones(self.prob.sigmaMap.nP)
        self.prob.pair(self.survey)
        print('Solving Fields for problem {}'.format(self.formulation))
        t = time.time()
        self.fields = self.prob.fields(self.m)
        print('... done. Time: {}\n'.format(time.time()-t))

        # create a prob where will be re-computing fields at each jvec
        # iteration
        mesh = get_mesh()
        mapping = get_mapping(mesh)
        self.probfwd = get_prob(mesh, mapping, self.formulation)
        self.surveyfwd = get_survey(self.probfwd, self.t0)
        self.probfwd.pair(self.surveyfwd)

    def get_rx(self, rxcomp):
        rxOffset = 15.

        timerx = self.t0 + np.logspace(-5, -3, 20)
        return getattr(EM.TDEM.Rx, 'Point_{}'.format(rxcomp[:-1]))(
            np.array([[rxOffset, 0., 0.]]), timerx, rxcomp[-1]
        )

    def set_rxList(self, rxcomp):
        # append the right rxlist to the surveys
        rx = [self.get_rx(rxcomp)]
        rxfwd = [self.get_rx(rxcomp)]

        # append to srclists
        for srcList, rxlist in zip(
            [self.survey.srcList, self.surveyfwd.srcList], [rx, rxfwd]
        ):
            for src in srcList:
                src.rxList = rxlist

    def JvecTest(self, rxcomp):
        self.set_rxList(rxcomp)

        def derChk(m):
            return [
                self.probfwd.survey.dpred(m),
                lambda mx: self.prob.Jvec(self.m, mx, f=self.fields)
            ]
        print('test_Jvec_{prbtype}_{rxcomp}'.format(
            prbtype=self.formulation, rxcomp=rxcomp)
        )
        Tests.checkDerivative(derChk, self.m, plotIt=False, num=2, eps=1e-20)

    def JvecVsJtvecTest(self, rxcomp):
        self.set_rxList(rxcomp)
        print(
            '\nAdjoint Testing Jvec, Jtvec prob {}, {}'.format(
                self.formulation, rxcomp
            )
        )

        m = np.random.rand(self.prob.sigmaMap.nP)
        d = np.random.randn(self.prob.survey.nD)
        V1 = d.dot(self.prob.Jvec(self.m, m, f=self.fields))
        V2 = m.dot(self.prob.Jtvec(self.m, d, f=self.fields))
        tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.
        passed = np.abs(V1-V2) < tol

        print('    {v1} {v2} {passed}'.format(
            prbtype=self.formulation, v1=V1, v2=V2, passed=passed))
        self.assertTrue(passed)


class DerivAdjoint_E(Base_DerivAdjoint_Test):

    formulation = 'e'

    if testDeriv:
        def test_Jvec_e_dbxdt(self):
            self.JvecTest('dbdtx')

        def test_Jvec_e_dbzdt(self):
            self.JvecTest('dbdtz')

        def test_Jvec_e_ey(self):
            self.JvecTest('ey')

    if testAdjoint:
        def test_Jvec_adjoint_e_ey(self):
            self.JvecVsJtvecTest('dbdtx')

        def test_Jvec_adjoint_e_ey(self):
            self.JvecVsJtvecTest('dbdtz')

        def test_Jvec_adjoint_e_ey(self):
            self.JvecVsJtvecTest('ey')


class DerivAdjoint_B(Base_DerivAdjoint_Test):

    formulation = 'b'

    if testDeriv:
        def test_Jvec_b_bx(self):
            self.JvecTest('bx')

        def test_Jvec_b_bz(self):
            self.JvecTest('bz')

        def test_Jvec_b_dbdtx(self):
            self.JvecTest('dbdtx')

        def test_Jvec_b_dbdtz(self):
            self.JvecTest('dbdtz')

        def test_Jvec_b_ey(self):
            self.JvecTest('ey')

    if testAdjoint:
        def test_Jvec_adjoint_b_bx(self):
            self.JvecVsJtvecTest('bx')

        def test_Jvec_adjoint_b_bz(self):
            self.JvecVsJtvecTest('bz')

        def test_Jvec_adjoint_b_dbdtz(self):
            self.JvecVsJtvecTest('dbdtx')

        def test_Jvec_adjoint_b_dbdtx(self):
            self.JvecVsJtvecTest('dbdtz')

        def test_Jvec_adjoint_b_ey(self):
            self.JvecVsJtvecTest('ey')


class DerivAdjoint_H(Base_DerivAdjoint_Test):

    formulation = 'h'

    if testDeriv:
        def test_Jvec_h_hx(self):
            self.JvecTest('hx')

        def test_Jvec_h_hz(self):
            self.JvecTest('hz')

        def test_Jvec_h_dhdtx(self):
            self.JvecTest('dhdtx')

        def test_Jvec_h_dhdtz(self):
            self.JvecTest('dhdtz')

    if testAdjoint:
        def test_Jvec_adjoint_h_hx(self):
            self.JvecVsJtvecTest('hx')

        def test_Jvec_adjoint_h_hz(self):
            self.JvecVsJtvecTest('hz')

        def test_Jvec_adjoint_h_dhdtx(self):
            self.JvecVsJtvecTest('dhdtx')

        def test_Jvec_adjoint_h_dhdtz(self):
            self.JvecVsJtvecTest('dhdtz')

        def test_Jvec_adjoint_h_jy(self):
            self.JvecVsJtvecTest('jy')


class DerivAdjoint_J(Base_DerivAdjoint_Test):

    formulation = 'j'

    if testDeriv:
        def test_Jvec_j_jy(self):
            self.JvecTest('jy')

        def test_Jvec_j_dhdtx(self):
            self.JvecTest('dhdtx')

        def test_Jvec_j_dhdtz(self):
            self.JvecTest('dhdtz')

    if testAdjoint:
        def test_Jvec_adjoint_j_jy(self):
            self.JvecVsJtvecTest('jy')

        def test_Jvec_adjoint_j_dhdtx(self):
            self.JvecVsJtvecTest('dhdtx')

        def test_Jvec_adjoint_j_dhdtz(self):
            self.JvecVsJtvecTest('dhdtz')


# class TDEM_DerivTests(unittest.TestCase):

# # ====== TEST A ========== #

#     def AderivTest(self, prbtype):
#         prb, m0, mesh = setUp_TDEM(prbtype)
#         tInd = 2
#         if prbtype == 'b':
#             nu = mesh.nF
#         elif prbtype == 'e':
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
#         Tests.checkDerivative(AderivFun, m0, plotIt=False, num=4, eps=EPS)

#     def A_adjointTest(self, prbtype):
#         prb, m0, mesh = setUp_TDEM(prbtype)
#         tInd = 2

#         print('\n Testing A_adjoint')
#         m = np.random.rand(prb.sigmaMap.nP)
#         if prbtype == 'b':
#             nu = prb.mesh.nF
#         elif prbtype == 'e':
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
#         self.AderivTest(prbtype='b')

#     def test_Aderiv_e(self):
#         self.AderivTest(prbtype='e')

#     def test_Aadjoint_b(self):
#         self.A_adjointTest(prbtype='b')

#     def test_Aadjoint_e(self):
#         self.A_adjointTest(prbtype='e')

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
#         V1 = e.dot(f._eDeriv_m(1, prb.survey.srcList[0], m))
#         V2 = m.dot(f._eDeriv_m(1, prb.survey.srcList[0], e, adjoint=True))
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
#         V1 = e.dot(f._eDeriv_u(1, prb.survey.srcList[0], b))
#         V2 = b.dot(f._eDeriv_u(1, prb.survey.srcList[0], e, adjoint=True))
#         tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.
#         passed = np.abs(V1-V2) < tol

#         print(
#             '     {v1}, {v2}, {diff}, {tol}, {passed}'.format(
#                 v1=V1, v2=V2, diff=np.abs(V1-V2), tol=tol, passed=passed
#             )
#         )
#         self.assertTrue(passed)



if __name__ == '__main__':
    unittest.main()
