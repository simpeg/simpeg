from __future__ import division, print_function
import unittest
import numpy as np
from SimPEG import Mesh, Maps, SolverLU, Tests
from SimPEG import EM

try:
    from pymatsolver import PardisoSolver
    Solver = PardisoSolver
except ImportError:
    Solver = SolverLU

plotIt = False

testDeriv = True
testAdjoint = True

TOL = 1e-5

np.random.seed(42)


def setUp_TDEM(prbtype='b', rxcomp='bz'):
    cs = 5.
    ncx = 20
    ncy = 15
    npad = 20
    hx = [(cs, ncx), (cs, npad, 1.3)]
    hy = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
    mesh = Mesh.CylMesh([hx, 1, hy], '00C')
#
    active = mesh.vectorCCz < 0.
    activeMap = Maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * activeMap

    rxOffset = 10.
    rx = EM.TDEM.Rx(np.array([[rxOffset, 0., -1e-2]]),
                    np.logspace(-4, -3, 20), rxcomp)
    src = EM.TDEM.Src.MagDipole([rx], loc=np.array([0., 0., 0.]))

    survey = EM.TDEM.Survey([src])

    if prbtype == 'b':
        prb = EM.TDEM.Problem3D_b(mesh, sigmaMap=mapping)
    elif prbtype == 'e':
        prb = EM.TDEM.Problem3D_e(mesh, sigmaMap=mapping)

    prb.timeSteps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]
    # prb.timeSteps = [(1e-05, 10), (1e-05, 50), (1e-05, 50) ] #, (2.5e-4, 10)]

    prb.Solver = Solver

    m = (np.log(1e-1)*np.ones(prb.sigmaMap.nP) +
         1e-3*np.random.randn(prb.sigmaMap.nP))

    prb.pair(survey)
    mesh = mesh

    return prb, m, mesh


class TDEM_DerivTests(unittest.TestCase):

# ====== TEST A ========== #

    def AderivTest(self, prbtype):
        prb, m0, mesh = setUp_TDEM(prbtype)
        tInd = 2
        if prbtype == 'b':
            nu = mesh.nF
        elif prbtype == 'e':
            nu = mesh.nE
        v = np.random.rand(nu)

        def AderivFun(m):
            prb.model = m
            A = prb.getAdiag(tInd)
            Av = A*v
            prb.model = m0

            def ADeriv_dm(dm):
                return prb.getAdiagDeriv(tInd, v, dm)

            return Av, ADeriv_dm

        print('\n Testing ADeriv {}'.format(prbtype))
        Tests.checkDerivative(AderivFun, m0, plotIt=False, num=3, eps=1e-20)

    def A_adjointTest(self, prbtype):
        prb, m0, mesh = setUp_TDEM(prbtype)
        tInd = 2

        print('\n Testing A_adjoint')
        m = np.random.rand(prb.sigmaMap.nP)
        if prbtype == 'b':
            nu = prb.mesh.nF
        elif prbtype == 'e':
            nu = prb.mesh.nE

        v = np.random.rand(nu)
        u = np.random.rand(nu)
        prb.model = m0

        tInd = 2  # not actually used
        V1 = v.dot(prb.getAdiagDeriv(tInd, u, m))
        V2 = m.dot(prb.getAdiagDeriv(tInd, u, v, adjoint=True))
        passed = np.abs(V1-V2) < TOL * (np.abs(V1) + np.abs(V2))/2.
        print('AdjointTest {prbtype} {v1} {v2} {passed}'.format(
            prbtype=prbtype, v1=V1, v2=V2, passed=passed))
        self.assertTrue(passed)

    def test_Aderiv_b(self):
        self.AderivTest(prbtype='b')

    def test_Aderiv_e(self):
        self.AderivTest(prbtype='e')

    def test_Aadjoint_b(self):
        self.A_adjointTest(prbtype='b')

    def test_Aadjoint_e(self):
        self.A_adjointTest(prbtype='e')

# ====== TEST Fields Deriv Pieces ========== #

    # def test_eDeriv_m_adjoint(self):
    #     prb, m0, mesh = setUp_TDEM()
    #     tInd = 0

    #     v = np.random.rand(mesh.nF)

    #     print '\n Testing eDeriv_m Adjoint'

    #     prb, m0, mesh = setUp_TDEM()
    #     f = prb.fields(m0)

    #     m = np.random.rand(prb.mapping.nP)
    #     e = np.random.randn(prb.mesh.nE)
    #     V1 = e.dot(f._eDeriv_m(1, prb.survey.srcList[0], m))
    #     V2 = m.dot(f._eDeriv_m(1, prb.survey.srcList[0], e, adjoint=True))
    #     tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.
    #     passed = np.abs(V1-V2) < tol

    #     print '    ', V1, V2, np.abs(V1-V2), tol, passed
    #     self.assertTrue(passed)

    # def test_eDeriv_u_adjoint(self):
    #     print '\n Testing eDeriv_u Adjoint'

    #     prb, m0, mesh = setUp_TDEM()
    #     f = prb.fields(m0)

    #     b = np.random.rand(prb.mesh.nF)
    #     e = np.random.randn(prb.mesh.nE)
    #     V1 = e.dot(f._eDeriv_u(1, prb.survey.srcList[0], b))
    #     V2 = b.dot(f._eDeriv_u(1, prb.survey.srcList[0], e, adjoint=True))
    #     tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.
    #     passed = np.abs(V1-V2) < tol

    #     print '    ', V1, V2, np.abs(V1-V2), tol, passed
    #     self.assertTrue(passed)


# ====== TEST Jvec ========== #

    if testDeriv:

        def JvecTest(self, prbtype, rxcomp):
            prb, m, mesh = setUp_TDEM(prbtype, rxcomp)

            def derChk(m):
                return [prb.survey.dpred(m), lambda mx: prb.Jvec(m, mx)]
            print('test_Jvec_{prbtype}_{rxcomp}'.format(prbtype=prbtype,
                                                        rxcomp=rxcomp))
            Tests.checkDerivative(derChk, m, plotIt=False, num=3, eps=1e-20)

        # def test_Jvec_b_bx(self):
        #     self.JvecTest('b', 'bx')

        # def test_Jvec_b_bz(self):
        #     self.JvecTest('b', 'bz')

        # def test_Jvec_b_dbxdt(self):
        #     self.JvecTest('b', 'dbxdt')

        # def test_Jvec_b_dbzdt(self):
        #     self.JvecTest('b', 'dbzdt')

        # def test_Jvec_b_ey(self):
        #     self.JvecTest('b', 'ey')

        def test_Jvec_e_ey(self):
            self.JvecTest('e', 'ey')


# ====== TEST Jtvec ========== #

    if testAdjoint:

        def JvecVsJtvecTest(self, prbtype='b', rxcomp='bz'):

            print('\nAdjoint Testing Jvec, Jtvec {}'.format(rxcomp))

            prb, m0, mesh = setUp_TDEM(prbtype, rxcomp)
            m = np.random.rand(prb.sigmaMap.nP)
            d = np.random.randn(prb.survey.nD)
            V1 = d.dot(prb.Jvec(m0, m))
            V2 = m.dot(prb.Jtvec(m0, d))
            tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.
            passed = np.abs(V1-V2) < tol

            print('AdjointTest {prbtype} {v1} {v2} {passed}'.format(
                prbtype=prbtype, v1=V1, v2=V2, passed=passed))
            self.assertTrue(passed)

        def test_Jvec_adjoint_b_bx(self):
            self.JvecVsJtvecTest('b', 'bx')

        def test_Jvec_adjoint_b_bz(self):
            self.JvecVsJtvecTest('b', 'bz')

        def test_Jvec_adjoint_b_dbxdt(self):
            self.JvecVsJtvecTest('b', 'bx')

        def test_Jvec_adjoint_b_dbzdt(self):
            self.JvecVsJtvecTest('b', 'bz')

        def test_Jvec_adjoint_b_ey(self):
            self.JvecVsJtvecTest('b', 'ey')

        # This is not working because Problem3D_e has not done
        def test_Jvec_adjoint_e_ey(self):
            self.JvecVsJtvecTest('e', 'ey')


if __name__ == '__main__':
    unittest.main()
