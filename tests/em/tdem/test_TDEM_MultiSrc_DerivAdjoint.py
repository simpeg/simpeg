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
testAdjoint = False

TOL = 1e-5

def setUp_TDEM(prob='b', rxcomp='bz'):

    cs = 5.
    ncx = 8
    ncy = 8
    ncz = 8
    npad = 4

    mesh = Mesh.TensorMesh(
        [
            [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)],
            [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)],
            [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
        ], 'CCC'
    )

    active = mesh.vectorCCz < 0.
    activeMap = Maps.InjectActiveCells(mesh, active, np.log(1e-8),
                                       nC=mesh.nCz)
    mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * activeMap

    rxOffset = 40.
    rx = getattr(EM.TDEM.Rx, 'Point_{}'.format(rxcomp[:-1]))(
        np.array([[rxOffset, 0., 0.]]), np.logspace(-4, -3, 20),
        rxcomp[-1]
    )

    src = EM.TDEM.Src.MagDipole([rx], loc=np.array([0., 0., 0.]))
    rx2 = getattr(EM.TDEM.Rx, 'Point_{}'.format(rxcomp[:-1]))(
        np.array([[rxOffset-10, 0., 0.]]), np.logspace(-5, -4, 25),
        rxcomp[-1]
    )
    src2 = EM.TDEM.Src.MagDipole( [rx2], loc=np.array([0., 0., 0.]))

    survey = EM.TDEM.Survey([src, src2])

    prb = getattr(EM.TDEM, 'Problem3D_{}'.format(prob))(mesh, sigmaMap=mapping)
    prb.timeSteps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]

    prb.Solver = Solver

    m = (
        np.log(1e-1)*np.ones(prb.sigmaMap.nP) +
        1e-2*np.random.randn(prb.sigmaMap.nP)
    )

    prb.pair(survey)

    return mesh, prb, m


class TDEM_bDerivTests(unittest.TestCase):

    if testDeriv:
        def Deriv_J(self, prob='b', rxcomp='bz'):

            mesh, prb, m0 = setUp_TDEM(prob=prob, rxcomp=rxcomp)

            prb.timeSteps = [(1e-05, 10), (0.0001, 10), (0.001, 10)]

            def derChk(m):
                return [prb.survey.dpred(m), lambda mx: prb.Jvec(m0, mx)]

            print('test_Deriv_J problem {}, {}'.format(prob, rxcomp))
            Tests.checkDerivative(derChk, m0, plotIt=False, num=2, eps=1e-20)

        def test_Jvec_b_bx(self):
            self.Deriv_J(prob='b', rxcomp='bx')

        def test_Jvec_b_dbdtx(self):
            self.Deriv_J(prob='b', rxcomp='dbdtx')

        def test_Jvec_b_ey(self):
            self.Deriv_J(prob='b', rxcomp='ey')

        def test_Jvec_e_dbdtx(self):
            self.Deriv_J(prob='e', rxcomp='dbdtx')

        def test_Jvec_e_ey(self):
            self.Deriv_J(prob='e', rxcomp='ey')

        def test_Jvec_h_hx(self):
            self.Deriv_J(prob='h', rxcomp='hx')

        def test_Jvec_h_dhdtx(self):
            self.Deriv_J(prob='h', rxcomp='dhdtx')

        def test_Jvec_h_jy(self):
            self.Deriv_J(prob='h', rxcomp='jy')

        def test_Jvec_j_dhdtx(self):
            self.Deriv_J(prob='j', rxcomp='dhdtx')

        def test_Jvec_j_jy(self):
            self.Deriv_J(prob='j', rxcomp='jy')


    if testAdjoint:
        def adjointJvecVsJtvec(self, prob='b', rxcomp='bz'):
            print(' \n Testing Adjoint problem {}, {}'.format(prob, rxcomp))
            mesh, prb, m0 = setUp_TDEM(prob, rxcomp)

            m = np.random.rand(prb.sigmaMap.nP)
            d = np.random.rand(prb.survey.nD)

            V1 = d.dot(prb.Jvec(m0, m))
            V2 = m.dot(prb.Jtvec(m0, d))

            tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.
            passed = np.abs(V1-V2) < tol
            print('    {v1}, {v2}, {diff}, {tol}, {passed} '.format(
                v1=V1, v2=V2, diff=np.abs(V1-V2), tol=tol, passed=passed))
            self.assertTrue(passed)

        def test_adjoint_b_bx(self):
            self.Deriv_J(prob='b', rxcomp='bx')

        def test_adjoint_b_dbdtx(self):
            self.Deriv_J(prob='b', rxcomp='dbdtx')

        def test_adjoint_b_ey(self):
            self.Deriv_J(prob='b', rxcomp='ey')

        def test_adjoint_e_dbdtx(self):
            self.Deriv_J(prob='e', rxcomp='dbdtx')

        def test_adjoint_e_ey(self):
            self.Deriv_J(prob='e', rxcomp='ey')

        def test_adjoint_h_hx(self):
            self.Deriv_J(prob='h', rxcomp='hx')

        def test_adjoint_h_dhdtx(self):
            self.Deriv_J(prob='h', rxcomp='dhdtx')

        def test_adjoint_h_jy(self):
            self.Deriv_J(prob='h', rxcomp='jy')

        def test_adjoint_j_dhdtx(self):
            self.Deriv_J(prob='e', rxcomp='dhdtx')

        def test_adjoint_j_jy(self):
            self.Deriv_J(prob='e', rxcomp='jy')

if __name__ == '__main__':
    unittest.main()
