import unittest
import numpy as np
from SimPEG import Mesh, Maps, SolverLU, Tests, Survey
from SimPEG import EM

plotIt = False
testDeriv = True
testAdjoint = True

TOL = 1e-5

def setUp_TDEM(self, rxcomp='bz'):

        cs = 5.
        ncx = 20
        ncy = 6
        npad = 20
        hx = [(cs, ncx), (cs, npad, 1.3)]
        hy = [(cs, npad, -1.3), (cs,ncy), (cs,npad,1.3)]
        mesh = Mesh.CylMesh([hx,1,hy], '00C')

        active = mesh.vectorCCz<0.
        activeMap = Maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
        mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * activeMap

        rxOffset = 40.
        rx = EM.TDEM.Rx(np.array([[rxOffset, 0., 0.]]), np.logspace(-4,-3, 20), rxcomp)
        src = EM.TDEM.Src.MagDipole( [rx], loc=np.array([0., 0., 0.]))
        rx2 = EM.TDEM.Rx(np.array([[rxOffset-10, 0., 0.]]), np.logspace(-5,-4, 25), rxcomp)
        src2 = EM.TDEM.Src.MagDipole( [rx2], loc=np.array([0., 0., 0.]))

        survey = EM.TDEM.Survey([src,src2])

        prb = EM.TDEM.Problem3D_b(mesh, mapping=mapping)
        # prb.timeSteps = [1e-5]
        prb.timeSteps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]
        # prb.timeSteps = [(1e-05, 100)]

        try:
            from pymatsolver import MumpsSolver
            prb.Solver = MumpsSolver
        except ImportError, e:
            prb.Solver  = SolverLU

        m = np.log(1e-1)*np.ones(prb.mapping.nP) + 1e-2*np.random.randn(prb.mapping.nP)

        prb.pair(survey)

        return mesh, prb, m

class TDEM_bDerivTests(unittest.TestCase):


    if testDeriv:
        def Deriv_J(self, rxcomp='bz'):

            mesh, prb, m0 = setUp_TDEM(rxcomp)

            prb.timeSteps = [(1e-05, 10), (0.0001, 10), (0.001, 10)]

            derChk = lambda m: [prb.survey.dpred(m), lambda mx: prb.Jvec(m0, mx)]
            print '\n'
            print 'test_Deriv_J %s'%rxcomp
            Tests.checkDerivative(derChk, m0, plotIt=False, num=3, eps=1e-20)

        def test_Jvec_bx(self):
            self.Deriv_J(rxcomp='bx')

        def test_Jvec_bz(self):
            self.Deriv_J(rxcomp='bz')

        def test_Jvec_ey(self):
            self.Deriv_J(rxcomp='ey')

    if testAdjoint:
        def adjointJvecVsJtvec(self, rxcomp='bz'):
            print ' \n Testing Adjoint %s' %rxcomp
            mesh, prb, m0 = setUp_TDEM(rxcomp)

            m = np.random.rand(prb.mapping.nP)
            d = np.random.rand(prb.survey.nD)

            V1 = d.dot(prb.Jvec(m0, m))
            V2 = m.dot(prb.Jtvec(m0, d))

            tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.
            passed = np.abs(V1-V2) < tol
            print '    ', V1, V2, np.abs(V1-V2), tol, passed
            self.assertTrue(passed)

        def test_JvecVsJtvec_bx(self):
            self.adjointJvecVsJtvec(rxcomp='bx')

        def test_JvecVsJtvec_bz(self):
            self.adjointJvecVsJtvec(rxcomp='bz')

        def test_JvecVsJtvec_ey(self):
            self.adjointJvecVsJtvec(rxcomp='ey')

if __name__ == '__main__':
    unittest.main()
