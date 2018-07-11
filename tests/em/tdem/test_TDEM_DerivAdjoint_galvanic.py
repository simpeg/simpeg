from __future__ import division, print_function
import unittest
import numpy as np
from SimPEG import Mesh, Maps, SolverLU, Tests
from SimPEG import EM
from pymatsolver import Pardiso as Solver

plotIt = False

testDeriv = True
testAdjoint = True

TOL = 1e-4

np.random.seed(10)


def setUp_TDEM(prbtype='e', rxcomp='ex'):
    cs = 5.
    ncx = 8
    ncy = 8
    ncz = 8
    npad = 0
    # hx = [(cs, ncx), (cs, npad, 1.3)]
    # hz = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
    mesh = Mesh.TensorMesh(
        [
            [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)],
            [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)],
            [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
        ], 'CCC'
    )
#
    active = mesh.vectorCCz < 0.
    activeMap = Maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * activeMap

    rxOffset = 0.
    rxlocs = np.array([[20, 20., 0.]])
    rxtimes = np.logspace(-4, -3, 20)
    rx = getattr(EM.TDEM.Rx, 'Point_{}'.format(rxcomp[:-1]))(
        locs=rxlocs, times=rxtimes, orientation=rxcomp[-1]
    )
    Aloc = np.r_[-10., 0., 0.]
    Bloc = np.r_[10., 0., 0.]
    srcloc = np.vstack((Aloc, Bloc))

    src = EM.TDEM.Src.LineCurrent([rx], loc=srcloc, waveform = EM.TDEM.Src.StepOffWaveform())
    survey = EM.TDEM.Survey([src])

    prb = getattr(EM.TDEM, 'Problem3D_{}'.format(prbtype))(mesh, sigmaMap=mapping)

    prb.timeSteps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]

    prb.Solver = Solver

    m = (
        np.log(1e-1)*np.ones(prb.sigmaMap.nP) +
        1e-3*np.random.randn(prb.sigmaMap.nP)
    )

    prb.pair(survey)
    mesh = mesh

    return prb, m, mesh


class TDEM_DerivTests(unittest.TestCase):

# ====== TEST Jvec ========== #

    if testDeriv:
        def JvecTest(self, prbtype, rxcomp):
            prb, m, mesh = setUp_TDEM(prbtype, rxcomp)

            def derChk(m):
                return [prb.survey.dpred(m), lambda mx: prb.Jvec(m, mx)]
            print('test_Jvec_{prbtype}_{rxcomp}'.format(
                prbtype=prbtype, rxcomp=rxcomp)
            )
            Tests.checkDerivative(derChk, m, plotIt=False, num=2, eps=1e-20)

        def test_Jvec_e_dbzdt(self):
            self.JvecTest('e', 'dbdtz')

        def test_Jvec_e_ex(self):
            self.JvecTest('e', 'ex')

        def test_Jvec_e_ey(self):
            self.JvecTest('e', 'ey')


# ====== TEST Jtvec ========== #

    if testAdjoint:
        def JvecVsJtvecTest(self, prbtype='b', rxcomp='bz'):

            print(
                '\nAdjoint Testing Jvec, Jtvec prob {}, {}'.format(
                    prbtype, rxcomp
                )
            )

            prb, m0, mesh = setUp_TDEM(prbtype, rxcomp)
            m = np.random.rand(prb.sigmaMap.nP)
            d = np.random.randn(prb.survey.nD)

            print(m.shape, d.shape, m0.shape)

            V1 = d.dot(prb.Jvec(m0, m))
            V2 = m.dot(prb.Jtvec(m0, d))
            tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.
            passed = np.abs(V1-V2) < tol

            print('AdjointTest {prbtype} {v1} {v2} {passed}'.format(
                prbtype=prbtype, v1=V1, v2=V2, passed=passed)
            )
            self.assertTrue(passed)

        def test_Jvec_adjoint_e_dbzdt(self):
            self.JvecVsJtvecTest('e', 'dbdtz')

        def test_Jvec_adjoint_e_ex(self):
            self.JvecVsJtvecTest('e', 'ex')

        def test_Jvec_adjoint_e_ey(self):
            self.JvecVsJtvecTest('e', 'ey')

if __name__ == '__main__':
    unittest.main()
