from __future__ import division, print_function
import unittest
from SimPEG import Maps, Mesh
from SimPEG import EM
import numpy as np

try:
    from pymatsolver import PardisoSolver as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

TOL = 1e-5
FLR = 1e-20

# set a seed so that the same conductivity model is used for all runs
np.random.seed(seed=25)


def setUp_TDEM(prbtype='b', rxcomp='bz'):
    cs = 5.
    ncx = 20
    ncy = 15
    npad = 20
    hx = [(cs, ncx), (cs, npad, 1.3)]
    hy = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
    mesh = Mesh.CylMesh([hx, 1, hy], '00C')

    active = mesh.vectorCCz < 0.
    activeMap = Maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * activeMap

    rxOffset = 10.
    rx = getattr(EM.TDEM.Rx, 'Point_{}'.format(rxcomp[:-1]))(
        np.array([[rxOffset, 0., -1e-2]]), np.logspace(-4, -3, 20), rxcomp[-1]
    )
    src = EM.TDEM.Src.MagDipole([rx], loc=np.array([0., 0., 0.]))

    survey = EM.TDEM.Survey([src])

    if prbtype == 'b':
        prb = EM.TDEM.Problem3D_b(mesh, sigmaMap=mapping)
    elif prbtype == 'e':
        prb = EM.TDEM.Problem3D_e(mesh, sigmaMap=mapping)
    else:
        raise NotImplementedError()

    prb.timeSteps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]

    prb.Solver = Solver

    m = (np.log(1e-1)*np.ones(prb.sigmaMap.nP) +
         1e-2*np.random.rand(prb.sigmaMap.nP))

    prb.pair(survey)
    mesh = mesh

    return prb, m, mesh


def CrossCheck(prbtype1='b', prbtype2='e', rxcomp='bz'):

    prb1, m1, mesh1 = setUp_TDEM(prbtype1, rxcomp)
    prb2, _, mesh2 = setUp_TDEM(prbtype2, rxcomp)

    d1 = prb1.survey.dpred(m1)
    d2 = prb2.survey.dpred(m1)

    check = np.linalg.norm(d1 - d2)
    tol = 0.5 * (np.linalg.norm(d1) + np.linalg.norm(d2)) * TOL
    passed = check < tol

    print('Checking {}, {} for {} data'.format(prbtype1, prbtype2, rxcomp))
    print('{}, {}, {}'.format(np.linalg.norm(d1), np.linalg.norm(d2),
          np.linalg.norm(check), tol, passed))

    assert passed


class TDEM_cross_check_EB(unittest.TestCase):
    def test_EB_ey(self):
        CrossCheck(prbtype1='b', prbtype2='e', rxcomp='ey')

    def test_EB_dbdtx(self):
        CrossCheck(prbtype1='b', prbtype2='e', rxcomp='dbdtx')

    def test_EB_dbdtz(self):
        CrossCheck(prbtype1='b', prbtype2='e', rxcomp='dbdtz')

    # def test_EB_bx(self):
    #     CrossCheck(prbtype1='b', prbtype2='e', rxcomp='bx')

    # def test_EB_bz(self):
    #     CrossCheck(prbtype1='b', prbtype2='e', rxcomp='bz')


if __name__ == '__main__':
    unittest.main()
