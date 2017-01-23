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


def setUp_TDEM(prbtype='b', rxcomp='bz', waveform='stepoff'):
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

    prb = getattr(EM.TDEM, 'Problem3D_{}'.format(prbtype))(mesh, sigmaMap=mapping)

    if waveform.upper() == 'RAW':
        out = EM.Utils.VTEMFun(prb.times, 0.00595, 0.006, 100)
        wavefun = interp1d(prb.times, out)
        t0 = 0.006
        waveform = EM.TDEM.Src.RawWaveform(offTime=t0, waveFct=wavefun)
        prb.timeSteps = [(1e-3, 5), (1e-4, 5), (5e-5, 10), (5e-5, 10), (1e-4, 10)]

    else:

        prb.timeSteps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]




    prb.Solver = Solver

    m = (np.log(1e-1)*np.ones(prb.sigmaMap.nP) +
         1e-2*np.random.rand(prb.sigmaMap.nP))

    prb.pair(survey)
    mesh = mesh

    return prb, m, mesh


def CrossCheck(prbtype1='b', prbtype2='e', rxcomp='bz', waveform='stepoff'):

    prb1, m1, mesh1 = setUp_TDEM(prbtype1, rxcomp, waveform)
    prb2, _, mesh2 = setUp_TDEM(prbtype2, rxcomp, waveform)

    d1 = prb1.survey.dpred(m1)
    d2 = prb2.survey.dpred(m1)

    check = np.linalg.norm(d1 - d2)
    tol = 0.5 * (np.linalg.norm(d1) + np.linalg.norm(d2)) * TOL
    passed = check < tol

    print(
        'Checking {}, {} for {} data, {} waveform'.format(
        prbtype1, prbtype2, rxcomp, waveform
        )
    )
    print('{}, {}, {}'.format(np.linalg.norm(d1), np.linalg.norm(d2),
          np.linalg.norm(check), tol, passed))

    assert passed


class TDEM_cross_check_EB(unittest.TestCase):
    def test_EB_ey_stepoff(self):
        CrossCheck(prbtype1='b', prbtype2='e', rxcomp='ey', waveform='stepoff')

    def test_EB_dbdtx_stepoff(self):
        CrossCheck(prbtype1='b', prbtype2='e', rxcomp='dbdtx', waveform='stepoff')

    def test_EB_dbdtz_stepoff(self):
        CrossCheck(prbtype1='b', prbtype2='e', rxcomp='dbdtz', waveform='stepoff')

    def test_HJ_j_stepoff(self):
        CrossCheck(prbtype1='h', prbtype2='j', rxcomp='jy', waveform='stepoff')

    # def test_EB_bx(self):
    #     CrossCheck(prbtype1='b', prbtype2='e', rxcomp='bx')

    # def test_EB_bz(self):
    #     CrossCheck(prbtype1='b', prbtype2='e', rxcomp='bz')


if __name__ == '__main__':
    unittest.main()
