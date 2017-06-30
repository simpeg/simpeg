from __future__ import division, print_function
import unittest
from SimPEG import Maps, Mesh
from SimPEG import EM
import numpy as np

import warnings
from pymatsolver import Pardiso as Solver

TOL = 1e-4
FLR = 1e-20

# set a seed so that the same conductivity model is used for all runs
np.random.seed(25)


def setUp_TDEM(prbtype='b', rxcomp='bz', waveform='stepoff'):
    cs = 5.
    ncx = 8
    ncy = 8
    ncz = 8
    npad = 4
    # hx = [(cs, ncx), (cs, npad, 1.3)]
    # hz = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
    mesh = Mesh.TensorMesh(
        [
            [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)],
            [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)],
            [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
        ], 'CCC'
    )

    active = mesh.vectorCCz < 0.
    activeMap = Maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * activeMap

    prb = getattr(EM.TDEM, 'Problem3D_{}'.format(prbtype))(mesh, sigmaMap=mapping)

    rxtimes = np.logspace(-4, -3, 20)

    if waveform.upper() == 'RAW':
        out = EM.Utils.VTEMFun(prb.times, 0.00595, 0.006, 100)
        wavefun = interp1d(prb.times, out)
        t0 = 0.006
        waveform = EM.TDEM.Src.RawWaveform(offTime=t0, waveFct=wavefun)
        prb.timeSteps = [(1e-3, 5), (1e-4, 5), (5e-5, 10), (5e-5, 10), (1e-4, 10)]
        rxtimes = t0+rxtimes

    else:
        waveform = EM.TDEM.Src.StepOffWaveform()
        prb.timeSteps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]

    rxOffset = 10.
    rx = getattr(EM.TDEM.Rx, 'Point_{}'.format(rxcomp[:-1]))(
        np.r_[rxOffset, 0., -1e-2], rxtimes, rxcomp[-1]
    )
    src = EM.TDEM.Src.MagDipole(
        [rx], loc=np.array([0., 0., 0.]), waveform=waveform
    )

    survey = EM.TDEM.Survey([src])



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
    print(
        "    {}, {}, {} < {} ? {}".format(
            np.linalg.norm(d1), np.linalg.norm(d2), np.linalg.norm(check),
            tol, passed)
    )

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

    def test_HJ_j_stepoff(self):
        CrossCheck(prbtype1='h', prbtype2='j', rxcomp='jy', waveform='stepoff')

    def test_HJ_dhdtx_stepoff(self):
        CrossCheck(prbtype1='h', prbtype2='j', rxcomp='dhdtx', waveform='stepoff')

    def test_HJ_dhdtz_stepoff(self):
        CrossCheck(prbtype1='h', prbtype2='j', rxcomp='dhdtx', waveform='stepoff')


    def test_EB_ey_vtem(self):
        CrossCheck(prbtype1='b', prbtype2='e', rxcomp='ey', waveform='vtem')

    def test_EB_dbdtx_vtem(self):
        CrossCheck(prbtype1='b', prbtype2='e', rxcomp='dbdtx', waveform='vtem')

    def test_EB_dbdtz_vtem(self):
        CrossCheck(prbtype1='b', prbtype2='e', rxcomp='dbdtz', waveform='vtem')

    def test_HJ_j_vtem(self):
        CrossCheck(prbtype1='h', prbtype2='j', rxcomp='jy', waveform='vtem')

    def test_HJ_j_vtem(self):
        CrossCheck(prbtype1='h', prbtype2='j', rxcomp='jy', waveform='vtem')

    def test_HJ_dhdtx_vtem(self):
        CrossCheck(prbtype1='h', prbtype2='j', rxcomp='dhdtx', waveform='vtem')

    def test_HJ_dhdtz_vtem(self):
        CrossCheck(prbtype1='h', prbtype2='j', rxcomp='dhdtx', waveform='vtem')


    def test_MagDipoleSimpleFail(self):

        print('\ntesting MagDipole error handling')


        with warnings.catch_warnings(record=True):
            EM.TDEM.Src.MagDipole(
                [], loc=np.r_[0., 0., 0.],
                orientation=np.r_[1., 1., 0.]
            )

if __name__ == '__main__':
    unittest.main()
