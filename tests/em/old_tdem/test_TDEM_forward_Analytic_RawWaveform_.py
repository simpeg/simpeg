from __future__ import division, print_function
import unittest
from SimPEG import Mesh, Maps
from SimPEG import EM
import numpy as np
from scipy.constants import mu_0
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pymatsolver import Pardiso as Solver


def halfSpaceProblemAnaDiff(
    meshType, srctype="MagDipole", sig_half=1e-2, rxOffset=50., bounds=None,
    plotIt=False, rxType='bz'
):

    if bounds is None:
        bounds = [1e-5, 1e-3]
    if meshType == 'CYL':
        cs, ncx, ncz, npad = 15., 30, 10, 15
        hx = [(cs, ncx), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
        mesh = Mesh.CylMesh([hx, 1, hz], '00C')

    elif meshType == 'TENSOR':
        cs, nc, npad = 20., 13, 5
        hx = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]
        hy = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]
        mesh = Mesh.TensorMesh([hx, hy, hz], 'CCC')

    active = mesh.vectorCCz < 0.
    actMap = Maps.InjectActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    mapping = Maps.ExpMap(mesh) * Maps.SurjectVertical1D(mesh) * actMap

    prb = EM.TDEM.Problem3D_b(mesh, sigmaMap=mapping)
    prb.Solver = Solver
    prb.timeSteps = [(1e-3, 5), (1e-4, 5), (5e-5, 10), (5e-5, 10), (1e-4, 10)]
    out = EM.Utils.VTEMFun(prb.times, 0.00595, 0.006, 100)
    wavefun = interp1d(prb.times, out)
    t0 = 0.006
    waveform = EM.TDEM.Src.RawWaveform(offTime=t0, waveFct=wavefun)

    rx = getattr(EM.TDEM.Rx, 'Point_{}'.format(rxType[:-1]))(
        np.array([[rxOffset, 0., 0.]]), np.logspace(-4, -3, 31)+t0, rxType[-1]
    )

    if srctype == "MagDipole":
        src = EM.TDEM.Src.MagDipole(
            [rx], waveform=waveform, loc=np.array([0, 0., 0.])
        )
    elif srctype == "CircularLoop":
        src = EM.TDEM.Src.CircularLoop(
            [rx], waveform=waveform, loc=np.array([0., 0., 0.]), radius=13.
        )

    survey = EM.TDEM.Survey([src])
    prb.pair(survey)

    sigma = np.ones(mesh.nCz)*1e-8
    sigma[active] = sig_half
    sigma = np.log(sigma[active])

    if srctype == "MagDipole":
        bz_ana = mu_0*EM.Analytics.hzAnalyticDipoleT(rx.locs[0][0]+1e-3,
                                                     rx.times-t0, sig_half)
    elif srctype == "CircularLoop":
        bz_ana = mu_0*EM.Analytics.hzAnalyticCentLoopT(13, rx.times-t0,
                                                       sig_half)

    bz_calc = survey.dpred(sigma)
    ind = np.logical_and(rx.times-t0 > bounds[0], rx.times-t0 < bounds[1])
    log10diff = (np.linalg.norm(np.log10(np.abs(bz_calc[ind])) -
                 np.log10(np.abs(bz_ana[ind]))) /
                 np.linalg.norm(np.log10(np.abs(bz_ana[ind]))))

    print(' |bz_ana| = {ana} |bz_num| = {num} |bz_ana-bz_num| = {diff}'.format(
          ana=np.linalg.norm(bz_ana), num=np.linalg.norm(bz_calc),
          diff=np.linalg.norm(bz_ana-bz_calc)))
    print('Difference: {}'.format(log10diff))

    if plotIt is True:
        plt.loglog(rx.times[bz_calc > 0]-t0, bz_calc[bz_calc > 0], 'r',
                   rx.times[bz_calc < 0]-t0, -bz_calc[bz_calc < 0], 'r--')
        plt.loglog(rx.times-t0, abs(bz_ana), 'b*')
        plt.title('sig_half = {:e}'.format(sig_half))
        plt.show()

    return log10diff


class TDEM_SimpleSrcTests(unittest.TestCase):
    def test_source(self):
        waveform = EM.TDEM.Src.StepOffWaveform()
        assert waveform.eval(0.) == 1.


class TDEM_bTests(unittest.TestCase):

    def test_analytic_p0_CYL_1m_MagDipole(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=1.0,
                        sig_half=1e+0) < 0.01)

    def test_analytic_m1_CYL_1m_MagDipole(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=1.0,
                        sig_half=1e-1) < 0.01)

    def test_analytic_m2_CYL_1m_MagDipole(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=1.0,
                        sig_half=1e-2) < 0.01)

    def test_analytic_m3_CYL_1m_MagDipole(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=1.0,
                        sig_half=1e-3) < 0.01)

    def test_analytic_p0_CYL_0m_CircularLoop(self):
        self.assertTrue(halfSpaceProblemAnaDiff(
            'CYL', srctype="CircularLoop", rxOffset=.0, sig_half=1e+0) < 0.02)

    def test_analytic_m1_CYL_0m_CircularLoop(self):
        self.assertTrue(halfSpaceProblemAnaDiff(
            'CYL', srctype="CircularLoop", rxOffset=.0, sig_half=1e-1) < 0.01)

    def test_analytic_m2_CYL_0m_CircularLoop(self):
        self.assertTrue(halfSpaceProblemAnaDiff(
            'CYL', srctype="CircularLoop", rxOffset=.0, sig_half=1e-2) < 0.01)

    def test_analytic_m3_CYL_0m_CircularLoop(self):
        self.assertTrue(halfSpaceProblemAnaDiff(
            'CYL', srctype="CircularLoop", rxOffset=.0, sig_half=1e-3) < 0.01)


if __name__ == '__main__':
    unittest.main()
