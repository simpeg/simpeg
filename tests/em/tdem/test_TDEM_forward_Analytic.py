import unittest
from SimPEG import *
from SimPEG import EM
from scipy.constants import mu_0
import matplotlib.pyplot as plt

try:
    from pymatsolver import MumpsSolver
except ImportError, e:
    MumpsSolver = SolverLU


def halfSpaceProblemAnaDiff(meshType, sig_half=1e-2, rxOffset=50., bounds=None, showIt=False):

    print '\nTesting sig_half = {0}, rxOffset= {1}'.format(sig_half, rxOffset)

    if bounds is None:
        bounds = [1e-5, 1e-3]
    if meshType == 'CYL':
        cs, ncx, ncz, npad = 5., 30, 10, 20
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

    rx = EM.TDEM.RxTDEM(np.array([[rxOffset, 0., 0.]]),
                        np.logspace(-5, -4, 21), 'bz')
    src = EM.TDEM.SrcTDEM_VMD_MVP([rx], loc=np.array([0., 0., 0.]))
    # src = EM.TDEM.SrcTDEM([rx], loc=np.array([0., 0., 0.]))

    survey = EM.TDEM.SurveyTDEM([src])
    prb = EM.TDEM.ProblemTDEM_b(mesh, mapping=mapping)
    prb.Solver = MumpsSolver

    prb.timeSteps = [(1e-06, 40), (5e-06, 40), (1e-05, 40), (5e-05, 40), (0.0001, 40), (0.0005, 40)]

    sigma = np.ones(mesh.nCz)*1e-8
    sigma[active] = sig_half
    sigma = np.log(sigma[active])
    prb.pair(survey)

    bz_ana = mu_0*EM.Analytics.hzAnalyticDipoleT(rx.locs[0][0]+1e-3, rx.times, sig_half)

    bz_calc = survey.dpred(sigma)

    ind = np.logical_and(rx.times > bounds[0], rx.times < bounds[1])
    log10diff = np.linalg.norm(np.log10(np.abs(bz_calc[ind])) - np.log10(np.abs(bz_ana[ind])))/np.linalg.norm(np.log10(np.abs(bz_ana[ind])))
    print 'Difference: ', log10diff

    if showIt == True:
        plt.loglog(rx.times[bz_calc>0], bz_calc[bz_calc>0], 'r', rx.times[bz_calc<0], -bz_calc[bz_calc<0], 'r--')
        plt.loglog(rx.times, abs(bz_ana), 'b*')
        plt.title('sig_half = %e'%sig_half)
        plt.show()

    return log10diff


class TDEM_bTests(unittest.TestCase):

    def test_analytic_p2_TENSOR_50m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('TENSOR', rxOffset=50., sig_half=1e+2) < 0.01)

    def test_analytic_p2_CYL_50m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=50., sig_half=1e+2) < 0.01)
    def test_analytic_p1_CYL_50m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=50., sig_half=1e+1) < 0.01)
    def test_analytic_p0_CYL_50m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=50., sig_half=1e+0) < 0.01)
    def test_analytic_m1_CYL_50m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=50., sig_half=1e-1) < 0.01)
    def test_analytic_m2_CYL_50m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=50., sig_half=1e-2) < 0.01)
    def test_analytic_m3_CYL_50m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=50., sig_half=1e-3) < 0.02)

    def test_analytic_p0_CYL_1m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=1.0, sig_half=1e+0) < 0.01)
    def test_analytic_m1_CYL_1m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=1.0, sig_half=1e-1) < 0.01)
    def test_analytic_m2_CYL_1m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=1.0, sig_half=1e-2) < 0.01)
    def test_analytic_m3_CYL_1m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=1.0, sig_half=1e-3) < 0.02)



if __name__ == '__main__':
    unittest.main()
