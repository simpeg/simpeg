import unittest
from SimPEG import *
import simpegEM as EM
from scipy.constants import mu_0
import matplotlib.pyplot as plt


def halfSpaceProblemAnaDiff(meshType, sig_half=1e-2, rxOffset=50., bounds=[1e-5,1e-3], showIt=False):
    if meshType == 'CYL':
        cs, ncx, ncz, npad = 5., 30, 10, 15
        hx = [(cs,ncx), (cs,npad,1.3)]
        hz = [(cs,npad,-1.3), (cs,ncz), (cs,npad,1.3)]
        mesh = Mesh.CylMesh([hx,1,hz], '00C')
    elif meshType == 'TENSOR':
        cs, nc, npad = 20., 13, 5
        hx = [(cs,npad,-1.3), (cs,nc), (cs,npad,1.3)]
        hy = [(cs,npad,-1.3), (cs,nc), (cs,npad,1.3)]
        hz = [(cs,npad,-1.3), (cs,nc), (cs,npad,1.3)]
        mesh = Mesh.TensorMesh([hx,hy,hz], 'CCC')

    active = mesh.vectorCCz<0.
    actMap = Maps.ActiveCells(mesh, active, np.log(1e-8), nC=mesh.nCz)
    mapping = Maps.ComboMap(mesh, [Maps.ExpMap, Maps.Vertical1DMap, actMap])


    opts = {'txLoc':np.array([0., 0., 0.]),
            'txType':'VMD_MVP',
            'rxLoc':np.array([rxOffset, 0., 0.]),
            'rxType':'bz',
            'timeCh':np.logspace(-5,-4, 21),
            }

    survey = EM.TDEM.SurveyTDEM1D(**opts)
    prb = EM.TDEM.ProblemTDEM_b(mesh, mapping=mapping)
    prb.Solver = Utils.SolverUtils.DSolverWrap(sp.linalg.splu, factorize=True)
    # try:
    #     from mumpsSCI import MumpsSolver
    #     prb.Solver = MumpsSolver
    # except ImportError, e:
    #     pass

    prb.timeSteps = [(1e-06, 40), (5e-06, 40), (1e-05, 40), (5e-05, 40), (0.0001, 40), (0.0005, 40)]

    sigma = np.ones(mesh.nCz)*1e-8
    sigma[active] = sig_half
    sigma = np.log(sigma[active])
    prb.pair(survey)

    bz_ana = mu_0*EM.Utils.Ana.hzAnalyticDipoleT(survey.rxLoc[0]+1e-3, prb.times[1:], sig_half)

    bz_calc = survey.dpred(sigma)
    ind = np.logical_and(prb.times[1:] > bounds[0],prb.times[1:] < bounds[1])
    log10diff = np.linalg.norm(np.log10(np.abs(bz_calc[ind])) - np.log10(np.abs(bz_ana[ind])))/np.linalg.norm(np.log10(np.abs(bz_ana[ind])))
    print 'Difference: ', log10diff

    if showIt == True:
        plt.loglog(prb.times[1:][bz_calc>0], bz_calc[bz_calc>0], 'r', prb.times[1:][bz_calc<0], -bz_calc[bz_calc<0], 'r--')
        plt.loglog(prb.times[1:], abs(bz_ana), 'b*')
        plt.title('sig_half = %e'%sig_half)
        plt.show()

    return log10diff


class TDEM_bTests(unittest.TestCase):

    def test_analitic_p2_CYL_50m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=50., sig_half=1e+2) < 0.01)
    def test_analitic_p1_CYL_50m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=50., sig_half=1e+1) < 0.01)
    def test_analitic_p0_CYL_50m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=50., sig_half=1e+0) < 0.01)
    def test_analitic_m1_CYL_50m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=50., sig_half=1e-1) < 0.01)
    def test_analitic_m2_CYL_50m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=50., sig_half=1e-2) < 0.01)
    def test_analitic_m3_CYL_50m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=50., sig_half=1e-3) < 0.02)

    def test_analitic_p0_CYL_1m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=1.0, sig_half=1e+0) < 0.01)
    def test_analitic_m1_CYL_1m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=1.0, sig_half=1e-1) < 0.01)
    def test_analitic_m2_CYL_1m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=1.0, sig_half=1e-2) < 0.01)
    def test_analitic_m3_CYL_1m(self):
        self.assertTrue(halfSpaceProblemAnaDiff('CYL', rxOffset=1.0, sig_half=1e-3) < 0.02)



if __name__ == '__main__':
    unittest.main()
