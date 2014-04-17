import unittest
from SimPEG import *
import simpegEM as EM
from scipy.constants import mu_0
import matplotlib.pyplot as plt


def halfSpaceProblemAnaDiff(meshType, sig_half=1e-2, rxOffset=50., bounds=[1e-5,1e-3], showIt=False):
    if meshType == 'CYL':
        cs, ncx, ncz, npad = 5., 30, 10, 15
        hx = Utils.meshTensors(((0,cs), (ncx,cs), (npad,cs)))
        hz = Utils.meshTensors(((npad,cs), (ncz,cs), (npad,cs)))
        mesh = Mesh.CylMesh([hx,1,hz], [0,0,-hz.sum()/2])
    elif meshType == 'TENSOR':
        cs, nc, npad = 20., 13, 5
        hx = Utils.meshTensors(((npad,cs), (nc,cs), (npad,cs)))
        hy = Utils.meshTensors(((npad,cs), (nc,cs), (npad,cs)))
        hz = Utils.meshTensors(((npad,cs), (nc,cs), (npad,cs)))
        mesh = Mesh.TensorMesh([hx,hy,hz], [-hx.sum()/2.,-hy.sum()/2.,-hz.sum()/2.])

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

    prb.setTimes([1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4], [40, 40, 40, 40, 40, 40])

    sigma = np.ones(mesh.nCz)*1e-8
    sigma[active] = sig_half
    sigma = np.log(sigma[active])
    prb.pair(survey)

    bz_ana = mu_0*EM.Utils.Ana.hzAnalyticDipoleT(survey.rxLoc[0], prb.times, sig_half)

    bz_calc = survey.dpred(sigma)
    ind = np.logical_and(prb.times > bounds[0],prb.times < bounds[1])
    log10diff = np.linalg.norm(np.log10(np.abs(bz_calc[ind])) - np.log10(np.abs(bz_ana[ind])))/np.linalg.norm(np.log10(np.abs(bz_ana[ind])))
    print 'Difference: ', log10diff

    if showIt == True:
        plt.loglog(prb.times[bz_calc>0], bz_calc[bz_calc>0], 'r', prb.times[bz_calc<0], -bz_calc[bz_calc<0], 'r--')
        plt.loglog(prb.times, abs(bz_ana), 'b*')
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
