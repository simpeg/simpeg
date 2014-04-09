import unittest
from SimPEG import *
import simpegEM as EM
from scipy.constants import mu_0
from simpegEM.Utils.Ana import hzAnalyticDipoleT
import matplotlib.pyplot as plt

class TDEM_bTests(unittest.TestCase):

    def setUp(self):

        cs = 10.
        ncx = 15
        ncy = 10
        npad = 20
        hx = Utils.meshTensors(((0,cs), (ncx,cs), (npad,cs)))        
        hy = Utils.meshTensors(((npad,cs), (ncy,cs), (npad,cs)))
        mesh = Mesh.Cyl1DMesh([hx,hy], -hy.sum()/2)

        active = mesh.vectorCCz<0.
        model = Model.ActiveModel(mesh, active, -8, nC=mesh.nCz)
        model = Model.ComboModel(mesh,
                    [Model.LogModel, Model.Vertical1DModel, model])

        opts = {'txLoc':0.,
                'txType':'VMD_MVP',
                'rxLoc':np.r_[30., 0.],
                'rxType':'bz',
                'timeCh':np.logspace(-4,-2.5, 21),
                }

        self.dat = EM.TDEM.SurveyTDEM1D(**opts)
        self.prb = EM.TDEM.ProblemTDEM_b(model)
        self.prb.setTimes([1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4], [40, 40, 40, 40, 40, 40])

        self.sigma = np.ones(mesh.nCz)*1e-8
        self.sigma[mesh.vectorCCz<0] = 1e-3
        self.sigma = np.log(self.sigma[active])
        self.showIt = True
        self.prb.pair(self.dat)

    def test_analitic_b(self):
        bz_calc = self.dat.dpred(self.sigma)
        bz_ana = mu_0*hzAnalyticDipoleT(self.dat.rxLoc[0], self.prb.times, np.exp(self.sigma[0]))
        ind = self.prb.times > 1e-5
        diff = np.linalg.norm(bz_calc[ind].flatten() - bz_ana[ind].flatten())/np.linalg.norm(bz_ana[ind].flatten())

        if self.showIt == True:

            plt.loglog(self.prb.times[bz_calc>0], bz_calc[bz_calc>0], 'b', self.prb.times[bz_calc<0], -bz_calc[bz_calc<0], 'b--')
            plt.loglog(self.prb.times, abs(bz_ana), 'b*')
            plt.xlim(1e-5, 1e-2)
            plt.show()

        print diff
        self.assertTrue(diff < 0.10)



if __name__ == '__main__':
    unittest.main()
