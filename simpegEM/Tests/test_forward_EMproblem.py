import unittest
from SimPEG import *
import simpegEM as EM
from scipy.constants import mu_0
from simpegEM.Utils.Ana import hzAnalyticDipoleT


class EMProblemTests(unittest.TestCase):

    def setUp(self):

       cs = 5.
       ncx = 20
       ncy = 6
       npad = 20
       hx = Utils.meshTensors(((0,cs), (ncx,cs), (npad,cs)))
       hy = Utils.meshTensors(((npad,cs), (ncy,cs), (npad,cs)))
       mesh = Mesh.Cyl1DMesh([hx,hy], -hy.sum()/2)
       model = Model.Vertical1DModel(mesh)

       opts = {'txLoc':0.,
               'txType':'VMD_MVP',
               'rxLoc':np.r_[150., 0.],
               'rxType':'bz',
               'timeCh':np.logspace(-4,-2,20),
               }
       self.dat = EM.TDEM.DataTDEM1D(**opts)

       self.prb = EM.TDEM.ProblemTDEM_b(mesh, model)
       self.prb.setTimes([1e-5, 5e-5, 2.5e-4], [150, 150, 150])
       self.sigma = np.ones(mesh.nCz)*1e-8
       self.sigma[mesh.vectorCCz<0] = 0.1
       self.prb.pair(self.dat)


    def test_analitic_b(self):
       bz_calc = self.dat.dpred(self.sigma)
       bz_ana = mu_0*hzAnalyticDipoleT(self.dat.rxLoc[0], self.prb.times, self.sigma[0])

       diff = np.linalg.norm(bz_calc.flatten() - bz_ana.flatten())/np.linalg.norm(bz_ana.flatten())
       print diff
       self.assertTrue(diff<0.05)

    def test_awesome(self):
        self.assertTrue(True)





if __name__ == '__main__':
    unittest.main()
