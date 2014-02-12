import unittest
from SimPEG import *
import simpegEM as EM
from scipy.constants import mu_0
from simpegEM.Utils.Ana import hzAnalyticDipoleT


class TDEM_bTests(unittest.TestCase):

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
       self.assertTrue(diff<0.05)


class TDEM_bDerivTests(unittest.TestCase):

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
       self.prb.setTimes([1e-5, 5e-5, 2.5e-4], [10, 10, 10])
       self.sigma = np.ones(mesh.nCz)*1e-8
       self.sigma[mesh.vectorCCz<0] = 0.1
       self.prb.pair(self.dat)

    def test_AhVec(self):
        """
            Test that fields and AhVec produce consistent results
        """

        sigma = np.ones(self.prb.mesh.nCz)*1e-8
        sigma[self.prb.mesh.vectorCCz<0] = 0.1
        u = self.prb.fields(sigma)
        Ahu = self.prb.AhVec(sigma, u)
        self.assertTrue(np.linalg.norm(Ahu.get_b(0)-1/self.prb.getDt(0)*u.get_b(-1))/np.linalg.norm(u.get_b(0)) < 1.e-2)
        self.assertTrue(np.linalg.norm(Ahu.get_e(0))/np.linalg.norm(u.get_e(0)) < 1.e-2)
        for i in range(1,u.nTimes):
            self.assertTrue(np.linalg.norm(Ahu.get_b(i))/np.linalg.norm(u.get_b(i)) < 1.e-2)
            self.assertTrue(np.linalg.norm(Ahu.get_e(i))/np.linalg.norm(u.get_e(i)) < 1.e-2)


    def test_DerivG(self):
        """
            Test the derivative of c with respect to sigma
        """

        # Random model and perturbation
        sigma = np.random.rand(self.prb.mesh.nCz)
        f = self.prb.fields(sigma)
        dm = np.random.rand(self.prb.mesh.nCz)
        h = 1.

        a = np.linalg.norm(self.prb.AhVec(sigma+h*dm, f).fieldVec() - self.prb.AhVec(sigma, f).fieldVec())
        b = np.linalg.norm(self.prb.AhVec(sigma+h*dm, f).fieldVec() - self.prb.AhVec(sigma, f).fieldVec() - h*self.prb.G(sigma, dm, u=f).fieldVec())
        # Assuming that the gradient is exact to machine precision
        self.assertTrue(b<1e-16)


if __name__ == '__main__':
    unittest.main()
