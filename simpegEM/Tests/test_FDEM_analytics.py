import unittest
from SimPEG import *
import simpegEM as EM
from scipy.constants import mu_0

plotIt = False

class FDEM_analyticTests(unittest.TestCase):

    def setUp(self):

        cs = 10.
        ncx, ncy, ncz = 8, 8, 8
        npad = 5
        hx = Utils.meshTensors(((npad,cs), (ncx,cs), (npad,cs)))
        hy = Utils.meshTensors(((npad,cs), (ncy,cs), (npad,cs)))
        hz = Utils.meshTensors(((npad,cs), (ncz,cs), (npad,cs)))
        mesh = Mesh.TensorMesh([hx,hy,hz], x0=[-hx.sum()/2.,-hy.sum()/2.,-hz.sum()/2.,])

        model = Model.LogModel(mesh)

        x = np.linspace(-10,10,5)
        XYZ = Utils.ndgrid(x,np.r_[0],np.r_[0])
        rxList = EM.FDEM.RxListFDEM(XYZ, 'exi')
        Tx0 = EM.FDEM.TxFDEM(np.r_[0.,0.,0.], 'VMD', 1e2, rxList)

        survey = EM.FDEM.SurveyFDEM([Tx0])

        prb = EM.FDEM.ProblemFDEM_b(model)
        prb.pair(survey)
        prb.Solver = Utils.SolverUtils.DSolverWrap(sp.linalg.splu, checkAccuracy=False)

        sig = 1e-1
        sigma = np.ones(mesh.nC)*sig
        sigma[mesh.gridCC[:,2] > 0] = 1e-8
        m = np.log(sigma)

        self.prb = prb
        self.mesh = mesh
        self.m = m
        self.Tx0 = Tx0
        self.sig = sig

    def test_Transect(self):
        print 'Testing Transect for analytic'

        u = self.prb.fields(self.m)

        bfz = self.mesh.r(u[self.Tx0, 'b'],'F','Fz','M')

        x = np.linspace(-55,55,12)
        XYZ = Utils.ndgrid(x,np.r_[0],np.r_[0])

        P = self.mesh.getInterpolationMat(XYZ, 'Fz')

        an = EM.Utils.Ana.FEM.hzAnalyticDipoleF(x, self.Tx0.freq, self.sig)

        diff = np.log10(np.abs(P*np.imag(u[self.Tx0, 'b']) - np.abs(mu_0*np.imag(an))))

        if plotIt:
            import matplotlib.pyplot as plt
            plt.plot(x,np.log10(np.abs(P*np.imag(u[self.Tx0, 'b']))))
            plt.plot(x,np.log10(np.abs(mu_0*np.imag(an))), 'r')
            plt.plot(x,diff,'g')
            plt.show()

        # We want the difference to be an orderMag less
        # than the analytic solution. Note that right at
        # the source, both the analytic and the numerical
        # solution will be poor. Use plotIt up top to see that...
        orderMag = 1.6
        passed = np.abs(np.mean(diff - np.log10(np.abs(mu_0*np.imag(an))))) > orderMag
        self.assertTrue(passed)


if __name__ == '__main__':
    unittest.main()
