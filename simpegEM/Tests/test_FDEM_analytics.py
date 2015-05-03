import unittest
from SimPEG import *
import simpegEM as EM
from scipy.constants import mu_0

plotIt = False
freq = 1e2

class FDEM_analyticTests(unittest.TestCase):

    def setUp(self):

        cs = 10.
        ncx, ncy, ncz = 10, 10, 10
        npad = 4
        hx = [(cs,npad,-1.3), (cs,ncx), (cs,npad,1.3)]
        hy = [(cs,npad,-1.3), (cs,ncy), (cs,npad,1.3)]
        hz = [(cs,npad,-1.3), (cs,ncz), (cs,npad,1.3)]
        mesh = Mesh.TensorMesh([hx,hy,hz], 'CCC')

        mapping = Maps.ExpMap(mesh)

        x = np.linspace(-10,10,5)
        XYZ = Utils.ndgrid(x,np.r_[0],np.r_[0])
        rxList = EM.FDEM.RxFDEM(XYZ, 'exi')
        # Src0 = EM.FDEM.SrcFDEM(np.r_[0.,0.,0.], 'VMD', 1e2, [rxList])
        Src0 = EM.FDEM.SrcFDEM_MagDipole(np.r_[0.,0.,0.], freq, [rxList])

        survey = EM.FDEM.SurveyFDEM([Src0])

        prb = EM.FDEM.ProblemFDEM_b(mesh, mapping=mapping)
        prb.pair(survey)

        try:
            from pymatsolver import MumpsSolver
            prb.Solver = MumpsSolver
        except ImportError, e:
            prb.Solver = SolverLU

        sig = 1e-1
        sigma = np.ones(mesh.nC)*sig
        sigma[mesh.gridCC[:,2] > 0] = 1e-8
        m = np.log(sigma)

        self.prb = prb
        self.mesh = mesh
        self.m = m
        self.Src0 = Src0
        self.sig = sig

    def test_Transect(self):
        print 'Testing Transect for analytic'

        u = self.prb.fields(self.m)

        bfz = self.mesh.r(u[self.Src0, 'b'],'F','Fz','M')

        x = np.linspace(-55,55,12)
        XYZ = Utils.ndgrid(x,np.r_[0],np.r_[0])

        P = self.mesh.getInterpolationMat(XYZ, 'Fz')

        an = EM.Analytics.FDEM.hzAnalyticDipoleF(x, self.Src0.freq, self.sig)

        diff = np.log10(np.abs(P*np.imag(u[self.Src0, 'b']) - mu_0*np.imag(an)))

        if plotIt:
            import matplotlib.pyplot as plt
            plt.plot(x,np.log10(np.abs(P*np.imag(u[self.Src0, 'b']))))
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
