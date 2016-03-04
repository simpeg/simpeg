import unittest
import SimPEG.DCIP as DC
from SimPEG import *
from pymatsolver import MumpsSolver

class IPforwardTests(unittest.TestCase):

    def test_IPforward(self):

        cs = 12.5
        nc = 500/cs+1
        hx = [(cs,7, -1.3),(cs,nc),(cs,7, 1.3)]
        hy = [(cs,7, -1.3),(cs,int(nc/2+1)),(cs,7, 1.3)]
        hz = [(cs,7, -1.3),(cs,int(nc/2+1))]
        mesh = Mesh.TensorMesh([hx, hy, hz], 'CCN')
        sighalf = 1e-2
        sigma = np.ones(mesh.nC)*sighalf
        p0 = np.r_[-50., 50., -50.]
        p1 = np.r_[ 50.,-50., -150.]
        blk_ind = Utils.ModelBuilder.getIndicesBlock(p0, p1, mesh.gridCC)
        sigma[blk_ind] = 1e-3
        eta = np.zeros_like(sigma)
        eta[blk_ind] = 0.1
        sigmaInf = sigma.copy()
        sigma0 = sigma*(1-eta)

        nElecs = 11
        x_temp = np.linspace(-250, 250, nElecs)
        aSpacing = x_temp[1]-x_temp[0]
        y_temp = 0.
        xyz = Utils.ndgrid(x_temp, np.r_[y_temp], np.r_[0.])
        srcList = DC.Utils.WennerSrcList(nElecs,aSpacing)
        survey = DC.SurveyDC(srcList)

        imap   = Maps.IdentityMap(mesh)
        problem = DC.ProblemDC_CC(mesh, mapping=imap)
        problem.Solver = MumpsSolver
        problem.pair(survey)

        phi0 = survey.dpred(sigma0)
        phiInf = survey.dpred(sigmaInf)

        phiIP_true = phi0-phiInf

        surveyIP = DC.SurveyIP(srcList)
        problemIP = DC.ProblemIP(mesh, sigma=sigma)
        problemIP.pair(surveyIP)

        problemIP.Solver = MumpsSolver
        phiIP_approx = surveyIP.dpred(eta)

        err =  np.linalg.norm(phiIP_true-phiIP_approx) / np.linalg.norm(phiIP_true)
        passed = err < 0.02
        self.assertTrue(passed)


if __name__ == '__main__':
    unittest.main()
