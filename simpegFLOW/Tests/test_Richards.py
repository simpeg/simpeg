import unittest
from SimPEG import *
from SimPEG.Tests.TestUtils import OrderTest, checkDerivative
from scipy.sparse.linalg import dsolve
from simpegFLOW import Richards

TOL = 1E-8

class TestModels(unittest.TestCase):

    def test_BaseHaverkamp_Theta(self):
        mesh = Mesh.TensorMesh([50])
        hav = Richards.Empirical._haverkamp_theta(mesh)
        m = np.random.randn(50)
        def wrapper(u):
            return hav.transform(u, m), hav.transformDerivU(u, m)
        passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
        self.assertTrue(passed,True)

    def test_vangenuchten_theta(self):
        mesh = Mesh.TensorMesh([50])
        hav = Richards.Empirical._vangenuchten_theta(mesh)
        m = np.random.randn(50)
        def wrapper(u):
            return hav.transform(u, m), hav.transformDerivU(u, m)
        passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
        self.assertTrue(passed,True)

    def test_BaseHaverkamp_k(self):
        mesh = Mesh.TensorMesh([50])
        hav = Richards.Empirical._haverkamp_k(mesh)
        m = np.random.randn(50)
        def wrapper(u):
            return hav.transform(u, m), hav.transformDerivU(u, m)
        passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
        self.assertTrue(passed,True)

        hav = Richards.Empirical._haverkamp_k(mesh)
        u = np.random.randn(50)
        def wrapper(m):
            return hav.transform(u, m), hav.transformDerivM(u, m)
        passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
        self.assertTrue(passed,True)

    def test_vangenuchten_k(self):
        mesh = Mesh.TensorMesh([50])
        hav = Richards.Empirical._vangenuchten_k(mesh)
        m = np.random.randn(50)
        def wrapper(u):
            return hav.transform(u, m), hav.transformDerivU(u, m)
        passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
        self.assertTrue(passed,True)

        hav = Richards.Empirical._vangenuchten_k(mesh)
        u = np.random.randn(50)
        def wrapper(m):
            return hav.transform(u, m), hav.transformDerivM(u, m)
        passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
        self.assertTrue(passed,True)



class RichardsTests1D(unittest.TestCase):

    def setUp(self):
        M = Mesh.TensorMesh([np.ones(20)])
        M.setCellGradBC('dirichlet')

        params = Richards.Empirical.HaverkampParams().celia1990
        E = Richards.Empirical.Haverkamp(M, **params)

        bc = np.array([-61.5,-20.7])
        h = np.zeros(M.nC) + bc[0]

        prob = Richards.RichardsProblem(M, mapping=E, timeSteps=[(40,3),(60,3)],
                                    boundaryConditions=bc, initialConditions=h,
                                    doNewton=False, method='mixed')

        locs = np.r_[5.,10,15]
        times = prob.times[3:5]
        rxSat = Richards.RichardsRx(locs, times, 'saturation')
        rxPre = Richards.RichardsRx(locs, times, 'pressureHead')
        survey = Richards.RichardsSurvey([rxSat, rxPre])

        prob.pair(survey)

        self.h0 = h
        self.M = M
        self.Ks = params['Ks']
        self.prob = prob
        self.survey = survey

    def test_Richards_getResidual_Newton(self):
        self.prob.doNewton = True
        m = self.Ks
        passed = checkDerivative(lambda hn1: self.prob.getResidual(m, self.h0, hn1, self.prob.timeSteps[0]), self.h0, plotIt=False)
        self.assertTrue(passed,True)

    def test_Richards_getResidual_Picard(self):
        self.prob.doNewton = False
        m = self.Ks
        passed = checkDerivative(lambda hn1: self.prob.getResidual(m, self.h0, hn1, self.prob.timeSteps[0]), self.h0, plotIt=False, expectedOrder=1)
        self.assertTrue(passed,True)

    def test_Adjoint(self):
        v = np.random.rand(self.survey.nD)
        z = np.random.rand(self.M.nC)
        Hs = self.prob.fields(self.Ks)
        vJz = v.dot(self.prob.Jvec(self.Ks,z,u=Hs))
        zJv = z.dot(self.prob.Jtvec(self.Ks,v,u=Hs))
        tol = TOL*(10**int(np.log10(zJv)))
        passed = np.abs(vJz - zJv) < tol
        print 'Richards Adjoint Test - PressureHead'
        print '%4.4e === %4.4e, diff=%4.4e < %4.e'%(vJz, zJv,np.abs(vJz - zJv),tol)
        self.assertTrue(passed,True)

    def test_Sensitivity(self):
        mTrue = self.Ks*np.ones(self.M.nC)
        derChk = lambda m: [self.survey.dpred(m), lambda v: self.prob.Jvec(m, v)]
        print 'Testing Richards Derivative'
        passed = checkDerivative(derChk, mTrue, num=4, plotIt=False)
        self.assertTrue(passed,True)


# class RichardsTests2D(object):

#     def setUp(self):
#         M = mesh.TensorMesh([np.ones(8),np.ones(30)])
#         Ks = 9.4400e-03
#         E = Richards.Haverkamp(Ks=np.log(Ks), A=1.1750e+06, gamma=4.74, alpha=1.6110e+06, theta_s=0.287, theta_r=0.075, beta=3.96)

#         bc = np.array([-61.5,-20.7])
#         bc = np.r_[np.zeros(M.nCy*2),np.ones(M.nCx)*bc[0],np.ones(M.nCx)*bc[1]]
#         h = np.zeros(M.nC) + bc[0]
#         prob = Richards.RichardsProblem(M,E, timeStep=60, timeEnd=180, boundaryConditions=bc, initialConditions=h, doNewton=False, method='mixed')


#         XY = utils.ndgrid(np.array([5,7.]),np.array([5,15,25.]))
#         q = M.getInterpolationMat(XY,'CC')
#         P = sp.kron(sp.identity(prob.numIts),q)
#         prob.P = P

#         self.h0 = h
#         self.M = M
#         self.Ks = Ks
#         self.prob = prob

#     def test_Richards_getResidual_Newton(self):
#         self.prob.doNewton = True
#         passed = checkDerivative(lambda hn1: self.prob.getResidual(self.h0,hn1), self.h0, plotIt=False)
#         self.assertTrue(passed,True)

#     def test_Richards_getResidual_Picard(self):
#         self.prob.doNewton = False
#         passed = checkDerivative(lambda hn1: self.prob.getResidual(self.h0,hn1), self.h0, plotIt=False, expectedOrder=1)
#         self.assertTrue(passed,True)

#     def test_Adjoint_PressureHead(self):
#         self.prob.dataType = 'pressureHead'
#         Ks = self.Ks
#         v = np.random.rand(self.prob.P.shape[0])
#         z = np.random.rand(self.M.nC)
#         Hs = self.prob.field(np.log(Ks))
#         vJz = v.dot(self.prob.J(np.log(Ks),z,u=Hs))
#         zJv = z.dot(self.prob.Jt(np.log(Ks),v,u=Hs))
#         tol = TOL*(10**int(np.log10(zJv)))
#         passed = np.abs(vJz - zJv) < tol
#         print 'Richards Adjoint Test - PressureHead'
#         print '%4.4e === %4.4e, diff=%4.4e < %4.e'%(vJz, zJv,np.abs(vJz - zJv),tol)
#         self.assertTrue(passed,True)


#     def test_Adjoint_Saturation(self):
#         self.prob.dataType = 'saturation'
#         Ks = self.Ks
#         v = np.random.rand(self.prob.P.shape[0])
#         z = np.random.rand(self.M.nC)
#         Hs = self.prob.field(np.log(Ks))
#         vJz = v.dot(self.prob.J(np.log(Ks),z,u=Hs))
#         zJv = z.dot(self.prob.Jt(np.log(Ks),v,u=Hs))
#         tol = TOL #*(10**int(np.log10(zJv)))
#         passed = np.abs(vJz - zJv) < tol
#         print 'Richards Adjoint Test - Saturation'
#         print '%4.4e === %4.4e, diff=%4.4e < %4.e'%(vJz, zJv,np.abs(vJz - zJv),tol)
#         self.assertTrue(passed,True)

#     def test_Sensitivity(self):
#         self.prob.dataType = 'pressureHead'
#         mTrue = np.ones(self.M.nC)*self.Ks
#         stdev = 0.01  # The standard deviation for the noise
#         dobs = self.prob.createSyntheticSurvey(mTrue,std=stdev)[0]
#         self.prob.dobs = dobs
#         self.prob.std = dobs*0 + stdev
#         Hs = self.prob.field(mTrue)
#         opt = inverse.InexactGaussNewton(maxIterLS=20, maxIter=10, tolF=1e-6, tolX=1e-6, tolG=1e-6, maxIterCG=6)
#         reg = regularization.Regularization(self.M)
#         inv = inverse.Inversion(self.prob, reg, opt, beta0=1e4)
#         derChk = lambda m: [inv.dataObj(m), inv.dataObjDeriv(m)]
#         print 'Testing Richards Derivative'
#         passed = checkDerivative(derChk, mTrue, num=5, plotIt=False)
#         self.assertTrue(passed,True)

if __name__ == '__main__':
    unittest.main()
