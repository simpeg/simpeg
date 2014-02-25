import unittest
from SimPEG import *
from SimPEG.Tests.TestUtils import OrderTest, checkDerivative
from scipy.sparse.linalg import dsolve
import simpegFLOW.Richards as Richards

TOL = 1E-8

class TestModels(unittest.TestCase):

    def test_BaseHaverkamp_Theta(self):
        mesh = Mesh.TensorMesh([50])
        hav = Richards.BaseHaverkamp_theta(mesh)
        m = np.random.randn(50)
        def wrapper(u):
            return hav.transform(u, m), hav.transformDerivU(u, m)
        passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
        self.assertTrue(passed,True)


    def test_BaseHaverkamp_k(self):
        mesh = Mesh.TensorMesh([50])
        hav = Richards.BaseHaverkamp_k(mesh)
        m = np.random.randn(50)
        def wrapper(u):
            return hav.transform(u, m), hav.transformDerivU(u, m)
        passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
        self.assertTrue(passed,True)

        hav = Richards.BaseHaverkamp_k(mesh)
        u = np.random.randn(50)
        def wrapper(m):
            return hav.transform(u, m), hav.transformDerivM(u, m)
        passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
        self.assertTrue(passed,True)

    # def test_Haverkamp_hydraulicConductivity(self):
    #     print 'Haverkamp_hydraulicConductivity'
    #     hav = Richards.Haverkamp()
    #     def wrapper(x):
    #         return hav.hydraulicConductivity(x), hav.hydraulicConductivityDeriv(x)
    #     passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
    #     self.assertTrue(passed,True)

    # def test_Haverkamp_hydraulicConductivity_FullKs(self):
    #     print 'Haverkamp_hydraulicConductivity_FullKs'
    #     n = 50
    #     hav = Richards.Haverkamp(Ks=np.random.rand(n))
    #     def wrapper(x):
    #         return hav.hydraulicConductivity(x), hav.hydraulicConductivityDeriv(x)
    #     passed = checkDerivative(wrapper, np.random.randn(n), plotIt=False)
    #     self.assertTrue(passed,True)

    # def test_VanGenuchten_moistureContent(self):
    #     print 'VanGenuchten_moistureContent'
    #     vanG = Richards.VanGenuchten()
    #     def wrapper(x):
    #         return vanG.moistureContent(x), vanG.moistureContentDeriv(x)
    #     passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
    #     self.assertTrue(passed,True)

    # def test_VanGenuchten_hydraulicConductivity(self):
    #     print 'VanGenuchten_hydraulicConductivity'
    #     hav = Richards.VanGenuchten()
    #     def wrapper(x):
    #         return hav.hydraulicConductivity(x), hav.hydraulicConductivityDeriv(x)
    #     passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
    #     self.assertTrue(passed,True)

    # def test_VanGenuchten_hydraulicConductivity_FullKs(self):
    #     print 'VanGenuchten_hydraulicConductivity_FullKs'
    #     n = 50
    #     hav = Richards.VanGenuchten(Ks=np.random.rand(n))
    #     def wrapper(x):
    #         return hav.hydraulicConductivity(x), hav.hydraulicConductivityDeriv(x)
    #     passed = checkDerivative(wrapper, np.random.randn(n), plotIt=False)
    #     self.assertTrue(passed,True)

    # def test_Haverkamp_moistureContent(self):
    #     print 'Haverkamp_moistureContent'
    #     hav = Richards.Haverkamp()
    #     def wrapper(x):
    #         return hav.moistureContent(x), hav.moistureContentDeriv(x)
    #     passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
    #     self.assertTrue(passed,True)

    # def test_Haverkamp_hydraulicConductivity(self):
    #     print 'Haverkamp_hydraulicConductivity'
    #     hav = Richards.Haverkamp()
    #     def wrapper(x):
    #         return hav.hydraulicConductivity(x), hav.hydraulicConductivityDeriv(x)
    #     passed = checkDerivative(wrapper, np.random.randn(50), plotIt=False)
    #     self.assertTrue(passed,True)

    # def test_Haverkamp_hydraulicConductivity_FullKs(self):
    #     print 'Haverkamp_hydraulicConductivity_FullKs'
    #     n = 50
    #     hav = Richards.Haverkamp(Ks=np.random.rand(n))
    #     def wrapper(x):
    #         return hav.hydraulicConductivity(x), hav.hydraulicConductivityDeriv(x)
    #     passed = checkDerivative(wrapper, np.random.randn(n), plotIt=False)
    #     self.assertTrue(passed,True)


# class RichardsTests1D(unittest.TestCase):

#     def setUp(self):
#         M = Mesh.TensorMesh([np.ones(20)])
#         M.setCellGradBC('dirichlet')

#         Ks = 9.4400e-03
#         E = Richards.Haverkamp(Ks=np.log(Ks), A=1.1750e+06, gamma=4.74, alpha=1.6110e+06, theta_s=0.287, theta_r=0.075, beta=3.96)

#         bc = np.array([-61.5,-20.7])
#         h = np.zeros(M.nC) + bc[0]
#         prob = Richards.RichardsProblem(M,E, timeStep=60, timeEnd=180, boundaryConditions=bc, initialConditions=h, doNewton=False, method='mixed')

#         q = sp.csr_matrix((np.ones(3),(np.arange(3),np.array([5,10,15]))),shape=(3,M.nC))
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
#         tol = TOL*(10**int(np.log10(zJv)))
#         passed = np.abs(vJz - zJv) < tol
#         print 'Richards Adjoint Test - Saturation'
#         print '%4.4e === %4.4e, diff=%4.4e < %4.e'%(vJz, zJv,np.abs(vJz - zJv),tol)
#         self.assertTrue(passed,True)

#     def test_Sensitivity(self):
#         self.prob.dataType = 'pressureHead'
#         mTrue = np.ones(self.M.nC)*np.log(self.Ks)
#         stdev = 0.01  # The standard deviation for the noise
#         dobs = self.prob.createSyntheticData(mTrue,std=stdev)[0]
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
#         mTrue = np.ones(self.M.nC)*np.log(self.Ks)
#         stdev = 0.01  # The standard deviation for the noise
#         dobs = self.prob.createSyntheticData(mTrue,std=stdev)[0]
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
