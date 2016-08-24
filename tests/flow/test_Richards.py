from __future__ import print_function
import unittest
import numpy as np
from SimPEG import Mesh, Solver, Utils
from SimPEG.Tests import OrderTest, checkDerivative
from scipy.sparse.linalg import dsolve
from SimPEG.FLOW import Richards
try:
    from pymatsolver import PardisoSolver
    Solver = PardisoSolver
except Exception:
    pass


TOL = 1E-8

np.random.seed(0)


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
        params['Ks'] = np.log(params['Ks'])
        E = Richards.Empirical.Haverkamp(M, **params)

        bc = np.array([-61.5,-20.7])
        h = np.zeros(M.nC) + bc[0]

        prob = Richards.RichardsProblem(M, mapping=E, timeSteps=[(40,3),(60,3)], tolRootFinder=1e-6, debug=False,
                                    boundaryConditions=bc, initialConditions=h,
                                    doNewton=False, method='mixed')
        prob.Solver = Solver

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
        passed = checkDerivative(lambda hn1: self.prob.getResidual(m, self.h0, hn1, self.prob.timeSteps[0], self.prob.boundaryConditions), self.h0, plotIt=False)
        self.assertTrue(passed,True)

    def test_Richards_getResidual_Picard(self):
        self.prob.doNewton = False
        m = self.Ks
        passed = checkDerivative(lambda hn1: self.prob.getResidual(m, self.h0, hn1, self.prob.timeSteps[0], self.prob.boundaryConditions), self.h0, plotIt=False, expectedOrder=1)
        self.assertTrue(passed,True)

    def test_Adjoint(self):
        v = np.random.rand(self.survey.nD)
        z = np.random.rand(self.M.nC)
        Hs = self.prob.fields(self.Ks)
        vJz = v.dot(self.prob.Jvec(self.Ks,z,f=Hs))
        zJv = z.dot(self.prob.Jtvec(self.Ks,v,f=Hs))
        tol = TOL*(10**int(np.log10(np.abs(zJv))))
        passed = np.abs(vJz - zJv) < tol
        print('Richards Adjoint Test - PressureHead')
        print('{0:4.4e} === {1:4.4e}, diff={2:4.4e} < {3:4e}'.format(vJz, zJv, np.abs(vJz - zJv), tol))
        self.assertTrue(passed,True)

    def test_Sensitivity(self):
        mTrue = self.Ks*np.ones(self.M.nC)
        derChk = lambda m: [self.survey.dpred(m), lambda v: self.prob.Jvec(m, v)]
        print('Testing Richards Derivative')
        passed = checkDerivative(derChk, mTrue, num=4, plotIt=False)
        self.assertTrue(passed,True)


    def test_Sensitivity_full(self):
        mTrue = self.Ks*np.ones(self.M.nC)
        J = self.prob.Jfull(mTrue)
        derChk = lambda m: [self.survey.dpred(m), J]
        print('Testing Richards Derivative FULL')
        passed = checkDerivative(derChk, mTrue, num=4, plotIt=False)
        self.assertTrue(passed,True)


class RichardsTests2D(unittest.TestCase):

    def setUp(self):
        M = Mesh.TensorMesh([np.ones(8),np.ones(30)])

        M.setCellGradBC(['neumann','dirichlet'])

        params = Richards.Empirical.HaverkampParams().celia1990
        params['Ks'] = np.log(params['Ks'])
        E = Richards.Empirical.Haverkamp(M, **params)

        bc = np.array([-61.5,-20.7])
        bc = np.r_[np.zeros(M.nCy*2),np.ones(M.nCx)*bc[0],np.ones(M.nCx)*bc[1]]
        h = np.zeros(M.nC) + bc[0]
        prob = Richards.RichardsProblem(M,E, timeSteps=[(40,3),(60,3)], boundaryConditions=bc, initialConditions=h, doNewton=False, method='mixed', tolRootFinder=1e-6, debug=False)
        prob.Solver = Solver

        locs = Utils.ndgrid(np.array([5,7.]),np.array([5,15,25.]))
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
        passed = checkDerivative(lambda hn1: self.prob.getResidual(m, self.h0, hn1, self.prob.timeSteps[0], self.prob.boundaryConditions), self.h0, plotIt=False)
        self.assertTrue(passed,True)

    def test_Richards_getResidual_Picard(self):
        self.prob.doNewton = False
        m = self.Ks
        passed = checkDerivative(lambda hn1: self.prob.getResidual(m, self.h0, hn1, self.prob.timeSteps[0], self.prob.boundaryConditions), self.h0, plotIt=False, expectedOrder=1)
        self.assertTrue(passed,True)

    def test_Adjoint(self):
        v = np.random.rand(self.survey.nD)
        z = np.random.rand(self.M.nC)
        Hs = self.prob.fields(self.Ks)
        vJz = v.dot(self.prob.Jvec(self.Ks,z,f=Hs))
        zJv = z.dot(self.prob.Jtvec(self.Ks,v,f=Hs))
        tol = TOL*(10**int(np.log10(np.abs(zJv))))
        passed = np.abs(vJz - zJv) < tol
        print('2D: Richards Adjoint Test - PressureHead')
        print('{0:4.4e} === {1:4.4e}, diff={2:4.4e} < {3:4e}'.format(vJz, zJv, np.abs(vJz - zJv), tol))
        self.assertTrue(passed,True)

    def test_Sensitivity(self):
        mTrue = self.Ks*np.ones(self.M.nC)
        derChk = lambda m: [self.survey.dpred(m), lambda v: self.prob.Jvec(m, v)]
        print('2D: Testing Richards Derivative')
        passed = checkDerivative(derChk, mTrue, num=3, plotIt=False)
        self.assertTrue(passed,True)

    def test_Sensitivity_full(self):
        mTrue = self.Ks*np.ones(self.M.nC)
        J = self.prob.Jfull(mTrue)
        derChk = lambda m: [self.survey.dpred(m), J]
        print('2D: Testing Richards Derivative FULL')
        passed = checkDerivative(derChk, mTrue, num=4, plotIt=False)
        self.assertTrue(passed,True)



class RichardsTests3D(unittest.TestCase):

    def setUp(self):
        M = Mesh.TensorMesh([np.ones(8),np.ones(20),np.ones(10)])

        M.setCellGradBC(['neumann','neumann','dirichlet'])

        params = Richards.Empirical.HaverkampParams().celia1990
        params['Ks'] = np.log(params['Ks'])
        E = Richards.Empirical.Haverkamp(M, **params)

        bc = np.array([-61.5,-20.7])
        bc = np.r_[np.zeros(M.nCy*M.nCz*2),np.zeros(M.nCx*M.nCz*2),np.ones(M.nCx*M.nCy)*bc[0],np.ones(M.nCx*M.nCy)*bc[1]]
        h = np.zeros(M.nC) + bc[0]
        prob = Richards.RichardsProblem(M,E, timeSteps=[(40,3),(60,3)], boundaryConditions=bc, initialConditions=h, doNewton=False, method='mixed', tolRootFinder=1e-6, debug=False)
        prob.Solver = Solver

        locs = Utils.ndgrid(np.r_[5,7.],np.r_[5,15.],np.r_[6,8.])
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
        passed = checkDerivative(lambda hn1: self.prob.getResidual(m, self.h0, hn1, self.prob.timeSteps[0], self.prob.boundaryConditions), self.h0, plotIt=False)
        self.assertTrue(passed,True)

    def test_Richards_getResidual_Picard(self):
        self.prob.doNewton = False
        m = self.Ks
        passed = checkDerivative(lambda hn1: self.prob.getResidual(m, self.h0, hn1, self.prob.timeSteps[0], self.prob.boundaryConditions), self.h0, plotIt=False, expectedOrder=1)
        self.assertTrue(passed,True)

    def test_Adjoint(self):
        v = np.random.rand(self.survey.nD)
        z = np.random.rand(self.M.nC)
        Hs = self.prob.fields(self.Ks)
        vJz = v.dot(self.prob.Jvec(self.Ks,z,f=Hs))
        zJv = z.dot(self.prob.Jtvec(self.Ks,v,f=Hs))
        tol = TOL*(10**int(np.log10(np.abs(zJv))))
        passed = np.abs(vJz - zJv) < tol
        print('3D: Richards Adjoint Test - PressureHead')
        print('{0:4.4e} === {1:4.4e}, diff={2:4.4e} < {3:4e}'.format(vJz, zJv, np.abs(vJz - zJv), tol))
        self.assertTrue(passed,True)

    def test_Sensitivity(self):
        mTrue = self.Ks*np.ones(self.M.nC)
        derChk = lambda m: [self.survey.dpred(m), lambda v: self.prob.Jvec(m, v)]
        print('3D: Testing Richards Derivative')
        passed = checkDerivative(derChk, mTrue, num=4, plotIt=False)
        self.assertTrue(passed,True)

    # def test_Sensitivity_full(self):
    #     mTrue = self.Ks*np.ones(self.M.nC)
    #     J = self.prob.Jfull(mTrue)
    #     derChk = lambda m: [self.survey.dpred(m), J]
    #     print('3D: Testing Richards Derivative FULL')
    #     passed = checkDerivative(derChk, mTrue, num=4, plotIt=False)
    #     self.assertTrue(passed,True)


if __name__ == '__main__':
    unittest.main()
