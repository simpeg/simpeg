from __future__ import print_function
import unittest
import numpy as np
from SimPEG import Mesh
from SimPEG import Maps
from SimPEG import Utils
from SimPEG.Tests import checkDerivative
from SimPEG.FLOW import Richards
try:
    from pymatsolver import PardisoSolver as Solver
except Exception:
    from SimPEG import Solver


TOL = 1E-8

np.random.seed(0)


class RichardsTests1D(unittest.TestCase):

    def setUp(self):
        mesh = Mesh.TensorMesh([np.ones(20)])
        mesh.setCellGradBC('dirichlet')

        params = Richards.Empirical.HaverkampParams().celia1990
        params['Ks'] = np.log(params['Ks'])
        E = Richards.Empirical.Haverkamp(mesh, **params)
        E.kModel.KsMap = Maps.ExpMap(nP=mesh.nC)

        bc = np.array([-61.5, -20.7])
        h = np.zeros(mesh.nC) + bc[0]

        prob = Richards.RichardsProblem(
            mesh, mapping=E, tolRootFinder=1e-6, debug=False,
            boundaryConditions=bc, initialConditions=h,
            doNewton=False, method='mixed'
        )
        prob.timeSteps = [(40, 3), (60, 3)]
        prob.Solver = Solver

        locs = np.r_[5., 10, 15]
        times = prob.times[3:5]
        rxSat = Richards.RichardsRx(locs, times, 'saturation')
        rxPre = Richards.RichardsRx(locs, times, 'pressureHead')
        survey = Richards.RichardsSurvey([rxSat, rxPre])

        prob.pair(survey)

        self.h0 = h
        self.mesh = mesh
        self.Ks = params['Ks'] * np.ones(self.mesh.nC)
        self.prob = prob
        self.survey = survey

    def test_Richards_getResidual_Newton(self):
        self.prob.doNewton = True
        m = self.Ks
        passed = checkDerivative(
            lambda hn1: self.prob.getResidual(
                m,
                self.h0,
                hn1,
                self.prob.timeSteps[0],
                self.prob.boundaryConditions
            ),
            self.h0,
            plotIt=False
        )
        self.assertTrue(passed, True)

    def test_Richards_getResidual_Picard(self):
        self.prob.doNewton = False
        m = self.Ks

        passed = checkDerivative(
            lambda hn1:
            self.prob.getResidual(
                m,
                self.h0,
                hn1,
                self.prob.timeSteps[0],
                self.prob.boundaryConditions
            ),
            self.h0,
            plotIt=False,
            expectedOrder=1
        )

        self.assertTrue(passed, True)

    def test_Adjoint(self):
        v = np.random.rand(self.survey.nD)
        z = np.random.rand(self.mesh.nC)
        Hs = self.prob.fields(self.Ks)
        vJz = v.dot(self.prob.Jvec(self.Ks, z, f=Hs))
        zJv = z.dot(self.prob.Jtvec(self.Ks, v, f=Hs))
        tol = TOL*(10**int(np.log10(np.abs(zJv))))
        passed = np.abs(vJz - zJv) < tol
        print('Richards Adjoint Test - PressureHead')
        print('{0:4.4e} === {1:4.4e}, diff={2:4.4e} < {3:4e}'.format(
            vJz, zJv, np.abs(vJz - zJv), tol)
        )
        self.assertTrue(passed, True)

    def test_Sensitivity(self):
        mTrue = self.Ks*np.ones(self.mesh.nC)
        print('Testing Richards Derivative')
        passed = checkDerivative(
            lambda m: [self.survey.dpred(m), lambda v: self.prob.Jvec(m, v)],
            mTrue,
            num=4,
            plotIt=False
        )
        self.assertTrue(passed, True)

    def test_Sensitivity_full(self):
        print('Testing Richards Derivative FULL')
        mTrue = self.Ks*np.ones(self.mesh.nC)
        J = self.prob.Jfull(mTrue)
        passed = checkDerivative(
            lambda m: [self.survey.dpred(m), J],
            mTrue,
            num=4,
            plotIt=False
        )
        self.assertTrue(passed, True)


class RichardsTests2D(unittest.TestCase):

    def setUp(self):
        mesh = Mesh.TensorMesh([np.ones(8), np.ones(30)])

        mesh.setCellGradBC(['neumann', 'dirichlet'])

        params = Richards.Empirical.HaverkampParams().celia1990
        params['Ks'] = np.log(params['Ks'])
        E = Richards.Empirical.Haverkamp(mesh, **params)
        E.kModel.KsMap = Maps.ExpMap(nP=mesh.nC)

        bc = np.array([-61.5, -20.7])
        bc = np.r_[
            np.zeros(mesh.nCy*2),
            np.ones(mesh.nCx)*bc[0],
            np.ones(mesh.nCx)*bc[1]
        ]
        h = np.zeros(mesh.nC) + bc[0]

        prob = Richards.RichardsProblem(
            mesh,
            mapping=E,
            timeSteps=[(40, 3), (60, 3)],
            Solver=Solver,
            boundaryConditions=bc,
            initialConditions=h,
            doNewton=False,
            method='mixed',
            tolRootFinder=1e-6,
            debug=False
        )

        # kModel = Richards.Empirical.Haverkamp_k(mesh)
        # thetaModel = Richards.Empirical.Haverkamp_theta(mesh)
        # kModel.KsMap = Maps.ExpMap(nP=mesh.nC)

        # prob = Richards.RichardsProblem(
        #     mesh,
        #     kModel=kModel,
        #     thetaModel=thetaModel,
        #     timeSteps=[(40, 3), (60, 3)],
        #     Solver=Solver,
        #     boundaryConditions=bc,
        #     initialConditions=h,
        #     doNewton=False,
        #     method='mixed',
        #     tolRootFinder=1e-6,
        #     debug=False
        # )

        locs = Utils.ndgrid(np.array([5, 7.]), np.array([5, 15, 25.]))
        times = prob.times[3:5]
        rxSat = Richards.RichardsRx(locs, times, 'saturation')
        rxPre = Richards.RichardsRx(locs, times, 'pressureHead')
        survey = Richards.RichardsSurvey([rxSat, rxPre])

        prob.pair(survey)

        self.h0 = h
        self.mesh = mesh
        self.Ks = params['Ks'] * np.ones(mesh.nC)
        self.prob = prob
        self.survey = survey

    def test_Richards_getResidual_Newton(self):
        self.prob.doNewton = True
        m = self.Ks
        passed = checkDerivative(
            lambda hn1: self.prob.getResidual(
                m,
                self.h0,
                hn1,
                self.prob.timeSteps[0],
                self.prob.boundaryConditions
            ),
            self.h0,
            plotIt=False
        )
        self.assertTrue(passed, True)

    def test_Richards_getResidual_Picard(self):
        self.prob.doNewton = False
        m = self.Ks
        passed = checkDerivative(
            lambda hn1: self.prob.getResidual(
                m,
                self.h0,
                hn1,
                self.prob.timeSteps[0],
                self.prob.boundaryConditions
            ),
            self.h0,
            plotIt=False,
            expectedOrder=1
        )
        self.assertTrue(passed, True)

    def test_Adjoint(self):
        v = np.random.rand(self.survey.nD)
        z = np.random.rand(self.mesh.nC)
        Hs = self.prob.fields(self.Ks)
        vJz = v.dot(self.prob.Jvec(self.Ks, z, f=Hs))
        zJv = z.dot(self.prob.Jtvec(self.Ks, v, f=Hs))
        tol = TOL*(10**int(np.log10(np.abs(zJv))))
        passed = np.abs(vJz - zJv) < tol
        print('2D: Richards Adjoint Test - PressureHead')
        print('{0:4.4e} === {1:4.4e}, diff={2:4.4e} < {3:4e}'.format(
            vJz, zJv, np.abs(vJz - zJv), tol)
        )
        self.assertTrue(passed, True)

    def test_Sensitivity(self):
        print('2D: Testing Richards Derivative')
        mTrue = self.Ks*np.ones(self.mesh.nC)
        passed = checkDerivative(
            lambda m: [self.survey.dpred(m), lambda v: self.prob.Jvec(m, v)],
            mTrue,
            num=3,
            plotIt=False
        )
        self.assertTrue(passed, True)

    def test_Sensitivity_full(self):
        mTrue = self.Ks*np.ones(self.mesh.nC)
        J = self.prob.Jfull(mTrue)
        print('2D: Testing Richards Derivative FULL')
        passed = checkDerivative(
            lambda m: [self.survey.dpred(m), J],
            mTrue,
            num=4,
            plotIt=False
        )
        self.assertTrue(passed, True)


class RichardsTests3D(unittest.TestCase):

    def setUp(self):
        mesh = Mesh.TensorMesh([np.ones(8), np.ones(20), np.ones(10)])

        mesh.setCellGradBC(['neumann', 'neumann', 'dirichlet'])

        params = Richards.Empirical.HaverkampParams().celia1990
        params['Ks'] = np.log(params['Ks'])
        E = Richards.Empirical.Haverkamp(mesh, **params)
        E.kModel.KsMap = Maps.ExpMap(nP=mesh.nC)

        bc = np.array([-61.5, -20.7])
        bc = np.r_[
            np.zeros(mesh.nCy*mesh.nCz*2),
            np.zeros(mesh.nCx*mesh.nCz*2),
            np.ones(mesh.nCx*mesh.nCy)*bc[0],
            np.ones(mesh.nCx*mesh.nCy)*bc[1]
        ]
        h = np.zeros(mesh.nC) + bc[0]
        prob = Richards.RichardsProblem(
            mesh,
            mapping=E,
            timeSteps=[(40, 3), (60, 3)],
            Solver=Solver,
            boundaryConditions=bc,
            initialConditions=h,
            doNewton=False,
            method='mixed',
            tolRootFinder=1e-6,
            debug=False
        )

        locs = Utils.ndgrid(np.r_[5, 7.], np.r_[5, 15.], np.r_[6, 8.])
        times = prob.times[3:5]
        rxSat = Richards.RichardsRx(locs, times, 'saturation')
        rxPre = Richards.RichardsRx(locs, times, 'pressureHead')
        survey = Richards.RichardsSurvey([rxSat, rxPre])

        prob.pair(survey)

        self.h0 = h
        self.mesh = mesh
        self.Ks = params['Ks'] * np.ones(mesh.nC)
        self.prob = prob
        self.survey = survey

    def test_Richards_getResidual_Newton(self):
        self.prob.doNewton = True
        m = self.Ks
        passed = checkDerivative(
            lambda hn1: self.prob.getResidual(
                m,
                self.h0,
                hn1,
                self.prob.timeSteps[0],
                self.prob.boundaryConditions
            ),
            self.h0,
            plotIt=False
        )
        self.assertTrue(passed, True)

    def test_Richards_getResidual_Picard(self):
        self.prob.doNewton = False
        m = self.Ks
        passed = checkDerivative(
            lambda hn1: self.prob.getResidual(
                m,
                self.h0,
                hn1,
                self.prob.timeSteps[0],
                self.prob.boundaryConditions
            ),
            self.h0,
            plotIt=False,
            expectedOrder=1
        )
        self.assertTrue(passed, True)

    def test_Adjoint(self):
        v = np.random.rand(self.survey.nD)
        z = np.random.rand(self.mesh.nC)
        Hs = self.prob.fields(self.Ks)
        vJz = v.dot(self.prob.Jvec(self.Ks, z, f=Hs))
        zJv = z.dot(self.prob.Jtvec(self.Ks, v, f=Hs))
        tol = TOL*(10**int(np.log10(np.abs(zJv))))
        passed = np.abs(vJz - zJv) < tol
        print('3D: Richards Adjoint Test - PressureHead')
        print('{0:4.4e} === {1:4.4e}, diff={2:4.4e} < {3:4e}'.format(
            vJz, zJv, np.abs(vJz - zJv), tol)
        )
        self.assertTrue(passed, True)

    def test_Sensitivity(self):
        mTrue = self.Ks*np.ones(self.mesh.nC)
        print('3D: Testing Richards Derivative')
        passed = checkDerivative(
            lambda m: [self.survey.dpred(m), lambda v: self.prob.Jvec(m, v)],
            mTrue,
            num=4,
            plotIt=False
        )
        self.assertTrue(passed, True)

    # def test_Sensitivity_full(self):
    #     mTrue = self.Ks*np.ones(self.mesh.nC)
    #     J = self.prob.Jfull(mTrue)
    #     derChk = lambda m: [self.survey.dpred(m), J]
    #     print('3D: Testing Richards Derivative FULL')
    #     passed = checkDerivative(derChk, mTrue, num=4, plotIt=False)
    #     self.assertTrue(passed,True)


if __name__ == '__main__':
    unittest.main()
