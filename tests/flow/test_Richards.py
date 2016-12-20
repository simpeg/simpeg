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


class BaseRichardsTest(unittest.TestCase):

    def _dotest_getResidual(self, newton):
        print('Testing richards get residual newton={}, dim={}'.format(
            newton,
            self.mesh.dim
        ))
        self.prob.do_newton = newton
        passed = checkDerivative(
            lambda hn1: self.prob.getResidual(
                self.mtrue,
                self.h0,
                hn1,
                self.prob.timeSteps[0],
                self.prob.boundary_conditions
            ),
            self.h0,
            expectedOrder=2 if newton else 1,
            plotIt=False
        )
        self.assertTrue(passed, True)

    def _dotest_adjoint(self):
        v = np.random.rand(self.survey.nD)
        z = np.random.rand(len(self.mtrue))
        Hs = self.prob.fields(self.mtrue)
        vJz = v.dot(self.prob.Jvec(self.mtrue, z, f=Hs))
        zJv = z.dot(self.prob.Jtvec(self.mtrue, v, f=Hs))
        tol = TOL*(10**int(np.log10(np.abs(zJv))))
        passed = np.abs(vJz - zJv) < tol
        print('Richards Adjoint Test - PressureHead dim={}'.format(
            self.mesh.dim
        ))
        print('{0:4.4e} === {1:4.4e}, diff={2:4.4e} < {3:4e}'.format(
            vJz, zJv, np.abs(vJz - zJv), tol)
        )
        self.assertTrue(passed, True)

    def _dotest_sensitivity(self):
        print('Testing Richards Derivative dim={}'.format(
            self.mesh.dim
        ))
        passed = checkDerivative(
            lambda m: [self.survey.dpred(m), lambda v: self.prob.Jvec(m, v)],
            self.mtrue,
            num=4,
            plotIt=False
        )
        self.assertTrue(passed, True)

    def _dotest_sensitivity_full(self):
        print('Testing Richards Derivative FULL dim={}'.format(
            self.mesh.dim
        ))
        J = self.prob.Jfull(self.mtrue)
        passed = checkDerivative(
            lambda m: [self.survey.dpred(m), J],
            self.mtrue,
            num=4,
            plotIt=False
        )
        self.assertTrue(passed, True)


class RichardsTests1D(BaseRichardsTest):

    def setUp(self):
        mesh = Mesh.TensorMesh([np.ones(20)])
        mesh.setCellGradBC('dirichlet')

        params = Richards.Empirical.HaverkampParams().celia1990
        k_fun, theta_fun = Richards.Empirical.haverkamp(mesh, **params)
        k_fun.KsMap = Maps.ExpMap(nP=mesh.nC)

        bc = np.array([-61.5, -20.7])
        h = np.zeros(mesh.nC) + bc[0]

        prob = Richards.RichardsProblem(
            mesh,
            hydraulic_conductivity=k_fun,
            water_retention=theta_fun,
            root_finder_tol=1e-6, debug=False,
            boundary_conditions=bc, initial_conditions=h,
            do_newton=False, method='mixed'
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
        self.mtrue = np.log(self.Ks)
        self.prob = prob
        self.survey = survey

    def test_Richards_getResidual_Newton(self):
        self._dotest_getResidual(True)

    def test_Richards_getResidual_Picard(self):
        self._dotest_getResidual(False)

    def test_adjoint(self):
        self._dotest_adjoint()

    def test_sensitivity(self):
        self._dotest_sensitivity()

    def test_sensitivity_full(self):
        self._dotest_sensitivity_full()


class RichardsTests1D_Multi(BaseRichardsTest):

    def setUp(self):
        mesh = Mesh.TensorMesh([np.ones(20)])
        mesh.setCellGradBC('dirichlet')

        params = Richards.Empirical.HaverkampParams().celia1990
        k_fun, theta_fun = Richards.Empirical.haverkamp(mesh, **params)
        wires = Maps.Wires(('Ks', mesh.nC), ('A', mesh.nC))

        k_fun.KsMap = Maps.ExpMap(nP=mesh.nC) * wires.Ks
        k_fun.AMap = wires.A

        bc = np.array([-61.5, -20.7])
        h = np.zeros(mesh.nC) + bc[0]

        prob = Richards.RichardsProblem(
            mesh,
            hydraulic_conductivity=k_fun,
            water_retention=theta_fun,
            root_finder_tol=1e-6, debug=False,
            boundary_conditions=bc, initial_conditions=h,
            do_newton=False, method='mixed'
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
        self.A = params['A'] * np.ones(self.mesh.nC)
        self.mtrue = np.r_[np.log(self.Ks), self.A]
        self.prob = prob
        self.survey = survey

    def test_adjoint(self):
        self._dotest_adjoint()

    def test_sensitivity(self):
        self._dotest_sensitivity()

    def test_sensitivity_full(self):
        self._dotest_sensitivity_full()


class RichardsTests2D(BaseRichardsTest):

    def setUp(self):
        mesh = Mesh.TensorMesh([np.ones(8), np.ones(30)])

        mesh.setCellGradBC(['neumann', 'dirichlet'])

        params = Richards.Empirical.HaverkampParams().celia1990
        k_fun, theta_fun = Richards.Empirical.haverkamp(mesh, **params)
        k_fun.KsMap = Maps.ExpMap(nP=mesh.nC)

        bc = np.array([-61.5, -20.7])
        bc = np.r_[
            np.zeros(mesh.nCy*2),
            np.ones(mesh.nCx)*bc[0],
            np.ones(mesh.nCx)*bc[1]
        ]
        h = np.zeros(mesh.nC) + bc[0]

        prob = Richards.RichardsProblem(
            mesh,
            hydraulic_conductivity=k_fun,
            water_retention=theta_fun,
            timeSteps=[(40, 3), (60, 3)],
            Solver=Solver,
            boundary_conditions=bc,
            initial_conditions=h,
            do_newton=False,
            method='mixed',
            root_finder_tol=1e-6,
            debug=False
        )

        locs = Utils.ndgrid(np.array([5, 7.]), np.array([5, 15, 25.]))
        times = prob.times[3:5]
        rxSat = Richards.RichardsRx(locs, times, 'saturation')
        rxPre = Richards.RichardsRx(locs, times, 'pressureHead')
        survey = Richards.RichardsSurvey([rxSat, rxPre])

        prob.pair(survey)

        self.h0 = h
        self.mesh = mesh
        self.Ks = params['Ks'] * np.ones(mesh.nC)
        self.mtrue = np.log(self.Ks)
        self.prob = prob
        self.survey = survey

    def test_Richards_getResidual_Newton(self):
        self._dotest_getResidual(True)

    def test_Richards_getResidual_Picard(self):
        self._dotest_getResidual(False)

    def test_adjoint(self):
        self._dotest_adjoint()

    def test_sensitivity(self):
        self._dotest_sensitivity()

    def test_sensitivity_full(self):
        self._dotest_sensitivity_full()


class RichardsTests3D(BaseRichardsTest):

    def setUp(self):
        mesh = Mesh.TensorMesh([np.ones(8), np.ones(20), np.ones(10)])

        mesh.setCellGradBC(['neumann', 'neumann', 'dirichlet'])

        params = Richards.Empirical.HaverkampParams().celia1990
        k_fun, theta_fun = Richards.Empirical.haverkamp(mesh, **params)
        k_fun.KsMap = Maps.ExpMap(nP=mesh.nC)

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
            hydraulic_conductivity=k_fun,
            water_retention=theta_fun,
            timeSteps=[(40, 3), (60, 3)],
            Solver=Solver,
            boundary_conditions=bc,
            initial_conditions=h,
            do_newton=False,
            method='mixed',
            root_finder_tol=1e-6,
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
        self.mtrue = np.log(self.Ks)
        self.prob = prob
        self.survey = survey

    def test_Richards_getResidual_Newton(self):
        self._dotest_getResidual(True)

    def test_Richards_getResidual_Picard(self):
        self._dotest_getResidual(False)

    def test_adjoint(self):
        self._dotest_adjoint()

    def test_sensitivity(self):
        self._dotest_sensitivity()

    # def test_sensitivity_full(self):
    #     self._dotest_sensitivity_full()


if __name__ == '__main__':
    unittest.main()
