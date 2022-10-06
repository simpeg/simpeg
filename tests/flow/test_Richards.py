from __future__ import print_function
import unittest
import numpy as np

from discretize.tests import check_derivative
import discretize

from SimPEG import maps
from SimPEG import utils
from SimPEG.flow import richards

try:
    from pymatsolver import Pardiso as Solver
except Exception:
    from SimPEG import Solver


TOL = 1e-8

np.random.seed(0)


class BaseRichardsTest(unittest.TestCase):
    def setUp(self):
        mesh = self.get_mesh()
        params = richards.empirical.HaverkampParams().celia1990
        k_fun, theta_fun = richards.empirical.haverkamp(mesh, **params)

        self.setup_maps(mesh, k_fun, theta_fun)
        bc, h = self.get_conditions(mesh)

        time_steps = [(40, 3), (60, 3)]
        time_mesh = discretize.TensorMesh([time_steps])
        rx_list = self.get_rx_list(time_mesh.nodes_x)
        survey = richards.Survey(rx_list)

        prob = richards.SimulationNDCellCentered(
            mesh,
            survey=survey,
            hydraulic_conductivity=k_fun,
            water_retention=theta_fun,
            root_finder_tol=1e-6,
            debug=False,
            boundary_conditions=bc,
            initial_conditions=h,
            do_newton=False,
            method="mixed",
        )
        prob.time_steps = time_steps
        prob.solver = Solver

        self.h0 = h
        self.mesh = mesh
        self.Ks = params["Ks"] * np.ones(self.mesh.nC)
        self.A = params["A"] * np.ones(self.mesh.nC)
        self.theta_s = params["theta_s"] * np.ones(self.mesh.nC)
        self.prob = prob
        self.survey = survey
        self.setup_model()

    def _dotest_getResidual(self, newton):
        print(
            "Testing richards get residual newton={}, dim={}".format(
                newton, self.mesh.dim
            )
        )
        self.prob.do_newton = newton
        passed = check_derivative(
            lambda hn1: self.prob.getResidual(
                self.mtrue,
                self.h0,
                hn1,
                self.prob.time_steps[0],
                self.prob.boundary_conditions,
            ),
            self.h0,
            expectedOrder=2 if newton else 1,
            plotIt=False,
        )
        self.assertTrue(passed, True)

    def _dotest_adjoint(self):
        v = np.random.rand(self.survey.nD)
        z = np.random.rand(len(self.mtrue))
        Hs = self.prob.fields(self.mtrue)
        vJz = v.dot(self.prob.Jvec(self.mtrue, z, f=Hs))
        zJv = z.dot(self.prob.Jtvec(self.mtrue, v, f=Hs))
        tol = TOL * (10 ** int(np.log10(np.abs(zJv))))
        passed = np.abs(vJz - zJv) < tol
        print("Richards Adjoint Test - PressureHead dim={}".format(self.mesh.dim))
        print(
            "{0:4.4e} === {1:4.4e}, diff={2:4.4e} < {3:4e}".format(
                vJz, zJv, np.abs(vJz - zJv), tol
            )
        )
        self.assertTrue(passed, True)

    def _dotest_sensitivity(self):
        print("Testing Richards Derivative dim={}".format(self.mesh.dim))
        passed = check_derivative(
            lambda m: [self.prob.dpred(m), lambda v: self.prob.Jvec(m, v)],
            self.mtrue,
            num=3,
            plotIt=False,
        )
        self.assertTrue(passed, True)

    def _dotest_sensitivity_full(self):
        print("Testing Richards Derivative FULL dim={}".format(self.mesh.dim))
        J = self.prob.Jfull(self.mtrue)
        passed = check_derivative(
            lambda m: [self.prob.dpred(m), J], self.mtrue, num=4, plotIt=False
        )
        self.assertTrue(passed, True)


class RichardsTests1D(BaseRichardsTest):
    def get_mesh(self):
        mesh = discretize.TensorMesh([np.ones(20)])
        mesh.setCellGradBC("dirichlet")
        print(mesh.dim)
        return mesh

    def get_rx_list(self, times):
        locs = np.array([[5.0], [10], [15]])
        rxSat = richards.receivers.Saturation(locations=locs, times=times)
        rxPre = richards.receivers.Pressure(locations=locs, times=times)
        return [rxSat, rxPre]

    def get_conditions(self, mesh):
        bc = np.array([-61.5, -20.7])
        h = np.zeros(mesh.nC) + bc[0]
        return bc, h

    def setup_maps(self, mesh, k_fun, theta_fun):
        k_fun.KsMap = maps.ExpMap(nP=mesh.nC)

    def setup_model(self):
        self.mtrue = np.log(self.Ks)

    def _test_Richards_getResidual_Newton(self):
        self._dotest_getResidual(True)

    def _test_Richards_getResidual_Picard(self):
        self._dotest_getResidual(False)

    def test_adjoint(self):
        self._dotest_adjoint()

    def test_sensitivity(self):
        self._dotest_sensitivity()

    def test_sensitivity_full(self):
        self._dotest_sensitivity_full()


class RichardsTests1D_Saturation(RichardsTests1D):
    def setup_maps(self, mesh, k_fun, theta_fun):
        theta_fun.theta_sMap = maps.IdentityMap(nP=mesh.nC)

    def setup_model(self):
        self.prob.hydraulic_conductivity.Ks = self.Ks
        self.mtrue = self.theta_s

    def get_rx_list(self, times):
        locs = np.array([[5.0], [10], [15]])
        rxSat = richards.receivers.Saturation(locs, times)
        rxPre = richards.receivers.Pressure(locs, times)
        return [rxSat, rxPre]

    def test_adjoint(self):
        self._dotest_adjoint()

    def test_sensitivity(self):
        self._dotest_sensitivity()

    def test_sensitivity_full(self):
        self._dotest_sensitivity_full()


class RichardsTests1D_Multi(RichardsTests1D):
    def setup_maps(self, mesh, k_fun, theta_fun):
        wires = maps.Wires(("Ks", mesh.nC), ("A", mesh.nC), ("theta_s", mesh.nC))
        k_fun.KsMap = maps.ExpMap(nP=mesh.nC) * wires.Ks
        k_fun.AMap = wires.A
        theta_fun.theta_sMap = wires.theta_s

    def setup_model(self):
        self.mtrue = np.r_[np.log(self.Ks), self.A, self.theta_s]

    def test_adjoint(self):
        self._dotest_adjoint()

    def test_sensitivity(self):
        self._dotest_sensitivity()

    def test_sensitivity_full(self):
        self._dotest_sensitivity_full()


class RichardsTests2D(BaseRichardsTest):
    def get_mesh(self):
        mesh = discretize.TensorMesh([np.ones(8), np.ones(30)])
        mesh.setCellGradBC(["neumann", "dirichlet"])
        return mesh

    def get_rx_list(self, times):
        locs = utils.ndgrid(np.array([5, 7.0]), np.array([5, 15, 25.0]))
        rxSat = richards.receivers.Saturation(locs, times)
        rxPre = richards.receivers.Pressure(locs, times)
        return [rxSat, rxPre]

    def get_conditions(self, mesh):
        bc = np.array([-61.5, -20.7])
        bc = np.r_[
            np.zeros(mesh.nCy * 2), np.ones(mesh.nCx) * bc[0], np.ones(mesh.nCx) * bc[1]
        ]
        h = np.zeros(mesh.nC) + bc[0]
        return bc, h

    def setup_maps(self, mesh, k_fun, theta_fun):
        k_fun.KsMap = maps.ExpMap(nP=mesh.nC)

    def setup_model(self):
        self.mtrue = np.log(self.Ks)

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
    def get_mesh(self):
        mesh = discretize.TensorMesh([np.ones(8), np.ones(20), np.ones(10)])
        mesh.setCellGradBC(["neumann", "neumann", "dirichlet"])
        return mesh

    def get_rx_list(self, times):
        locs = utils.ndgrid(np.r_[5, 7.0], np.r_[5, 15.0], np.r_[6, 8.0])
        rxSat = richards.receivers.Saturation(locs, times)
        rxPre = richards.receivers.Pressure(locs, times)
        return [rxSat, rxPre]

    def get_conditions(self, mesh):
        bc = np.array([-61.5, -20.7])
        bc = np.r_[
            np.zeros(mesh.nCy * mesh.nCz * 2),
            np.zeros(mesh.nCx * mesh.nCz * 2),
            np.ones(mesh.nCx * mesh.nCy) * bc[0],
            np.ones(mesh.nCx * mesh.nCy) * bc[1],
        ]
        h = np.zeros(mesh.nC) + bc[0]
        return bc, h

    def setup_maps(self, mesh, k_fun, theta_fun):
        k_fun.KsMap = maps.ExpMap(nP=mesh.nC)

    def setup_model(self):
        self.mtrue = np.log(self.Ks)

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


if __name__ == "__main__":
    unittest.main()
