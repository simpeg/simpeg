import unittest

# import SimPEG.dask as simpeg
from SimPEG import maps, utils, data, tests
import discretize
from discretize.utils import mkvc, refine_tree_xyz
from SimPEG.electromagnetics import natural_source as ns
import numpy as np
from pymatsolver import Pardiso as Solver
from discretize.utils import volume_average

TOLr = 5e-2
TOL = 1e-4
FLR = 1e-20


class ComplexResistivityTest(unittest.TestCase):
    def setUp(self):

        csx = 2000.0
        csz = 2000.0

        mesh = discretize.TensorMesh(
            [
                [(csx, 1, -3), (csx, 6), (csx, 1, 3)],
                [(csx, 1, -3), (csx, 6), (csx, 1, 3)],
                [(csz, 1, -3), (csz, 6), (csz, 1, 3)],
            ],
            x0="CCC",
        )

        active = mesh.gridCC[:, 2] < 50

        # create background conductivity model
        sigma_back = 1e-2
        sigma_background = np.zeros(mesh.nC) * sigma_back
        sigma_background[~active] = 1e-8

        # create a model to test with
        block = [csx * np.r_[-3, 3], csx * np.r_[-3, 3], csz * np.r_[-6, -1]]

        block_sigma = 3e-1

        block_inds = (
            (mesh.gridCC[:, 0] >= block[0].min())
            & (mesh.gridCC[:, 0] <= block[0].max())
            & (mesh.gridCC[:, 1] >= block[1].min())
            & (mesh.gridCC[:, 1] <= block[1].max())
            & (mesh.gridCC[:, 2] >= block[2].min())
            & (mesh.gridCC[:, 2] <= block[2].max())
        )

        m = sigma_background.copy()
        m[block_inds] = block_sigma
        m = np.log(m[active])

        self.mesh = mesh
        self.sigma_background = sigma_background
        self.model = m
        self.active = active

    def create_simulation(self, rx_type="apparent_resistivity", rx_orientation="xy"):

        rx_x, rx_y = np.meshgrid(
            np.linspace(-5000, 5000, 10), np.linspace(-5000, 5000, 10)
        )
        rx_loc = np.hstack(
            (mkvc(rx_x, 2), mkvc(rx_y, 2), np.zeros((np.prod(rx_x.shape), 1)))
        )
        rx_loc[:, 2] = -50

        # Make a receiver list
        rxList = [ns.Rx.PointNaturalSource(rx_loc, rx_orientation, rx_type)]

        # Source list
        freqs = [10, 50, 200]
        srcList = [ns.Src.PlanewaveXYPrimary(rxList, freq) for freq in freqs]

        # Survey MT
        survey_ns = ns.Survey(srcList)

        # Set the mapping
        actMap = maps.InjectActiveCells(
            mesh=self.mesh, indActive=self.active, valInactive=np.log(1e-8)
        )
        mapping = maps.ExpMap(self.mesh) * actMap
        # print(survey_ns.source_list)
        # # Setup the problem object
        sim = ns.simulation.Simulation3DPrimarySecondary(
            self.mesh,
            survey=survey_ns,
            sigmaPrimary=self.sigma_background,
            sigmaMap=mapping,
            solver=Solver,
        )
        return sim

    def create_simulation_rx(self, rx_type="apparent_resistivity", rx_orientation="xy"):

        rx_x, rx_y = np.meshgrid(
            np.linspace(-5000, 5000, 10), np.linspace(-5000, 5000, 10)
        )
        rx_loc = np.hstack(
            (mkvc(rx_x, 2), mkvc(rx_y, 2), np.zeros((np.prod(rx_x.shape), 1)))
        )
        rx_loc[:, 2] = -50

        # Make a receiver list
        rxList = [
            ns.Rx.PointNaturalSource(
                orientation=rx_orientation,
                component=rx_type,
                locations_e=rx_loc,
                locations_h=rx_loc,
            )
        ]

        # Source list
        freqs = [10, 50, 200]
        srcList = [ns.Src.PlanewaveXYPrimary(rxList, freq) for freq in freqs]

        # Survey MT
        survey_ns = ns.Survey(srcList)

        # Set the mapping
        actMap = maps.InjectActiveCells(
            mesh=self.mesh, indActive=self.active, valInactive=np.log(1e-8)
        )
        mapping = maps.ExpMap(self.mesh) * actMap
        # print(survey_ns.source_list)
        # # Setup the problem object
        sim = ns.simulation.Simulation3DPrimarySecondary(
            self.mesh,
            survey=survey_ns,
            sigmaPrimary=self.sigma_background,
            sigmaMap=mapping,
            solver=Solver,
        )
        return sim

    def create_simulation_1dprimary_assign_mesh1d(
        self, rx_type="apparent_resistivity", rx_orientation="xy"
    ):

        rx_x, rx_y = np.meshgrid(
            np.linspace(-5000, 5000, 10), np.linspace(-5000, 5000, 10)
        )
        rx_loc = np.hstack(
            (mkvc(rx_x, 2), mkvc(rx_y, 2), np.zeros((np.prod(rx_x.shape), 1)))
        )
        rx_loc[:, 2] = -50

        # Make a receiver list
        rxList = [ns.Rx.PointNaturalSource(rx_loc, rx_orientation, rx_type)]

        # give background a value
        x0 = self.mesh.x0
        hs = [
            [self.mesh.vectorNx[-1] - x0[0]],
            [self.mesh.vectorNy[-1] - x0[1]],
            self.mesh.h[-1],
        ]
        mesh1d = discretize.TensorMesh(hs, x0=x0)
        sigma1d = np.exp(
            volume_average(self.mesh, mesh1d, np.log(self.sigma_background))
        )

        # Source list
        freqs = [10, 50, 200]
        srcList = [
            ns.Src.PlanewaveXYPrimary(rxList, freq, sigma_primary=sigma1d)
            for freq in freqs
        ]

        # Survey MT
        survey_ns = ns.Survey(srcList)

        # Set the mapping
        actMap = maps.InjectActiveCells(
            mesh=self.mesh, indActive=self.active, valInactive=np.log(1e-8)
        )
        mapping = maps.ExpMap(self.mesh) * actMap
        # print(survey_ns.source_list)
        # # Setup the problem object
        sim = ns.simulation.Simulation3DPrimarySecondary(
            self.mesh,
            survey=survey_ns,
            sigmaMap=mapping,
            solver=Solver,
        )
        return sim

    def create_simulation_1dprimary_assign(
        self, rx_type="apparent_resistivity", rx_orientation="xy"
    ):

        rx_x, rx_y = np.meshgrid(
            np.linspace(-5000, 5000, 10), np.linspace(-5000, 5000, 10)
        )
        rx_loc = np.hstack(
            (mkvc(rx_x, 2), mkvc(rx_y, 2), np.zeros((np.prod(rx_x.shape), 1)))
        )
        rx_loc[:, 2] = -50

        # Make a receiver list
        rxList = [ns.Rx.PointNaturalSource(rx_loc, rx_orientation, rx_type)]

        # Source list
        freqs = [10, 50, 200]
        srcList = [
            ns.Src.PlanewaveXYPrimary(rxList, freq, sigma_primary=self.sigma_background)
            for freq in freqs
        ]

        # Survey MT
        survey_ns = ns.Survey(srcList)

        # Set the mapping
        actMap = maps.InjectActiveCells(
            mesh=self.mesh, indActive=self.active, valInactive=np.log(1e-8)
        )
        mapping = maps.ExpMap(self.mesh) * actMap
        # print(survey_ns.source_list)
        # # Setup the problem object
        sim = ns.simulation.Simulation3DPrimarySecondary(
            self.mesh,
            survey=survey_ns,
            sigmaMap=mapping,
            solver=Solver,
        )
        return sim

    def check_deriv(self, sim):
        def fun(x):
            return sim.dpred(x), lambda x: sim.Jvec(self.model, x)

        passed = tests.checkDerivative(fun, self.model, num=3, plotIt=False)
        self.assertTrue(passed)

    def check_adjoint(self, sim):
        w = np.random.rand(len(self.model))
        v = np.random.rand(sim.survey.nD)
        f = sim.fields(self.model)

        vJw = v.ravel().dot(sim.Jvec(self.model, w, f))
        wJtv = w.ravel().dot(sim.Jtvec(self.model, v, f))
        tol = np.max([TOL * (10 ** int(np.log10(np.abs(vJw)))), FLR])
        passed = np.abs(vJw - wJtv) < tol

        print("\nvJw   wJtv  vJw - wJtv     tol    abs(vJw - wJtv) < tol")
        print(f"{vJw:1.2e}, {wJtv:1.2e}, {vJw - wJtv:1.2e} < {tol:1.2e}?, {passed}")

        self.assertTrue(passed)

    def check_deriv_adjoint(self, component, orientation):
        print(f"\n\n============= Testing {component} {orientation} =============\n")
        sim = self.create_simulation(component, orientation)
        sim2 = self.create_simulation_1dprimary_assign(component, orientation)
        sim3 = self.create_simulation_1dprimary_assign_mesh1d(component, orientation)
        sim4 = self.create_simulation_rx(component, orientation)
        self.check_deriv(sim)
        self.check_adjoint(sim)
        self.check_deriv(sim2)
        self.check_adjoint(sim2)
        self.check_deriv(sim3)
        self.check_adjoint(sim3)
        self.check_deriv(sim4)
        self.check_adjoint(sim4)
        print(f"... done")

    def test_apparent_resistivity_xx(self):
        self.check_deriv_adjoint("apparent_resistivity", "xx")

    def test_apparent_resistivity_xy(self):
        self.check_deriv_adjoint("apparent_resistivity", "xy")

    def test_apparent_resistivity_yx(self):
        self.check_deriv_adjoint("apparent_resistivity", "yx")

    def test_apparent_resistivity_yy(self):
        self.check_deriv_adjoint("apparent_resistivity", "yy")

    def test_phase_xx(self):
        self.check_deriv_adjoint("phase", "xx")

    def test_phase_xy(self):
        self.check_deriv_adjoint("phase", "xy")

    def test_phase_yx(self):
        self.check_deriv_adjoint("phase", "yx")

    def test_phase_yy(self):
        self.check_deriv_adjoint("phase", "yy")


if __name__ == "__main__":
    unittest.main()
