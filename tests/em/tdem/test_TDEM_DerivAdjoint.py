import unittest
import numpy as np
import time
import discretize
from simpeg import maps, tests
from simpeg.electromagnetics import time_domain as tdem

testDeriv = True
testAdjoint = True

TOL = 1e-4


def get_mesh():
    cs = 10.0
    ncx = 4
    ncy = 4
    ncz = 4
    npad = 2
    # hx = [(cs, ncx), (cs, npad, 1.3)]
    # hz = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
    return discretize.TensorMesh(
        [
            [(cs, npad, -1.5), (cs, ncx), (cs, npad, 1.5)],
            [(cs, npad, -1.5), (cs, ncy), (cs, npad, 1.5)],
            [(cs, npad, -1.5), (cs, ncz), (cs, npad, 1.5)],
        ],
        "CCC",
    )


def get_sigma_mapping(mesh):
    # H AND J FORMULATIONS UNSTABLE WITHOUT SURJECT VERTICAL 1D
    active = mesh.cell_centers_z < 0.0
    activeMap = maps.InjectActiveCells(
        mesh, active, np.log(1e-8), nC=mesh.shape_cells[2]
    )
    return maps.ExpMap(mesh) * maps.SurjectVertical1D(mesh) * activeMap


def get_wire_mappings(mesh):
    # active cells, faces + edges
    active_cells = mesh.cell_centers[:, -1] < 0.0
    active_faces = mesh.faces[:, -1] < 0.0
    active_edges = mesh.edges[:, -1] < 0.0
    n_active_cells = np.sum(active_cells)
    n_active_faces = np.sum(active_faces)
    n_active_edges = np.sum(active_edges)

    # wire map
    wire_map = maps.Wires(
        ("log_sigma", n_active_cells),
        ("log_tau", n_active_faces),
        ("log_kappa", n_active_edges),
    )

    sigma_map = (
        maps.InjectActiveCells(mesh, active_cells, 1e-8)
        * maps.ExpMap(nP=n_active_cells)
        * wire_map.log_sigma
    )
    tau_map = (
        maps.InjectActiveFaces(mesh, active_faces, 0)
        * maps.ExpMap(nP=n_active_faces)
        * wire_map.log_tau
    )
    kappa_map = (
        maps.InjectActiveEdges(mesh, active_edges, 0)
        * maps.ExpMap(nP=n_active_edges)
        * wire_map.log_kappa
    )

    return sigma_map, tau_map, kappa_map


def get_prob(mesh, formulation, sigma_map, **kwargs):
    prb = getattr(tdem, "Simulation3D{}".format(formulation))(
        mesh, sigmaMap=sigma_map, **kwargs
    )
    prb.time_steps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]
    return prb


def get_hierarchical_prob(
    mesh, formulation, sigma_map=None, tau_map=None, kappa_map=None, **kwargs
):
    prb = getattr(tdem, "Simulation3D{}".format(formulation))(
        mesh, sigmaMap=sigma_map, tauMap=tau_map, kappaMap=kappa_map, **kwargs
    )
    prb.time_steps = [(1e-05, 10), (5e-05, 10), (2.5e-4, 10)]
    return prb


def get_survey():
    src1 = tdem.Src.MagDipole([], location=np.array([0.0, 0.0, 0.0]))
    src2 = tdem.Src.MagDipole([], location=np.array([0.0, 0.0, 8.0]))
    return tdem.Survey([src1, src2])


# ====== TEST Jvec ========== #


class Base_DerivAdjoint_Test(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # create a prob where we will store the fields
        mesh = get_mesh()
        self.survey = get_survey()

        if "Hierarchical" in self.formulation:
            active_cells = mesh.cell_centers[:, -1] < 0.0
            active_faces = mesh.faces[:, -1] < 0.0
            active_edges = mesh.edges[:, -1] < 0.0

            sigma_map, tau_map, kappa_map = get_wire_mappings(mesh)
            self.prob = get_hierarchical_prob(
                mesh,
                self.formulation,
                sigma_map=sigma_map,
                tau_map=tau_map,
                kappa_map=kappa_map,
                survey=self.survey,
            )
            self.m = np.r_[
                np.log(1e-1) * np.ones(np.sum(active_cells))
                + 1e-3 * np.random.randn(np.sum(active_cells)),
                np.log(10 * 1e-1) * np.ones(np.sum(active_faces))
                + 1e-3 * np.random.randn(np.sum(active_faces)),
                np.log(100 * 1e-1) * np.ones(np.sum(active_edges))
                + 1e-3 * np.random.randn(np.sum(active_edges)),
            ]

        else:
            sigma_map = get_sigma_mapping(mesh)
            self.prob = get_prob(mesh, self.formulation, sigma_map, survey=self.survey)
            rng = np.random.default_rng(seed=42)
            self.m = np.log(1e-1) * np.ones(self.prob.sigmaMap.nP) + 1e-3 * rng.normal(
                size=self.prob.sigmaMap.nP
            )
        print("Solving Fields for problem {}".format(self.formulation))
        t = time.time()
        self.fields = self.prob.fields(self.m)
        print("... done. Time: {}\n".format(time.time() - t))

        # create a prob where will be re-computing fields at each jvec
        # iteration
        mesh = get_mesh()
        self.surveyfwd = get_survey()
        if "Hierarchical" in self.formulation:
            sigma_map, tau_map, kappa_map = get_wire_mappings(mesh)
            self.probfwd = get_hierarchical_prob(
                mesh,
                self.formulation,
                sigma_map=sigma_map,
                tau_map=tau_map,
                kappa_map=kappa_map,
                survey=self.surveyfwd,
            )
        else:
            sigma_map = get_sigma_mapping(mesh)
            self.probfwd = get_prob(
                mesh, self.formulation, sigma_map, survey=self.surveyfwd
            )

    def get_rx(self, rxcomp):
        rxOffset = 15.0
        rxlocs = np.array([[rxOffset, 0.0, -1e-2]])
        rxtimes = np.logspace(-4, -3, 20)
        return getattr(tdem.Rx, "Point{}".format(rxcomp[:-1]))(
            locations=rxlocs, times=rxtimes, orientation=rxcomp[-1]
        )

    def set_receiver_list(self, rxcomp):
        # append the right rxlist to the surveys
        rx = [self.get_rx(rxcomp)]
        rxfwd = [self.get_rx(rxcomp)]

        # append to srclists
        for source_list, rxlist in zip(
            [self.survey.source_list, self.surveyfwd.source_list], [rx, rxfwd]
        ):
            for src in source_list:
                src.receiver_list = rxlist

    def JvecTest(self, rxcomp):
        self.set_receiver_list(rxcomp)

        def derChk(m):
            return [
                self.probfwd.dpred(m),
                lambda mx: self.prob.Jvec(self.m, mx, f=self.fields),
            ]

        print(
            "test_Jvec_{prbtype}_{rxcomp}".format(
                prbtype=self.formulation, rxcomp=rxcomp
            )
        )
        np.random.seed(10)  # use seed for check_derivative
        tests.check_derivative(derChk, self.m, plotIt=False, num=2, eps=1e-20)

    def JvecVsJtvecTest(self, rxcomp):
        self.set_receiver_list(rxcomp)
        print(
            "\nAdjoint Testing Jvec, Jtvec prob {}, {}".format(self.formulation, rxcomp)
        )

        rng = np.random.default_rng(seed=42)
        m = rng.uniform(size=self.prob.sigmaMap.nP)
        d = rng.normal(size=self.prob.survey.nD)
        V1 = d.dot(self.prob.Jvec(self.m, m, f=self.fields))
        V2 = m.dot(self.prob.Jtvec(self.m, d, f=self.fields))
        tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.0
        passed = np.abs(V1 - V2) < tol

        print(f"{self.formulation} {V1} {V2} {passed}")
        self.assertTrue(passed)


class TDEM_Fields_B_Pieces(Base_DerivAdjoint_Test):
    formulation = "MagneticFluxDensity"

    def test_eDeriv_m_adjoint(self):
        prb = self.prob
        f = self.fields

        print("\n Testing eDeriv_m Adjoint")

        rng = np.random.default_rng(seed=42)
        m = rng.uniform(size=len(self.m))
        e = rng.normal(size=prb.mesh.nE)
        V1 = e.dot(f._eDeriv_m(1, prb.survey.source_list[0], m))
        V2 = m.dot(f._eDeriv_m(1, prb.survey.source_list[0], e, adjoint=True))
        tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.0
        passed = np.abs(V1 - V2) < tol

        print("    ", V1, V2, np.abs(V1 - V2), tol, passed)
        self.assertTrue(passed)

    def test_eDeriv_u_adjoint(self):
        print("\n Testing eDeriv_u Adjoint")

        prb = self.prob
        f = self.fields

        rng = np.random.default_rng(seed=42)
        b = rng.uniform(size=prb.mesh.nF)
        e = rng.normal(size=prb.mesh.nE)
        V1 = e.dot(f._eDeriv_u(1, prb.survey.source_list[0], b))
        V2 = b.dot(f._eDeriv_u(1, prb.survey.source_list[0], e, adjoint=True))
        tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.0
        passed = np.abs(V1 - V2) < tol

        print("    ", V1, V2, np.abs(V1 - V2), tol, passed)
        self.assertTrue(passed)


class DerivAdjoint_E(Base_DerivAdjoint_Test):
    formulation = "ElectricField"

    if testDeriv:

        def test_Jvec_e_dbxdt(self):
            self.JvecTest("MagneticFluxTimeDerivativex")

        def test_Jvec_e_dbzdt(self):
            self.JvecTest("MagneticFluxTimeDerivativez")

        def test_Jvec_e_ey(self):
            self.JvecTest("ElectricFieldy")

        def test_Jvec_e_dhxdt(self):
            self.JvecTest("MagneticFieldTimeDerivativex")

        def test_Jvec_e_dhzdt(self):
            self.JvecTest("MagneticFieldTimeDerivativez")

        def test_Jvec_e_jy(self):
            self.JvecTest("CurrentDensityy")

    if testAdjoint:

        def test_Jvec_adjoint_e_dbdtx(self):
            self.JvecVsJtvecTest("MagneticFluxTimeDerivativex")

        def test_Jvec_adjoint_e_dbdtz(self):
            self.JvecVsJtvecTest("MagneticFluxTimeDerivativez")

        def test_Jvec_adjoint_e_ey(self):
            self.JvecVsJtvecTest("ElectricFieldy")

        def test_Jvec_adjoint_e_dhdtx(self):
            self.JvecVsJtvecTest("MagneticFieldTimeDerivativex")

        def test_Jvec_adjoint_e_dhdtz(self):
            self.JvecVsJtvecTest("MagneticFieldTimeDerivativez")

        def test_Jvec_adjoint_e_jy(self):
            self.JvecVsJtvecTest("CurrentDensityy")

        pass


class DerivAdjoint_HierarchicalE(Base_DerivAdjoint_Test):
    formulation = "HierarchicalElectricField"

    if testDeriv:

        def test_Jvec_e_dbxdt(self):
            self.JvecTest("MagneticFluxTimeDerivativex")

        def test_Jvec_e_dbzdt(self):
            self.JvecTest("MagneticFluxTimeDerivativez")

        def test_Jvec_e_ey(self):
            self.JvecTest("ElectricFieldy")

        def test_Jvec_e_dhxdt(self):
            self.JvecTest("MagneticFieldTimeDerivativex")

        def test_Jvec_e_dhzdt(self):
            self.JvecTest("MagneticFieldTimeDerivativez")

        def test_Jvec_e_jy(self):
            self.JvecTest("CurrentDensityy")

    if testAdjoint:

        def test_Jvec_adjoint_e_dbdtx(self):
            self.JvecVsJtvecTest("MagneticFluxTimeDerivativex")

        def test_Jvec_adjoint_e_dbdtz(self):
            self.JvecVsJtvecTest("MagneticFluxTimeDerivativez")

        def test_Jvec_adjoint_e_ey(self):
            self.JvecVsJtvecTest("ElectricFieldy")

        def test_Jvec_adjoint_e_dhdtx(self):
            self.JvecVsJtvecTest("MagneticFieldTimeDerivativex")

        def test_Jvec_adjoint_e_dhdtz(self):
            self.JvecVsJtvecTest("MagneticFieldTimeDerivativez")

        def test_Jvec_adjoint_e_jy(self):
            self.JvecVsJtvecTest("CurrentDensityy")


class DerivAdjoint_B(Base_DerivAdjoint_Test):
    formulation = "MagneticFluxDensity"

    if testDeriv:

        def test_Jvec_b_bx(self):
            self.JvecTest("MagneticFluxDensityx")

        def test_Jvec_b_bz(self):
            self.JvecTest("MagneticFluxDensityz")

        def test_Jvec_b_dbdtx(self):
            self.JvecTest("MagneticFluxTimeDerivativex")

        def test_Jvec_b_dbdtz(self):
            self.JvecTest("MagneticFluxTimeDerivativez")

        def test_Jvec_b_hx(self):
            self.JvecTest("MagneticFieldx")

        def test_Jvec_b_hz(self):
            self.JvecTest("MagneticFieldz")

        def test_Jvec_b_dhdtx(self):
            self.JvecTest("MagneticFieldTimeDerivativex")

        def test_Jvec_b_dhdtz(self):
            self.JvecTest("MagneticFieldTimeDerivativez")

        def test_Jvec_b_jy(self):
            self.JvecTest("CurrentDensityy")

    if testAdjoint:

        def test_Jvec_adjoint_b_bx(self):
            self.JvecVsJtvecTest("MagneticFluxDensityx")

        def test_Jvec_adjoint_b_bz(self):
            self.JvecVsJtvecTest("MagneticFluxDensityz")

        def test_Jvec_adjoint_b_dbdtx(self):
            self.JvecVsJtvecTest("MagneticFluxTimeDerivativex")

        def test_Jvec_adjoint_b_dbdtz(self):
            self.JvecVsJtvecTest("MagneticFluxTimeDerivativez")

        def test_Jvec_adjoint_b_ey(self):
            self.JvecVsJtvecTest("ElectricFieldy")

        def test_Jvec_adjoint_b_hx(self):
            self.JvecVsJtvecTest("MagneticFieldx")

        def test_Jvec_adjoint_b_hz(self):
            self.JvecVsJtvecTest("MagneticFieldz")

        def test_Jvec_adjoint_b_dhdtx(self):
            self.JvecVsJtvecTest("MagneticFieldTimeDerivativex")

        def test_Jvec_adjoint_b_dhdtz(self):
            self.JvecVsJtvecTest("MagneticFieldTimeDerivativez")

        def test_Jvec_adjoint_b_jy(self):
            self.JvecVsJtvecTest("CurrentDensityy")


class DerivAdjoint_HierarchicalB(Base_DerivAdjoint_Test):
    formulation = "HierarchicalMagneticFluxDensity"

    if testDeriv:

        def test_Jvec_b_bx(self):
            self.JvecTest("MagneticFluxDensityx")

        def test_Jvec_b_bz(self):
            self.JvecTest("MagneticFluxDensityz")

        def test_Jvec_b_dbdtx(self):
            self.JvecTest("MagneticFluxTimeDerivativex")

        def test_Jvec_b_dbdtz(self):
            self.JvecTest("MagneticFluxTimeDerivativez")

        def test_Jvec_b_ey(self):
            self.JvecTest("ElectricFieldy")

        def test_Jvec_b_hx(self):
            self.JvecTest("MagneticFieldx")

        def test_Jvec_b_hz(self):
            self.JvecTest("MagneticFieldz")

        def test_Jvec_b_dhdtx(self):
            self.JvecTest("MagneticFieldTimeDerivativex")

        def test_Jvec_b_dhdtz(self):
            self.JvecTest("MagneticFieldTimeDerivativez")

        def test_Jvec_b_jy(self):
            self.JvecTest("CurrentDensityy")

    if testAdjoint:

        def test_Jvec_adjoint_b_bx(self):
            self.JvecVsJtvecTest("MagneticFluxDensityx")

        def test_Jvec_adjoint_b_bz(self):
            self.JvecVsJtvecTest("MagneticFluxDensityz")

        def test_Jvec_adjoint_b_dbdtx(self):
            self.JvecVsJtvecTest("MagneticFluxTimeDerivativex")

        def test_Jvec_adjoint_b_dbdtz(self):
            self.JvecVsJtvecTest("MagneticFluxTimeDerivativez")

        def test_Jvec_adjoint_b_ey(self):
            self.JvecVsJtvecTest("ElectricFieldy")

        def test_Jvec_adjoint_b_hx(self):
            self.JvecVsJtvecTest("MagneticFieldx")

        def test_Jvec_adjoint_b_hz(self):
            self.JvecVsJtvecTest("MagneticFieldz")

        def test_Jvec_adjoint_b_dhdtx(self):
            self.JvecVsJtvecTest("MagneticFieldTimeDerivativex")

        def test_Jvec_adjoint_b_dhdtz(self):
            self.JvecVsJtvecTest("MagneticFieldTimeDerivativez")

        def test_Jvec_adjoint_b_jy(self):
            self.JvecVsJtvecTest("CurrentDensityy")


class DerivAdjoint_H(Base_DerivAdjoint_Test):
    formulation = "MagneticField"

    if testDeriv:

        def test_Jvec_h_hx(self):
            self.JvecTest("MagneticFieldx")

        def test_Jvec_h_hz(self):
            self.JvecTest("MagneticFieldz")

        def test_Jvec_h_dhdtx(self):
            self.JvecTest("MagneticFieldTimeDerivativex")

        def test_Jvec_h_dhdtz(self):
            self.JvecTest("MagneticFieldTimeDerivativez")

        def test_Jvec_h_jy(self):
            self.JvecTest("CurrentDensityy")

        def test_Jvec_h_bx(self):
            self.JvecTest("MagneticFluxDensityx")

        def test_Jvec_h_bz(self):
            self.JvecTest("MagneticFluxDensityz")

        def test_Jvec_h_dbdtx(self):
            self.JvecTest("MagneticFluxTimeDerivativex")

        def test_Jvec_h_dbdtz(self):
            self.JvecTest("MagneticFluxTimeDerivativez")

        def test_Jvec_h_ey(self):
            self.JvecTest("ElectricFieldy")

    if testAdjoint:

        def test_Jvec_adjoint_h_hx(self):
            self.JvecVsJtvecTest("MagneticFieldx")

        def test_Jvec_adjoint_h_hz(self):
            self.JvecVsJtvecTest("MagneticFieldz")

        def test_Jvec_adjoint_h_dhdtx(self):
            self.JvecVsJtvecTest("MagneticFieldTimeDerivativex")

        def test_Jvec_adjoint_h_dhdtz(self):
            self.JvecVsJtvecTest("MagneticFieldTimeDerivativez")

        def test_Jvec_adjoint_h_jy(self):
            self.JvecVsJtvecTest("CurrentDensityy")

        def test_Jvec_adjoint_h_bx(self):
            self.JvecVsJtvecTest("MagneticFluxDensityx")

        def test_Jvec_adjoint_h_bz(self):
            self.JvecVsJtvecTest("MagneticFluxDensityz")

        def test_Jvec_adjoint_h_dbdtx(self):
            self.JvecVsJtvecTest("MagneticFluxTimeDerivativex")

        def test_Jvec_adjoint_h_dbdtz(self):
            self.JvecVsJtvecTest("MagneticFluxTimeDerivativez")

        def test_Jvec_adjoint_h_ey(self):
            self.JvecVsJtvecTest("ElectricFieldy")


class DerivAdjoint_J(Base_DerivAdjoint_Test):
    formulation = "CurrentDensity"

    if testDeriv:

        def test_Jvec_j_jy(self):
            self.JvecTest("CurrentDensityy")

        def test_Jvec_j_dhdtx(self):
            self.JvecTest("MagneticFieldTimeDerivativex")

        def test_Jvec_j_dhdtz(self):
            self.JvecTest("MagneticFieldTimeDerivativez")

        def test_Jvec_j_ey(self):
            self.JvecTest("ElectricFieldy")

        def test_Jvec_j_dbdtx(self):
            self.JvecTest("MagneticFluxTimeDerivativex")

        def test_Jvec_j_dbdtz(self):
            self.JvecTest("MagneticFluxTimeDerivativez")

    if testAdjoint:

        def test_Jvec_adjoint_j_jy(self):
            self.JvecVsJtvecTest("CurrentDensityy")

        def test_Jvec_adjoint_j_dhdtx(self):
            self.JvecVsJtvecTest("MagneticFieldTimeDerivativex")

        def test_Jvec_adjoint_j_dhdtz(self):
            self.JvecVsJtvecTest("MagneticFieldTimeDerivativez")

        def test_Jvec_adjoint_j_ey(self):
            self.JvecVsJtvecTest("ElectricFieldy")

        def test_Jvec_adjoint_j_dbdtx(self):
            self.JvecVsJtvecTest("MagneticFluxTimeDerivativex")

        def test_Jvec_adjoint_j_dbdtz(self):
            self.JvecVsJtvecTest("MagneticFluxTimeDerivativez")


if __name__ == "__main__":
    unittest.main()
