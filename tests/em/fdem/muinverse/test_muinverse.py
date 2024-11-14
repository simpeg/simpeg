import discretize
from simpeg import maps, utils, tests
from simpeg.electromagnetics import frequency_domain as fdem
import numpy as np

import unittest

MuMax = 50.0
TOL = 1e-8
EPS = 1e-10


def setupMeshModel():
    cs = 10.0
    nc = 20.0
    npad = 15.0
    hx = [(cs, nc), (cs, npad, 1.3)]
    hz = [(cs, npad, -1.3), (cs, nc), (cs, npad, 1.3)]

    mesh = discretize.CylindricalMesh([hx, 1.0, hz], "0CC")
    rng = np.random.default_rng(seed=2016)
    muMod = 1 + MuMax * rng.normal(size=mesh.nC)
    conductivityMod = rng.normal(size=mesh.nC)

    return mesh, muMod, conductivityMod


def setupProblem(
    mesh,
    muMod,
    conductivityMod,
    prbtype="ElectricField",
    invertMui=False,
    conductivityInInversion=False,
    freq=1.0,
):
    rxcomp = ["real", "imag"]

    loc = utils.ndgrid([mesh.cell_centers_x, np.r_[0.0], mesh.cell_centers_z])

    if prbtype in ["ElectricField", "MagneticFluxDensity"]:
        rxfields_y = ["ElectricField", "CurrentDensity"]
        rxfields_xz = ["MagneticFluxDensity", "MagneticField"]

    elif prbtype in ["MagneticField", "CurrentDensity"]:
        rxfields_y = ["MagneticFluxDensity", "MagneticField"]
        rxfields_xz = ["ElectricField", "CurrentDensity"]

    receiver_list_edge = [
        getattr(fdem.receivers, "Point{f}".format(f=f))(
            loc, component=comp, orientation=orient
        )
        for f in rxfields_y
        for comp in rxcomp
        for orient in ["y"]
    ]

    receiver_list_face = [
        getattr(fdem.receivers, "Point{f}".format(f=f))(
            loc, component=comp, orientation=orient
        )
        for f in rxfields_xz
        for comp in rxcomp
        for orient in ["x", "z"]
    ]

    receiver_list = receiver_list_edge + receiver_list_face

    src_loc = np.r_[0.0, 0.0, 0.0]

    if prbtype in ["ElectricField", "MagneticFluxDensity"]:
        src = fdem.sources.MagDipole(
            receiver_list=receiver_list, location=src_loc, frequency=freq
        )

    elif prbtype in ["MagneticField", "CurrentDensity"]:
        ind = utils.closest_points_index(mesh, src_loc, "Fz") + mesh.vnF[0]
        vec = np.zeros(mesh.nF)
        vec[ind] = 1.0

        src = fdem.sources.RawVec_e(
            receiver_list=receiver_list, frequency=freq, s_e=vec
        )

    survey = fdem.Survey([src])

    if conductivityInInversion:
        wires = maps.Wires(("mu", mesh.nC), ("conductivity", mesh.nC))

        muMap = maps.MuRelative(mesh) * wires.mu
        conductivity_map = maps.ExpMap(mesh) * wires.conductivity

        if invertMui:
            muiMap = maps.ReciprocalMap(mesh) * muMap
            prob = getattr(fdem, "Simulation3D{}".format(prbtype))(
                mesh, muiMap=muiMap, conductivity_map=conductivity_map
            )
            # m0 = np.hstack([1./muMod, conductivityMod])
        else:
            prob = getattr(fdem, "Simulation3D{}".format(prbtype))(
                mesh, muMap=muMap, conductivity_map=conductivity_map
            )
        m0 = np.hstack([muMod, conductivityMod])

    else:
        muMap = maps.MuRelative(mesh)

        if invertMui:
            muiMap = maps.ReciprocalMap(mesh) * muMap
            prob = getattr(fdem, "Simulation3D{}".format(prbtype))(
                mesh, conductivity=conductivityMod, muiMap=muiMap
            )
            # m0 = 1./muMod
        else:
            prob = getattr(fdem, "Simulation3D{}".format(prbtype))(
                mesh, conductivity=conductivityMod, muMap=muMap
            )
        m0 = muMod

    prob.survey = survey

    return m0, prob, survey


class MuTests(unittest.TestCase):
    def setUpProb(
        self, prbtype="ElectricField", conductivityInInversion=False, invertMui=False
    ):
        self.mesh, muMod, conductivityMod = setupMeshModel()
        self.m0, self.simulation, self.survey = setupProblem(
            self.mesh,
            muMod,
            conductivityMod,
            prbtype=prbtype,
            conductivityInInversion=conductivityInInversion,
            invertMui=invertMui,
        )

    def test_mats_cleared(self):
        self.setUpProb()
        u = self.simulation.fields(self.m0)

        self.simulation._Me_permeability
        self.simulation._inv_Me_permeability
        self.simulation._Mf__perm_inv
        self.simulation._inv_Mf__perm_inv
        self.simulation._Me_permeability_deriv(u[:, "e"])
        self.simulation._Mf__perm_inv_deriv(u[:, "b"])
        MfMuiDeriv_zero = self.simulation._Mf__perm_inv_deriv(utils.Zero())
        MfMuiIDeriv_zero = self.simulation._inv_Mf__perm_inv_deriv(utils.Zero())
        MeMuDeriv_zero = self.simulation._Me_permeability_deriv(utils.Zero())

        rng = np.random.default_rng(seed=2016)
        m1 = rng.uniform(size=self.mesh.nC)
        self.simulation.model = m1

        self.assertTrue(getattr(self, "_Me_permeability", None) is None)
        self.assertTrue(getattr(self, "_inv_Me_permeability", None) is None)
        self.assertTrue(getattr(self, "_Mf__perm_inv", None) is None)
        self.assertTrue(getattr(self, "_inv_Mf__perm_inv", None) is None)
        self.assertTrue(getattr(self, "_Mf__perm_inv_deriv", None) is None)
        self.assertTrue(getattr(self, "_Me_permeability_deriv", None) is None)
        self.assertTrue(isinstance(MfMuiDeriv_zero, utils.Zero))
        self.assertTrue(isinstance(MfMuiIDeriv_zero, utils.Zero))
        self.assertTrue(isinstance(MeMuDeriv_zero, utils.Zero))

    def JvecTest(
        self, prbtype="ElectricField", conductivityInInversion=False, invertMui=False
    ):
        self.setUpProb(prbtype, conductivityInInversion, invertMui)
        print("Testing Jvec {}".format(prbtype))

        mod = self.m0

        def fun(x):
            return (
                self.simulation.dpred(x),
                lambda x: self.simulation.Jvec(mod, x),
            )

        rng = np.random.default_rng(seed=3321)
        dx = rng.uniform(size=mod.shape) * (mod.max() - mod.min()) * 0.01

        return tests.check_derivative(
            fun, mod, dx=dx, num=4, plotIt=False, random_seed=55
        )

    def JtvecTest(
        self, prbtype="ElectricField", conductivityInInversion=False, invertMui=False
    ):
        self.setUpProb(prbtype, conductivityInInversion, invertMui)
        print("Testing Jvec {}".format(prbtype))

        rng = np.random.default_rng(seed=3321)
        u = rng.uniform(size=self.simulation.muMap.nP)
        v = rng.uniform(size=self.survey.nD)

        self.simulation.model = self.m0

        V1 = v.dot(self.simulation.Jvec(self.m0, u))
        V2 = u.dot(self.simulation.Jtvec(self.m0, v))
        diff = np.abs(V1 - V2)
        tol = TOL * (np.abs(V1) + np.abs(V2)) / 2.0
        passed = (diff < tol) | (diff < EPS)
        print(
            "AdjointTest {prbtype} {v1} {v2} {diff} {tol} {passed}".format(
                prbtype=prbtype, v1=V1, v2=V2, diff=diff, tol=tol, passed=passed
            )
        )
        return passed

    def test_Jvec_e(self):
        self.assertTrue(self.JvecTest("ElectricField", conductivityInInversion=False))

    def test_Jvec_b(self):
        self.assertTrue(
            self.JvecTest("MagneticFluxDensity", conductivityInInversion=False)
        )

    def test_Jvec_j(self):
        self.assertTrue(self.JvecTest("CurrentDensity", conductivityInInversion=False))

    def test_Jvec_h(self):
        self.assertTrue(self.JvecTest("MagneticField", conductivityInInversion=False))

    def test_Jtvec_e(self):
        self.assertTrue(self.JtvecTest("ElectricField", conductivityInInversion=False))

    def test_Jtvec_b(self):
        self.assertTrue(
            self.JtvecTest("MagneticFluxDensity", conductivityInInversion=False)
        )

    def test_Jtvec_j(self):
        self.assertTrue(self.JtvecTest("CurrentDensity", conductivityInInversion=False))

    def test_Jtvec_h(self):
        self.assertTrue(self.JtvecTest("MagneticField", conductivityInInversion=False))

    def test_Jvec_musig_e(self):
        self.assertTrue(self.JvecTest("ElectricField", conductivityInInversion=True))

    def test_Jvec_musig_b(self):
        self.assertTrue(
            self.JvecTest("MagneticFluxDensity", conductivityInInversion=True)
        )

    def test_Jvec_musig_j(self):
        self.assertTrue(self.JvecTest("CurrentDensity", conductivityInInversion=True))

    def test_Jvec_musig_h(self):
        self.assertTrue(self.JvecTest("MagneticField", conductivityInInversion=True))

    def test_Jtvec_musig_e(self):
        self.assertTrue(self.JtvecTest("ElectricField", conductivityInInversion=True))

    def test_Jtvec_musig_b(self):
        self.assertTrue(
            self.JtvecTest("MagneticFluxDensity", conductivityInInversion=True)
        )

    def test_Jtvec_musig_j(self):
        self.assertTrue(self.JtvecTest("CurrentDensity", conductivityInInversion=True))

    def test_Jtvec_musig_h(self):
        self.assertTrue(self.JtvecTest("MagneticField", conductivityInInversion=True))

    def test_Jvec_e_mui(self):
        self.assertTrue(
            self.JvecTest(
                "ElectricField", conductivityInInversion=False, invertMui=True
            )
        )

    def test_Jvec_b_mui(self):
        self.assertTrue(
            self.JvecTest(
                "MagneticFluxDensity", conductivityInInversion=False, invertMui=True
            )
        )

    def test_Jvec_j_mui(self):
        self.assertTrue(
            self.JvecTest(
                "CurrentDensity", conductivityInInversion=False, invertMui=True
            )
        )

    def test_Jvec_h_mui(self):
        self.assertTrue(
            self.JvecTest(
                "MagneticField", conductivityInInversion=False, invertMui=True
            )
        )

    def test_Jtvec_e_mui(self):
        self.assertTrue(
            self.JtvecTest(
                "ElectricField", conductivityInInversion=False, invertMui=True
            )
        )

    def test_Jtvec_b_mui(self):
        self.assertTrue(
            self.JtvecTest(
                "MagneticFluxDensity", conductivityInInversion=False, invertMui=True
            )
        )

    def test_Jtvec_j_mui(self):
        self.assertTrue(
            self.JtvecTest(
                "CurrentDensity", conductivityInInversion=False, invertMui=True
            )
        )

    def test_Jtvec_h_mui(self):
        self.assertTrue(
            self.JtvecTest(
                "MagneticField", conductivityInInversion=False, invertMui=True
            )
        )

    def test_Jvec_musig_e_mui(self):
        self.assertTrue(
            self.JvecTest("ElectricField", conductivityInInversion=True, invertMui=True)
        )

    def test_Jvec_musig_b_mui(self):
        self.assertTrue(
            self.JvecTest(
                "MagneticFluxDensity", conductivityInInversion=True, invertMui=True
            )
        )

    def test_Jvec_musig_j_mui(self):
        self.assertTrue(
            self.JvecTest(
                "CurrentDensity", conductivityInInversion=True, invertMui=True
            )
        )

    def test_Jvec_musig_h_mui(self):
        self.assertTrue(
            self.JvecTest("MagneticField", conductivityInInversion=True, invertMui=True)
        )

    def test_Jtvec_musig_e_mui(self):
        self.assertTrue(
            self.JtvecTest(
                "ElectricField", conductivityInInversion=True, invertMui=True
            )
        )

    def test_Jtvec_musig_b_mui(self):
        self.assertTrue(
            self.JtvecTest(
                "MagneticFluxDensity", conductivityInInversion=True, invertMui=True
            )
        )

    def test_Jtvec_musig_j_mui(self):
        self.assertTrue(
            self.JtvecTest(
                "CurrentDensity", conductivityInInversion=True, invertMui=True
            )
        )

    def test_Jtvec_musig_h_mui(self):
        self.assertTrue(
            self.JtvecTest(
                "MagneticField", conductivityInInversion=True, invertMui=True
            )
        )


if __name__ == "__main__":
    unittest.main()
