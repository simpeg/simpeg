import discretize
from SimPEG import maps, utils, tests
from SimPEG.electromagnetics import frequency_domain as fdem
import numpy as np
from scipy.constants import mu_0

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
    muMod = 1 + MuMax * np.random.randn(mesh.nC)
    sigmaMod = np.random.randn(mesh.nC)

    return mesh, muMod, sigmaMod


def setupProblem(
    mesh,
    muMod,
    sigmaMod,
    prbtype="ElectricField",
    invertMui=False,
    sigmaInInversion=False,
    freq=1.0,
):
    rxcomp = ["real", "imag"]

    loc = utils.ndgrid([mesh.vectorCCx, np.r_[0.0], mesh.vectorCCz])

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

    if sigmaInInversion:

        wires = maps.Wires(("mu", mesh.nC), ("sigma", mesh.nC))

        muMap = maps.MuRelative(mesh) * wires.mu
        sigmaMap = maps.ExpMap(mesh) * wires.sigma

        if invertMui:
            muiMap = maps.ReciprocalMap(mesh) * muMap
            prob = getattr(fdem, "Simulation3D{}".format(prbtype))(
                mesh, muiMap=muiMap, sigmaMap=sigmaMap
            )
            # m0 = np.hstack([1./muMod, sigmaMod])
        else:
            prob = getattr(fdem, "Simulation3D{}".format(prbtype))(
                mesh, muMap=muMap, sigmaMap=sigmaMap
            )
        m0 = np.hstack([muMod, sigmaMod])

    else:
        muMap = maps.MuRelative(mesh)

        if invertMui:
            muiMap = maps.ReciprocalMap(mesh) * muMap
            prob = getattr(fdem, "Simulation3D{}".format(prbtype))(
                mesh, sigma=sigmaMod, muiMap=muiMap
            )
            # m0 = 1./muMod
        else:
            prob = getattr(fdem, "Simulation3D{}".format(prbtype))(
                mesh, sigma=sigmaMod, muMap=muMap
            )
        m0 = muMod

    prob.survey = survey

    return m0, prob, survey


class MuTests(unittest.TestCase):
    def setUpProb(
        self, prbtype="ElectricField", sigmaInInversion=False, invertMui=False
    ):
        self.mesh, muMod, sigmaMod = setupMeshModel()
        self.m0, self.simulation, self.survey = setupProblem(
            self.mesh,
            muMod,
            sigmaMod,
            prbtype=prbtype,
            sigmaInInversion=sigmaInInversion,
            invertMui=invertMui,
        )

    def test_mats_cleared(self):
        self.setUpProb()
        u = self.simulation.fields(self.m0)

        MeMu = self.simulation.MeMu
        MeMuI = self.simulation.MeMuI
        MfMui = self.simulation.MfMui
        MfMuiI = self.simulation.MfMuiI
        MeMuDeriv = self.simulation.MeMuDeriv(u[:, "e"])
        MfMuiDeriv = self.simulation.MfMuiDeriv(u[:, "b"])
        MfMuiDeriv_zero = self.simulation.MfMuiDeriv(utils.Zero())
        MfMuiIDeriv_zero = self.simulation.MfMuiIDeriv(utils.Zero())
        MeMuDeriv_zero = self.simulation.MeMuDeriv(utils.Zero())

        m1 = np.random.rand(self.mesh.nC)
        self.simulation.model = m1

        self.assertTrue(getattr(self, "_MeMu", None) is None)
        self.assertTrue(getattr(self, "_MeMuI", None) is None)
        self.assertTrue(getattr(self, "_MfMui", None) is None)
        self.assertTrue(getattr(self, "_MfMuiI", None) is None)
        self.assertTrue(getattr(self, "_MfMuiDeriv", None) is None)
        self.assertTrue(getattr(self, "_MeMuDeriv", None) is None)
        self.assertTrue(isinstance(MfMuiDeriv_zero, utils.Zero))
        self.assertTrue(isinstance(MfMuiIDeriv_zero, utils.Zero))
        self.assertTrue(isinstance(MeMuDeriv_zero, utils.Zero))

    def JvecTest(
        self, prbtype="ElectricField", sigmaInInversion=False, invertMui=False
    ):
        self.setUpProb(prbtype, sigmaInInversion, invertMui)
        print("Testing Jvec {}".format(prbtype))

        np.random.seed(3321)
        mod = self.m0

        def fun(x):
            return (
                self.simulation.dpred(x),
                lambda x: self.simulation.Jvec(mod, x),
            )

        dx = np.random.rand(*mod.shape) * (mod.max() - mod.min()) * 0.01

        return tests.check_derivative(fun, mod, dx=dx, num=3, plotIt=False)

    def JtvecTest(
        self, prbtype="ElectricField", sigmaInInversion=False, invertMui=False
    ):
        self.setUpProb(prbtype, sigmaInInversion, invertMui)
        print("Testing Jvec {}".format(prbtype))

        np.random.seed(31345)
        u = np.random.rand(self.simulation.muMap.nP)
        v = np.random.rand(self.survey.nD)

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
        self.assertTrue(self.JvecTest("ElectricField", sigmaInInversion=False))

    def test_Jvec_b(self):
        self.assertTrue(self.JvecTest("MagneticFluxDensity", sigmaInInversion=False))

    def test_Jvec_j(self):
        self.assertTrue(self.JvecTest("CurrentDensity", sigmaInInversion=False))

    def test_Jvec_h(self):
        self.assertTrue(self.JvecTest("MagneticField", sigmaInInversion=False))

    def test_Jtvec_e(self):
        self.assertTrue(self.JtvecTest("ElectricField", sigmaInInversion=False))

    def test_Jtvec_b(self):
        self.assertTrue(self.JtvecTest("MagneticFluxDensity", sigmaInInversion=False))

    def test_Jtvec_j(self):
        self.assertTrue(self.JtvecTest("CurrentDensity", sigmaInInversion=False))

    def test_Jtvec_h(self):
        self.assertTrue(self.JtvecTest("MagneticField", sigmaInInversion=False))

    def test_Jvec_musig_e(self):
        self.assertTrue(self.JvecTest("ElectricField", sigmaInInversion=True))

    def test_Jvec_musig_b(self):
        self.assertTrue(self.JvecTest("MagneticFluxDensity", sigmaInInversion=True))

    def test_Jvec_musig_j(self):
        self.assertTrue(self.JvecTest("CurrentDensity", sigmaInInversion=True))

    def test_Jvec_musig_h(self):
        self.assertTrue(self.JvecTest("MagneticField", sigmaInInversion=True))

    def test_Jtvec_musig_e(self):
        self.assertTrue(self.JtvecTest("ElectricField", sigmaInInversion=True))

    def test_Jtvec_musig_b(self):
        self.assertTrue(self.JtvecTest("MagneticFluxDensity", sigmaInInversion=True))

    def test_Jtvec_musig_j(self):
        self.assertTrue(self.JtvecTest("CurrentDensity", sigmaInInversion=True))

    def test_Jtvec_musig_h(self):
        self.assertTrue(self.JtvecTest("MagneticField", sigmaInInversion=True))

    def test_Jvec_e_mui(self):
        self.assertTrue(
            self.JvecTest("ElectricField", sigmaInInversion=False, invertMui=True)
        )

    def test_Jvec_b_mui(self):
        self.assertTrue(
            self.JvecTest("MagneticFluxDensity", sigmaInInversion=False, invertMui=True)
        )

    def test_Jvec_j_mui(self):
        self.assertTrue(
            self.JvecTest("CurrentDensity", sigmaInInversion=False, invertMui=True)
        )

    def test_Jvec_h_mui(self):
        self.assertTrue(
            self.JvecTest("MagneticField", sigmaInInversion=False, invertMui=True)
        )

    def test_Jtvec_e_mui(self):
        self.assertTrue(
            self.JtvecTest("ElectricField", sigmaInInversion=False, invertMui=True)
        )

    def test_Jtvec_b_mui(self):
        self.assertTrue(
            self.JtvecTest(
                "MagneticFluxDensity", sigmaInInversion=False, invertMui=True
            )
        )

    def test_Jtvec_j_mui(self):
        self.assertTrue(
            self.JtvecTest("CurrentDensity", sigmaInInversion=False, invertMui=True)
        )

    def test_Jtvec_h_mui(self):
        self.assertTrue(
            self.JtvecTest("MagneticField", sigmaInInversion=False, invertMui=True)
        )

    def test_Jvec_musig_e_mui(self):
        self.assertTrue(
            self.JvecTest("ElectricField", sigmaInInversion=True, invertMui=True)
        )

    def test_Jvec_musig_b_mui(self):
        self.assertTrue(
            self.JvecTest("MagneticFluxDensity", sigmaInInversion=True, invertMui=True)
        )

    def test_Jvec_musig_j_mui(self):
        self.assertTrue(
            self.JvecTest("CurrentDensity", sigmaInInversion=True, invertMui=True)
        )

    def test_Jvec_musig_h_mui(self):
        self.assertTrue(
            self.JvecTest("MagneticField", sigmaInInversion=True, invertMui=True)
        )

    def test_Jtvec_musig_e_mui(self):
        self.assertTrue(
            self.JtvecTest("ElectricField", sigmaInInversion=True, invertMui=True)
        )

    def test_Jtvec_musig_b_mui(self):
        self.assertTrue(
            self.JtvecTest("MagneticFluxDensity", sigmaInInversion=True, invertMui=True)
        )

    def test_Jtvec_musig_j_mui(self):
        self.assertTrue(
            self.JtvecTest("CurrentDensity", sigmaInInversion=True, invertMui=True)
        )

    def test_Jtvec_musig_h_mui(self):
        self.assertTrue(
            self.JtvecTest("MagneticField", sigmaInInversion=True, invertMui=True)
        )


if __name__ == "__main__":
    unittest.main()
