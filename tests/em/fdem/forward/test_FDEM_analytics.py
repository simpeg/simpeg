from __future__ import print_function

import unittest

import discretize
import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0
from SimPEG import SolverLU, utils
from SimPEG.electromagnetics import analytics
from SimPEG.electromagnetics import frequency_domain as fdem

# import matplotlib
# matplotlib.use('Agg')


plotIt = False
tol_Transect = 2e-1
tol_EBdipole = 1e-2


class FDEM_analyticTests(unittest.TestCase):
    def setUp(self):
        print("\nTesting Transect for analytic")

        cs = 10.0
        ncx, ncy, ncz = 10, 10, 10
        npad = 5
        freq = 1e2

        hx = [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)]
        hy = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
        mesh = discretize.TensorMesh([hx, hy, hz], "CCC")

        x = np.linspace(-10, 10, 5)
        XYZ = utils.ndgrid(x, np.r_[0], np.r_[0])
        receiver_list = fdem.Rx.PointElectricField(
            XYZ, orientation="x", component="imag"
        )
        SrcList = [
            fdem.Src.MagDipole(
                [receiver_list], location=np.r_[0.0, 0.0, 0.0], frequency=freq
            ),
            fdem.Src.CircularLoop(
                [receiver_list],
                location=np.r_[0.0, 0.0, 0.0],
                frequency=freq,
                radius=np.sqrt(1.0 / np.pi),
                # test number of turns and current
                n_turns=2,
                current=0.5,
            ),
        ]

        survey = fdem.Survey(SrcList)

        sig = 1e-1
        sigma = np.ones(mesh.nC) * sig
        sigma[mesh.gridCC[:, 2] > 0] = 1e-8

        prb = fdem.Simulation3DMagneticFluxDensity(mesh, survey=survey, sigma=sigma)

        try:
            from pymatsolver import Pardiso

            prb.solver = Pardiso
        except ImportError:
            prb.solver = SolverLU

        self.prb = prb
        self.mesh = mesh
        self.sig = sig

        print(" starting solve ...")
        u = self.prb.fields()
        print(" ... done")
        self.u = u

    def test_Transect(self, plotIt=plotIt):

        for src in self.prb.survey.source_list:
            print(" --- testing {} --- ".format(src.__class__.__name__))
            x = np.linspace(-55, 55, 12)
            XYZ = utils.ndgrid(x, np.r_[0], np.r_[0])

            P = self.mesh.getInterpolationMat(XYZ, "Fz")

            ana = mu_0 * np.imag(
                analytics.FDEM.hzAnalyticDipoleF(x, src.frequency, self.sig)
            )
            num = P * np.imag(self.u[src, "b"])

            diff = np.linalg.norm(num - ana)

            if plotIt:
                import matplotlib.pyplot as plt

                plt.plot(x, np.log10(np.abs(num)))
                plt.plot(x, np.log10(np.abs(ana)), "r")
                plt.plot(x, diff, "g")
                plt.show()

            norm_num = np.linalg.norm(num)
            norm_ana = np.linalg.norm(ana)
            tol = tol_Transect * (norm_num + norm_ana) / 2.0

            passed = diff < tol
            print(
                "analytic: {}, numeric {}, difference {} < tolerance {} ? "
                " {}".format(norm_ana, norm_num, diff, tol, passed)
            )

            self.assertTrue(passed)


class TestDipoles(unittest.TestCase):
    def test_CylindricalMeshEBDipoles(self, plotIt=plotIt):
        print(
            "Testing CylindricalMesh Electric and Magnetic Dipoles in a wholespace-"
            " Analytic: J-formulation"
        )
        sigmaback = 1.0
        mur = 2.0
        freq = 1.0
        skdpth = 500.0 / np.sqrt(sigmaback * freq)

        csx, ncx, npadx = 5, 50, 25
        csz, ncz, npadz = 5, 50, 25

        hx = utils.meshTensor([(csx, ncx), (csx, npadx, 1.3)])
        hz = utils.meshTensor([(csz, npadz, -1.3), (csz, ncz), (csz, npadz, 1.3)])

        # define the cylindrical mesh
        mesh = discretize.CylindricalMesh([hx, 1, hz], [0.0, 0.0, -hz.sum() / 2])

        if plotIt:
            mesh.plotGrid()

        # make sure mesh is big enough
        self.assertTrue(mesh.hz.sum() > skdpth * 2.0)
        self.assertTrue(mesh.hx.sum() > skdpth * 2.0)

        # set up source
        # test electric dipole
        src_loc = np.r_[0.0, 0.0, 0.0]
        s_ind = utils.closest_points_index(mesh, src_loc, "Fz") + mesh.nFx

        de = np.zeros(mesh.nF, dtype=complex)
        de[s_ind] = 1.0 / csz
        de_p = [fdem.Src.RawVec_e([], freq, de / mesh.area)]

        dm_p = [fdem.Src.MagDipole([], freq, src_loc)]

        # Pair the problem and survey
        surveye = fdem.Survey(de_p)
        surveym = fdem.Survey(dm_p)

        prbe = fdem.Simulation3DMagneticField(
            mesh, survey=surveye, sigma=sigmaback, mu=mur * mu_0
        )
        prbm = fdem.Simulation3DElectricField(
            mesh, survey=surveym, sigma=sigmaback, mu=mur * mu_0
        )

        # solve
        fieldsBackE = prbe.fields()
        fieldsBackM = prbm.fields()

        rlim = [20.0, 500.0]
        # lookAtTx = de_p
        r = mesh.vectorCCx[
            np.argmin(np.abs(mesh.vectorCCx - rlim[0])) : np.argmin(
                np.abs(mesh.vectorCCx - rlim[1])
            )
        ]
        z = 100.0

        # where we choose to measure
        XYZ = utils.ndgrid(r, np.r_[0.0], np.r_[z])

        Pf = mesh.getInterpolationMat(XYZ, "CC")
        Zero = sp.csr_matrix(Pf.shape)
        Pfx, Pfz = sp.hstack([Pf, Zero]), sp.hstack([Zero, Pf])

        jn = fieldsBackE[de_p, "j"]
        bn = fieldsBackM[dm_p, "b"]

        SigmaBack = sigmaback * np.ones((mesh.nC))
        Rho = utils.sdiag(1.0 / SigmaBack)
        Rho = sp.block_diag([Rho, Rho])

        en = Rho * mesh.aveF2CCV * jn
        bn = mesh.aveF2CCV * bn

        ex, ez = Pfx * en, Pfz * en
        bx, bz = Pfx * bn, Pfz * bn

        # get analytic solution
        exa, eya, eza = analytics.FDEM.ElectricDipoleWholeSpace(
            XYZ, src_loc, sigmaback, freq, "Z", mu_r=mur
        )

        exa = utils.mkvc(exa, 2)
        eya = utils.mkvc(eya, 2)
        eza = utils.mkvc(eza, 2)

        bxa, bya, bza = analytics.FDEM.MagneticDipoleWholeSpace(
            XYZ, src_loc, sigmaback, freq, "Z", mu_r=mur
        )
        bxa = utils.mkvc(bxa, 2)
        bya = utils.mkvc(bya, 2)
        bza = utils.mkvc(bza, 2)

        print(
            " comp,       anayltic,       numeric,       num - ana,       (num - ana)/ana"
        )
        print(
            "  ex:",
            np.linalg.norm(exa),
            np.linalg.norm(ex),
            np.linalg.norm(exa - ex),
            np.linalg.norm(exa - ex) / np.linalg.norm(exa),
        )
        print(
            "  ez:",
            np.linalg.norm(eza),
            np.linalg.norm(ez),
            np.linalg.norm(eza - ez),
            np.linalg.norm(eza - ez) / np.linalg.norm(eza),
        )

        print(
            "  bx:",
            np.linalg.norm(bxa),
            np.linalg.norm(bx),
            np.linalg.norm(bxa - bx),
            np.linalg.norm(bxa - bx) / np.linalg.norm(bxa),
        )
        print(
            "  bz:",
            np.linalg.norm(bza),
            np.linalg.norm(bz),
            np.linalg.norm(bza - bz),
            np.linalg.norm(bza - bz) / np.linalg.norm(bza),
        )

        if plotIt is True:
            # Edipole
            plt.subplot(221)
            plt.plot(r, ex.real, "o", r, exa.real, linewidth=2)
            plt.grid(which="both")
            plt.title("Ex Real")
            plt.xlabel("r (m)")

            plt.subplot(222)
            plt.plot(r, ex.imag, "o", r, exa.imag, linewidth=2)
            plt.grid(which="both")
            plt.title("Ex Imag")
            plt.legend(["Num", "Ana"], bbox_to_anchor=(1.5, 0.5))
            plt.xlabel("r (m)")

            plt.subplot(223)
            plt.plot(r, ez.real, "o", r, eza.real, linewidth=2)
            plt.grid(which="both")
            plt.title("Ez Real")
            plt.xlabel("r (m)")

            plt.subplot(224)
            plt.plot(r, ez.imag, "o", r, eza.imag, linewidth=2)
            plt.grid(which="both")
            plt.title("Ez Imag")
            plt.xlabel("r (m)")

            plt.tight_layout()

            # Bdipole
            plt.subplot(221)
            plt.plot(r, bx.real, "o", r, bxa.real, linewidth=2)
            plt.grid(which="both")
            plt.title("Bx Real")
            plt.xlabel("r (m)")

            plt.subplot(222)
            plt.plot(r, bx.imag, "o", r, bxa.imag, linewidth=2)
            plt.grid(which="both")
            plt.title("Bx Imag")
            plt.legend(["Num", "Ana"], bbox_to_anchor=(1.5, 0.5))
            plt.xlabel("r (m)")

            plt.subplot(223)
            plt.plot(r, bz.real, "o", r, bza.real, linewidth=2)
            plt.grid(which="both")
            plt.title("Bz Real")
            plt.xlabel("r (m)")

            plt.subplot(224)
            plt.plot(r, bz.imag, "o", r, bza.imag, linewidth=2)
            plt.grid(which="both")
            plt.title("Bz Imag")
            plt.xlabel("r (m)")

            plt.tight_layout()

        self.assertTrue(np.linalg.norm(exa - ex) / np.linalg.norm(exa) < tol_EBdipole)
        self.assertTrue(np.linalg.norm(eza - ez) / np.linalg.norm(eza) < tol_EBdipole)

        self.assertTrue(np.linalg.norm(bxa - bx) / np.linalg.norm(bxa) < tol_EBdipole)
        self.assertTrue(np.linalg.norm(bza - bz) / np.linalg.norm(bza) < tol_EBdipole)