from __future__ import print_function
import unittest
import numpy as np
import scipy.sparse as sp
from SimPEG import Mesh
from SimPEG import Utils
from SimPEG import SolverLU
from SimPEG import EM
from scipy.constants import mu_0

# import matplotlib
# matplotlib.use('Agg')

import matplotlib.pylab as plt

plotIt = False
tol_Transect = 2e-1
tol_EBdipole = 1e-2


class FDEM_analyticTests(unittest.TestCase):

    def setUp(self):
        print('\nTesting Transect for analytic')

        cs = 10.
        ncx, ncy, ncz = 10, 10, 10
        npad = 5
        freq = 1e2

        hx = [(cs, npad, -1.3), (cs, ncx), (cs, npad, 1.3)]
        hy = [(cs, npad, -1.3), (cs, ncy), (cs, npad, 1.3)]
        hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
        mesh = Mesh.TensorMesh([hx, hy, hz], 'CCC')

        x = np.linspace(-10, 10, 5)
        XYZ = Utils.ndgrid(x, np.r_[0], np.r_[0])
        rxList = EM.FDEM.Rx.Point_e(XYZ, orientation='x', component='imag')
        SrcList = [
            EM.FDEM.Src.MagDipole(
                [rxList], loc=np.r_[0., 0., 0.],
                freq=freq
            ),
            EM.FDEM.Src.CircularLoop(
                [rxList], loc=np.r_[0., 0., 0.],
                freq=freq, radius=np.sqrt(1./np.pi)
            ),
            # EM.FDEM.Src.MagDipole_Bfield(
            #     [rxList], loc=np.r_[0., 0., 0.],
            #     freq=freq
            # ), # less accurate
        ]

        survey = EM.FDEM.Survey(SrcList)

        sig = 1e-1
        sigma = np.ones(mesh.nC)*sig
        sigma[mesh.gridCC[:, 2] > 0] = 1e-8

        prb = EM.FDEM.Problem3D_b(mesh, sigma=sigma)
        prb.pair(survey)

        try:
            from pymatsolver import Pardiso
            prb.Solver = Pardiso
        except ImportError:
            prb.Solver = SolverLU

        self.prb = prb
        self.mesh = mesh
        self.sig = sig

        print(' starting solve ...')
        u = self.prb.fields()
        print(' ... done')
        self.u = u

    def test_Transect(self, plotIt=plotIt):

        for src in self.prb.survey.srcList:
            print(' --- testing {} --- '.format(src.__class__.__name__))
            x = np.linspace(-55, 55, 12)
            XYZ = Utils.ndgrid(x, np.r_[0], np.r_[0])

            P = self.mesh.getInterpolationMat(XYZ, 'Fz')

            ana = mu_0*np.imag(
                EM.Analytics.FDEM.hzAnalyticDipoleF(x, src.freq, self.sig)
            )
            num = P*np.imag(self.u[src, 'b'])

            diff = np.linalg.norm(num - ana)

            if plotIt:
                import matplotlib.pyplot as plt
                plt.plot(x, np.log10(np.abs(num)))
                plt.plot(x, np.log10(np.abs(ana)), 'r')
                plt.plot(x, diff, 'g')
                plt.show()

            norm_num = np.linalg.norm(num)
            norm_ana = np.linalg.norm(ana)
            tol = tol_Transect*(norm_num + norm_ana)/2.

            passed = diff < tol
            print('analytic: {}, numeric {}, difference {} < tolerance {} ? '
                   ' {}'.format(norm_ana, norm_num, diff,
                                tol, passed))

            self.assertTrue(passed)


class TestDipoles(unittest.TestCase):

    def test_CylMeshEBDipoles(self, plotIt=plotIt):
        print("Testing CylMesh Electric and Magnetic Dipoles in a wholespace-"
               " Analytic: J-formulation")
        sigmaback = 1.
        mur = 2.
        freq = 1.
        skdpth = 500./np.sqrt(sigmaback*freq)

        csx, ncx, npadx = 5, 50, 25
        csz, ncz, npadz = 5, 50, 25

        hx = Utils.meshTensor(
            [(csx, ncx), (csx, npadx, 1.3)]
        )
        hz = Utils.meshTensor(
            [(csz, npadz, -1.3), (csz, ncz), (csz, npadz, 1.3)]
        )

        # define the cylindrical mesh
        mesh = Mesh.CylMesh([hx, 1, hz], [0., 0., -hz.sum()/2])

        if plotIt:
            mesh.plotGrid()

        # make sure mesh is big enough
        self.assertTrue(mesh.hz.sum() > skdpth*2.)
        self.assertTrue(mesh.hx.sum() > skdpth*2.)

        # set up source
        # test electric dipole
        src_loc = np.r_[0., 0., 0.]
        s_ind = Utils.closestPoints(mesh, src_loc, 'Fz') + mesh.nFx

        de = np.zeros(mesh.nF, dtype=complex)
        de[s_ind] = 1./csz
        de_p = [EM.FDEM.Src.RawVec_e([], freq, de/mesh.area)]

        dm_p = [EM.FDEM.Src.MagDipole([], freq, src_loc)]

        # Pair the problem and survey
        surveye = EM.FDEM.Survey(de_p)
        surveym = EM.FDEM.Survey(dm_p)

        prbe = EM.FDEM.Problem3D_h(mesh, sigma=sigmaback, mu=mur*mu_0)
        prbm = EM.FDEM.Problem3D_e(mesh, sigma=sigmaback, mu=mur*mu_0)

        # pair problem and survey
        prbe.pair(surveye)
        prbm.pair(surveym)

        # solve
        fieldsBackE = prbe.fields()
        fieldsBackM = prbm.fields()

        rlim = [20., 500.]
        # lookAtTx = de_p
        r = mesh.vectorCCx[
            np.argmin(np.abs(mesh.vectorCCx-rlim[0])):
            np.argmin(np.abs(mesh.vectorCCx-rlim[1]))
        ]
        z = 100.

        # where we choose to measure
        XYZ = Utils.ndgrid(r, np.r_[0.], np.r_[z])

        Pf = mesh.getInterpolationMat(XYZ, 'CC')
        Zero = sp.csr_matrix(Pf.shape)
        Pfx, Pfz = sp.hstack([Pf, Zero]), sp.hstack([Zero, Pf])

        jn = fieldsBackE[de_p, 'j']
        bn = fieldsBackM[dm_p, 'b']

        SigmaBack = sigmaback*np.ones((mesh.nC))
        Rho = Utils.sdiag(1./SigmaBack)
        Rho = sp.block_diag([Rho, Rho])

        en = Rho*mesh.aveF2CCV*jn
        bn = mesh.aveF2CCV*bn

        ex, ez = Pfx*en, Pfz*en
        bx, bz = Pfx*bn, Pfz*bn

        # get analytic solution
        exa, eya, eza = EM.Analytics.FDEM.ElectricDipoleWholeSpace(
            XYZ, src_loc, sigmaback, freq, orientation='Z', mu=mur*mu_0
        )

        exa = Utils.mkvc(exa, 2)
        eya = Utils.mkvc(eya, 2)
        eza = Utils.mkvc(eza, 2)

        bxa, bya, bza = EM.Analytics.FDEM.MagneticDipoleWholeSpace(
            XYZ, src_loc, sigmaback, freq, orientation='Z', mu=mur*mu_0
        )
        bxa = Utils.mkvc(bxa, 2)
        bya = Utils.mkvc(bya, 2)
        bza = Utils.mkvc(bza, 2)

        print(' comp,       anayltic,       numeric,       num - ana,       (num - ana)/ana')
        print('  ex:', np.linalg.norm(exa), np.linalg.norm(ex), np.linalg.norm(exa-ex), np.linalg.norm(exa-ex)/np.linalg.norm(exa))
        print('  ez:', np.linalg.norm(eza), np.linalg.norm(ez), np.linalg.norm(eza-ez), np.linalg.norm(eza-ez)/np.linalg.norm(eza))

        print('  bx:', np.linalg.norm(bxa), np.linalg.norm(bx), np.linalg.norm(bxa-bx), np.linalg.norm(bxa-bx)/np.linalg.norm(bxa))
        print('  bz:', np.linalg.norm(bza), np.linalg.norm(bz), np.linalg.norm(bza-bz), np.linalg.norm(bza-bz)/np.linalg.norm(bza))

        if plotIt is True:
            # Edipole
            plt.subplot(221)
            plt.plot(r, ex.real, 'o', r, exa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('Ex Real')
            plt.xlabel('r (m)')

            plt.subplot(222)
            plt.plot(r, ex.imag, 'o', r, exa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('Ex Imag')
            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.xlabel('r (m)')

            plt.subplot(223)
            plt.plot(r, ez.real, 'o', r, eza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('Ez Real')
            plt.xlabel('r (m)')

            plt.subplot(224)
            plt.plot(r, ez.imag, 'o', r, eza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('Ez Imag')
            plt.xlabel('r (m)')

            plt.tight_layout()

            # Bdipole
            plt.subplot(221)
            plt.plot(r, bx.real, 'o', r, bxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('Bx Real')
            plt.xlabel('r (m)')

            plt.subplot(222)
            plt.plot(r, bx.imag, 'o', r, bxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('Bx Imag')
            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.xlabel('r (m)')

            plt.subplot(223)
            plt.plot(r, bz.real, 'o', r, bza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('Bz Real')
            plt.xlabel('r (m)')

            plt.subplot(224)
            plt.plot(r, bz.imag, 'o', r, bza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('Bz Imag')
            plt.xlabel('r (m)')

            plt.tight_layout()

        self.assertTrue(
            np.linalg.norm(exa-ex)/np.linalg.norm(exa) < tol_EBdipole
        )
        self.assertTrue(
            np.linalg.norm(eza-ez)/np.linalg.norm(eza) < tol_EBdipole
        )

        self.assertTrue(
            np.linalg.norm(bxa-bx)/np.linalg.norm(bxa) < tol_EBdipole
        )
        self.assertTrue(
            np.linalg.norm(bza-bz)/np.linalg.norm(bza) < tol_EBdipole
        )


if __name__ == '__main__':
    unittest.main()
