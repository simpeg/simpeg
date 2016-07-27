import unittest
from SimPEG import EM, Mesh, Utils, np, Maps
# import sys
from scipy.constants import mu_0


plotIt = False
tol_ElecDipole = 1e-2

if plotIt:
    import matplotlib.pylab


class FDEM_analytic_DipoleTests(unittest.TestCase):

    def setUp(self):

        # Define model parameters
        sigmaback = 1.
        # mu = mu_0*(1+kappa)
        kappa = 1.

        # Set source parameters
        freq = 1.
        src_loc = np.r_[0., 0., 0.]

        # Compute skin depth
        skdpth = 500. / np.sqrt(sigmaback * freq)

        # Create cylindrical mesh
        csx, ncx, npadx = 5, 50, 25
        csz, ncz, npadz = 5, 50, 25
        hx = Utils.meshTensor([(csx, ncx), (csx, npadx, 1.3)])
        hz = Utils.meshTensor([(csz, npadz, -1.3), (csz, ncz), (csz, npadz, 1.3)])
        mesh = Mesh.CylMesh([hx, 1, hz], [0., 0., -hz.sum()/2])

        if plotIt:
            mesh.plotGrid()

        # make sure mesh is big enough
        self.assertTrue(mesh.hz.sum() > skdpth*2.)
        self.assertTrue(mesh.hx.sum() > skdpth*2.)

        # Create wholespace models
        SigmaBack = sigmaback*np.ones((mesh.nC))
        MuBack = (mu_0*(1 + kappa))*np.ones((mesh.nC))

        # Define reciever locations
        rlim = [20., 500.]
        r = mesh.vectorCCx[np.argmin(np.abs(mesh.vectorCCx-rlim[0])):np.argmin(np.abs(mesh.vectorCCx-rlim[1]))]
        z = 100.

        # where we choose to measure
        XYZ = Utils.ndgrid(r, np.r_[0.], np.r_[z])

        # Form data interpolation matrix
        Pf = mesh.getInterpolationMat(XYZ, 'CC')
        Pey = mesh.getInterpolationMat(XYZ, 'Ey')
        Zero = sp.csr_matrix(Pf.shape)
        Pfx, Pfz = sp.hstack([Pf, Zero]), sp.hstack([Zero, Pf])

    def test_CylMesh_ElecDipoleTest_Z(self):
        print('Testing various componemts of the field and fluxes from a Z-oriented analytic harmonic electric dipole against a numerical solution on a cylindrical mesh.')

        # Define the source
        # Search over z-faces to find face nearest src_loc then add nFx to get to global face index [nFx][nFy = 0][nFz]
        s_ind = Utils.closestPoints(mesh, src_loc, 'Fz') + mesh.nFx
        de = np.zeros(mesh.nF, dtype=complex)
        de[s_ind] = 1./csz
        de_z = [EM.FDEM.Src.RawVec_e([], freq, de/mesh.area)]

        # Pair the problem and survey
        survey = EM.FDEM.Survey(de_z)

        mapping = [('sigma', Maps.IdentityMap(mesh)), ('mu', Maps.IdentityMap(mesh))]

        problem = EM.FDEM.Problem3D_h(mesh, mapping=mapping)

        # pair problem and survey
        problem.pair(survey)

        try:
            from pymatsolver import MumpsSolver
            problem.Solver = MumpsSolver
        except ImportError, e:
            problem.Solver = SolverLU

        # solve
        numFields_ElecDipole = problem.fields(np.r_[SigmaBack, MuBack])

        # J lives on faces
        j_num = numFields_ElecDipole[de_z, 'j']
        Rho = Utils.sdiag(1./SigmaBack)
        Rho = sp.block_diag([Rho, Rho])
        e_numTest = Rho*mesh.aveF2CCV*j_num
        j_num = mesh.aveF2CCV*j_num

        # E lives on cell centres
        e_num = numFields_ElecDipole[de_z, 'e']

        # H lives on edges
        h_num_e = numFields_ElecDipole[de_z, 'h']
        h_num = mesh.aveE2CCV*h_num_e
        # B lives on cell centers
        b_num = numFields_ElecDipole[de_z, 'b']
        MuBack_ey = (mu_0*(1 + kappa))*np.ones((mesh.nEy))
        Mu = Utils.sdiag(MuBack_ey)
        b_numTest = Mu*h_num_e

        # Interpolate numeric fields and fluxes to cell cetres for easy comparison with analytics
        ex_num, ez_num = Pfx*e_num, Pfz*e_num
        ey_num = np.zeros_like(ex_num)
        ex_numTest, ez_numTest = Pfx*e_numTest, Pfz*e_numTest
        ey_numTest = np.zeros_like(ex_numTest)

        # Check E values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  E_x:', np.linalg.norm(ex_num), np.linalg.norm(ex_numTest), np.linalg.norm(ex_num-ex_numTest), np.linalg.norm(ex_num-ex_numTest)/np.linalg.norm(ex_numTest))
        print('')
        self.assertTrue(np.linalg.norm(ex_num-ex_numTest)/np.linalg.norm(ex_numTest) < tol_ElecDipole, msg='The two ways of calculating the numeric E field do not agree.')

        jx_num, jz_num = Pfx*j_num, Pfz*j_num
        jy_num = np.zeros_like(jx_num)

        # Since we are evaluating along the plane y=0 the b_theta == b_y in cartesian coordiantes
        btheta_num = Pf*b_num
        bx_num = np.zeros_like(btheta_num)
        bz_num = np.zeros_like(btheta_num)

        btheta_numTest = Pf*b_numTest
        bx_numTest = np.zeros_like(btheta_numTest)
        bz_numTest = np.zeros_like(btheta_numTest)

        # Check B values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  B_theta:', np.linalg.norm(btheta_num), np.linalg.norm(btheta_numTest), np.linalg.norm(btheta_num - btheta_numTest), np.linalg.norm(btheta_num-btheta_numTest)/np.linalg.norm(btheta_numTest))
        print('')
        self.assertTrue(np.linalg.norm(btheta_num-btheta_numTest)/np.linalg.norm(btheta_numTest) < tol_ElecDipole, msg='The two ways of calculating the numeric B field do not agree.')

        htheta_num = Pey*h_num_e
        hx_num = np.zeros_like(htheta_num)
        hz_num = np.zeros_like(htheta_num)

        # get analytic solution
        exa, eya, eza = EM.Analytics.FDEMDipolarfields.E_from_ElectricDipoleWholeSpace(XYZ, src_loc, sigmaback, Utils.mkvc(np.array(freq)),orientation='Z',kappa= kappa)
        exa, eya, eza = Utils.mkvc(exa, 2), Utils.mkvc(eya, 2), Utils.mkvc(eza, 2)

        jxa, jya, jza = EM.Analytics.FDEMDipolarfields.J_from_ElectricDipoleWholeSpace(XYZ, src_loc, sigmaback, Utils.mkvc(np.array(freq)),orientation='Z',kappa= kappa)
        jxa, jya, jza = Utils.mkvc(jxa, 2), Utils.mkvc(jya, 2), Utils.mkvc(jza, 2)

        bxa, bya, bza = EM.Analytics.FDEMDipolarfields.B_from_ElectricDipoleWholeSpace(XYZ, src_loc, sigmaback, Utils.mkvc(np.array(freq)),orientation='Z',kappa= kappa)
        bxa, bya, bza = Utils.mkvc(bxa, 2), Utils.mkvc(bya, 2), Utils.mkvc(bza, 2)

        hxa, hya, hza = EM.Analytics.FDEMDipolarfields.H_from_ElectricDipoleWholeSpace(XYZ, src_loc, sigmaback, Utils.mkvc(np.array(freq)),orientation='Z',kappa= kappa)
        hxa, hya, hza = Utils.mkvc(hxa, 2), Utils.mkvc(hya, 2), Utils.mkvc(hza, 2)

        print ' comp,       anayltic,       numeric,       num - ana,       (num - ana)/ana'
        print '  E_x:', np.linalg.norm(exa), np.linalg.norm(ex_numTest), np.linalg.norm(exa-ex_numTest), np.linalg.norm(exa-ex_numTest)/np.linalg.norm(exa)
        print '  E_y:', np.linalg.norm(eya), np.linalg.norm(ey_numTest), np.linalg.norm(eya-ey_numTest)
        print '  E_z:', np.linalg.norm(eza), np.linalg.norm(ez_numTest), np.linalg.norm(eza-ez_numTest), np.linalg.norm(eza-ez_numTest)/np.linalg.norm(eza)
        print ''
        print '  J_x:', np.linalg.norm(jxa), np.linalg.norm(jx_num), np.linalg.norm(jxa-jx_num), np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa)
        print '  J_y:', np.linalg.norm(jya), np.linalg.norm(jy_num), np.linalg.norm(jya-jy_num)
        print '  J_z:', np.linalg.norm(jza), np.linalg.norm(jz_num), np.linalg.norm(jza-jz_num), np.linalg.norm(jza-jz_num)/np.linalg.norm(jza)
        print ''
        print '  H_x:', np.linalg.norm(hxa), np.linalg.norm(hx_num), np.linalg.norm(hxa-hx_num)
        print '  H_y:', np.linalg.norm(hya), np.linalg.norm(htheta_num), np.linalg.norm(hya-htheta_num), np.linalg.norm(hya-htheta_num)/np.linalg.norm(hya)
        print '  H_z:', np.linalg.norm(hza), np.linalg.norm(hz_num), np.linalg.norm(hza-hz_num)
        print ''
        print '  B_x:', np.linalg.norm(bxa), np.linalg.norm(bx_num), np.linalg.norm(bxa-bx_num)
        print '  B_y:', np.linalg.norm(bya), np.linalg.norm(btheta_numTest), np.linalg.norm(bya-btheta_numTest), np.linalg.norm(bya-btheta_numTest)/np.linalg.norm(bya)
        print '  B_z:', np.linalg.norm(bza), np.linalg.norm(bz_num), np.linalg.norm(bza-bz_num)

        if plotIt:
            # Plot E
            plt.subplot(221)
            plt.plot(r, ex_numTest.real, 'o', r, exa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_x Real')
            plt.xlabel('r (m)')

            plt.subplot(222)
            plt.plot(r, ex_numTest.imag, 'o', r, exa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_x Imag')
            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.xlabel('r (m)')

            plt.subplot(223)
            plt.plot(r, ez_numTest.real, 'o', r, eza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_z Real')
            plt.xlabel('r (m)')

            plt.subplot(224)
            plt.plot(r, ez_numTest.imag, 'o', r, eza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_z Imag')
            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.xlabel('r (m)')

            plt.tight_layout()

            # Plot J
            plt.subplot(221)
            plt.plot(r, jx_num.real, 'o', r, jxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_x Real')
            plt.xlabel('r (m)')

            plt.subplot(222)
            plt.plot(r, jx_num.imag, 'o', r, jxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_x Imag')
            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.xlabel('r (m)')

            plt.subplot(223)
            plt.plot(r, jz_num.real, 'o', r, jza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_z Real')
            plt.xlabel('r (m)')

            plt.subplot(224)
            plt.plot(r, jz_num.imag, 'o', r, jza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_z Imag')
            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.xlabel('r (m)')

            plt.tight_layout()

            # Plot H
            plt.subplot(211)
            plt.plot(r, htheta_num.real, 'o', r, hya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_y Real')
            plt.xlabel('r (m)')

            plt.subplot(212)
            plt.plot(r, htheta_num.imag, 'o', r, hya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_y Imag')
            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.xlabel('r (m)')

            plt.tight_layout()

            # Plot B
            plt.subplot(211)
            plt.plot(r, btheta_numTest.real, 'o', r, bya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_y Real')
            plt.xlabel('r (m)')

            plt.subplot(212)
            plt.plot(r, btheta_numTest.imag, 'o', r, bya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_y Imag')
            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.xlabel('r (m)')

            plt.tight_layout()


        self.assertTrue(np.linalg.norm(exa-ex_numTest)/np.linalg.norm(exa) < tol_ElecDipole)
        self.assertTrue(np.linalg.norm(eya-ey_numTest) < tol_ElecDipole)
        self.assertTrue(np.linalg.norm(eza-ez_numTest)/np.linalg.norm(eza) < tol_ElecDipole)

        self.assertTrue(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa) < tol_ElecDipole)
        self.assertTrue(np.linalg.norm(jya-jy_num) < tol_ElecDipole)
        self.assertTrue(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza) < tol_ElecDipole)

        self.assertTrue(np.linalg.norm(hxa-hx_num) < tol_ElecDipole)
        self.assertTrue(np.linalg.norm(hya-htheta_num)/np.linalg.norm(hya) < tol_ElecDipole)
        self.assertTrue(np.linalg.norm(hza-hz_num) < tol_ElecDipole)

        self.assertTrue(np.linalg.norm(bxa-bx_num) < tol_ElecDipole)
        self.assertTrue(np.linalg.norm(bya-btheta_numTest)/np.linalg.norm(bya) < tol_ElecDipole)
        self.assertTrue(np.linalg.norm(bza-bz_num) < tol_ElecDipole)

        # checkEx = np.linalg.norm(exa-ex_numTest)/np.linalg.norm(exa) < tol_ElecDipole
        # print checkEx
        # checkEy = np.linalg.norm(eya-ey_numTest) < tol_ElecDipole
        # print checkEy
        # checkEz = np.linalg.norm(eza-ez_numTest)/np.linalg.norm(eza) < tol_ElecDipole
        # print checkEz

        # checkJx = np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa) < tol_ElecDipole
        # print checkJx
        # checkJy = np.linalg.norm(jya-jy_num) < tol_ElecDipole
        # print checkJy
        # checkJz = np.linalg.norm(jza-jz_num)/np.linalg.norm(jza) < tol_ElecDipole
        # print checkJz

        # checkHx = np.linalg.norm(hxa-hx_num) < tol_ElecDipole
        # print checkHx
        # checkHy = np.linalg.norm(hya-htheta_num)/np.linalg.norm(hya) < tol_ElecDipole
        # print checkHy
        # checkHz = np.linalg.norm(hza-hz_num) < tol_ElecDipole
        # print checkHz

        # checkBx = np.linalg.norm(bxa-bx_num) < tol_ElecDipole
        # print checkBx
        # checkBy = np.linalg.norm(bya-btheta_num)/np.linalg.norm(bya) < tol_ElecDipole
        # print checkBy
        # checkBz = np.linalg.norm(bza-bz_num) < tol_ElecDipole
        # print checkBz


if __name__ == '__main__':
    unittest.main()
