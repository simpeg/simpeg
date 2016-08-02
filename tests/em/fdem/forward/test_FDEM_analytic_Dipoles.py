import unittest
from SimPEG import EM, Mesh, Utils, np, Maps, sp
from pymatsolver import MumpsSolver
# import sys
from scipy.constants import mu_0


plotIt = False

if plotIt:
    import matplotlib.pylab
    import matplotlib.pyplot as plt


class FDEM_analytic_DipoleTests_CylMesh(unittest.TestCase):

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

        tol_ElecDipole = 1e-2

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

class FDEM_analytic_DipoleTests_3DMesh(unittest.TestCase):

    def setUp(self):

        # Define model parameters
        sigmaback = 1.
        # mu = mu_0*(1+kappa)
        kappa = 1.

        # Create 3D mesh
        npad = 5
        csx, ncx, npadx = 5, 30, npad
        csy, ncy, npady = 5, 30, npad
        csz, ncz, npadz = 5, 30, npad
        hx = Utils.meshTensor([(csx, npadx, -1.3), (csx, ncx), (csx, npadx, 1.3)])
        hy = Utils.meshTensor([(csy, npady, -1.3), (csy, ncy), (csy, npady, 1.3)])
        hz = Utils.meshTensor([(csz, npadz, -1.3), (csz, ncz), (csz, npadz, 1.3)])
        mesh = Mesh.TensorMesh([hx, hy, hz], 'CCC')

        # Set source parameters
        freq = 100.
        src_loc = np.r_[0., 0., -35]
        src_loc_CCInd = Utils.closestPoints(mesh, src_loc, 'CC')
        src_loc_CC = mesh.gridCC[src_loc_CCInd,:]
        src_loc_CC = src_loc_CC[0]

        # Compute skin depth
        skdpth = 500. / np.sqrt(sigmaback * freq)

        # make sure mesh is big enough
        self.assertTrue(mesh.hx.sum() > skdpth*2.)
        self.assertTrue(mesh.hy.sum() > skdpth*2.)
        self.assertTrue(mesh.hz.sum() > skdpth*2.)

        # Create wholespace models
        SigmaBack = sigmaback*np.ones((mesh.nC))
        MuBack = (mu_0*(1 + kappa))*np.ones((mesh.nC))

        # Define reciever locations
        xlim = 60. # x locations from -60 to 60
        xInd = np.where(np.abs(mesh.vectorCCx) < xlim)
        x = mesh.vectorCCx[xInd[0]]
        y = 10.
        z = 35.

        # # where we choose to measure
        XYZ = Utils.ndgrid(x, np.r_[y], np.r_[z])

        XYZ_CCInd = Utils.closestPoints(mesh, XYZ, 'CC')
        XYZ_CC = mesh.gridCC[XYZ_CCInd,:]

        # Form data interpolation matrices
        Pcc = mesh.getInterpolationMat(XYZ_CC, 'CC')
        Zero = sp.csr_matrix(Pcc.shape)
        Pccx, Pccy, Pccz = sp.hstack([Pcc, Zero, Zero]), sp.hstack([Zero, Pcc, Zero]), sp.hstack([Zero, Zero, Pcc])

        Pex, Pey, Pez = mesh.getInterpolationMat(XYZ_CC, 'Ex'), mesh.getInterpolationMat(XYZ_CC, 'Ey'), mesh.getInterpolationMat(XYZ_CC, 'Ez')
        Pfx, Pfy, Pfz = mesh.getInterpolationMat(XYZ_CC, 'Fx'), mesh.getInterpolationMat(XYZ_CC, 'Fy'), mesh.getInterpolationMat(XYZ_CC, 'Fz')

    def test_3DMesh_ElecDipoleTest_X(self):
        print('Testing various componemts of the fields and fluxes from a X-oriented analytic harmonic electric dipole against a numerical solution on a 3D tesnsor mesh.')

        tol_ElecDipole_X = 3e-2
        tol_NumErrZero = 1e-16

        # Define the source
        # Search over x-faces to find face nearest src_loc
        s_ind = Utils.closestPoints(mesh, src_loc_CC, 'Fx')
        de = np.zeros(mesh.nF, dtype=complex)
        de[s_ind] = 1./csx
        de_x = [EM.FDEM.Src.RawVec_e([], freq, de/mesh.area)]

        src_loc_Fx = mesh.gridFx[s_ind,:]
        src_loc_Fx = src_loc_Fx[0]

        # Plot Tx and Rx locations on mesh
        if plotIt:
            fig, ax = plt.subplots(1,1, figsize=(10,10))
            ax.plot(src_loc_Fx[0], src_loc_Fx[2], 'ro', ms=8)
            ax.plot(XYZ_CC[:,0], XYZ_CC[:,2], 'k.', ms=8)
            mesh.plotSlice(np.zeros(mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

        # Create survey and problem object
        survey = EM.FDEM.Survey(de_x)
        mapping = [('sigma', Maps.IdentityMap(mesh)), ('mu', Maps.IdentityMap(mesh))]
        problem = EM.FDEM.Problem3D_h(mesh, mapping=mapping)

        # Pair problem and survey
        problem.pair(survey)

        try:
            from pymatsolver import MumpsSolver
            problem.Solver = MumpsSolver
            print('solver set to Mumps')
        except ImportError, e:
            problem.Solver = SolverLU

        # Solve forward problem
        numFields_ElecDipole_X = problem.fields(np.r_[SigmaBack, MuBack])

        # Get fields and fluxes
        # J lives on faces
        j_numF = numFields_ElecDipole_X[de_x, 'j']
        j_numCC = mesh.aveF2CCV*j_numF

        # E lives on cell centres
        e_num = numFields_ElecDipole_X[de_x, 'e']
        Rho = Utils.sdiag(1./SigmaBack)
        Rho = sp.block_diag([Rho, Rho, Rho])
        e_numTest = Rho*mesh.aveF2CCV*j_numF

        # H lives on edges
        h_numE = numFields_ElecDipole_X[de_x, 'h']
        h_numCC = mesh.aveE2CCV*h_numE

        # B lives on cell centers
        b_num = numFields_ElecDipole_X[de_x, 'b']
        MuBack_E = (mu_0*(1 + kappa))*np.ones((mesh.nE))
        Mu = Utils.sdiag(MuBack_E)
        b_numTest = Mu*h_numE

        # Interpolate numeric fields and fluxes to cell cetres for easy comparison with analytics
        ex_num, ey_num, ez_num = Pccx*e_num, Pccy*e_num, Pccz*e_num
        ex_numTest, ey_numTest, ez_numTest = Pccx*e_numTest, Pccy*e_numTest, Pccz*e_numTest

        jx_num, jy_num, jz_num = Pfx*j_numF, Pfy*j_numF, Pfz*j_numF
        jx_numTest, jy_numTest, jz_numTest = Pccx*j_numCC, Pccy*j_numCC, Pccz*j_numCC

        hx_num, hy_num, hz_num = Pex*h_numE, Pey*h_numE, Pez*h_numE
        hx_numTest, hy_numTest, hz_numTest = Pccx*h_numCC, Pccy*h_numCC, Pccz*h_numCC

        bx_num, by_num, bz_num = Pccx*b_num, Pccy*b_num, Pccz*b_num
        bx_numTest, by_numTest, bz_numTest = Pex*b_numTest, Pey*b_numTest, Pez*b_numTest

        # Check E values computed from fields object
        tol_fieldObjCheck = 1e-14
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  E_x:', np.linalg.norm(ex_num), np.linalg.norm(ex_numTest), np.linalg.norm(ex_num-ex_numTest), np.linalg.norm(ex_num-ex_numTest)/np.linalg.norm(ex_numTest))
        print('  E_y:', np.linalg.norm(ey_num), np.linalg.norm(ey_numTest), np.linalg.norm(ey_num-ey_numTest), np.linalg.norm(ey_num-ey_numTest)/np.linalg.norm(ey_numTest))
        print('  E_z:', np.linalg.norm(ez_num), np.linalg.norm(ez_numTest), np.linalg.norm(ez_num-ez_numTest), np.linalg.norm(ez_num-ez_numTest)/np.linalg.norm(ez_numTest))
        print('')
        self.assertTrue(np.linalg.norm(ex_num-ex_numTest)/np.linalg.norm(ex_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ex field do not agree.')
        self.assertTrue(np.linalg.norm(ey_num-ey_numTest)/np.linalg.norm(ey_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ey field do not agree.')
        self.assertTrue(np.linalg.norm(ez_num-ez_numTest)/np.linalg.norm(ez_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ez field do not agree.')

        # Check J values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  J_x:', np.linalg.norm(jx_num), np.linalg.norm(jx_numTest), np.linalg.norm(jx_num-jx_numTest), np.linalg.norm(jx_num-jx_numTest)/np.linalg.norm(jx_numTest))
        print('  J_y:', np.linalg.norm(jy_num), np.linalg.norm(jy_numTest), np.linalg.norm(jy_num-jy_numTest), np.linalg.norm(jy_num-jy_numTest)/np.linalg.norm(jy_numTest))
        print('  J_z:', np.linalg.norm(jz_num), np.linalg.norm(jz_numTest), np.linalg.norm(jz_num-jz_numTest), np.linalg.norm(jz_num-jz_numTest)/np.linalg.norm(jz_numTest))
        print('')
        self.assertTrue(np.linalg.norm(jx_num-jx_numTest)/np.linalg.norm(jx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jx field do not agree.')
        self.assertTrue(np.linalg.norm(jy_num-jy_numTest)/np.linalg.norm(jy_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jy field do not agree.')
        self.assertTrue(np.linalg.norm(jz_num-jz_numTest)/np.linalg.norm(jz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jz field do not agree.')

        # Check H values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  H_x:', np.linalg.norm(hx_num), np.linalg.norm(hx_numTest), np.linalg.norm(hx_num-hx_numTest), np.linalg.norm(hx_num-hx_numTest)/np.linalg.norm(hx_numTest))
        print('  H_y:', np.linalg.norm(hy_num), np.linalg.norm(hy_numTest), np.linalg.norm(hy_num-hy_numTest), np.linalg.norm(hy_num-hy_numTest)/np.linalg.norm(hy_numTest))
        print('  H_z:', np.linalg.norm(hz_num), np.linalg.norm(hz_numTest), np.linalg.norm(hz_num-hz_numTest), np.linalg.norm(hz_num-hz_numTest)/np.linalg.norm(hz_numTest))
        print('')
        self.assertTrue(np.linalg.norm(hx_num-hx_numTest)/np.linalg.norm(hx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hx field do not agree.')
        self.assertTrue(np.linalg.norm(hy_num-hy_numTest)/np.linalg.norm(hy_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hy field do not agree.')
        self.assertTrue(np.linalg.norm(hz_num-hz_numTest)/np.linalg.norm(hz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hz field do not agree.')

        # Check B values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  B_x:', np.linalg.norm(bx_num), np.linalg.norm(bx_numTest), np.linalg.norm(bx_num-bx_numTest), np.linalg.norm(bx_num-bx_numTest)/np.linalg.norm(bx_numTest))
        print('  B_y:', np.linalg.norm(by_num), np.linalg.norm(by_numTest), np.linalg.norm(by_num-by_numTest), np.linalg.norm(by_num-by_numTest)/np.linalg.norm(by_numTest))
        print('  B_z:', np.linalg.norm(bz_num), np.linalg.norm(bz_numTest), np.linalg.norm(bz_num-bz_numTest), np.linalg.norm(bz_num-bz_numTest)/np.linalg.norm(bz_numTest))
        print('')
        self.assertTrue(np.linalg.norm(bx_num-bx_numTest)/np.linalg.norm(bx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Bx field do not agree.')
        self.assertTrue(np.linalg.norm(by_num-by_numTest)/np.linalg.norm(by_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric By field do not agree.')
        self.assertTrue(np.linalg.norm(bz_num-bz_numTest)/np.linalg.norm(bz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Bz field do not agree.')

        # get analytic solutions
        exa, eya, eza = EM.Analytics.FDEMDipolarfields.E_from_ElectricDipoleWholeSpace(XYZ_CC, src_loc_Fx, sigmaback, Utils.mkvc(np.array(freq)),orientation='X',kappa= kappa)
        exa, eya, eza = Utils.mkvc(exa, 2), Utils.mkvc(eya, 2), Utils.mkvc(eza, 2)

        jxa, jya, jza = EM.Analytics.FDEMDipolarfields.J_from_ElectricDipoleWholeSpace(XYZ_CC, src_loc_Fx, sigmaback, Utils.mkvc(np.array(freq)),orientation='X',kappa= kappa)
        jxa, jya, jza = Utils.mkvc(jxa, 2), Utils.mkvc(jya, 2), Utils.mkvc(jza, 2)

        hxa, hya, hza = EM.Analytics.FDEMDipolarfields.H_from_ElectricDipoleWholeSpace(XYZ_CC, src_loc_Fx, sigmaback, Utils.mkvc(np.array(freq)),orientation='X',kappa= kappa)
        hxa, hya, hza = Utils.mkvc(hxa, 2), Utils.mkvc(hya, 2), Utils.mkvc(hza, 2)

        bxa, bya, bza = EM.Analytics.FDEMDipolarfields.B_from_ElectricDipoleWholeSpace(XYZ_CC, src_loc_Fx, sigmaback, Utils.mkvc(np.array(freq)),orientation='X',kappa= kappa)
        bxa, bya, bza = Utils.mkvc(bxa, 2), Utils.mkvc(bya, 2), Utils.mkvc(bza, 2)

        print ' comp,       anayltic,       numeric,       num - ana,       (num - ana)/ana'
        print '  E_x:', np.linalg.norm(exa), np.linalg.norm(ex_num), np.linalg.norm(exa-ex_num), np.linalg.norm(exa-ex_num)/np.linalg.norm(exa)
        print '  E_y:', np.linalg.norm(eya), np.linalg.norm(ey_num), np.linalg.norm(eya-ey_num), np.linalg.norm(eya-ey_num)/np.linalg.norm(eya)
        print '  E_z:', np.linalg.norm(eza), np.linalg.norm(ez_num), np.linalg.norm(eza-ez_num), np.linalg.norm(eza-ez_num)/np.linalg.norm(eza)
        print ''
        print '  J_x:', np.linalg.norm(jxa), np.linalg.norm(jx_num), np.linalg.norm(jxa-jx_num), np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa)
        print '  J_y:', np.linalg.norm(jya), np.linalg.norm(jy_num), np.linalg.norm(jya-jy_num), np.linalg.norm(jya-jy_num)/np.linalg.norm(jya)
        print '  J_z:', np.linalg.norm(jza), np.linalg.norm(jz_num), np.linalg.norm(jza-jz_num), np.linalg.norm(jza-jz_num)/np.linalg.norm(jza)
        print ''
        print '  H_x:', np.linalg.norm(hxa), np.linalg.norm(hx_num), np.linalg.norm(hxa-hx_num), np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa)
        print '  H_y:', np.linalg.norm(hya), np.linalg.norm(hy_num), np.linalg.norm(hya-hy_num), np.linalg.norm(hya-hy_num)/np.linalg.norm(hya)
        print '  H_z:', np.linalg.norm(hza), np.linalg.norm(hz_num), np.linalg.norm(hza-hz_num), np.linalg.norm(hza-hz_num)/np.linalg.norm(hza)
        print ''
        print '  B_x:', np.linalg.norm(bxa), np.linalg.norm(bx_num), np.linalg.norm(bxa-bx_num), np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa)
        print '  B_y:', np.linalg.norm(bya), np.linalg.norm(by_num), np.linalg.norm(bya-by_num), np.linalg.norm(bya-by_num)/np.linalg.norm(bya)
        print '  B_z:', np.linalg.norm(bza), np.linalg.norm(bz_num), np.linalg.norm(bza-bz_num), np.linalg.norm(bza-bz_num)/np.linalg.norm(bza)
        print ''
        self.assertTrue(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Ex do not agree.')
        self.assertTrue(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Ey do not agree.')
        self.assertTrue(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Ez do not agree.')

        self.assertTrue(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Jx do not agree.')
        self.assertTrue(np.linalg.norm(jya-jy_num)/np.linalg.norm(jya) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Jy do not agree.')
        self.assertTrue(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Jz do not agree.')

        self.assertTrue(np.linalg.norm(hxa-hx_num) < tol_NumErrZero, msg='Analytic and numeric solutions for Hx do not agree.')
        self.assertTrue(np.linalg.norm(hya-hy_num)/np.linalg.norm(hya) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Hy do not agree.')
        self.assertTrue(np.linalg.norm(hza-hz_num)/np.linalg.norm(hza) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Hz do not agree.')

        self.assertTrue(np.linalg.norm(bxa-bx_num) < tol_NumErrZero, msg='Analytic and numeric solutions for Bx do not agree.')
        self.assertTrue(np.linalg.norm(bya-by_num)/np.linalg.norm(bya) < tol_ElecDipole_X, msg='Analytic and numeric solutions for By do not agree.')
        self.assertTrue(np.linalg.norm(bza-bz_num)/np.linalg.norm(bza) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Bz do not agree.')

        # Plot Tx and Rx locations on mesh
        if plotIt:
            # Plot E
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(x, ex_num.real, 'o', x, ex_numTest.real, 'd', x, exa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(x, ex_num.imag, 'o', x, ex_numTest.imag, 'd', x, exa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(x, ey_num.real, 'o', x, ey_numTest.real, 'd', x, eya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(x, ey_num.imag, 'o', x, ey_numTest.imag, 'd', x, eya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(x, ez_num.real, 'o', x, ez_numTest.real, 'd', x, eza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(x, ez_num.imag, 'o', x, ez_numTest.imag, 'd', x, eza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot J
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(x, jx_num.real, 'o', x, jx_numTest.real, 'd', x, jxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(x, jx_num.imag, 'o', x, jx_numTest.imag, 'd', x, jxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(x, jy_num.real, 'o', x, jy_numTest.real, 'd', x, jya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(x, jy_num.imag, 'o', x, jy_numTest.imag, 'd', x, jya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(x, jz_num.real, 'o', x, jz_numTest.real, 'd', x, jza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(x, jz_num.imag, 'o', x, jz_numTest.imag, 'd', x, jza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot H
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(x, hx_num.real, 'o', x, hx_numTest.real, 'd', x, hxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(x, hx_num.imag, 'o', x, hx_numTest.imag, 'd', x, hxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(x, hy_num.real, 'o', x, hy_numTest.real, 'd', x, hya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(x, hy_num.imag, 'o', x, hy_numTest.imag, 'd', x, hya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(x, hz_num.real, 'o', x, hz_numTest.real, 'd', x, hza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(x, hz_num.imag, 'o', x, hz_numTest.imag, 'd', x, hza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot B
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(x, bx_num.real, 'o', x, bx_numTest.real, 'd', x, bxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(x, bx_num.imag, 'o', x, bx_numTest.imag, 'd', x, bxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(x, by_num.real, 'o', x, by_numTest.real, 'd', x, bya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(x, by_num.imag, 'o', x, by_numTest.imag, 'd', x, bya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(x, bz_num.real, 'o', x, bz_numTest.real, 'd', x, bza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(x, bz_num.imag, 'o', x, bz_numTest.imag, 'd', x, bza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()


    def test_3DMesh_ElecDipoleTest_Y(self):
        print('Testing various componemts of the fields and fluxes from a Y-oriented analytic harmonic electric dipole against a numerical solution on a 3D tesnsor mesh.')


        print('Testing various componemts of the fields and fluxes from a Y-oriented analytic harmonic electric dipole against a numerical solution on a 3D tesnsor mesh.')

        tol_ElecDipole_Y = 4e-2
        tol_NumErrZero = 1e-16

        # Define the source
        # Search over y-faces to find face nearest src_loc
        s_ind = Utils.closestPoints(mesh, src_loc_CC, 'Fy') + mesh.nFx
        de = np.zeros(mesh.nF, dtype=complex)
        de[s_ind] = 1./csy
        de_y = [EM.FDEM.Src.RawVec_e([], freq, de/mesh.area)]

        src_loc_Fy = mesh.gridFy[Utils.closestPoints(mesh, src_loc_CC, 'Fy'),:]
        src_loc_Fy = src_loc_Fy[0]

        # Plot Tx and Rx locations on mesh
        if plotIt:
            fig, ax = plt.subplots(1,1, figsize=(10,10))
            ax.plot(src_loc_Fy[0], src_loc_Fy[2], 'ro', ms=8)
            ax.plot(XYZ_CC[:,0], XYZ_CC[:,2], 'k.', ms=8)
            mesh.plotSlice(np.zeros(mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

        # Create survey and problem object
        survey = EM.FDEM.Survey(de_y)
        mapping = [('sigma', Maps.IdentityMap(mesh)), ('mu', Maps.IdentityMap(mesh))]
        problem = EM.FDEM.Problem3D_h(mesh, mapping=mapping)

        # Pair problem and survey
        problem.pair(survey)

        try:
            from pymatsolver import MumpsSolver
            problem.Solver = MumpsSolver
            print('solver set to Mumps')
        except ImportError, e:
            problem.Solver = SolverLU

        # Solve forward problem
        numFields_ElecDipole_Y = problem.fields(np.r_[SigmaBack, MuBack])

        # Get fields and fluxes
        # J lives on faces
        j_numF = numFields_ElecDipole_Y[de_y, 'j']
        j_numCC = mesh.aveF2CCV*j_numF

        # E lives on cell centres
        e_num = numFields_ElecDipole_Y[de_y, 'e']
        Rho = Utils.sdiag(1./SigmaBack)
        Rho = sp.block_diag([Rho, Rho, Rho])
        e_numTest = Rho*mesh.aveF2CCV*j_numF

        # H lives on edges
        h_numE = numFields_ElecDipole_Y[de_y, 'h']
        h_numCC = mesh.aveE2CCV*h_numE

        # B lives on cell centers
        b_num = numFields_ElecDipole_Y[de_y, 'b']
        MuBack_E = (mu_0*(1 + kappa))*np.ones((mesh.nE))
        Mu = Utils.sdiag(MuBack_E)
        b_numTest = Mu*h_numE

        # Interpolate numeric fields and fluxes to cell cetres for easy comparison with analytics
        ex_num, ey_num, ez_num = Pccx*e_num, Pccy*e_num, Pccz*e_num
        ex_numTest, ey_numTest, ez_numTest = Pccx*e_numTest, Pccy*e_numTest, Pccz*e_numTest

        jx_num, jy_num, jz_num = Pfx*j_numF, Pfy*j_numF, Pfz*j_numF
        jx_numTest, jy_numTest, jz_numTest = Pccx*j_numCC, Pccy*j_numCC, Pccz*j_numCC

        hx_num, hy_num, hz_num = Pex*h_numE, Pey*h_numE, Pez*h_numE
        hx_numTest, hy_numTest, hz_numTest = Pccx*h_numCC, Pccy*h_numCC, Pccz*h_numCC

        bx_num, by_num, bz_num = Pccx*b_num, Pccy*b_num, Pccz*b_num
        bx_numTest, by_numTest, bz_numTest = Pex*b_numTest, Pey*b_numTest, Pez*b_numTest

        # Check E values computed from fields object
        tol_fieldObjCheck = 1e-14
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  E_x:', np.linalg.norm(ex_num), np.linalg.norm(ex_numTest), np.linalg.norm(ex_num-ex_numTest), np.linalg.norm(ex_num-ex_numTest)/np.linalg.norm(ex_numTest))
        print('  E_y:', np.linalg.norm(ey_num), np.linalg.norm(ey_numTest), np.linalg.norm(ey_num-ey_numTest), np.linalg.norm(ey_num-ey_numTest)/np.linalg.norm(ey_numTest))
        print('  E_z:', np.linalg.norm(ez_num), np.linalg.norm(ez_numTest), np.linalg.norm(ez_num-ez_numTest), np.linalg.norm(ez_num-ez_numTest)/np.linalg.norm(ez_numTest))
        print('')
        self.assertTrue(np.linalg.norm(ex_num-ex_numTest)/np.linalg.norm(ex_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ex field do not agree.')
        self.assertTrue(np.linalg.norm(ey_num-ey_numTest)/np.linalg.norm(ey_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ey field do not agree.')
        self.assertTrue(np.linalg.norm(ez_num-ez_numTest)/np.linalg.norm(ez_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ez field do not agree.')

        # Check J values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  J_x:', np.linalg.norm(jx_num), np.linalg.norm(jx_numTest), np.linalg.norm(jx_num-jx_numTest), np.linalg.norm(jx_num-jx_numTest)/np.linalg.norm(jx_numTest))
        print('  J_y:', np.linalg.norm(jy_num), np.linalg.norm(jy_numTest), np.linalg.norm(jy_num-jy_numTest), np.linalg.norm(jy_num-jy_numTest)/np.linalg.norm(jy_numTest))
        print('  J_z:', np.linalg.norm(jz_num), np.linalg.norm(jz_numTest), np.linalg.norm(jz_num-jz_numTest), np.linalg.norm(jz_num-jz_numTest)/np.linalg.norm(jz_numTest))
        print('')
        self.assertTrue(np.linalg.norm(jx_num-jx_numTest)/np.linalg.norm(jx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jx field do not agree.')
        self.assertTrue(np.linalg.norm(jy_num-jy_numTest)/np.linalg.norm(jy_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jy field do not agree.')
        self.assertTrue(np.linalg.norm(jz_num-jz_numTest)/np.linalg.norm(jz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jz field do not agree.')

        # Check H values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  H_x:', np.linalg.norm(hx_num), np.linalg.norm(hx_numTest), np.linalg.norm(hx_num-hx_numTest), np.linalg.norm(hx_num-hx_numTest)/np.linalg.norm(hx_numTest))
        print('  H_y:', np.linalg.norm(hy_num), np.linalg.norm(hy_numTest), np.linalg.norm(hy_num-hy_numTest), np.linalg.norm(hy_num-hy_numTest)/np.linalg.norm(hy_numTest))
        print('  H_z:', np.linalg.norm(hz_num), np.linalg.norm(hz_numTest), np.linalg.norm(hz_num-hz_numTest), np.linalg.norm(hz_num-hz_numTest)/np.linalg.norm(hz_numTest))
        print('')
        self.assertTrue(np.linalg.norm(hx_num-hx_numTest)/np.linalg.norm(hx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hx field do not agree.')
        self.assertTrue(np.linalg.norm(hy_num-hy_numTest)/np.linalg.norm(hy_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hy field do not agree.')
        self.assertTrue(np.linalg.norm(hz_num-hz_numTest)/np.linalg.norm(hz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hz field do not agree.')

        # Check B values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  B_x:', np.linalg.norm(bx_num), np.linalg.norm(bx_numTest), np.linalg.norm(bx_num-bx_numTest), np.linalg.norm(bx_num-bx_numTest)/np.linalg.norm(bx_numTest))
        print('  B_y:', np.linalg.norm(by_num), np.linalg.norm(by_numTest), np.linalg.norm(by_num-by_numTest), np.linalg.norm(by_num-by_numTest)/np.linalg.norm(by_numTest))
        print('  B_z:', np.linalg.norm(bz_num), np.linalg.norm(bz_numTest), np.linalg.norm(bz_num-bz_numTest), np.linalg.norm(bz_num-bz_numTest)/np.linalg.norm(bz_numTest))
        print('')
        self.assertTrue(np.linalg.norm(bx_num-bx_numTest)/np.linalg.norm(bx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Bx field do not agree.')
        self.assertTrue(np.linalg.norm(by_num-by_numTest)/np.linalg.norm(by_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric By field do not agree.')
        self.assertTrue(np.linalg.norm(bz_num-bz_numTest)/np.linalg.norm(bz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Bz field do not agree.')


        # get analytic solutions
        exa, eya, eza = EM.Analytics.FDEMDipolarfields.E_from_ElectricDipoleWholeSpace(XYZ_CC, src_loc_Fy, sigmaback, Utils.mkvc(np.array(freq)),orientation='Y',kappa= kappa)
        exa, eya, eza = Utils.mkvc(exa, 2), Utils.mkvc(eya, 2), Utils.mkvc(eza, 2)

        jxa, jya, jza = EM.Analytics.FDEMDipolarfields.J_from_ElectricDipoleWholeSpace(XYZ_CC, src_loc_Fy, sigmaback, Utils.mkvc(np.array(freq)),orientation='Y',kappa= kappa)
        jxa, jya, jza = Utils.mkvc(jxa, 2), Utils.mkvc(jya, 2), Utils.mkvc(jza, 2)

        hxa, hya, hza = EM.Analytics.FDEMDipolarfields.H_from_ElectricDipoleWholeSpace(XYZ_CC, src_loc_Fy, sigmaback, Utils.mkvc(np.array(freq)),orientation='Y',kappa= kappa)
        hxa, hya, hza = Utils.mkvc(hxa, 2), Utils.mkvc(hya, 2), Utils.mkvc(hza, 2)

        bxa, bya, bza = EM.Analytics.FDEMDipolarfields.B_from_ElectricDipoleWholeSpace(XYZ_CC, src_loc_Fy, sigmaback, Utils.mkvc(np.array(freq)),orientation='Y',kappa= kappa)
        bxa, bya, bza = Utils.mkvc(bxa, 2), Utils.mkvc(bya, 2), Utils.mkvc(bza, 2)

        print ' comp,       anayltic,       numeric,       num - ana,       (num - ana)/ana'
        print '  E_x:', np.linalg.norm(exa), np.linalg.norm(ex_num), np.linalg.norm(exa-ex_num), np.linalg.norm(exa-ex_num)/np.linalg.norm(exa)
        print '  E_y:', np.linalg.norm(eya), np.linalg.norm(ey_num), np.linalg.norm(eya-ey_num), np.linalg.norm(eya-ey_num)/np.linalg.norm(eya)
        print '  E_z:', np.linalg.norm(eza), np.linalg.norm(ez_num), np.linalg.norm(eza-ez_num), np.linalg.norm(eza-ez_num)/np.linalg.norm(eza)
        print ''
        print '  J_x:', np.linalg.norm(jxa), np.linalg.norm(jx_num), np.linalg.norm(jxa-jx_num), np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa)
        print '  J_y:', np.linalg.norm(jya), np.linalg.norm(jy_num), np.linalg.norm(jya-jy_num), np.linalg.norm(jya-jy_num)/np.linalg.norm(jya)
        print '  J_z:', np.linalg.norm(jza), np.linalg.norm(jz_num), np.linalg.norm(jza-jz_num), np.linalg.norm(jza-jz_num)/np.linalg.norm(jza)
        print ''
        print '  H_x:', np.linalg.norm(hxa), np.linalg.norm(hx_num), np.linalg.norm(hxa-hx_num), np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa)
        print '  H_y:', np.linalg.norm(hya), np.linalg.norm(hy_num), np.linalg.norm(hya-hy_num), np.linalg.norm(hya-hy_num)/np.linalg.norm(hya)
        print '  H_z:', np.linalg.norm(hza), np.linalg.norm(hz_num), np.linalg.norm(hza-hz_num), np.linalg.norm(hza-hz_num)/np.linalg.norm(hza)
        print ''
        print '  B_x:', np.linalg.norm(bxa), np.linalg.norm(bx_num), np.linalg.norm(bxa-bx_num), np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa)
        print '  B_y:', np.linalg.norm(bya), np.linalg.norm(by_num), np.linalg.norm(bya-by_num), np.linalg.norm(bya-by_num)/np.linalg.norm(bya)
        print '  B_z:', np.linalg.norm(bza), np.linalg.norm(bz_num), np.linalg.norm(bza-bz_num), np.linalg.norm(bza-bz_num)/np.linalg.norm(bza)
        print ''
        self.assertTrue(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Ex do not agree.')
        self.assertTrue(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Ey do not agree.')
        self.assertTrue(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Ez do not agree.')

        self.assertTrue(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Jx do not agree.')
        self.assertTrue(np.linalg.norm(jya-jy_num)/np.linalg.norm(jya) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Jy do not agree.')
        self.assertTrue(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Jz do not agree.')

        self.assertTrue(np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Hx do not agree.')
        self.assertTrue(np.linalg.norm(hya-hy_num) < tol_NumErrZero, msg='Analytic and numeric solutions for Hy do not agree.')
        self.assertTrue(np.linalg.norm(hza-hz_num)/np.linalg.norm(hza) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Hz do not agree.')

        self.assertTrue(np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Bx do not agree.')
        self.assertTrue(np.linalg.norm(bya-by_num) < tol_NumErrZero, msg='Analytic and numeric solutions for By do not agree.')
        self.assertTrue(np.linalg.norm(bza-bz_num)/np.linalg.norm(bza) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Bz do not agree.')

        # Plot Tx and Rx locations on mesh
        if plotIt:
            # Plot E
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(x, ex_num.real, 'o', x, ex_numTest.real, 'd', x, exa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(x, ex_num.imag, 'o', x, ex_numTest.imag, 'd', x, exa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(x, ey_num.real, 'o', x, ey_numTest.real, 'd', x, eya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(x, ey_num.imag, 'o', x, ey_numTest.imag, 'd', x, eya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(x, ez_num.real, 'o', x, ez_numTest.real, 'd', x, eza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(x, ez_num.imag, 'o', x, ez_numTest.imag, 'd', x, eza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot J
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(x, jx_num.real, 'o', x, jx_numTest.real, 'd', x, jxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(x, jx_num.imag, 'o', x, jx_numTest.imag, 'd', x, jxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(x, jy_num.real, 'o', x, jy_numTest.real, 'd', x, jya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(x, jy_num.imag, 'o', x, jy_numTest.imag, 'd', x, jya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(x, jz_num.real, 'o', x, jz_numTest.real, 'd', x, jza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(x, jz_num.imag, 'o', x, jz_numTest.imag, 'd', x, jza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot H
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(x, hx_num.real, 'o', x, hx_numTest.real, 'd', x, hxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(x, hx_num.imag, 'o', x, hx_numTest.imag, 'd', x, hxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(x, hy_num.real, 'o', x, hy_numTest.real, 'd', x, hya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(x, hy_num.imag, 'o', x, hy_numTest.imag, 'd', x, hya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(x, hz_num.real, 'o', x, hz_numTest.real, 'd', x, hza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(x, hz_num.imag, 'o', x, hz_numTest.imag, 'd', x, hza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot B
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(x, bx_num.real, 'o', x, bx_numTest.real, 'd', x, bxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(x, bx_num.imag, 'o', x, bx_numTest.imag, 'd', x, bxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(x, by_num.real, 'o', x, by_numTest.real, 'd', x, bya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(x, by_num.imag, 'o', x, by_numTest.imag, 'd', x, bya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(x, bz_num.real, 'o', x, bz_numTest.real, 'd', x, bza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(x, bz_num.imag, 'o', x, bz_numTest.imag, 'd', x, bza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

    def test_3DMesh_ElecDipoleTest_Z(self):
        print('Testing various componemts of the fields and fluxes from a Z-oriented analytic harmonic electric dipole against a numerical solution on a 3D tesnsor mesh.')

        tol_ElecDipole_Z = 3e-2
        tol_NumErrZero = 1e-16

        # Define the source
        # Search over x-faces to find face nearest src_loc
        s_ind = Utils.closestPoints(mesh, src_loc_CC, 'Fz') + mesh.nFx + mesh.nFy
        de = np.zeros(mesh.nF, dtype=complex)
        de[s_ind] = 1./csz
        de_z = [EM.FDEM.Src.RawVec_e([], freq, de/mesh.area)]

        src_loc_Fz = mesh.gridFz[Utils.closestPoints(mesh, src_loc_CC, 'Fz'),:]
        src_loc_Fz = src_loc_Fz[0]

        # Plot Tx and Rx locations on mesh
        if plotIt:
            fig, ax = plt.subplots(1,1, figsize=(10,10))
            ax.plot(src_loc_Fz[0], src_loc_Fz[2], 'ro', ms=8)
            ax.plot(XYZ_CC[:,0], XYZ_CC[:,2], 'k.', ms=8)
            mesh.plotSlice(np.zeros(mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

        # Create survey and problem object
        survey = EM.FDEM.Survey(de_z)
        mapping = [('sigma', Maps.IdentityMap(mesh)), ('mu', Maps.IdentityMap(mesh))]
        problem = EM.FDEM.Problem3D_h(mesh, mapping=mapping)

        # Pair problem and survey
        problem.pair(survey)

        try:
            from pymatsolver import MumpsSolver
            problem.Solver = MumpsSolver
            print('solver set to Mumps')
        except ImportError, e:
            problem.Solver = SolverLU

        # Solve forward problem
        numFields_ElecDipole_Z = problem.fields(np.r_[SigmaBack, MuBack])

        # Get fields and fluxes
        # J lives on faces
        j_numF = numFields_ElecDipole_Z[de_z, 'j']
        j_numCC = mesh.aveF2CCV*j_numF

        # E lives on cell centres
        e_num = numFields_ElecDipole_Z[de_z, 'e']
        Rho = Utils.sdiag(1./SigmaBack)
        Rho = sp.block_diag([Rho, Rho, Rho])
        e_numTest = Rho*mesh.aveF2CCV*j_numF

        # H lives on edges
        h_numE = numFields_ElecDipole_Z[de_z, 'h']
        h_numCC = mesh.aveE2CCV*h_numE

        # B lives on cell centers
        b_num = numFields_ElecDipole_Z[de_z, 'b']
        MuBack_E = (mu_0*(1 + kappa))*np.ones((mesh.nE))
        Mu = Utils.sdiag(MuBack_E)
        b_numTest = Mu*h_numE

        # Interpolate numeric fields and fluxes to cell cetres for easy comparison with analytics
        ex_num, ey_num, ez_num = Pccx*e_num, Pccy*e_num, Pccz*e_num
        ex_numTest, ey_numTest, ez_numTest = Pccx*e_numTest, Pccy*e_numTest, Pccz*e_numTest

        jx_num, jy_num, jz_num = Pfx*j_numF, Pfy*j_numF, Pfz*j_numF
        jx_numTest, jy_numTest, jz_numTest = Pccx*j_numCC, Pccy*j_numCC, Pccz*j_numCC

        hx_num, hy_num, hz_num = Pex*h_numE, Pey*h_numE, Pez*h_numE
        hx_numTest, hy_numTest, hz_numTest = Pccx*h_numCC, Pccy*h_numCC, Pccz*h_numCC

        bx_num, by_num, bz_num = Pccx*b_num, Pccy*b_num, Pccz*b_num
        bx_numTest, by_numTest, bz_numTest = Pex*b_numTest, Pey*b_numTest, Pez*b_numTest

        # Check E values computed from fields object
        tol_fieldObjCheck = 1e-14
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  E_x:', np.linalg.norm(ex_num), np.linalg.norm(ex_numTest), np.linalg.norm(ex_num-ex_numTest), np.linalg.norm(ex_num-ex_numTest)/np.linalg.norm(ex_numTest))
        print('  E_y:', np.linalg.norm(ey_num), np.linalg.norm(ey_numTest), np.linalg.norm(ey_num-ey_numTest), np.linalg.norm(ey_num-ey_numTest)/np.linalg.norm(ey_numTest))
        print('  E_z:', np.linalg.norm(ez_num), np.linalg.norm(ez_numTest), np.linalg.norm(ez_num-ez_numTest), np.linalg.norm(ez_num-ez_numTest)/np.linalg.norm(ez_numTest))
        print('')
        self.assertTrue(np.linalg.norm(ex_num-ex_numTest)/np.linalg.norm(ex_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ex field do not agree.')
        self.assertTrue(np.linalg.norm(ey_num-ey_numTest)/np.linalg.norm(ey_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ey field do not agree.')
        self.assertTrue(np.linalg.norm(ez_num-ez_numTest)/np.linalg.norm(ez_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ez field do not agree.')

        # Check J values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  J_x:', np.linalg.norm(jx_num), np.linalg.norm(jx_numTest), np.linalg.norm(jx_num-jx_numTest), np.linalg.norm(jx_num-jx_numTest)/np.linalg.norm(jx_numTest))
        print('  J_y:', np.linalg.norm(jy_num), np.linalg.norm(jy_numTest), np.linalg.norm(jy_num-jy_numTest), np.linalg.norm(jy_num-jy_numTest)/np.linalg.norm(jy_numTest))
        print('  J_z:', np.linalg.norm(jz_num), np.linalg.norm(jz_numTest), np.linalg.norm(jz_num-jz_numTest), np.linalg.norm(jz_num-jz_numTest)/np.linalg.norm(jz_numTest))
        print('')
        self.assertTrue(np.linalg.norm(jx_num-jx_numTest)/np.linalg.norm(jx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jx field do not agree.')
        self.assertTrue(np.linalg.norm(jy_num-jy_numTest)/np.linalg.norm(jy_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jy field do not agree.')
        self.assertTrue(np.linalg.norm(jz_num-jz_numTest)/np.linalg.norm(jz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jz field do not agree.')

        # Check H values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  H_x:', np.linalg.norm(hx_num), np.linalg.norm(hx_numTest), np.linalg.norm(hx_num-hx_numTest), np.linalg.norm(hx_num-hx_numTest)/np.linalg.norm(hx_numTest))
        print('  H_y:', np.linalg.norm(hy_num), np.linalg.norm(hy_numTest), np.linalg.norm(hy_num-hy_numTest), np.linalg.norm(hy_num-hy_numTest)/np.linalg.norm(hy_numTest))
        print('  H_z:', np.linalg.norm(hz_num), np.linalg.norm(hz_numTest), np.linalg.norm(hz_num-hz_numTest), np.linalg.norm(hz_num-hz_numTest)/np.linalg.norm(hz_numTest))
        print('')
        self.assertTrue(np.linalg.norm(hx_num-hx_numTest)/np.linalg.norm(hx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hx field do not agree.')
        self.assertTrue(np.linalg.norm(hy_num-hy_numTest)/np.linalg.norm(hy_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hy field do not agree.')
        self.assertTrue(np.linalg.norm(hz_num-hz_numTest)/np.linalg.norm(hz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hz field do not agree.')

        # Check B values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  B_x:', np.linalg.norm(bx_num), np.linalg.norm(bx_numTest), np.linalg.norm(bx_num-bx_numTest), np.linalg.norm(bx_num-bx_numTest)/np.linalg.norm(bx_numTest))
        print('  B_y:', np.linalg.norm(by_num), np.linalg.norm(by_numTest), np.linalg.norm(by_num-by_numTest), np.linalg.norm(by_num-by_numTest)/np.linalg.norm(by_numTest))
        print('  B_z:', np.linalg.norm(bz_num), np.linalg.norm(bz_numTest), np.linalg.norm(bz_num-bz_numTest), np.linalg.norm(bz_num-bz_numTest)/np.linalg.norm(bz_numTest))
        print('')
        self.assertTrue(np.linalg.norm(bx_num-bx_numTest)/np.linalg.norm(bx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Bx field do not agree.')
        self.assertTrue(np.linalg.norm(by_num-by_numTest)/np.linalg.norm(by_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric By field do not agree.')
        self.assertTrue(np.linalg.norm(bz_num-bz_numTest)/np.linalg.norm(bz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Bz field do not agree.')

        # get analytic solutions
        exa, eya, eza = EM.Analytics.FDEMDipolarfields.E_from_ElectricDipoleWholeSpace(XYZ_CC, src_loc_Fz, sigmaback, Utils.mkvc(np.array(freq)),orientation='Z',kappa= kappa)
        exa, eya, eza = Utils.mkvc(exa, 2), Utils.mkvc(eya, 2), Utils.mkvc(eza, 2)

        jxa, jya, jza = EM.Analytics.FDEMDipolarfields.J_from_ElectricDipoleWholeSpace(XYZ_CC, src_loc_Fz, sigmaback, Utils.mkvc(np.array(freq)),orientation='Z',kappa= kappa)
        jxa, jya, jza = Utils.mkvc(jxa, 2), Utils.mkvc(jya, 2), Utils.mkvc(jza, 2)

        hxa, hya, hza = EM.Analytics.FDEMDipolarfields.H_from_ElectricDipoleWholeSpace(XYZ_CC, src_loc_Fz, sigmaback, Utils.mkvc(np.array(freq)),orientation='Z',kappa= kappa)
        hxa, hya, hza = Utils.mkvc(hxa, 2), Utils.mkvc(hya, 2), Utils.mkvc(hza, 2)

        bxa, bya, bza = EM.Analytics.FDEMDipolarfields.B_from_ElectricDipoleWholeSpace(XYZ_CC, src_loc_Fz, sigmaback, Utils.mkvc(np.array(freq)),orientation='Z',kappa= kappa)
        bxa, bya, bza = Utils.mkvc(bxa, 2), Utils.mkvc(bya, 2), Utils.mkvc(bza, 2)

        print ' comp,       anayltic,       numeric,       num - ana,       (num - ana)/ana'
        print '  E_x:', np.linalg.norm(exa), np.linalg.norm(ex_num), np.linalg.norm(exa-ex_num), np.linalg.norm(exa-ex_num)/np.linalg.norm(exa)
        print '  E_y:', np.linalg.norm(eya), np.linalg.norm(ey_num), np.linalg.norm(eya-ey_num), np.linalg.norm(eya-ey_num)/np.linalg.norm(eya)
        print '  E_z:', np.linalg.norm(eza), np.linalg.norm(ez_num), np.linalg.norm(eza-ez_num), np.linalg.norm(eza-ez_num)/np.linalg.norm(eza)
        print ''
        print '  J_x:', np.linalg.norm(jxa), np.linalg.norm(jx_num), np.linalg.norm(jxa-jx_num), np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa)
        print '  J_y:', np.linalg.norm(jya), np.linalg.norm(jy_num), np.linalg.norm(jya-jy_num), np.linalg.norm(jya-jy_num)/np.linalg.norm(jya)
        print '  J_z:', np.linalg.norm(jza), np.linalg.norm(jz_num), np.linalg.norm(jza-jz_num), np.linalg.norm(jza-jz_num)/np.linalg.norm(jza)
        print ''
        print '  H_x:', np.linalg.norm(hxa), np.linalg.norm(hx_num), np.linalg.norm(hxa-hx_num), np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa)
        print '  H_y:', np.linalg.norm(hya), np.linalg.norm(hy_num), np.linalg.norm(hya-hy_num), np.linalg.norm(hya-hy_num)/np.linalg.norm(hya)
        print '  H_z:', np.linalg.norm(hza), np.linalg.norm(hz_num), np.linalg.norm(hza-hz_num), np.linalg.norm(hza-hz_num)/np.linalg.norm(hza)
        print ''
        print '  B_x:', np.linalg.norm(bxa), np.linalg.norm(bx_num), np.linalg.norm(bxa-bx_num), np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa)
        print '  B_y:', np.linalg.norm(bya), np.linalg.norm(by_num), np.linalg.norm(bya-by_num), np.linalg.norm(bya-by_num)/np.linalg.norm(bya)
        print '  B_z:', np.linalg.norm(bza), np.linalg.norm(bz_num), np.linalg.norm(bza-bz_num), np.linalg.norm(bza-bz_num)/np.linalg.norm(bza)
        print ''
        self.assertTrue(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for Ex do not agree.')
        self.assertTrue(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for Ey do not agree.')
        self.assertTrue(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for Ez do not agree.')

        self.assertTrue(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for Jx do not agree.')
        self.assertTrue(np.linalg.norm(jya-jy_num)/np.linalg.norm(jya) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for Jy do not agree.')
        self.assertTrue(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for Jz do not agree.')

        self.assertTrue(np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for Hx do not agree.')
        self.assertTrue(np.linalg.norm(hya-hy_num)/np.linalg.norm(hya) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for Hy do not agree.')
        self.assertTrue(np.linalg.norm(hza-hz_num) < tol_NumErrZero, msg='Analytic and numeric solutions for Hz do not agree.')

        self.assertTrue(np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for Bx do not agree.')
        self.assertTrue(np.linalg.norm(bya-by_num)/np.linalg.norm(bya) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for By do not agree.')
        self.assertTrue(np.linalg.norm(bza-bz_num) < tol_NumErrZero, msg='Analytic and numeric solutions for Bz do not agree.')

        # Plot Tx and Rx locations on mesh
        if plotIt:
            # Plot E
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(x, ex_num.real, 'o', x, ex_numTest.real, 'd', x, exa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(x, ex_num.imag, 'o', x, ex_numTest.imag, 'd', x, exa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(x, ey_num.real, 'o', x, ey_numTest.real, 'd', x, eya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(x, ey_num.imag, 'o', x, ey_numTest.imag, 'd', x, eya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(x, ez_num.real, 'o', x, ez_numTest.real, 'd', x, eza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(x, ez_num.imag, 'o', x, ez_numTest.imag, 'd', x, eza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot J
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(x, jx_num.real, 'o', x, jx_numTest.real, 'd', x, jxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(x, jx_num.imag, 'o', x, jx_numTest.imag, 'd', x, jxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(x, jy_num.real, 'o', x, jy_numTest.real, 'd', x, jya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(x, jy_num.imag, 'o', x, jy_numTest.imag, 'd', x, jya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(x, jz_num.real, 'o', x, jz_numTest.real, 'd', x, jza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(x, jz_num.imag, 'o', x, jz_numTest.imag, 'd', x, jza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot H
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(x, hx_num.real, 'o', x, hx_numTest.real, 'd', x, hxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(x, hx_num.imag, 'o', x, hx_numTest.imag, 'd', x, hxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(x, hy_num.real, 'o', x, hy_numTest.real, 'd', x, hya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(x, hy_num.imag, 'o', x, hy_numTest.imag, 'd', x, hya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(x, hz_num.real, 'o', x, hz_numTest.real, 'd', x, hza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(x, hz_num.imag, 'o', x, hz_numTest.imag, 'd', x, hza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot B
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(x, bx_num.real, 'o', x, bx_numTest.real, 'd', x, bxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(x, bx_num.imag, 'o', x, bx_numTest.imag, 'd', x, bxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(x, by_num.real, 'o', x, by_numTest.real, 'd', x, bya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(x, by_num.imag, 'o', x, by_numTest.imag, 'd', x, bya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(x, bz_num.real, 'o', x, bz_numTest.real, 'd', x, bza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(x, bz_num.imag, 'o', x, bz_numTest.imag, 'd', x, bza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()


if __name__ == '__main__':
    unittest.main()
