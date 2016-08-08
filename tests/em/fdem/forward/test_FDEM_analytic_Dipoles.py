import unittest
from SimPEG import EM, Mesh, Utils, np, Maps, sp
try:
    from pymatsolver import MumpsSolver
    solver = MumpsSolver
except ImportError:
    from SimPEG import SolverLU
    solver = SolverLU
# import sys
from scipy.constants import mu_0

plotIt = False

if plotIt:
    import matplotlib.pylab
    import matplotlib.pyplot as plt

class FDEM_analytic_DipoleTests_CylMesh(unittest.TestCase):

    def setUp(self):

        # Define model parameters
        self.sigmaback = 1.
        # mu = mu_0*(1+kappa)
        self.kappa = 1.

        # Set source parameters
        self.freq = 1.
        self.src_loc = np.r_[0., 0., 0.]

        # Compute skin depth
        skdpth = 500. / np.sqrt(self.sigmaback * self.freq)

        # Create cylindrical mesh
        self.csx, self.ncx, self.npadx = 5, 50, 25
        self.csz, self.ncz, self.npadz = 5, 50, 25
        hx = Utils.meshTensor([(self.csx, self.ncx), (self.csx, self.npadx, 1.3)])
        hz = Utils.meshTensor([(self.csz, self.npadz, -1.3), (self.csz, self.ncz), (self.csz, self.npadz, 1.3)])
        self.mesh = Mesh.CylMesh([hx, 1, hz], [0., 0., -hz.sum()/2])

        # make sure mesh is big enough
        # checkZ = self.mesh.hz.sum() > skdpth*2.
        # checkX = self.mesh.hx.sum() > skdpth*2.
        # print checkX,checkZ
        # self.assertTrue(mesh.hz.sum() > skdpth*2.)
        # self.assertTrue(mesh.hx.sum() > skdpth*2.)

        # Create wholespace models
        self.SigmaBack = self.sigmaback*np.ones((self.mesh.nC))
        self.MuBack = (mu_0*(1 + self.kappa))*np.ones((self.mesh.nC))

        # Choose where to measure
        rlim = [20., 500.]
        self.r = self.mesh.vectorCCx[np.argmin(np.abs(self.mesh.vectorCCx-rlim[0])):np.argmin(np.abs(self.mesh.vectorCCx-rlim[1]))]
        z = 100.

        self.XYZ = Utils.ndgrid(self.r, np.r_[0.], np.r_[z])
        XYZ_CCInd = Utils.closestPoints(self.mesh, self.XYZ, 'CC')
        self.XYZ_CC = self.mesh.gridCC[XYZ_CCInd,:]

        # Form data interpolation matrices
        self.Pcc = self.mesh.getInterpolationMat(self.XYZ_CC, 'CC')
        Zero = sp.csr_matrix(self.Pcc.shape)
        self.Pccx, self.Pccz = sp.hstack([self.Pcc, Zero]), sp.hstack([Zero, self.Pcc])

        self.Pey = self.mesh.getInterpolationMat(self.XYZ_CC, 'Ey')
        self.Pfx, self.Pfz = self.mesh.getInterpolationMat(self.XYZ_CC, 'Fx'), self.mesh.getInterpolationMat(self.XYZ_CC, 'Fz')

    def test_CylMesh_ElecDipoleTest_Z(self):
        print('Testing various componemts of the field and fluxes from a Z-oriented analytic harmonic electric dipole against a numerical solution on a cylindrical mesh.')

        tol_ElecDipole_Z = 1e-2
        tol_NumErrZero = 1e-16

        # Define the source
        # Search over z-faces to find face nearest src_loc then add nFx to get to global face index [nFx][nFy = 0][nFz]
        s_ind = Utils.closestPoints(self.mesh, self.src_loc, 'Fz') + self.mesh.nFx
        de = np.zeros(self.mesh.nF, dtype=complex)
        de[s_ind] = 1./self.csz
        de_z = [EM.FDEM.Src.RawVec_e([], self.freq, de/self.mesh.area)]

        # src_loc_Fz = mesh.gridFz[Utils.closestPoints(mesh, src_loc, 'Fz'),:]
        # src_loc_Fz = src_loc_Fz[0]

        # Pair the problem and survey
        survey = EM.FDEM.Survey(de_z)

        mapping = [('sigma', Maps.IdentityMap(self.mesh)), ('mu', Maps.IdentityMap(self.mesh))]

        problem = EM.FDEM.Problem3D_h(self.mesh, mapping=mapping)

        # pair problem and survey
        problem.pair(survey)

        problem.Solver = solver

        # solve
        numFields_ElecDipole_Z = problem.fields(np.r_[self.SigmaBack, self.MuBack])


        # Temporary fudge factor
        fudge = 1.5

        # Get fields and fluxes
        # J lives on faces
        j_numF = numFields_ElecDipole_Z[de_z, 'j']
        j_numCC = self.mesh.aveF2CCV*j_numF

        # E lives on cell centres
        e_num = numFields_ElecDipole_Z[de_z, 'e']/fudge
        Rho = Utils.sdiag(1./self.SigmaBack)
        Rho = sp.block_diag([Rho, Rho])
        e_numTest = Rho*self.mesh.aveF2CCV*j_numF

        # H lives on edges
        h_numE = numFields_ElecDipole_Z[de_z, 'h']
        h_numCC = self.mesh.aveE2CCV*h_numE

        # B lives on cell centers
        b_num = numFields_ElecDipole_Z[de_z, 'b']/fudge
        MuBack_E = (mu_0*(1 + self.kappa))*np.ones((self.mesh.nE))
        Mu = Utils.sdiag(MuBack_E)
        b_numTest = Mu*h_numE

        # Interpolate numeric fields and fluxes to cell centres for easy comparison with analytics
        ex_num, ez_num = self.Pccx*e_num, self.Pccz*e_num
        ey_num = np.zeros_like(ex_num)
        ex_numTest, ez_numTest = self.Pccx*e_numTest, self.Pccz*e_numTest
        ey_numTest = np.zeros_like(ex_numTest)

        jx_num, jz_num = self.Pfx*j_numF, self.Pfz*j_numF
        jy_num = np.zeros_like(jx_num)
        jx_numTest, jz_numTest = self.Pccx*j_numCC, self.Pccz*j_numCC
        jy_numTest = np.zeros_like(jx_numTest)

        # Since we are evaluating along the plane y=0 the h_theta == h_y in cartesian coordiantes
        htheta_num = self.Pey*h_numE
        hx_num = np.zeros_like(htheta_num)
        hz_num = np.zeros_like(htheta_num)

        htheta_numTest = self.Pcc*h_numCC
        hx_numTest = np.zeros_like(htheta_num)
        hz_numTest = np.zeros_like(htheta_num)

        btheta_num = self.Pcc*b_num
        bx_num = np.zeros_like(btheta_num)
        bz_num = np.zeros_like(btheta_num)

        btheta_numTest = self.Pey*b_numTest
        bx_numTest = np.zeros_like(btheta_numTest)
        bz_numTest = np.zeros_like(btheta_numTest)

        # Check E values computed from fields object
        tol_fieldObjCheck = 1e-8
        print' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual'
        print'  E_x:', np.linalg.norm(ex_num), np.linalg.norm(ex_numTest), np.linalg.norm(ex_num-ex_numTest), np.linalg.norm(ex_num-ex_numTest)/np.linalg.norm(ex_numTest)
        print'  E_y:', np.linalg.norm(ey_num), np.linalg.norm(ey_numTest), np.linalg.norm(ey_num-ey_numTest)
        print'  E_z:', np.linalg.norm(ez_num), np.linalg.norm(ez_numTest), np.linalg.norm(ez_num-ez_numTest), np.linalg.norm(ez_num-ez_numTest)/np.linalg.norm(ez_numTest)
        print''
        # self.assertTrue(np.linalg.norm(ex_num-ex_numTest)/np.linalg.norm(ex_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ex field do not agree.')
        # self.assertTrue(np.linalg.norm(ey_num-ey_numTest) < tol_NumErrZero, msg='The two ways of calculating the numeric Ey field do not agree.')
        # self.assertTrue(np.linalg.norm(ez_num-ez_numTest)/np.linalg.norm(ez_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ez field do not agree.')

        # Check J values computed from fields object
        print' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual'
        print'  J_x:', np.linalg.norm(jx_num), np.linalg.norm(jx_numTest), np.linalg.norm(jx_num-jx_numTest), np.linalg.norm(jx_num-jx_numTest)/np.linalg.norm(jx_numTest)
        print'  J_y:', np.linalg.norm(jy_num), np.linalg.norm(jy_numTest), np.linalg.norm(jy_num-jy_numTest)
        print'  J_z:', np.linalg.norm(jz_num), np.linalg.norm(jz_numTest), np.linalg.norm(jz_num-jz_numTest), np.linalg.norm(jz_num-jz_numTest)/np.linalg.norm(jz_numTest)
        print''
        # self.assertTrue(np.linalg.norm(jx_num-jx_numTest)/np.linalg.norm(jx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jx field do not agree.')
        # self.assertTrue(np.linalg.norm(jy_num-jy_numTest) < tol_NumErrZero, msg='The two ways of calculating the numeric Jy field do not agree.')
        # self.assertTrue(np.linalg.norm(jz_num-jz_numTest)/np.linalg.norm(jz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jz field do not agree.')

        # Check H values computed from fields object
        print' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual'
        print'  H_x:', np.linalg.norm(hx_num), np.linalg.norm(hx_numTest), np.linalg.norm(hx_num-hx_numTest)
        print'  H_y:', np.linalg.norm(htheta_num), np.linalg.norm(htheta_numTest), np.linalg.norm(htheta_num-htheta_numTest), np.linalg.norm(htheta_num-htheta_numTest)/np.linalg.norm(htheta_numTest)
        print'  H_z:', np.linalg.norm(hz_num), np.linalg.norm(hz_numTest), np.linalg.norm(hz_num-hz_numTest)
        print ''
        # self.assertTrue(np.linalg.norm(hx_num-hx_numTest) < tol_NumErrZero, msg='The two ways of calculating the numeric Hx field do not agree.')
        # self.assertTrue(np.linalg.norm(htheta_num-htheta_numTest)/np.linalg.norm(htheta_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hy field do not agree.')
        # self.assertTrue(np.linalg.norm(hz_num-hz_numTest) < tol_NumErrZero, msg='The two ways of calculating the numeric Hz field do not agree.')

        # Check B values computed from fields object
        print' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual'
        print'  B_x:', np.linalg.norm(bx_num), np.linalg.norm(bx_numTest), np.linalg.norm(bx_num-bx_numTest)
        print'  B_y:', np.linalg.norm(btheta_num), np.linalg.norm(btheta_numTest), np.linalg.norm(btheta_num-btheta_numTest), np.linalg.norm(btheta_num-btheta_numTest)/np.linalg.norm(btheta_numTest)
        print'  B_z:', np.linalg.norm(bz_num), np.linalg.norm(bz_numTest), np.linalg.norm(bz_num-bz_numTest)
        print''
        # self.assertTrue(np.linalg.norm(bx_num-bx_numTest) < tol_NumErrZero, msg='The two ways of calculating the numeric Bx field do not agree.')
        # self.assertTrue(np.linalg.norm(by_num-by_numTest)/np.linalg.norm(by_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric By field do not agree.')
        # self.assertTrue(np.linalg.norm(bz_num-bz_numTest) < tol_NumErrZero, msg='The two ways of calculating the numeric Bz field do not agree.')

        # get analytic solutions
        exa, eya, eza = EM.Analytics.FDEMDipolarfields.E_from_ElectricDipoleWholeSpace(self.XYZ_CC, self.src_loc, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        exa, eya, eza = Utils.mkvc(exa, 2), Utils.mkvc(eya, 2), Utils.mkvc(eza, 2)

        jxa, jya, jza = EM.Analytics.FDEMDipolarfields.J_from_ElectricDipoleWholeSpace(self.XYZ_CC, self.src_loc, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        jxa, jya, jza = Utils.mkvc(jxa, 2), Utils.mkvc(jya, 2), Utils.mkvc(jza, 2)

        hxa, hya, hza = EM.Analytics.FDEMDipolarfields.H_from_ElectricDipoleWholeSpace(self.XYZ_CC, self.src_loc, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        hxa, hya, hza = Utils.mkvc(hxa, 2), Utils.mkvc(hya, 2), Utils.mkvc(hza, 2)

        bxa, bya, bza = EM.Analytics.FDEMDipolarfields.B_from_ElectricDipoleWholeSpace(self.XYZ_CC, self.src_loc, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        bxa, bya, bza = Utils.mkvc(bxa, 2), Utils.mkvc(bya, 2), Utils.mkvc(bza, 2)

        print ' comp,       anayltic,       numeric,       num - ana,       (num - ana)/ana'
        # print '  E_x:', np.linalg.norm(exa), np.linalg.norm(ex_num), np.linalg.norm(exa-ex_num), np.linalg.norm(exa-ex_num)/np.linalg.norm(exa)
        print '  E_x:', np.linalg.norm(exa), np.linalg.norm(ex_numTest), np.linalg.norm(exa-ex_numTest), np.linalg.norm(exa-ex_numTest)/np.linalg.norm(exa)
        # print '  E_y:', np.linalg.norm(eya), np.linalg.norm(ey_num), np.linalg.norm(eya-ey_num), np.linalg.norm(eya-ey_num)/np.linalg.norm(eya)
        print '  E_y:', np.linalg.norm(eya), np.linalg.norm(ey_numTest), np.linalg.norm(eya-ey_numTest)
        # print '  E_z:', np.linalg.norm(eza), np.linalg.norm(ez_num), np.linalg.norm(eza-ez_num), np.linalg.norm(eza-ez_num)/np.linalg.norm(eza)
        print '  E_z:', np.linalg.norm(eza), np.linalg.norm(ez_numTest), np.linalg.norm(eza-ez_numTest), np.linalg.norm(eza-ez_numTest)/np.linalg.norm(eza)
        print ''
        print '  J_x:', np.linalg.norm(jxa), np.linalg.norm(jx_num), np.linalg.norm(jxa-jx_num), np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa)
        # print '  J_x:', np.linalg.norm(jxa), np.linalg.norm(jx_numTest), np.linalg.norm(jxa-jx_numTest), np.linalg.norm(jxa-jx_numTest)/np.linalg.norm(jxa)
        print '  J_y:', np.linalg.norm(jya), np.linalg.norm(jy_num), np.linalg.norm(jya-jy_num)
        # print '  J_y:', np.linalg.norm(jya), np.linalg.norm(jy_numTest), np.linalg.norm(jya-jy_numTest), np.linalg.norm(eya-ey_numTest)/np.linalg.norm(jya)
        print '  J_z:', np.linalg.norm(jza), np.linalg.norm(jz_num), np.linalg.norm(jza-jz_num), np.linalg.norm(jza-jz_num)/np.linalg.norm(jza)
        # print '  J_z:', np.linalg.norm(jza), np.linalg.norm(jz_numTest), np.linalg.norm(jza-jz_numTest), np.linalg.norm(jza-jz_numTest)/np.linalg.norm(jza)
        print ''
        print '  H_x:', np.linalg.norm(hxa), np.linalg.norm(hx_num), np.linalg.norm(hxa-hx_num)
        # print '  H_x:', np.linalg.norm(hxa), np.linalg.norm(hx_numTest), np.linalg.norm(hxa-hx_numTest), np.linalg.norm(hxa-hx_numTest)/np.linalg.norm(hxa)
        print '  H_y:', np.linalg.norm(hya), np.linalg.norm(htheta_num), np.linalg.norm(hya-htheta_num), np.linalg.norm(hya-htheta_num)/np.linalg.norm(hya)
        # print '  H_y:', np.linalg.norm(hya), np.linalg.norm(htheta_numTest), np.linalg.norm(hya-htheta_numTest), np.linalg.norm(hya-htheta_numTest)/np.linalg.norm(hya)
        print '  H_z:', np.linalg.norm(hza), np.linalg.norm(hz_num), np.linalg.norm(hza-hz_num)
        # print '  H_z:', np.linalg.norm(hza), np.linalg.norm(hz_numTest), np.linalg.norm(hza-hz_numTest), np.linalg.norm(hza-hz_numTest)/np.linalg.norm(hza)
        print ''
        # print '  B_x:', np.linalg.norm(bxa), np.linalg.norm(bx_num), np.linalg.norm(bxa-bx_num), np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa)
        print '  B_x:', np.linalg.norm(bxa), np.linalg.norm(bx_numTest), np.linalg.norm(bxa-bx_numTest)
        # print '  B_y:', np.linalg.norm(bya), np.linalg.norm(btheta_num), np.linalg.norm(bya-btheta_num), np.linalg.norm(bya-btheta_num)/np.linalg.norm(bya)
        print '  B_y:', np.linalg.norm(bya), np.linalg.norm(btheta_numTest), np.linalg.norm(bya-btheta_numTest), np.linalg.norm(bya-btheta_numTest)/np.linalg.norm(bya)
        # print '  B_z:', np.linalg.norm(bza), np.linalg.norm(bz_num), np.linalg.norm(bza-bz_num), np.linalg.norm(bza-bz_num)/np.linalg.norm(bza)
        print '  B_z:', np.linalg.norm(bza), np.linalg.norm(bz_numTest), np.linalg.norm(bza-bz_numTest)
        print ''

        self.assertTrue(np.linalg.norm(exa-ex_numTest)/np.linalg.norm(exa) < tol_ElecDipole_Z)
        self.assertTrue(np.linalg.norm(eya-ey_numTest) < tol_NumErrZero)
        self.assertTrue(np.linalg.norm(eza-ez_numTest)/np.linalg.norm(eza) < tol_ElecDipole_Z)

        self.assertTrue(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa) < tol_ElecDipole_Z)
        self.assertTrue(np.linalg.norm(jya-jy_num) < tol_NumErrZero)
        self.assertTrue(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza) < tol_ElecDipole_Z)

        self.assertTrue(np.linalg.norm(hxa-hx_num) < tol_NumErrZero)
        self.assertTrue(np.linalg.norm(hya-htheta_num)/np.linalg.norm(hya) < tol_ElecDipole_Z)
        self.assertTrue(np.linalg.norm(hza-hz_num) < tol_NumErrZero)

        self.assertTrue(np.linalg.norm(bxa-bx_num) < tol_NumErrZero)
        self.assertTrue(np.linalg.norm(bya-btheta_numTest)/np.linalg.norm(bya) < tol_ElecDipole_Z)
        self.assertTrue(np.linalg.norm(bza-bz_num) < tol_NumErrZero)

        if plotIt:

            # Plot E
            fig, ax = plt.subplots(2,3, figsize=(20,10))
            plt.subplot(231)
            plt.plot(self.r, ex_num.real, 'o', self.r, ex_numTest.real, 'd', self.r, exa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_x Real')
            plt.xlabel('r (m)')

            plt.subplot(234)
            plt.plot(self.r, ex_num.imag, 'o', self.r, ex_numTest.imag, 'd', self.r, exa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_x Imag')
            plt.xlabel('r (m)')

            plt.subplot(232)
            plt.plot(self.r, ey_num.real, 'o', self.r, ey_numTest.real, 'd', self.r, eya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_y Real')
            plt.xlabel('r (m)')

            plt.subplot(235)
            plt.plot(self.r, ey_num.imag, 'o', self.r, ey_numTest.imag, 'd', self.r, eya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_y Imag')
            plt.xlabel('r (m)')

            plt.subplot(233)
            plt.plot(self.r, ez_num.real, 'o', self.r, ez_numTest.real, 'd', self.r, eza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_z Real')
            plt.xlabel('r (m)')

            plt.subplot(236)
            plt.plot(self.r, ez_num.imag, 'o', self.r, ez_numTest.imag, 'd', self.r, eza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('r (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot J
            fig, ax = plt.subplots(2,3, figsize=(20,10))
            plt.subplot(231)
            plt.plot(self.r, jx_num.real, 'o', self.r, jx_numTest.real, 'd', self.r, jxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_x Real')
            plt.xlabel('r (m)')

            plt.subplot(234)
            plt.plot(self.r, jx_num.imag, 'o', self.r, jx_numTest.imag, 'd', self.r, jxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_x Imag')
            plt.xlabel('r (m)')

            plt.subplot(232)
            plt.plot(self.r, jy_num.real, 'o', self.r, jy_numTest.real, 'd', self.r, jya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_y Real')
            plt.xlabel('r (m)')

            plt.subplot(235)
            plt.plot(self.r, jy_num.imag, 'o', self.r, jy_numTest.imag, 'd', self.r, jya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_y Imag')
            plt.xlabel('r (m)')

            plt.subplot(233)
            plt.plot(self.r, jz_num.real, 'o', self.r, jz_numTest.real, 'd', self.r, jza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_z Real')
            plt.xlabel('r (m)')

            plt.subplot(236)
            plt.plot(self.r, jz_num.imag, 'o', self.r, jz_numTest.imag, 'd', self.r, jza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('r (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot H
            fig, ax = plt.subplots(2,3, figsize=(20,10))
            plt.subplot(231)
            plt.plot(self.r, hx_num.real, 'o', self.r, hx_numTest.real, 'd', self.r, hxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_x Real')
            plt.xlabel('r (m)')

            plt.subplot(234)
            plt.plot(self.r, hx_num.imag, 'o', self.r, hx_numTest.imag, 'd', self.r, hxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_x Imag')
            plt.xlabel('r (m)')

            plt.subplot(232)
            plt.plot(self.r, htheta_num.real, 'o', self.r, htheta_numTest.real, 'd', self.r, hya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_y Real')
            plt.xlabel('r (m)')
            plt.subplot(235)
            plt.plot(self.r, htheta_num.imag, 'o', self.r, htheta_numTest.imag, 'd', self.r, hya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_y Imag')
            plt.xlabel('r (m)')

            plt.subplot(233)
            plt.plot(self.r, hz_num.real, 'o', self.r, hz_numTest.real, 'd', self.r, hza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_z Real')
            plt.xlabel('r (m)')

            plt.subplot(236)
            plt.plot(self.r, hz_num.imag, 'o', self.r, hz_numTest.imag, 'd', self.r, hza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('r (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

             # Plot B
            fig, ax = plt.subplots(2,3, figsize=(20,10))
            plt.subplot(231)
            plt.plot(self.r, bx_num.real, 'o', self.r, bx_numTest.real, 'd', self.r, bxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_x Real')
            plt.xlabel('r (m)')

            plt.subplot(234)
            plt.plot(self.r, bx_num.imag, 'o', self.r, bx_numTest.imag, 'd', self.r, bxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_x Imag')
            plt.xlabel('r (m)')

            plt.subplot(232)
            plt.plot(self.r, btheta_num.real, 'o', self.r, btheta_numTest.real, 'd', self.r, bya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_y Real')
            plt.xlabel('r (m)')

            plt.subplot(235)
            plt.plot(self.r, btheta_num.imag, 'o', self.r, btheta_numTest.imag, 'd', self.r, bya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_y Imag')
            plt.xlabel('r (m)')

            plt.subplot(233)
            plt.plot(self.r, bz_num.real, 'o', self.r, bz_numTest.real, 'd', self.r, bza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_z Real')
            plt.xlabel('r (m)')

            plt.subplot(236)
            plt.plot(self.r, bz_num.imag, 'o', self.r, bz_numTest.imag, 'd', self.r, bza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('r (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()


class FDEM_analytic_DipoleTests_3DMesh(unittest.TestCase):

    def setUp(self):

        # Define model parameters
        self.sigmaback = 1.
        # mu = mu_0*(1+kappa)
        self.kappa = 1.

        # Create 3D mesh
        self.csx, self.ncx, self.npadx = 5, 20, 7
        self.csy, self.ncy, self.npady = 5, 20, 7
        self.csz, self.ncz, self.npadz = 5, 25, 5
        self.hx = Utils.meshTensor([(self.csx, self.npadx, -1.3), (self.csx, self.ncx), (self.csx, self.npadx, 1.3)])
        self.hy = Utils.meshTensor([(self.csy, self.npady, -1.3), (self.csy, self.ncy), (self.csy, self.npady, 1.3)])
        self.hz = Utils.meshTensor([(self.csz, self.npadz, -1.3), (self.csz, self.ncz), (self.csz, self.npadz, 1.3)])
        self.mesh = Mesh.TensorMesh([self.hx, self.hy, self.hz], 'CCC')

        # Set source parameters
        self.freq = 100.
        self.src_loc = np.r_[0., 0., -35]
        src_loc_CCInd = Utils.closestPoints(self.mesh, self.src_loc, 'CC')
        self.src_loc_CC = self.mesh.gridCC[src_loc_CCInd,:]
        self.src_loc_CC = self.src_loc_CC[0]

        # Compute skin depth
        skdpth = 500. / np.sqrt(self.sigmaback * self.freq)

        # make sure mesh is big enough
        self.assertTrue(self.mesh.hx.sum() > skdpth*2.)
        self.assertTrue(self.mesh.hy.sum() > skdpth*2.)
        self.assertTrue(self.mesh.hz.sum() > skdpth*2.)

        # Create wholespace models
        self.SigmaBack = self.sigmaback*np.ones((self.mesh.nC))
        self.MuBack = (mu_0*(1 + self.kappa))*np.ones((self.mesh.nC))

        # Define reciever locations
        xlim = 50. # x locations from -50 to 50
        xInd = np.where(np.abs(self.mesh.vectorCCx) < xlim)
        self.x = self.mesh.vectorCCx[xInd[0]]
        y = 10.
        z = 35.

        # # where we choose to measure
        self.XYZ = Utils.ndgrid(self.x, np.r_[y], np.r_[z])

        XYZ_CCInd = Utils.closestPoints(self.mesh, self.XYZ, 'CC')
        self.XYZ_CC = self.mesh.gridCC[XYZ_CCInd,:]

        # Form data interpolation matrices
        self.Pcc = self.mesh.getInterpolationMat(self.XYZ_CC, 'CC')
        Zero = sp.csr_matrix(self.Pcc.shape)
        self.Pccx, self.Pccy, self.Pccz = sp.hstack([self.Pcc, Zero, Zero]), sp.hstack([Zero, self.Pcc, Zero]), sp.hstack([Zero, Zero, self.Pcc])

        self.Pex, self.Pey, self.Pez = self.mesh.getInterpolationMat(self.XYZ_CC, 'Ex'), self.mesh.getInterpolationMat(self.XYZ_CC, 'Ey'), self.mesh.getInterpolationMat(self.XYZ_CC, 'Ez')
        self.Pfx, self.Pfy, self.Pfz = self.mesh.getInterpolationMat(self.XYZ_CC, 'Fx'), self.mesh.getInterpolationMat(self.XYZ_CC, 'Fy'), self.mesh.getInterpolationMat(self.XYZ_CC, 'Fz')

    def test_3DMesh_ElecDipoleTest_X(self):
        print('Testing various componemts of the fields and fluxes from a X-oriented analytic harmonic electric dipole against a numerical solution on a 3D tesnsor mesh.')

        tol_ElecDipole_X = 3e-2
        tol_NumErrZero = 1e-16

        # Define the source
        # Search over x-faces to find face nearest src_loc
        s_ind = Utils.closestPoints(self.mesh, self.src_loc_CC, 'Fx')
        de = np.zeros(self.mesh.nF, dtype=complex)
        de[s_ind] = 1./self.csx
        de_x = [EM.FDEM.Src.RawVec_e([], self.freq, de/self.mesh.area)]

        src_loc_Fx = self.mesh.gridFx[s_ind,:]
        src_loc_Fx = src_loc_Fx[0]

        # Plot Tx and Rx locations on mesh
        if plotIt:
            fig, ax = plt.subplots(1,1, figsize=(10,10))
            ax.plot(src_loc_Fx[0], src_loc_Fx[2], 'ro', ms=8)
            ax.plot(self.XYZ_CC[:,0], self.XYZ_CC[:,2], 'k.', ms=8)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

        # Create survey and problem object
        survey = EM.FDEM.Survey(de_x)
        mapping = [('sigma', Maps.IdentityMap(self.mesh)), ('mu', Maps.IdentityMap(self.mesh))]
        problem = EM.FDEM.Problem3D_h(self.mesh, mapping=mapping)

        # Pair problem and survey
        problem.pair(survey)

        try:
            from pymatsolver import MumpsSolver
            problem.Solver = MumpsSolver
            print('solver set to Mumps')
        except ImportError, e:
            problem.Solver = SolverLU

        # Solve forward problem
        numFields_ElecDipole_X = problem.fields(np.r_[self.SigmaBack, self.MuBack])

        # Get fields and fluxes
        # J lives on faces
        j_numF = numFields_ElecDipole_X[de_x, 'j']
        j_numCC = self.mesh.aveF2CCV*j_numF

        # E lives on cell centres
        e_num = numFields_ElecDipole_X[de_x, 'e']
        Rho = Utils.sdiag(1./self.SigmaBack)
        Rho = sp.block_diag([Rho, Rho, Rho])
        e_numTest = Rho*self.mesh.aveF2CCV*j_numF

        # H lives on edges
        h_numE = numFields_ElecDipole_X[de_x, 'h']
        h_numCC = self.mesh.aveE2CCV*h_numE

        # B lives on cell centers
        b_num = numFields_ElecDipole_X[de_x, 'b']
        MuBack_E = (mu_0*(1 + self.kappa))*np.ones((self.mesh.nE))
        Mu = Utils.sdiag(MuBack_E)
        b_numTest = Mu*h_numE

        # Interpolate numeric fields and fluxes to cell cetres for easy comparison with analytics
        ex_num, ey_num, ez_num = self.Pccx*e_num, self.Pccy*e_num, self.Pccz*e_num
        ex_numTest, ey_numTest, ez_numTest = self.Pccx*e_numTest, self.Pccy*e_numTest, self.Pccz*e_numTest

        jx_num, jy_num, jz_num = self.Pfx*j_numF, self.Pfy*j_numF, self.Pfz*j_numF
        jx_numTest, jy_numTest, jz_numTest = self.Pccx*j_numCC, self.Pccy*j_numCC, self.Pccz*j_numCC

        hx_num, hy_num, hz_num = self.Pex*h_numE, self.Pey*h_numE, self.Pez*h_numE
        hx_numTest, hy_numTest, hz_numTest = self.Pccx*h_numCC, self.Pccy*h_numCC, self.Pccz*h_numCC

        bx_num, by_num, bz_num = self.Pccx*b_num, self.Pccy*b_num, self.Pccz*b_num
        bx_numTest, by_numTest, bz_numTest = self.Pex*b_numTest, self.Pey*b_numTest, self.Pez*b_numTest

        # Check E values computed from fields object
        tol_fieldObjCheck = 1e-14
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  E_x:', np.linalg.norm(ex_num), np.linalg.norm(ex_numTest), np.linalg.norm(ex_num-ex_numTest), np.linalg.norm(ex_num-ex_numTest)/np.linalg.norm(ex_numTest))
        print('  E_y:', np.linalg.norm(ey_num), np.linalg.norm(ey_numTest), np.linalg.norm(ey_num-ey_numTest), np.linalg.norm(ey_num-ey_numTest)/np.linalg.norm(ey_numTest))
        print('  E_z:', np.linalg.norm(ez_num), np.linalg.norm(ez_numTest), np.linalg.norm(ez_num-ez_numTest), np.linalg.norm(ez_num-ez_numTest)/np.linalg.norm(ez_numTest))
        print('')
        # self.assertTrue(np.linalg.norm(ex_num-ex_numTest)/np.linalg.norm(ex_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ex field do not agree.')
        # self.assertTrue(np.linalg.norm(ey_num-ey_numTest)/np.linalg.norm(ey_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ey field do not agree.')
        # self.assertTrue(np.linalg.norm(ez_num-ez_numTest)/np.linalg.norm(ez_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ez field do not agree.')

        # Check J values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  J_x:', np.linalg.norm(jx_num), np.linalg.norm(jx_numTest), np.linalg.norm(jx_num-jx_numTest), np.linalg.norm(jx_num-jx_numTest)/np.linalg.norm(jx_numTest))
        print('  J_y:', np.linalg.norm(jy_num), np.linalg.norm(jy_numTest), np.linalg.norm(jy_num-jy_numTest), np.linalg.norm(jy_num-jy_numTest)/np.linalg.norm(jy_numTest))
        print('  J_z:', np.linalg.norm(jz_num), np.linalg.norm(jz_numTest), np.linalg.norm(jz_num-jz_numTest), np.linalg.norm(jz_num-jz_numTest)/np.linalg.norm(jz_numTest))
        print('')
        # self.assertTrue(np.linalg.norm(jx_num-jx_numTest)/np.linalg.norm(jx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jx field do not agree.')
        # self.assertTrue(np.linalg.norm(jy_num-jy_numTest)/np.linalg.norm(jy_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jy field do not agree.')
        # self.assertTrue(np.linalg.norm(jz_num-jz_numTest)/np.linalg.norm(jz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jz field do not agree.')

        # Check H values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  H_x:', np.linalg.norm(hx_num), np.linalg.norm(hx_numTest), np.linalg.norm(hx_num-hx_numTest), np.linalg.norm(hx_num-hx_numTest)/np.linalg.norm(hx_numTest))
        print('  H_y:', np.linalg.norm(hy_num), np.linalg.norm(hy_numTest), np.linalg.norm(hy_num-hy_numTest), np.linalg.norm(hy_num-hy_numTest)/np.linalg.norm(hy_numTest))
        print('  H_z:', np.linalg.norm(hz_num), np.linalg.norm(hz_numTest), np.linalg.norm(hz_num-hz_numTest), np.linalg.norm(hz_num-hz_numTest)/np.linalg.norm(hz_numTest))
        print('')
        # self.assertTrue(np.linalg.norm(hx_num-hx_numTest)/np.linalg.norm(hx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hx field do not agree.')
        # self.assertTrue(np.linalg.norm(hy_num-hy_numTest)/np.linalg.norm(hy_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hy field do not agree.')
        # self.assertTrue(np.linalg.norm(hz_num-hz_numTest)/np.linalg.norm(hz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hz field do not agree.')

        # Check B values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  B_x:', np.linalg.norm(bx_num), np.linalg.norm(bx_numTest), np.linalg.norm(bx_num-bx_numTest), np.linalg.norm(bx_num-bx_numTest)/np.linalg.norm(bx_numTest))
        print('  B_y:', np.linalg.norm(by_num), np.linalg.norm(by_numTest), np.linalg.norm(by_num-by_numTest), np.linalg.norm(by_num-by_numTest)/np.linalg.norm(by_numTest))
        print('  B_z:', np.linalg.norm(bz_num), np.linalg.norm(bz_numTest), np.linalg.norm(bz_num-bz_numTest), np.linalg.norm(bz_num-bz_numTest)/np.linalg.norm(bz_numTest))
        print('')
        # self.assertTrue(np.linalg.norm(bx_num-bx_numTest)/np.linalg.norm(bx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Bx field do not agree.')
        # self.assertTrue(np.linalg.norm(by_num-by_numTest)/np.linalg.norm(by_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric By field do not agree.')
        # self.assertTrue(np.linalg.norm(bz_num-bz_numTest)/np.linalg.norm(bz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Bz field do not agree.')

        # get analytic solutions
        exa, eya, eza = EM.Analytics.FDEMDipolarfields.E_from_ElectricDipoleWholeSpace(self.XYZ_CC, src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
        exa, eya, eza = Utils.mkvc(exa, 2), Utils.mkvc(eya, 2), Utils.mkvc(eza, 2)

        jxa, jya, jza = EM.Analytics.FDEMDipolarfields.J_from_ElectricDipoleWholeSpace(self.XYZ_CC, src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
        jxa, jya, jza = Utils.mkvc(jxa, 2), Utils.mkvc(jya, 2), Utils.mkvc(jza, 2)

        hxa, hya, hza = EM.Analytics.FDEMDipolarfields.H_from_ElectricDipoleWholeSpace(self.XYZ_CC, src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
        hxa, hya, hza = Utils.mkvc(hxa, 2), Utils.mkvc(hya, 2), Utils.mkvc(hza, 2)

        bxa, bya, bza = EM.Analytics.FDEMDipolarfields.B_from_ElectricDipoleWholeSpace(self.XYZ_CC, src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
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
        print '  H_x:', np.linalg.norm(hxa), np.linalg.norm(hx_num), np.linalg.norm(hxa-hx_num)
        print '  H_y:', np.linalg.norm(hya), np.linalg.norm(hy_num), np.linalg.norm(hya-hy_num), np.linalg.norm(hya-hy_num)/np.linalg.norm(hya)
        print '  H_z:', np.linalg.norm(hza), np.linalg.norm(hz_num), np.linalg.norm(hza-hz_num), np.linalg.norm(hza-hz_num)/np.linalg.norm(hza)
        print ''
        print '  B_x:', np.linalg.norm(bxa), np.linalg.norm(bx_num), np.linalg.norm(bxa-bx_num)
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
            plt.plot(self.x, ex_num.real, 'o', self.x, ex_numTest.real, 'd', self.x, exa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(self.x, ex_num.imag, 'o', self.x, ex_numTest.imag, 'd', self.x, exa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(self.x, ey_num.real, 'o', self.x, ey_numTest.real, 'd', self.x, eya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(self.x, ey_num.imag, 'o', self.x, ey_numTest.imag, 'd', self.x, eya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(self.x, ez_num.real, 'o', self.x, ez_numTest.real, 'd', self.x, eza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(self.x, ez_num.imag, 'o', self.x, ez_numTest.imag, 'd', self.x, eza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot J
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(self.x, jx_num.real, 'o', self.x, jx_numTest.real, 'd', self.x, jxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(self.x, jx_num.imag, 'o', self.x, jx_numTest.imag, 'd', self.x, jxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(self.x, jy_num.real, 'o', self.x, jy_numTest.real, 'd', self.x, jya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(self.x, jy_num.imag, 'o', self.x, jy_numTest.imag, 'd', self.x, jya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(self.x, jz_num.real, 'o', self.x, jz_numTest.real, 'd', self.x, jza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(self.x, jz_num.imag, 'o', self.x, jz_numTest.imag, 'd', self.x, jza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot H
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(self.x, hx_num.real, 'o', self.x, hx_numTest.real, 'd', self.x, hxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(self.x, hx_num.imag, 'o', self.x, hx_numTest.imag, 'd', self.x, hxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(self.x, hy_num.real, 'o', self.x, hy_numTest.real, 'd', self.x, hya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(self.x, hy_num.imag, 'o', self.x, hy_numTest.imag, 'd', self.x, hya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(self.x, hz_num.real, 'o', self.x, hz_numTest.real, 'd', self.x, hza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(self.x, hz_num.imag, 'o', self.x, hz_numTest.imag, 'd', self.x, hza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot B
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(self.x, bx_num.real, 'o', self.x, bx_numTest.real, 'd', self.x, bxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(self.x, bx_num.imag, 'o', self.x, bx_numTest.imag, 'd', self.x, bxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(self.x, by_num.real, 'o', self.x, by_numTest.real, 'd', self.x, bya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(self.x, by_num.imag, 'o', self.x, by_numTest.imag, 'd', self.x, bya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(self.x, bz_num.real, 'o', self.x, bz_numTest.real, 'd', self.x, bza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(self.x, bz_num.imag, 'o', self.x, bz_numTest.imag, 'd', self.x, bza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()


    def test_3DMesh_ElecDipoleTest_Y(self):
        print('Testing various componemts of the fields and fluxes from a Y-oriented analytic harmonic electric dipole against a numerical solution on a 3D tesnsor mesh.')

        tol_ElecDipole_Y = 4e-2
        tol_NumErrZero = 1e-16

        # Define the source
        # Search over y-faces to find face nearest src_loc
        s_ind = Utils.closestPoints(self.mesh, self.src_loc_CC, 'Fy') + self.mesh.nFx
        de = np.zeros(self.mesh.nF, dtype=complex)
        de[s_ind] = 1./self.csy
        de_y = [EM.FDEM.Src.RawVec_e([], self.freq, de/self.mesh.area)]

        src_loc_Fy = self.mesh.gridFy[Utils.closestPoints(self.mesh, self.src_loc_CC, 'Fy'),:]
        src_loc_Fy = src_loc_Fy[0]

        # Plot Tx and Rx locations on mesh
        if plotIt:
            fig, ax = plt.subplots(1,1, figsize=(10,10))
            ax.plot(src_loc_Fy[0], src_loc_Fy[2], 'ro', ms=8)
            ax.plot(self.XYZ_CC[:,0], self.XYZ_CC[:,2], 'k.', ms=8)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

        # Create survey and problem object
        survey = EM.FDEM.Survey(de_y)
        mapping = [('sigma', Maps.IdentityMap(self.mesh)), ('mu', Maps.IdentityMap(self.mesh))]
        problem = EM.FDEM.Problem3D_h(self.mesh, mapping=mapping)

        # Pair problem and survey
        problem.pair(survey)

        try:
            from pymatsolver import MumpsSolver
            problem.Solver = MumpsSolver
            print('solver set to Mumps')
        except ImportError, e:
            problem.Solver = SolverLU

        # Solve forward problem
        numFields_ElecDipole_Y = problem.fields(np.r_[self.SigmaBack, self.MuBack])

        # Get fields and fluxes
        # J lives on faces
        j_numF = numFields_ElecDipole_Y[de_y, 'j']
        j_numCC = self.mesh.aveF2CCV*j_numF

        # E lives on cell centres
        e_num = numFields_ElecDipole_Y[de_y, 'e']
        Rho = Utils.sdiag(1./self.SigmaBack)
        Rho = sp.block_diag([Rho, Rho, Rho])
        e_numTest = Rho*self.mesh.aveF2CCV*j_numF

        # H lives on edges
        h_numE = numFields_ElecDipole_Y[de_y, 'h']
        h_numCC = self.mesh.aveE2CCV*h_numE

        # B lives on cell centers
        b_num = numFields_ElecDipole_Y[de_y, 'b']
        MuBack_E = (mu_0*(1 + self.kappa))*np.ones(self.mesh.nE)
        Mu = Utils.sdiag(MuBack_E)
        b_numTest = Mu*h_numE

        # Interpolate numeric fields and fluxes to cell cetres for easy comparison with analytics
        ex_num, ey_num, ez_num = self.Pccx*e_num, self.Pccy*e_num, self.Pccz*e_num
        ex_numTest, ey_numTest, ez_numTest = self.Pccx*e_numTest, self.Pccy*e_numTest, self.Pccz*e_numTest

        jx_num, jy_num, jz_num = self.Pfx*j_numF, self.Pfy*j_numF, self.Pfz*j_numF
        jx_numTest, jy_numTest, jz_numTest = self.Pccx*j_numCC, self.Pccy*j_numCC, self.Pccz*j_numCC

        hx_num, hy_num, hz_num = self.Pex*h_numE, self.Pey*h_numE, self.Pez*h_numE
        hx_numTest, hy_numTest, hz_numTest = self.Pccx*h_numCC, self.Pccy*h_numCC, self.Pccz*h_numCC

        bx_num, by_num, bz_num = self.Pccx*b_num, self.Pccy*b_num, self.Pccz*b_num
        bx_numTest, by_numTest, bz_numTest = self.Pex*b_numTest, self.Pey*b_numTest, self.Pez*b_numTest

        # Check E values computed from fields object
        tol_fieldObjCheck = 1e-14
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  E_x:', np.linalg.norm(ex_num), np.linalg.norm(ex_numTest), np.linalg.norm(ex_num-ex_numTest), np.linalg.norm(ex_num-ex_numTest)/np.linalg.norm(ex_numTest))
        print('  E_y:', np.linalg.norm(ey_num), np.linalg.norm(ey_numTest), np.linalg.norm(ey_num-ey_numTest), np.linalg.norm(ey_num-ey_numTest)/np.linalg.norm(ey_numTest))
        print('  E_z:', np.linalg.norm(ez_num), np.linalg.norm(ez_numTest), np.linalg.norm(ez_num-ez_numTest), np.linalg.norm(ez_num-ez_numTest)/np.linalg.norm(ez_numTest))
        print('')
        # self.assertTrue(np.linalg.norm(ex_num-ex_numTest)/np.linalg.norm(ex_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ex field do not agree.')
        # self.assertTrue(np.linalg.norm(ey_num-ey_numTest)/np.linalg.norm(ey_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ey field do not agree.')
        # self.assertTrue(np.linalg.norm(ez_num-ez_numTest)/np.linalg.norm(ez_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ez field do not agree.')

        # Check J values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  J_x:', np.linalg.norm(jx_num), np.linalg.norm(jx_numTest), np.linalg.norm(jx_num-jx_numTest), np.linalg.norm(jx_num-jx_numTest)/np.linalg.norm(jx_numTest))
        print('  J_y:', np.linalg.norm(jy_num), np.linalg.norm(jy_numTest), np.linalg.norm(jy_num-jy_numTest), np.linalg.norm(jy_num-jy_numTest)/np.linalg.norm(jy_numTest))
        print('  J_z:', np.linalg.norm(jz_num), np.linalg.norm(jz_numTest), np.linalg.norm(jz_num-jz_numTest), np.linalg.norm(jz_num-jz_numTest)/np.linalg.norm(jz_numTest))
        print('')
        # self.assertTrue(np.linalg.norm(jx_num-jx_numTest)/np.linalg.norm(jx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jx field do not agree.')
        # self.assertTrue(np.linalg.norm(jy_num-jy_numTest)/np.linalg.norm(jy_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jy field do not agree.')
        # self.assertTrue(np.linalg.norm(jz_num-jz_numTest)/np.linalg.norm(jz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jz field do not agree.')

        # Check H values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  H_x:', np.linalg.norm(hx_num), np.linalg.norm(hx_numTest), np.linalg.norm(hx_num-hx_numTest), np.linalg.norm(hx_num-hx_numTest)/np.linalg.norm(hx_numTest))
        print('  H_y:', np.linalg.norm(hy_num), np.linalg.norm(hy_numTest), np.linalg.norm(hy_num-hy_numTest), np.linalg.norm(hy_num-hy_numTest)/np.linalg.norm(hy_numTest))
        print('  H_z:', np.linalg.norm(hz_num), np.linalg.norm(hz_numTest), np.linalg.norm(hz_num-hz_numTest), np.linalg.norm(hz_num-hz_numTest)/np.linalg.norm(hz_numTest))
        print('')
        # self.assertTrue(np.linalg.norm(hx_num-hx_numTest)/np.linalg.norm(hx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hx field do not agree.')
        # self.assertTrue(np.linalg.norm(hy_num-hy_numTest)/np.linalg.norm(hy_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hy field do not agree.')
        # self.assertTrue(np.linalg.norm(hz_num-hz_numTest)/np.linalg.norm(hz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hz field do not agree.')

        # Check B values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  B_x:', np.linalg.norm(bx_num), np.linalg.norm(bx_numTest), np.linalg.norm(bx_num-bx_numTest), np.linalg.norm(bx_num-bx_numTest)/np.linalg.norm(bx_numTest))
        print('  B_y:', np.linalg.norm(by_num), np.linalg.norm(by_numTest), np.linalg.norm(by_num-by_numTest), np.linalg.norm(by_num-by_numTest)/np.linalg.norm(by_numTest))
        print('  B_z:', np.linalg.norm(bz_num), np.linalg.norm(bz_numTest), np.linalg.norm(bz_num-bz_numTest), np.linalg.norm(bz_num-bz_numTest)/np.linalg.norm(bz_numTest))
        print('')
        # self.assertTrue(np.linalg.norm(bx_num-bx_numTest)/np.linalg.norm(bx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Bx field do not agree.')
        # self.assertTrue(np.linalg.norm(by_num-by_numTest)/np.linalg.norm(by_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric By field do not agree.')
        # self.assertTrue(np.linalg.norm(bz_num-bz_numTest)/np.linalg.norm(bz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Bz field do not agree.')


        # get analytic solutions
        exa, eya, eza = EM.Analytics.FDEMDipolarfields.E_from_ElectricDipoleWholeSpace(self.XYZ_CC, src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
        exa, eya, eza = Utils.mkvc(exa, 2), Utils.mkvc(eya, 2), Utils.mkvc(eza, 2)

        jxa, jya, jza = EM.Analytics.FDEMDipolarfields.J_from_ElectricDipoleWholeSpace(self.XYZ_CC, src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
        jxa, jya, jza = Utils.mkvc(jxa, 2), Utils.mkvc(jya, 2), Utils.mkvc(jza, 2)

        hxa, hya, hza = EM.Analytics.FDEMDipolarfields.H_from_ElectricDipoleWholeSpace(self.XYZ_CC, src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
        hxa, hya, hza = Utils.mkvc(hxa, 2), Utils.mkvc(hya, 2), Utils.mkvc(hza, 2)

        bxa, bya, bza = EM.Analytics.FDEMDipolarfields.B_from_ElectricDipoleWholeSpace(self.XYZ_CC, src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
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
        print '  H_y:', np.linalg.norm(hya), np.linalg.norm(hy_num), np.linalg.norm(hya-hy_num)
        print '  H_z:', np.linalg.norm(hza), np.linalg.norm(hz_num), np.linalg.norm(hza-hz_num), np.linalg.norm(hza-hz_num)/np.linalg.norm(hza)
        print ''
        print '  B_x:', np.linalg.norm(bxa), np.linalg.norm(bx_num), np.linalg.norm(bxa-bx_num), np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa)
        print '  B_y:', np.linalg.norm(bya), np.linalg.norm(by_num), np.linalg.norm(bya-by_num)
        print '  B_z:', np.linalg.norm(bza), np.linalg.norm(bz_num), np.linalg.norm(bza-bz_num), np.linalg.norm(bza-bz_num)/np.linalg.norm(bza)
        print ''
        self.assertTrue(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Ex do not agree.')
        self.assertTrue(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Ey do not agree.')
        self.assertTrue(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Ez do not agree.')

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
            plt.plot(self.x, ex_num.real, 'o', self.x, ex_numTest.real, 'd', self.x, exa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(self.x, ex_num.imag, 'o', self.x, ex_numTest.imag, 'd', self.x, exa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(self.x, ey_num.real, 'o', self.x, ey_numTest.real, 'd', self.x, eya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(self.x, ey_num.imag, 'o', self.x, ey_numTest.imag, 'd', self.x, eya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(self.x, ez_num.real, 'o', self.x, ez_numTest.real, 'd', self.x, eza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(self.x, ez_num.imag, 'o', self.x, ez_numTest.imag, 'd', self.x, eza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot J
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(self.x, jx_num.real, 'o', self.x, jx_numTest.real, 'd', self.x, jxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(self.x, jx_num.imag, 'o', self.x, jx_numTest.imag, 'd', self.x, jxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(self.x, jy_num.real, 'o', self.x, jy_numTest.real, 'd', self.x, jya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(self.x, jy_num.imag, 'o', self.x, jy_numTest.imag, 'd', self.x, jya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(self.x, jz_num.real, 'o', self.x, jz_numTest.real, 'd', self.x, jza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(self.x, jz_num.imag, 'o', self.x, jz_numTest.imag, 'd', self.x, jza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot H
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(self.x, hx_num.real, 'o', self.x, hx_numTest.real, 'd', self.x, hxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(self.x, hx_num.imag, 'o', self.x, hx_numTest.imag, 'd', self.x, hxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(self.x, hy_num.real, 'o', self.x, hy_numTest.real, 'd', self.x, hya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(self.x, hy_num.imag, 'o', self.x, hy_numTest.imag, 'd', self.x, hya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(self.x, hz_num.real, 'o', self.x, hz_numTest.real, 'd', self.x, hza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(self.x, hz_num.imag, 'o', self.x, hz_numTest.imag, 'd', self.x, hza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot B
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(self.x, bx_num.real, 'o', self.x, bx_numTest.real, 'd', self.x, bxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(self.x, bx_num.imag, 'o', self.x, bx_numTest.imag, 'd', self.x, bxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(self.x, by_num.real, 'o', self.x, by_numTest.real, 'd', self.x, bya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(self.x, by_num.imag, 'o', self.x, by_numTest.imag, 'd', self.x, bya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(self.x, bz_num.real, 'o', self.x, bz_numTest.real, 'd', self.x, bza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(self.x, bz_num.imag, 'o', self.x, bz_numTest.imag, 'd', self.x, bza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

    def test_3DMesh_ElecDipoleTest_Z(self):
        print('Testing various componemts of the fields and fluxes from a Z-oriented analytic harmonic electric dipole against a numerical solution on a 3D tesnsor mesh.')

        tol_ElecDipole_Z = 4e-2
        tol_NumErrZero = 1e-16

        # Define the source
        # Search over x-faces to find face nearest src_loc
        s_ind = Utils.closestPoints(self.mesh, self.src_loc_CC, 'Fz') + self.mesh.nFx + self.mesh.nFy
        de = np.zeros(self.mesh.nF, dtype=complex)
        de[s_ind] = 1./self.csz
        de_z = [EM.FDEM.Src.RawVec_e([], self.freq, de/self.mesh.area)]

        src_loc_Fz = self.mesh.gridFz[Utils.closestPoints(self.mesh, self.src_loc_CC, 'Fz'),:]
        src_loc_Fz = src_loc_Fz[0]

        # Plot Tx and Rx locations on mesh
        if plotIt:
            fig, ax = plt.subplots(1,1, figsize=(10,10))
            ax.plot(src_loc_Fz[0], src_loc_Fz[2], 'ro', ms=8)
            ax.plot(self.XYZ_CC[:,0], self.XYZ_CC[:,2], 'k.', ms=8)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

        # Create survey and problem object
        survey = EM.FDEM.Survey(de_z)
        mapping = [('sigma', Maps.IdentityMap(self.mesh)), ('mu', Maps.IdentityMap(self.mesh))]
        problem = EM.FDEM.Problem3D_h(self.mesh, mapping=mapping)

        # Pair problem and survey
        problem.pair(survey)

        try:
            from pymatsolver import MumpsSolver
            problem.Solver = MumpsSolver
            print('solver set to Mumps')
        except ImportError, e:
            problem.Solver = SolverLU

        # Solve forward problem
        numFields_ElecDipole_Z = problem.fields(np.r_[self.SigmaBack, self.MuBack])

        # Get fields and fluxes
        # J lives on faces
        j_numF = numFields_ElecDipole_Z[de_z, 'j']
        j_numCC = self.mesh.aveF2CCV*j_numF

        # E lives on cell centres
        e_num = numFields_ElecDipole_Z[de_z, 'e']
        Rho = Utils.sdiag(1./self.SigmaBack)
        Rho = sp.block_diag([Rho, Rho, Rho])
        e_numTest = Rho*self.mesh.aveF2CCV*j_numF

        # H lives on edges
        h_numE = numFields_ElecDipole_Z[de_z, 'h']
        h_numCC = self.mesh.aveE2CCV*h_numE

        # B lives on cell centers
        b_num = numFields_ElecDipole_Z[de_z, 'b']
        MuBack_E = (mu_0*(1 + self.kappa))*np.ones((self.mesh.nE))
        Mu = Utils.sdiag(MuBack_E)
        b_numTest = Mu*h_numE

        # Interpolate numeric fields and fluxes to cell cetres for easy comparison with analytics
        ex_num, ey_num, ez_num = self.Pccx*e_num, self.Pccy*e_num, self.Pccz*e_num
        ex_numTest, ey_numTest, ez_numTest = self.Pccx*e_numTest, self.Pccy*e_numTest, self.Pccz*e_numTest

        jx_num, jy_num, jz_num = self.Pfx*j_numF, self.Pfy*j_numF, self.Pfz*j_numF
        jx_numTest, jy_numTest, jz_numTest = self.Pccx*j_numCC, self.Pccy*j_numCC, self.Pccz*j_numCC

        hx_num, hy_num, hz_num = self.Pex*h_numE, self.Pey*h_numE, self.Pez*h_numE
        hx_numTest, hy_numTest, hz_numTest = self.Pccx*h_numCC, self.Pccy*h_numCC, self.Pccz*h_numCC

        bx_num, by_num, bz_num = self.Pccx*b_num, self.Pccy*b_num, self.Pccz*b_num
        bx_numTest, by_numTest, bz_numTest = self.Pex*b_numTest, self.Pey*b_numTest, self.Pez*b_numTest

        # Check E values computed from fields object
        tol_fieldObjCheck = 1e-14
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  E_x:', np.linalg.norm(ex_num), np.linalg.norm(ex_numTest), np.linalg.norm(ex_num-ex_numTest), np.linalg.norm(ex_num-ex_numTest)/np.linalg.norm(ex_numTest))
        print('  E_y:', np.linalg.norm(ey_num), np.linalg.norm(ey_numTest), np.linalg.norm(ey_num-ey_numTest), np.linalg.norm(ey_num-ey_numTest)/np.linalg.norm(ey_numTest))
        print('  E_z:', np.linalg.norm(ez_num), np.linalg.norm(ez_numTest), np.linalg.norm(ez_num-ez_numTest), np.linalg.norm(ez_num-ez_numTest)/np.linalg.norm(ez_numTest))
        print('')
        # self.assertTrue(np.linalg.norm(ex_num-ex_numTest)/np.linalg.norm(ex_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ex field do not agree.')
        # self.assertTrue(np.linalg.norm(ey_num-ey_numTest)/np.linalg.norm(ey_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ey field do not agree.')
        # self.assertTrue(np.linalg.norm(ez_num-ez_numTest)/np.linalg.norm(ez_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Ez field do not agree.')

        # Check J values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  J_x:', np.linalg.norm(jx_num), np.linalg.norm(jx_numTest), np.linalg.norm(jx_num-jx_numTest), np.linalg.norm(jx_num-jx_numTest)/np.linalg.norm(jx_numTest))
        print('  J_y:', np.linalg.norm(jy_num), np.linalg.norm(jy_numTest), np.linalg.norm(jy_num-jy_numTest), np.linalg.norm(jy_num-jy_numTest)/np.linalg.norm(jy_numTest))
        print('  J_z:', np.linalg.norm(jz_num), np.linalg.norm(jz_numTest), np.linalg.norm(jz_num-jz_numTest), np.linalg.norm(jz_num-jz_numTest)/np.linalg.norm(jz_numTest))
        print('')
        # self.assertTrue(np.linalg.norm(jx_num-jx_numTest)/np.linalg.norm(jx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jx field do not agree.')
        # self.assertTrue(np.linalg.norm(jy_num-jy_numTest)/np.linalg.norm(jy_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jy field do not agree.')
        # self.assertTrue(np.linalg.norm(jz_num-jz_numTest)/np.linalg.norm(jz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Jz field do not agree.')

        # Check H values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  H_x:', np.linalg.norm(hx_num), np.linalg.norm(hx_numTest), np.linalg.norm(hx_num-hx_numTest), np.linalg.norm(hx_num-hx_numTest)/np.linalg.norm(hx_numTest))
        print('  H_y:', np.linalg.norm(hy_num), np.linalg.norm(hy_numTest), np.linalg.norm(hy_num-hy_numTest), np.linalg.norm(hy_num-hy_numTest)/np.linalg.norm(hy_numTest))
        print('  H_z:', np.linalg.norm(hz_num), np.linalg.norm(hz_numTest), np.linalg.norm(hz_num-hz_numTest), np.linalg.norm(hz_num-hz_numTest)/np.linalg.norm(hz_numTest))
        print('')
        # self.assertTrue(np.linalg.norm(hx_num-hx_numTest)/np.linalg.norm(hx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hx field do not agree.')
        # self.assertTrue(np.linalg.norm(hy_num-hy_numTest)/np.linalg.norm(hy_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hy field do not agree.')
        # self.assertTrue(np.linalg.norm(hz_num-hz_numTest)/np.linalg.norm(hz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Hz field do not agree.')

        # Check B values computed from fields object
        print(' comp,       fields obj,       manual,       fields - manual,       (fields - manual)/manual')
        print('  B_x:', np.linalg.norm(bx_num), np.linalg.norm(bx_numTest), np.linalg.norm(bx_num-bx_numTest), np.linalg.norm(bx_num-bx_numTest)/np.linalg.norm(bx_numTest))
        print('  B_y:', np.linalg.norm(by_num), np.linalg.norm(by_numTest), np.linalg.norm(by_num-by_numTest), np.linalg.norm(by_num-by_numTest)/np.linalg.norm(by_numTest))
        print('  B_z:', np.linalg.norm(bz_num), np.linalg.norm(bz_numTest), np.linalg.norm(bz_num-bz_numTest), np.linalg.norm(bz_num-bz_numTest)/np.linalg.norm(bz_numTest))
        print('')
        # self.assertTrue(np.linalg.norm(bx_num-bx_numTest)/np.linalg.norm(bx_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Bx field do not agree.')
        # self.assertTrue(np.linalg.norm(by_num-by_numTest)/np.linalg.norm(by_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric By field do not agree.')
        # self.assertTrue(np.linalg.norm(bz_num-bz_numTest)/np.linalg.norm(bz_numTest) < tol_fieldObjCheck, msg='The two ways of calculating the numeric Bz field do not agree.')

        # get analytic solutions
        exa, eya, eza = EM.Analytics.FDEMDipolarfields.E_from_ElectricDipoleWholeSpace(self.XYZ_CC, src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        exa, eya, eza = Utils.mkvc(exa, 2), Utils.mkvc(eya, 2), Utils.mkvc(eza, 2)

        jxa, jya, jza = EM.Analytics.FDEMDipolarfields.J_from_ElectricDipoleWholeSpace(self.XYZ_CC, src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        jxa, jya, jza = Utils.mkvc(jxa, 2), Utils.mkvc(jya, 2), Utils.mkvc(jza, 2)

        hxa, hya, hza = EM.Analytics.FDEMDipolarfields.H_from_ElectricDipoleWholeSpace(self.XYZ_CC, src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        hxa, hya, hza = Utils.mkvc(hxa, 2), Utils.mkvc(hya, 2), Utils.mkvc(hza, 2)

        bxa, bya, bza = EM.Analytics.FDEMDipolarfields.B_from_ElectricDipoleWholeSpace(self.XYZ_CC, src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
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
        print '  H_z:', np.linalg.norm(hza), np.linalg.norm(hz_num), np.linalg.norm(hza-hz_num)
        print ''
        print '  B_x:', np.linalg.norm(bxa), np.linalg.norm(bx_num), np.linalg.norm(bxa-bx_num), np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa)
        print '  B_y:', np.linalg.norm(bya), np.linalg.norm(by_num), np.linalg.norm(bya-by_num), np.linalg.norm(bya-by_num)/np.linalg.norm(bya)
        print '  B_z:', np.linalg.norm(bza), np.linalg.norm(bz_num), np.linalg.norm(bza-bz_num)
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
            plt.plot(self.x, ex_num.real, 'o', self.x, ex_numTest.real, 'd', self.x, exa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(self.x, ex_num.imag, 'o', self.x, ex_numTest.imag, 'd', self.x, exa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(self.x, ey_num.real, 'o', self.x, ey_numTest.real, 'd', self.x, eya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(self.x, ey_num.imag, 'o', self.x, ey_numTest.imag, 'd', self.x, eya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(self.x, ez_num.real, 'o', self.x, ez_numTest.real, 'd', self.x, eza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('E_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(self.x, ez_num.imag, 'o', self.x, ez_numTest.imag, 'd', self.x, eza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('E_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot J
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(self.x, jx_num.real, 'o', self.x, jx_numTest.real, 'd', self.x, jxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(self.x, jx_num.imag, 'o', self.x, jx_numTest.imag, 'd', self.x, jxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(self.x, jy_num.real, 'o', self.x, jy_numTest.real, 'd', self.x, jya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(self.x, jy_num.imag, 'o', self.x, jy_numTest.imag, 'd', self.x, jya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(self.x, jz_num.real, 'o', self.x, jz_numTest.real, 'd', self.x, jza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('J_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(self.x, jz_num.imag, 'o', self.x, jz_numTest.imag, 'd', self.x, jza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('J_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot H
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(self.x, hx_num.real, 'o', self.x, hx_numTest.real, 'd', self.x, hxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(self.x, hx_num.imag, 'o', self.x, hx_numTest.imag, 'd', self.x, hxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(self.x, hy_num.real, 'o', self.x, hy_numTest.real, 'd', self.x, hya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(self.x, hy_num.imag, 'o', self.x, hy_numTest.imag, 'd', self.x, hya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(self.x, hz_num.real, 'o', self.x, hz_numTest.real, 'd', self.x, hza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('H_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(self.x, hz_num.imag, 'o', self.x, hz_numTest.imag, 'd', self.x, hza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('H_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()

            # Plot B
            fig, ax = plt.subplots(2,3, figsize=(20,10))

            plt.subplot(231)
            plt.plot(self.x, bx_num.real, 'o', self.x, bx_numTest.real, 'd', self.x, bxa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_x Real')
            plt.xlabel('x (m)')

            plt.subplot(234)
            plt.plot(self.x, bx_num.imag, 'o', self.x, bx_numTest.imag, 'd', self.x, bxa.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_x Imag')
            plt.xlabel('x (m)')

            plt.subplot(232)
            plt.plot(self.x, by_num.real, 'o', self.x, by_numTest.real, 'd', self.x, bya.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_y Real')
            plt.xlabel('x (m)')

            plt.subplot(235)
            plt.plot(self.x, by_num.imag, 'o', self.x, by_numTest.imag, 'd', self.x, bya.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_y Imag')
            plt.xlabel('x (m)')

            plt.subplot(233)
            plt.plot(self.x, bz_num.real, 'o', self.x, bz_numTest.real, 'd', self.x, bza.real, linewidth=2)
            plt.grid(which='both')
            plt.title('B_z Real')
            plt.xlabel('x (m)')

            plt.subplot(236)
            plt.plot(self.x, bz_num.imag, 'o', self.x, bz_numTest.imag, 'd', self.x, bza.imag, linewidth=2)
            plt.grid(which='both')
            plt.title('B_z Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('x (m)')

            plt.legend(['Num', 'NumTest', 'Ana'],bbox_to_anchor=(1.5,0.5))
            plt.tight_layout()


if __name__ == '__main__':
    unittest.main()
