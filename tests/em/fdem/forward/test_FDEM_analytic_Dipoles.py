import unittest
from SimPEG import EM, Mesh, Utils, np, Maps, sp
try:
    from pymatsolver import PardisoSolver as Solver
except ImportError:
    from SimPEG import SolverLU as Solver
# import sys
from scipy.constants import mu_0

# Global Test Parameters
plotIt = False

SIGMABACK = 7e-2
KAPPA = 1

# Util functions
if plotIt:
    import matplotlib.pyplot as plt

    def plotLine_num_ana(ax, x, num, ana, title=None, xlabel='x (m)'):
        ax.plot(x, num, 'o', x, ana, linewidth=2)
        ax.grid(which='both')
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel('x (m)')
        return ax


def setUpMesh():
    csx, ncx, npadx = 5, 18, 8
    csy, ncy, npady = 5, 18, 8
    csz, ncz, npadz = 5, 18, 8
    hx = Utils.meshTensor([(csx, npadx, -1.3), (csx, ncx), (csx, npadx, 1.3)])
    hy = Utils.meshTensor([(csy, npady, -1.3), (csy, ncy), (csy, npady, 1.3)])
    hz = Utils.meshTensor([(csz, npadz, -1.3), (csz, ncz), (csz, npadz, 1.3)])

    return Mesh.TensorMesh([hx, hy, hz], 'CCC')



class X_ElecDipoleTest_3DMesh(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        print('Testing a X-oriented analytic harmonic electric dipole against the numerical solution on a 3D-tesnsor mesh.')

        # Define model parameters
        self.sigmaback = SIGMABACK
        # mu = mu_0*(1+kappa)
        self.kappa = KAPPA

        # Create 3D mesh
        self.mesh = setUpMesh()

        # Set source parameters
        self.freq = 500.
        src_loc = np.r_[0., 0., -35]
        src_loc_CCInd = Utils.closestPoints(self.mesh, src_loc, 'CC')
        self.src_loc_CC = self.mesh.gridCC[src_loc_CCInd,:]
        self.src_loc_CC = self.src_loc_CC[0]

        # Compute skin depth
        skdpth = 500. / np.sqrt(self.sigmaback * self.freq)

        # make sure mesh is big enough
        # print('skin depth =', skdpth)
        # self.assertTrue(self.mesh.hx.sum() > skdpth*2.)
        # self.assertTrue(self.mesh.hy.sum() > skdpth*2.)
        # self.assertTrue(self.mesh.hz.sum() > skdpth*2.)

        # Create wholespace models
        SigmaBack = self.sigmaback*np.ones((self.mesh.nC))
        MuBack = (mu_0*(1 + self.kappa))*np.ones((self.mesh.nC))

        # Define reciever locations
        xlim = 40. # x locations from -50 to 50
        xInd = np.where(np.abs(self.mesh.vectorCCx) < xlim)
        x = self.mesh.vectorCCx[xInd[0]]
        y = 10.
        z = 30.

        # where we choose to measure
        XYZ = Utils.ndgrid(x, np.r_[y], np.r_[z])

        # Cell centred recievers
        XYZ_CCInd = Utils.closestPoints(self.mesh, XYZ, 'CC')
        self.XYZ_CC = self.mesh.gridCC[XYZ_CCInd, :]

        # Edge recievers
        XYZ_ExInd = Utils.closestPoints(self.mesh, XYZ, 'Ex')
        self.XYZ_Ex = self.mesh.gridEx[XYZ_ExInd, :]
        XYZ_EyInd = Utils.closestPoints(self.mesh, XYZ, 'Ey')
        self.XYZ_Ey = self.mesh.gridEy[XYZ_EyInd, :]
        XYZ_EzInd = Utils.closestPoints(self.mesh, XYZ, 'Ez')
        self.XYZ_Ez = self.mesh.gridEz[XYZ_EzInd, :]

        # Face recievers
        XYZ_FxInd = Utils.closestPoints(self.mesh, XYZ, 'Fx')
        self.XYZ_Fx = self.mesh.gridFx[XYZ_FxInd, :]
        XYZ_FyInd = Utils.closestPoints(self.mesh, XYZ, 'Fy')
        self.XYZ_Fy = self.mesh.gridFy[XYZ_FyInd, :]
        XYZ_FzInd = Utils.closestPoints(self.mesh, XYZ, 'Fz')
        self.XYZ_Fz = self.mesh.gridFz[XYZ_FzInd, :]

        # Form data interpolation matrices
        Pcc = self.mesh.getInterpolationMat(self.XYZ_CC, 'CC')
        Zero = sp.csr_matrix(Pcc.shape)
        self.Pccx, self.Pccy, self.Pccz = sp.hstack([Pcc, Zero, Zero]), sp.hstack([Zero, Pcc, Zero]), sp.hstack([Zero, Zero, Pcc])

        self.Pex, self.Pey, self.Pez = self.mesh.getInterpolationMat(self.XYZ_Ex, 'Ex'), self.mesh.getInterpolationMat(self.XYZ_Ey, 'Ey'), self.mesh.getInterpolationMat(self.XYZ_Ez, 'Ez')
        self.Pfx, self.Pfy, self.Pfz = self.mesh.getInterpolationMat(self.XYZ_Fx, 'Fx'), self.mesh.getInterpolationMat(self.XYZ_Fy, 'Fy'), self.mesh.getInterpolationMat(self.XYZ_Fz, 'Fz')

        # Define the source
        # Search over x-faces to find face nearest src_loc
        s_ind = Utils.closestPoints(self.mesh, self.src_loc_CC, 'Fx')
        de = np.zeros(self.mesh.nF, dtype=complex)
        de[s_ind] = 1./self.mesh.hx.min()
        self.de_x = [EM.FDEM.Src.RawVec_e([], self.freq, de/self.mesh.area)]

        self.src_loc_Fx = self.mesh.gridFx[s_ind, :]
        self.src_loc_Fx = self.src_loc_Fx[0]

        # Create survey and problem object
        survey = EM.FDEM.Survey(self.de_x)
        mapping = [('sigma', Maps.IdentityMap(self.mesh)), ('mu', Maps.IdentityMap(self.mesh))]
        problem = EM.FDEM.Problem3D_h(self.mesh, mapping=mapping)

        # Pair problem and survey
        problem.pair(survey)
        problem.Solver = Solver

        # Solve forward problem
        self.numFields_ElecDipole_X = problem.fields(np.r_[SigmaBack, MuBack])

        # setUp_Done = True

    def test_3DMesh_X_ElecDipoleTest_E(self):
        print('Testing E components of a X-oriented analytic harmonic electric dipole.')

        # Specify toleraces
        tol_ElecDipole_X = 4e-2
        tol_NumErrZero = 1e-16

        # Get E which lives on cell centres
        e_numCC = self.numFields_ElecDipole_X[self.de_x, 'e']

        # Apply data projection matrix to get E at the reciever locations
        ex_num, ey_num, ez_num = self.Pccx*e_numCC, self.Pccy*e_numCC, self.Pccz*e_numCC

        # Get analytic solution
        exa, eya, eza = EM.Analytics.FDEMDipolarfields.E_from_ElectricDipoleWholeSpace(self.XYZ_CC, self.src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)), orientation='X', kappa= self.kappa)
        exa, eya, eza = Utils.mkvc(exa, 2), Utils.mkvc(eya, 2), Utils.mkvc(eza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('E_x:').rjust(4) , repr(np.linalg.norm(exa)).rjust(25), repr(np.linalg.norm(ex_num)).rjust(25), repr(np.linalg.norm(exa-ex_num)).rjust(25), repr(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa)).rjust(25), repr(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa) < tol_ElecDipole_X).center(12)
        print str('E_y:').rjust(4) , repr(np.linalg.norm(eya)).rjust(25), repr(np.linalg.norm(ey_num)).rjust(25), repr(np.linalg.norm(eya-ey_num)).rjust(25), repr(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya)).rjust(25), repr(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya) < tol_ElecDipole_X).center(12)
        print str('E_z:').rjust(4) , repr(np.linalg.norm(eza)).rjust(25), repr(np.linalg.norm(ez_num)).rjust(25), repr(np.linalg.norm(eza-ez_num)).rjust(25), repr(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza)).rjust(25), repr(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza) < tol_ElecDipole_X).center(12)
        print
        self.assertTrue(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Ex do not agree.')
        self.assertTrue(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Ey do not agree.')
        self.assertTrue(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Ez do not agree.')

        # Plot Tx and Rx locations on mesh
        if plotIt:

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fx[0], self.src_loc_Fx[2], 'ro', ms=8)
            ax.plot(self.XYZ_CC[:, 0], self.XYZ_CC[:, 2], 'k.', ms=8)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True,
                                normal="Y", ax = ax)

            # Plot E
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, ex_num.real, exa.real, title='E_x Real')
            plotLine_num_ana(ax[1], x, ex_num.imag, exa.imag, title='E_x Imag')

            plotLine_num_ana(ax[2], x, ey_num.real, eya.real, title='E_y Real')
            plotLine_num_ana(ax[3], x, ey_num.imag, eya.imag, title='E_y Imag')

            plotLine_num_ana(ax[4], x, ez_num.real, eza.real, title='E_z Real')
            plotLine_num_ana(ax[5], x, ez_num.imag, eza.imag, title='E_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

    def test_3DMesh_X_ElecDipoleTest_J(self):
        print('Testing J components of a X-oriented analytic harmonic electric dipole.')
        # Specify toleraces
        tol_ElecDipole_X = 4e-2
        tol_NumErrZero = 1e-16

        # Get J which lives on faces
        j_numF = self.numFields_ElecDipole_X[self.de_x, 'j']

        # Apply data projection matrix to get J at the reciever locations
        jx_num, jy_num, jz_num = self.Pfx*j_numF, self.Pfy*j_numF, self.Pfz*j_numF

        # Get analytic solution
        jxa, _ , _ = EM.Analytics.FDEMDipolarfields.J_from_ElectricDipoleWholeSpace(self.XYZ_Fx, self.src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
        _ , jya , _ = EM.Analytics.FDEMDipolarfields.J_from_ElectricDipoleWholeSpace(self.XYZ_Fy, self.src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
        _ , _ , jza = EM.Analytics.FDEMDipolarfields.J_from_ElectricDipoleWholeSpace(self.XYZ_Fz, self.src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
        jxa, jya, jza = Utils.mkvc(jxa, 2), Utils.mkvc(jya, 2), Utils.mkvc(jza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('J_x:').rjust(4) , repr(np.linalg.norm(jxa)).rjust(25), repr(np.linalg.norm(jx_num)).rjust(25), repr(np.linalg.norm(jxa-jx_num)).rjust(25), repr(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa)).rjust(25), repr(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa) < tol_ElecDipole_X).center(12)
        print str('J_y:').rjust(4) , repr(np.linalg.norm(jya)).rjust(25), repr(np.linalg.norm(jy_num)).rjust(25), repr(np.linalg.norm(jya-jy_num)).rjust(25), repr(np.linalg.norm(jya-jy_num)/np.linalg.norm(jya)).rjust(25), repr(np.linalg.norm(jya-jy_num)/np.linalg.norm(jya) < tol_ElecDipole_X).center(12)
        print str('J_z:').rjust(4) , repr(np.linalg.norm(jza)).rjust(25), repr(np.linalg.norm(jz_num)).rjust(25), repr(np.linalg.norm(jza-jz_num)).rjust(25), repr(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza)).rjust(25), repr(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza) < tol_ElecDipole_X).center(12)
        print
        self.assertTrue(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Jx do not agree.')
        self.assertTrue(np.linalg.norm(jya-jy_num)/np.linalg.norm(jya) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Jy do not agree.')
        self.assertTrue(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Jz do not agree.')

        # Plot Tx and Rx locations on mesh
        if plotIt:

            x_Jx = self.XYZ_Fx[:, 0]
            x_Jy = self.XYZ_Fy[:, 0]
            x_Jz = self.XYZ_Fz[:, 0]

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fx[0], self.src_loc_Fx[2], 'ro', ms=8)
            ax.plot(self.XYZ_Fx[:, 0], self.XYZ_Fx[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Fy[:, 0], self.XYZ_Fy[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Fz[:, 0], self.XYZ_Fz[:, 2], 'k.', ms=6)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

            # Plot J
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, jx_num.real, jxa.real, title='J_x Real')
            plotLine_num_ana(ax[1], x, jx_num.imag, jxa.imag, title='J_x Imag')

            plotLine_num_ana(ax[2], x, jy_num.real, jya.real, title='J_y Real')
            plotLine_num_ana(ax[3], x, jy_num.imag, jya.imag, title='J_y Imag')

            plotLine_num_ana(ax[4], x, jz_num.real, jza.real, title='J_z Real')
            plotLine_num_ana(ax[5], x, jz_num.imag, jza.imag, title='J_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

    def test_3DMesh_X_ElecDipoleTest_H(self):
        print('Testing H components of a X-oriented analytic harmonic electric dipole.')
        # Specify toleraces
        tol_ElecDipole_X = 2e-2
        tol_NumErrZero = 1e-16

        # Get H which lives on edges
        h_numE = self.numFields_ElecDipole_X[self.de_x, 'h']

        # Apply data projection matrix to get J at the reciever locations
        hx_num, hy_num, hz_num = self.Pex*h_numE, self.Pey*h_numE, self.Pez*h_numE

        # Get analytic solution
        hxa, _ , _  = EM.Analytics.FDEMDipolarfields.H_from_ElectricDipoleWholeSpace(self.XYZ_Ex, self.src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
        _ , hya , _ = EM.Analytics.FDEMDipolarfields.H_from_ElectricDipoleWholeSpace(self.XYZ_Ey, self.src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
        _ , _ , hza = EM.Analytics.FDEMDipolarfields.H_from_ElectricDipoleWholeSpace(self.XYZ_Ez, self.src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
        hxa, hya, hza = Utils.mkvc(hxa, 2), Utils.mkvc(hya, 2), Utils.mkvc(hza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('H_x:').rjust(4) , repr(np.linalg.norm(hxa)).rjust(25), repr(np.linalg.norm(hx_num)).rjust(25), repr(np.linalg.norm(hxa-hx_num)).rjust(25), str('').rjust(25)                                             , repr(np.linalg.norm(hxa-hx_num) < tol_NumErrZero).center(12)
        print str('H_y:').rjust(4) , repr(np.linalg.norm(hya)).rjust(25), repr(np.linalg.norm(hy_num)).rjust(25), repr(np.linalg.norm(hya-hy_num)).rjust(25), repr(np.linalg.norm(hya-hy_num)/np.linalg.norm(hya)).rjust(25), repr(np.linalg.norm(hya-hy_num)/np.linalg.norm(hya) < tol_ElecDipole_X).center(12)
        print str('H_z:').rjust(4) , repr(np.linalg.norm(hza)).rjust(25), repr(np.linalg.norm(hz_num)).rjust(25), repr(np.linalg.norm(hza-hz_num)).rjust(25), repr(np.linalg.norm(hza-hz_num)/np.linalg.norm(hza)).rjust(25), repr(np.linalg.norm(hza-hz_num)/np.linalg.norm(hza) < tol_ElecDipole_X).center(12)
        print
        self.assertTrue(np.linalg.norm(hxa-hx_num) < tol_NumErrZero, msg='Analytic and numeric solutions for Hx do not agree.')
        self.assertTrue(np.linalg.norm(hya-hy_num)/np.linalg.norm(hya) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Hy do not agree.')
        self.assertTrue(np.linalg.norm(hza-hz_num)/np.linalg.norm(hza) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Hz do not agree.')

        # Plot Tx and Rx locations on mesh
        if plotIt:

            x_Hx = self.XYZ_Ex[:, 0]
            x_Hy = self.XYZ_Ey[:, 0]
            x_Hz = self.XYZ_Ez[:, 0]

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fx[0], self.src_loc_Fx[2], 'ro', ms=8)
            ax.plot(self.XYZ_Ex[:, 0], self.XYZ_Ex[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Ey[:, 0], self.XYZ_Ey[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Ez[:, 0], self.XYZ_Ez[:, 2], 'k.', ms=6)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

            # Plot H
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, hx_num.real, hxa.real, title='H_x Real')
            plotLine_num_ana(ax[1], x, hx_num.imag, hxa.imag, title='H_x Imag')

            plotLine_num_ana(ax[2], x, hy_num.real, hya.real, title='H_y Real')
            plotLine_num_ana(ax[3], x, hy_num.imag, hya.imag, title='H_y Imag')

            plotLine_num_ana(ax[4], x, hz_num.real, hza.real, title='H_z Real')
            plotLine_num_ana(ax[5], x, hz_num.imag, hza.imag, title='H_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

    def test_3DMesh_X_ElecDipoleTest_B(self):
        print('Testing B components of a X-oriented analytic harmonic electric dipole.')

        # Specify toleraces
        tol_ElecDipole_X = 2e-2
        tol_NumErrZero = 1e-16

        # Get E which lives on cell centres
        b_numCC = self.numFields_ElecDipole_X[self.de_x, 'b']

        # Apply data projection matrix to get E at the reciever locations
        bx_num, by_num, bz_num = self.Pccx*b_numCC, self.Pccy*b_numCC, self.Pccz*b_numCC

        # Get analytic solution
        bxa, bya, bza = EM.Analytics.FDEMDipolarfields.B_from_ElectricDipoleWholeSpace(self.XYZ_CC, self.src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
        bxa, bya, bza = Utils.mkvc(bxa, 2), Utils.mkvc(bya, 2), Utils.mkvc(bza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('B_x:').rjust(4) , repr(np.linalg.norm(bxa)).rjust(25), repr(np.linalg.norm(bx_num)).rjust(25), repr(np.linalg.norm(bxa-bx_num)).rjust(25), str('').rjust(25)                                             , repr(np.linalg.norm(bxa-bx_num) < tol_NumErrZero).center(12)
        print str('B_y:').rjust(4) , repr(np.linalg.norm(bya)).rjust(25), repr(np.linalg.norm(by_num)).rjust(25), repr(np.linalg.norm(bya-by_num)).rjust(25), repr(np.linalg.norm(bya-by_num)/np.linalg.norm(bya)).rjust(25), repr(np.linalg.norm(bya-by_num)/np.linalg.norm(bya) < tol_ElecDipole_X).center(12)
        print str('B_z:').rjust(4) , repr(np.linalg.norm(bza)).rjust(25), repr(np.linalg.norm(bz_num)).rjust(25), repr(np.linalg.norm(bza-bz_num)).rjust(25), repr(np.linalg.norm(bza-bz_num)/np.linalg.norm(bza)).rjust(25), repr(np.linalg.norm(bza-bz_num)/np.linalg.norm(bza) < tol_ElecDipole_X).center(12)
        print
        self.assertTrue(np.linalg.norm(bxa-bx_num) < tol_NumErrZero, msg='Analytic and numeric solutions for Bx do not agree.')
        self.assertTrue(np.linalg.norm(bya-by_num)/np.linalg.norm(bya) < tol_ElecDipole_X, msg='Analytic and numeric solutions for By do not agree.')
        self.assertTrue(np.linalg.norm(bza-bz_num)/np.linalg.norm(bza) < tol_ElecDipole_X, msg='Analytic and numeric solutions for Bz do not agree.')

        # Plot Tx and Rx locations on mesh
        if plotIt:

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fx[0], self.src_loc_Fx[2], 'ro', ms=8)
            ax.plot(self.XYZ_CC[:, 0], self.XYZ_CC[:, 2], 'k.', ms=8)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

            # Plot B
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, bx_num.real, bxa.real, title='B_x Real')
            plotLine_num_ana(ax[1], x, bx_num.imag, bxa.imag, title='B_x Imag')

            plotLine_num_ana(ax[2], x, by_num.real, bya.real, title='B_y Real')
            plotLine_num_ana(ax[3], x, by_num.imag, bya.imag, title='B_y Imag')

            plotLine_num_ana(ax[4], x, bz_num.real, bza.real, title='B_z Real')
            plotLine_num_ana(ax[5], x, bz_num.imag, bza.imag, title='B_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()


class Y_ElecDipoleTest_3DMesh(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        print('Testing a Y-oriented analytic harmonic electric dipole against the numerical solution on a 3D-tesnsor mesh.')

        # Define model parameters
        self.sigmaback = SIGMABACK
        # mu = mu_0*(1+kappa)
        self.kappa = KAPPA

        # Create 3D mesh
        csx, ncx, npadx = 5, 18, 9
        csy, ncy, npady = 5, 18, 9
        csz, ncz, npadz = 5, 18, 9
        hx = Utils.meshTensor([(csx, npadx, -1.3), (csx, ncx), (csx, npadx, 1.3)])
        hy = Utils.meshTensor([(csy, npady, -1.3), (csy, ncy), (csy, npady, 1.3)])
        hz = Utils.meshTensor([(csz, npadz, -1.3), (csz, ncz), (csz, npadz, 1.3)])
        self.mesh = Mesh.TensorMesh([hx, hy, hz], 'CCC')

        # Set source parameters
        self.freq = 500.
        src_loc = np.r_[0., 0., -35]
        src_loc_CCInd = Utils.closestPoints(self.mesh, src_loc, 'CC')
        self.src_loc_CC = self.mesh.gridCC[src_loc_CCInd,:]
        self.src_loc_CC = self.src_loc_CC[0]

        # Compute skin depth
        skdpth = 500. / np.sqrt(self.sigmaback * self.freq)

        # make sure mesh is big enough
        # print('skin depth =', skdpth)
        # self.assertTrue(self.mesh.hx.sum() > skdpth*2.)
        # self.assertTrue(self.mesh.hy.sum() > skdpth*2.)
        # self.assertTrue(self.mesh.hz.sum() > skdpth*2.)

        # Create wholespace models
        SigmaBack = self.sigmaback*np.ones((self.mesh.nC))
        MuBack = (mu_0*(1 + self.kappa))*np.ones((self.mesh.nC))

        # Define reciever locations
        xlim = 40. # x locations from -50 to 50
        xInd = np.where(np.abs(self.mesh.vectorCCx) < xlim)
        x = self.mesh.vectorCCx[xInd[0]]
        y = 10.
        z = 30.

        # where we choose to measure
        XYZ = Utils.ndgrid(x, np.r_[y], np.r_[z])

        # Cell centred recievers
        XYZ_CCInd = Utils.closestPoints(self.mesh, XYZ, 'CC')
        self.XYZ_CC = self.mesh.gridCC[XYZ_CCInd,:]

        # Edge recievers
        XYZ_ExInd = Utils.closestPoints(self.mesh, XYZ, 'Ex')
        self.XYZ_Ex = self.mesh.gridEx[XYZ_ExInd,:]
        XYZ_EyInd = Utils.closestPoints(self.mesh, XYZ, 'Ey')
        self.XYZ_Ey = self.mesh.gridEy[XYZ_EyInd,:]
        XYZ_EzInd = Utils.closestPoints(self.mesh, XYZ, 'Ez')
        self.XYZ_Ez = self.mesh.gridEz[XYZ_EzInd,:]

        # Face recievers
        XYZ_FxInd = Utils.closestPoints(self.mesh, XYZ, 'Fx')
        self.XYZ_Fx = self.mesh.gridFx[XYZ_FxInd,:]
        XYZ_FyInd = Utils.closestPoints(self.mesh, XYZ, 'Fy')
        self.XYZ_Fy = self.mesh.gridFy[XYZ_FyInd,:]
        XYZ_FzInd = Utils.closestPoints(self.mesh, XYZ, 'Fz')
        self.XYZ_Fz = self.mesh.gridFz[XYZ_FzInd,:]

        # Form data interpolation matrices
        Pcc = self.mesh.getInterpolationMat(self.XYZ_CC, 'CC')
        Zero = sp.csr_matrix(Pcc.shape)
        self.Pccx, self.Pccy, self.Pccz = sp.hstack([Pcc, Zero, Zero]), sp.hstack([Zero, Pcc, Zero]), sp.hstack([Zero, Zero, Pcc])

        self.Pex, self.Pey, self.Pez = self.mesh.getInterpolationMat(self.XYZ_Ex, 'Ex'), self.mesh.getInterpolationMat(self.XYZ_Ey, 'Ey'), self.mesh.getInterpolationMat(self.XYZ_Ez, 'Ez')
        self.Pfx, self.Pfy, self.Pfz = self.mesh.getInterpolationMat(self.XYZ_Fx, 'Fx'), self.mesh.getInterpolationMat(self.XYZ_Fy, 'Fy'), self.mesh.getInterpolationMat(self.XYZ_Fz, 'Fz')

        # Define the source
        # Search over x-faces to find face nearest src_loc
        s_ind = Utils.closestPoints(self.mesh, self.src_loc_CC, 'Fy') + self.mesh.nFx
        de = np.zeros(self.mesh.nF, dtype=complex)
        de[s_ind] = 1./self.mesh.hy.min()
        self.de_y = [EM.FDEM.Src.RawVec_e([], self.freq, de/self.mesh.area)]

        self.src_loc_Fy = self.mesh.gridFy[Utils.closestPoints(self.mesh, self.src_loc_CC, 'Fy'),:]
        self.src_loc_Fy = self.src_loc_Fy[0]

        # Create survey and problem object
        survey = EM.FDEM.Survey(self.de_y)
        mapping = [('sigma', Maps.IdentityMap(self.mesh)), ('mu', Maps.IdentityMap(self.mesh))]
        problem = EM.FDEM.Problem3D_h(self.mesh, mapping=mapping)

        # Pair problem and survey
        problem.pair(survey)
        problem.Solver = Solver

        # Solve forward problem
        self.numFields_ElecDipole_Y = problem.fields(np.r_[SigmaBack, MuBack])

        # setUp_Done = True

    def test_3DMesh_Y_ElecDipoleTest_E(self):
        print('Testing E components of a Y-oriented analytic harmonic electric dipole.')

        # Specify toleraces
        tol_ElecDipole_Y = 4e-2
        tol_NumErrZero = 1e-16

        # Get E which lives on cell centres
        e_numCC = self.numFields_ElecDipole_Y[self.de_y, 'e']

        # Apply data projection matrix to get E at the reciever locations
        ex_num, ey_num, ez_num = self.Pccx*e_numCC, self.Pccy*e_numCC, self.Pccz*e_numCC

        # Get analytic solution
        exa, eya, eza = EM.Analytics.FDEMDipolarfields.E_from_ElectricDipoleWholeSpace(self.XYZ_CC, self.src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)), orientation='Y', kappa= self.kappa)
        exa, eya, eza = Utils.mkvc(exa, 2), Utils.mkvc(eya, 2), Utils.mkvc(eza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('E_x:').rjust(4) , repr(np.linalg.norm(exa)).rjust(25), repr(np.linalg.norm(ex_num)).rjust(25), repr(np.linalg.norm(exa-ex_num)).rjust(25), repr(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa)).rjust(25), repr(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa) < tol_ElecDipole_Y).center(12)
        print str('E_y:').rjust(4) , repr(np.linalg.norm(eya)).rjust(25), repr(np.linalg.norm(ey_num)).rjust(25), repr(np.linalg.norm(eya-ey_num)).rjust(25), repr(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya)).rjust(25), repr(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya) < tol_ElecDipole_Y).center(12)
        print str('E_z:').rjust(4) , repr(np.linalg.norm(eza)).rjust(25), repr(np.linalg.norm(ez_num)).rjust(25), repr(np.linalg.norm(eza-ez_num)).rjust(25), repr(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza)).rjust(25), repr(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza) < tol_ElecDipole_Y).center(12)
        print
        self.assertTrue(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Ex do not agree.')
        self.assertTrue(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Ey do not agree.')
        self.assertTrue(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Ez do not agree.')

        # Plot Tx and Rx locations on mesh
        if plotIt:

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fy[0], self.src_loc_Fy[2], 'ro', ms=8)
            ax.plot(self.XYZ_CC[:, 0], self.XYZ_CC[:, 2], 'k.', ms=8)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)


            # Plot E
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, ex_num.real, exa.real, title='E_x Real')
            plotLine_num_ana(ax[1], x, ex_num.imag, exa.imag, title='E_x Imag')

            plotLine_num_ana(ax[2], x, ey_num.real, eya.real, title='E_y Real')
            plotLine_num_ana(ax[3], x, ey_num.imag, eya.imag, title='E_y Imag')

            plotLine_num_ana(ax[4], x, ez_num.real, eza.real, title='E_z Real')
            plotLine_num_ana(ax[5], x, ez_num.imag, eza.imag, title='E_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

    def test_3DMesh_Y_ElecDipoleTest_J(self):
        print('Testing J components of a Y-oriented analytic harmonic electric dipole.')
        # Specify toleraces
        tol_ElecDipole_Y = 4e-2
        tol_NumErrZero = 1e-16

        # Get J which lives on faces
        j_numF = self.numFields_ElecDipole_Y[self.de_y, 'j']

        # Apply data projection matrix to get J at the reciever locations
        jx_num, jy_num, jz_num = self.Pfx*j_numF, self.Pfy*j_numF, self.Pfz*j_numF

        # Get analytic solution
        jxa, _ , _ = EM.Analytics.FDEMDipolarfields.J_from_ElectricDipoleWholeSpace(self.XYZ_Fx, self.src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
        _ , jya , _ = EM.Analytics.FDEMDipolarfields.J_from_ElectricDipoleWholeSpace(self.XYZ_Fy, self.src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
        _ , _ , jza = EM.Analytics.FDEMDipolarfields.J_from_ElectricDipoleWholeSpace(self.XYZ_Fz, self.src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
        jxa, jya, jza = Utils.mkvc(jxa, 2), Utils.mkvc(jya, 2), Utils.mkvc(jza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('J_x:').rjust(4) , repr(np.linalg.norm(jxa)).rjust(25), repr(np.linalg.norm(jx_num)).rjust(25), repr(np.linalg.norm(jxa-jx_num)).rjust(25), repr(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa)).rjust(25), repr(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa) < tol_ElecDipole_Y).center(12)
        print str('J_y:').rjust(4) , repr(np.linalg.norm(jya)).rjust(25), repr(np.linalg.norm(jy_num)).rjust(25), repr(np.linalg.norm(jya-jy_num)).rjust(25), repr(np.linalg.norm(jya-jy_num)/np.linalg.norm(jya)).rjust(25), repr(np.linalg.norm(jya-jy_num)/np.linalg.norm(jya) < tol_ElecDipole_Y).center(12)
        print str('J_z:').rjust(4) , repr(np.linalg.norm(jza)).rjust(25), repr(np.linalg.norm(jz_num)).rjust(25), repr(np.linalg.norm(jza-jz_num)).rjust(25), repr(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza)).rjust(25), repr(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza) < tol_ElecDipole_Y).center(12)
        print
        self.assertTrue(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Jx do not agree.')
        self.assertTrue(np.linalg.norm(jya-jy_num)/np.linalg.norm(jya) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Jy do not agree.')
        self.assertTrue(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Jz do not agree.')

        # Plot Tx and Rx locations on mesY
        if plotIt:

            x_Jx = self.XYZ_Fx[:, 0]
            x_Jy = self.XYZ_Fy[:, 0]
            x_Jz = self.XYZ_Fz[:, 0]

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fy[0], self.src_loc_Fy[2], 'ro', ms=8)
            ax.plot(self.XYZ_Fx[:, 0], self.XYZ_Fx[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Fy[:, 0], self.XYZ_Fy[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Fz[:, 0], self.XYZ_Fz[:, 2], 'k.', ms=6)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

            # Plot H
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, hx_num.real, hxa.real, title='H_x Real')
            plotLine_num_ana(ax[1], x, hx_num.imag, hxa.imag, title='H_x Imag')

            plotLine_num_ana(ax[2], x, hy_num.real, hya.real, title='H_y Real')
            plotLine_num_ana(ax[3], x, hy_num.imag, hya.imag, title='H_y Imag')

            plotLine_num_ana(ax[4], x, hz_num.real, hza.real, title='H_z Real')
            plotLine_num_ana(ax[5], x, hz_num.imag, hza.imag, title='H_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

    def test_3DMesh_Y_ElecDipoleTest_H(self):
        print('Testing H components of a Y-oriented analytic harmonic electric dipole.')
        # Specify toleraces
        tol_ElecDipole_Y = 2e-2
        tol_NumErrZero = 1e-16

        # Get H which lives on edges
        h_numE = self.numFields_ElecDipole_Y[self.de_y, 'h']

        # Apply data projection matrix to get J at the reciever locations
        hx_num, hy_num, hz_num = self.Pex*h_numE, self.Pey*h_numE, self.Pez*h_numE

        # Get analytic solution
        hxa, _ , _  = EM.Analytics.FDEMDipolarfields.H_from_ElectricDipoleWholeSpace(self.XYZ_Ex, self.src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
        _ , hya , _ = EM.Analytics.FDEMDipolarfields.H_from_ElectricDipoleWholeSpace(self.XYZ_Ey, self.src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
        _ , _ , hza = EM.Analytics.FDEMDipolarfields.H_from_ElectricDipoleWholeSpace(self.XYZ_Ez, self.src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
        hxa, hya, hza = Utils.mkvc(hxa, 2), Utils.mkvc(hya, 2), Utils.mkvc(hza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('H_x:').rjust(4) , repr(np.linalg.norm(hxa)).rjust(25), repr(np.linalg.norm(hx_num)).rjust(25), repr(np.linalg.norm(hxa-hx_num)).rjust(25), repr(np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa)).rjust(25), repr(np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa) < tol_ElecDipole_Y).center(12)
        print str('H_y:').rjust(4) , repr(np.linalg.norm(hya)).rjust(25), repr(np.linalg.norm(hy_num)).rjust(25), repr(np.linalg.norm(hya-hy_num)).rjust(25), str('').rjust(25)                                             , repr(np.linalg.norm(hya-hy_num) < tol_NumErrZero).center(12)
        print str('H_z:').rjust(4) , repr(np.linalg.norm(hza)).rjust(25), repr(np.linalg.norm(hz_num)).rjust(25), repr(np.linalg.norm(hza-hz_num)).rjust(25), repr(np.linalg.norm(hza-hz_num)/np.linalg.norm(hza)).rjust(25), repr(np.linalg.norm(hza-hz_num)/np.linalg.norm(hza) < tol_ElecDipole_Y).center(12)
        print
        self.assertTrue(np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Hx do not agree.')
        self.assertTrue(np.linalg.norm(hya-hy_num) < tol_NumErrZero, msg='Analytic and numeric solutions for Hy do not agree.')
        self.assertTrue(np.linalg.norm(hza-hz_num)/np.linalg.norm(hza) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Hz do not agree.')

        # Plot Tx and Rx locations on mesh
        if plotIt:

            x_Hx = self.XYZ_Ex[:, 0]
            x_Hy = self.XYZ_Ey[:, 0]
            x_Hz = self.XYZ_Ez[:, 0]

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fy[0], self.src_loc_Fy[2], 'ro', ms=8)
            ax.plot(self.XYZ_Ex[:, 0], self.XYZ_Ex[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Ey[:, 0], self.XYZ_Ey[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Ez[:, 0], self.XYZ_Ez[:, 2], 'k.', ms=6)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

            # Plot J
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, jx_num.real, jxa.real, title='J_x Real')
            plotLine_num_ana(ax[1], x, jx_num.imag, jxa.imag, title='J_x Imag')

            plotLine_num_ana(ax[2], x, jy_num.real, jya.real, title='J_y Real')
            plotLine_num_ana(ax[3], x, jy_num.imag, jya.imag, title='J_y Imag')

            plotLine_num_ana(ax[4], x, jz_num.real, jza.real, title='J_z Real')
            plotLine_num_ana(ax[5], x, jz_num.imag, jza.imag, title='J_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

    def test_3DMesh_Y_ElecDipoleTest_B(self):
        print('Testing B components of a Y-oriented analytic harmonic electric dipole.')

        # Specify toleraces
        tol_ElecDipole_Y = 2e-2
        tol_NumErrZero = 1e-16

        # Get E which lives on cell centres
        b_numCC = self.numFields_ElecDipole_Y[self.de_y, 'b']

        # Apply data projection matrix to get E at the reciever locations
        bx_num, by_num, bz_num = self.Pccx*b_numCC, self.Pccy*b_numCC, self.Pccz*b_numCC

        # Get analytic solution
        bxa, bya, bza = EM.Analytics.FDEMDipolarfields.B_from_ElectricDipoleWholeSpace(self.XYZ_CC, self.src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
        bxa, bya, bza = Utils.mkvc(bxa, 2), Utils.mkvc(bya, 2), Utils.mkvc(bza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('B_x:').rjust(4) , repr(np.linalg.norm(bxa)).rjust(25), repr(np.linalg.norm(bx_num)).rjust(25), repr(np.linalg.norm(bxa-bx_num)).rjust(25), repr(np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa)).rjust(25), repr(np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa) < tol_ElecDipole_Y).center(12)
        print str('B_y:').rjust(4) , repr(np.linalg.norm(bya)).rjust(25), repr(np.linalg.norm(by_num)).rjust(25), repr(np.linalg.norm(bya-by_num)).rjust(25), str('').rjust(25)                                             , repr(np.linalg.norm(bya-by_num) < tol_NumErrZero).center(12)
        print str('B_z:').rjust(4) , repr(np.linalg.norm(bza)).rjust(25), repr(np.linalg.norm(bz_num)).rjust(25), repr(np.linalg.norm(bza-bz_num)).rjust(25), repr(np.linalg.norm(bza-bz_num)/np.linalg.norm(bza)).rjust(25), repr(np.linalg.norm(bza-bz_num)/np.linalg.norm(bza) < tol_ElecDipole_Y).center(12)
        print
        self.assertTrue(np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Bx do not agree.')
        self.assertTrue(np.linalg.norm(bya-by_num) < tol_NumErrZero, msg='Analytic and numeric solutions for By do not agree.')
        self.assertTrue(np.linalg.norm(bza-bz_num)/np.linalg.norm(bza) < tol_ElecDipole_Y, msg='Analytic and numeric solutions for Bz do not agree.')

        # Plot Tx and Rx locations on mesh
        if plotIt:

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fy[0], self.src_loc_Fy[2], 'ro', ms=8)
            ax.plot(self.XYZ_CC[:, 0], self.XYZ_CC[:, 2], 'k.', ms=8)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)


            # Plot B
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, bx_num.real, bxa.real, title='B_x Real')
            plotLine_num_ana(ax[1], x, bx_num.imag, bxa.imag, title='B_x Imag')

            plotLine_num_ana(ax[2], x, by_num.real, bya.real, title='B_y Real')
            plotLine_num_ana(ax[3], x, by_num.imag, bya.imag, title='B_y Imag')

            plotLine_num_ana(ax[4], x, bz_num.real, bza.real, title='B_z Real')
            plotLine_num_ana(ax[5], x, bz_num.imag, bza.imag, title='B_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

class Z_ElecDipoleTest_3DMesh(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        print('Testing a Z-oriented analytic harmonic electric dipole against the numerical solution on a 3D-tesnsor mesh.')

        # Define model parameters
        self.sigmaback = SIGMABACK
        # mu = mu_0*(1+kappa)
        self.kappa = KAPPA

        # Create 3D mesh
        self.mesh = setUpMesh()

        # Set source parameters
        self.freq = 500.
        src_loc = np.r_[0., 0., -35]
        src_loc_CCInd = Utils.closestPoints(self.mesh, src_loc, 'CC')
        self.src_loc_CC = self.mesh.gridCC[src_loc_CCInd,:]
        self.src_loc_CC = self.src_loc_CC[0]

        # Compute skin depth
        skdpth = 500. / np.sqrt(self.sigmaback * self.freq)

        # make sure mesh is big enough
        # print('skin depth =', skdpth)
        # self.assertTrue(self.mesh.hx.sum() > skdpth*2.)
        # self.assertTrue(self.mesh.hy.sum() > skdpth*2.)
        # self.assertTrue(self.mesh.hz.sum() > skdpth*2.)

        # Create wholespace models
        SigmaBack = self.sigmaback*np.ones((self.mesh.nC))
        MuBack = (mu_0*(1 + self.kappa))*np.ones((self.mesh.nC))

        # Define reciever locations
        xlim = 40. # x locations from -50 to 50
        xInd = np.where(np.abs(self.mesh.vectorCCx) < xlim)
        x = self.mesh.vectorCCx[xInd[0]]
        y = 10.
        z = 30.

        # where we choose to measure
        XYZ = Utils.ndgrid(x, np.r_[y], np.r_[z])

        # Cell centred recievers
        XYZ_CCInd = Utils.closestPoints(self.mesh, XYZ, 'CC')
        self.XYZ_CC = self.mesh.gridCC[XYZ_CCInd,:]

        # Edge recievers
        XYZ_ExInd = Utils.closestPoints(self.mesh, XYZ, 'Ex')
        self.XYZ_Ex = self.mesh.gridEx[XYZ_ExInd,:]
        XYZ_EyInd = Utils.closestPoints(self.mesh, XYZ, 'Ey')
        self.XYZ_Ey = self.mesh.gridEy[XYZ_EyInd,:]
        XYZ_EzInd = Utils.closestPoints(self.mesh, XYZ, 'Ez')
        self.XYZ_Ez = self.mesh.gridEz[XYZ_EzInd,:]

        # Face recievers
        XYZ_FxInd = Utils.closestPoints(self.mesh, XYZ, 'Fx')
        self.XYZ_Fx = self.mesh.gridFx[XYZ_FxInd,:]
        XYZ_FyInd = Utils.closestPoints(self.mesh, XYZ, 'Fy')
        self.XYZ_Fy = self.mesh.gridFy[XYZ_FyInd,:]
        XYZ_FzInd = Utils.closestPoints(self.mesh, XYZ, 'Fz')
        self.XYZ_Fz = self.mesh.gridFz[XYZ_FzInd,:]

        # Form data interpolation matrices
        Pcc = self.mesh.getInterpolationMat(self.XYZ_CC, 'CC')
        Zero = sp.csr_matrix(Pcc.shape)
        self.Pccx, self.Pccy, self.Pccz = sp.hstack([Pcc, Zero, Zero]), sp.hstack([Zero, Pcc, Zero]), sp.hstack([Zero, Zero, Pcc])

        self.Pex, self.Pey, self.Pez = self.mesh.getInterpolationMat(self.XYZ_Ex, 'Ex'), self.mesh.getInterpolationMat(self.XYZ_Ey, 'Ey'), self.mesh.getInterpolationMat(self.XYZ_Ez, 'Ez')
        self.Pfx, self.Pfy, self.Pfz = self.mesh.getInterpolationMat(self.XYZ_Fx, 'Fx'), self.mesh.getInterpolationMat(self.XYZ_Fy, 'Fy'), self.mesh.getInterpolationMat(self.XYZ_Fz, 'Fz')

        # Define the source
        # Search over x-faces to find face nearest src_loc
        s_ind = Utils.closestPoints(self.mesh, self.src_loc_CC, 'Fz') + self.mesh.nFx + self.mesh.nFy
        de = np.zeros(self.mesh.nF, dtype=complex)
        de[s_ind] = 1./self.mesh.hz.min()
        self.de_z = [EM.FDEM.Src.RawVec_e([], self.freq, de/self.mesh.area)]

        self.src_loc_Fz = self.mesh.gridFz[Utils.closestPoints(self.mesh, self.src_loc_CC, 'Fz'),:]
        self.src_loc_Fz = self.src_loc_Fz[0]

        # Create survey and problem object
        survey = EM.FDEM.Survey(self.de_z)
        mapping = [('sigma', Maps.IdentityMap(self.mesh)), ('mu', Maps.IdentityMap(self.mesh))]
        problem = EM.FDEM.Problem3D_h(self.mesh, mapping=mapping)

        # Pair problem and survey
        problem.pair(survey)
        problem.Solver = Solver

        # Solve forward problem
        self.numFields_ElecDipole_Z = problem.fields(np.r_[SigmaBack, MuBack])

        # setUp_Done = True

    def test_3DMesh_Z_ElecDipoleTest_E(self):
        print('Testing E components of a Z-oriented analytic harmonic electric dipole.')

        # Specify toleraces
        tol_ElecDipole_Z = 2e-2
        tol_NumErrZero = 1e-16

        # Get E which lives on cell centres
        e_numCC = self.numFields_ElecDipole_Z[self.de_z, 'e']

        # Apply data projection matrix to get E at the reciever locations
        ex_num, ey_num, ez_num = self.Pccx*e_numCC, self.Pccy*e_numCC, self.Pccz*e_numCC

        # Get analytic solution
        exa, eya, eza = EM.Analytics.FDEMDipolarfields.E_from_ElectricDipoleWholeSpace(self.XYZ_CC, self.src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)), orientation='Z', kappa= self.kappa)
        exa, eya, eza = Utils.mkvc(exa, 2), Utils.mkvc(eya, 2), Utils.mkvc(eza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('E_x:').rjust(4) , repr(np.linalg.norm(exa)).rjust(25), repr(np.linalg.norm(ex_num)).rjust(25), repr(np.linalg.norm(exa-ex_num)).rjust(25), repr(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa)).rjust(25), repr(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa) < tol_ElecDipole_Z).center(12)
        print str('E_y:').rjust(4) , repr(np.linalg.norm(eya)).rjust(25), repr(np.linalg.norm(ey_num)).rjust(25), repr(np.linalg.norm(eya-ey_num)).rjust(25), repr(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya)).rjust(25), repr(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya) < tol_ElecDipole_Z).center(12)
        print str('E_z:').rjust(4) , repr(np.linalg.norm(eza)).rjust(25), repr(np.linalg.norm(ez_num)).rjust(25), repr(np.linalg.norm(eza-ez_num)).rjust(25), repr(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza)).rjust(25), repr(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza) < tol_ElecDipole_Z).center(12)
        print
        self.assertTrue(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for Ex do not agree.')
        self.assertTrue(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for Ey do not agree.')
        self.assertTrue(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for Ez do not agree.')

        # Plot Tx and Rx locations on mesh
        if plotIt:

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fz[0], self.src_loc_Fz[2], 'ro', ms=8)
            ax.plot(self.XYZ_CC[:, 0], self.XYZ_CC[:, 2], 'k.', ms=8)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)


            # Plot E
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, ex_num.real, exa.real, title='E_x Real')
            plotLine_num_ana(ax[1], x, ex_num.imag, exa.imag, title='E_x Imag')

            plotLine_num_ana(ax[2], x, ey_num.real, eya.real, title='E_y Real')
            plotLine_num_ana(ax[3], x, ey_num.imag, eya.imag, title='E_y Imag')

            plotLine_num_ana(ax[4], x, ez_num.real, eza.real, title='E_z Real')
            plotLine_num_ana(ax[5], x, ez_num.imag, eza.imag, title='E_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

    def test_3DMesh_Z_ElecDipoleTest_J(self):
        print('Testing J components of a Z-oriented analytic harmonic electric dipole.')
        # Specify toleraces
        tol_ElecDipole_Z = 2e-2
        tol_NumErrZero = 1e-16

        # Get J which lives on faces
        j_numF = self.numFields_ElecDipole_Z[self.de_z, 'j']

        # Apply data projection matrix to get J at the reciever locations
        jx_num, jy_num, jz_num = self.Pfx*j_numF, self.Pfy*j_numF, self.Pfz*j_numF

        # Get analytic solution
        jxa, _ , _ = EM.Analytics.FDEMDipolarfields.J_from_ElectricDipoleWholeSpace(self.XYZ_Fx, self.src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        _ , jya , _ = EM.Analytics.FDEMDipolarfields.J_from_ElectricDipoleWholeSpace(self.XYZ_Fy, self.src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        _ , _ , jza = EM.Analytics.FDEMDipolarfields.J_from_ElectricDipoleWholeSpace(self.XYZ_Fz, self.src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        jxa, jya, jza = Utils.mkvc(jxa, 2), Utils.mkvc(jya, 2), Utils.mkvc(jza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('J_x:').rjust(4) , repr(np.linalg.norm(jxa)).rjust(25), repr(np.linalg.norm(jx_num)).rjust(25), repr(np.linalg.norm(jxa-jx_num)).rjust(25), repr(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa)).rjust(25), repr(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa) < tol_ElecDipole_Z).center(12)
        print str('J_y:').rjust(4) , repr(np.linalg.norm(jya)).rjust(25), repr(np.linalg.norm(jy_num)).rjust(25), repr(np.linalg.norm(jya-jy_num)).rjust(25), repr(np.linalg.norm(jya-jy_num)/np.linalg.norm(jya)).rjust(25), repr(np.linalg.norm(jya-jy_num)/np.linalg.norm(jya) < tol_ElecDipole_Z).center(12)
        print str('J_z:').rjust(4) , repr(np.linalg.norm(jza)).rjust(25), repr(np.linalg.norm(jz_num)).rjust(25), repr(np.linalg.norm(jza-jz_num)).rjust(25), repr(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza)).rjust(25), repr(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza) < tol_ElecDipole_Z).center(12)
        print
        self.assertTrue(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for Jx do not agree.')
        self.assertTrue(np.linalg.norm(jya-jy_num)/np.linalg.norm(jya) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for Jy do not agree.')
        self.assertTrue(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for Jz do not agree.')

        # Plot Tx and Rx locations on mesY
        if plotIt:

            x_Jx = self.XYZ_Fx[:, 0]
            x_Jy = self.XYZ_Fy[:, 0]
            x_Jz = self.XYZ_Fz[:, 0]

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fz[0], self.src_loc_Fz[2], 'ro', ms=8)
            ax.plot(self.XYZ_Fx[:, 0], self.XYZ_Fx[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Fy[:, 0], self.XYZ_Fy[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Fz[:, 0], self.XYZ_Fz[:, 2], 'k.', ms=6)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

            # Plot J
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, jx_num.real, jxa.real, title='J_x Real')
            plotLine_num_ana(ax[1], x, jx_num.imag, jxa.imag, title='J_x Imag')

            plotLine_num_ana(ax[2], x, jy_num.real, jya.real, title='J_y Real')
            plotLine_num_ana(ax[3], x, jy_num.imag, jya.imag, title='J_y Imag')

            plotLine_num_ana(ax[4], x, jz_num.real, jza.real, title='J_z Real')
            plotLine_num_ana(ax[5], x, jz_num.imag, jza.imag, title='J_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

    def test_3DMesh_Z_ElecDipoleTest_H(self):
        print('Testing H components of a Z-oriented analytic harmonic electric dipole.')
        # Specify toleraces
        tol_ElecDipole_Z = 2e-2
        tol_NumErrZero = 1e-16

        # Get H which lives on edges
        h_numE = self.numFields_ElecDipole_Z[self.de_z, 'h']

        # Apply data projection matrix to get J at the reciever locations
        hx_num, hy_num, hz_num = self.Pex*h_numE, self.Pey*h_numE, self.Pez*h_numE

        # Get analytic solution
        hxa, _ , _  = EM.Analytics.FDEMDipolarfields.H_from_ElectricDipoleWholeSpace(self.XYZ_Ex, self.src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        _ , hya , _ = EM.Analytics.FDEMDipolarfields.H_from_ElectricDipoleWholeSpace(self.XYZ_Ey, self.src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        _ , _ , hza = EM.Analytics.FDEMDipolarfields.H_from_ElectricDipoleWholeSpace(self.XYZ_Ez, self.src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        hxa, hya, hza = Utils.mkvc(hxa, 2), Utils.mkvc(hya, 2), Utils.mkvc(hza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('H_x:').rjust(4) , repr(np.linalg.norm(hxa)).rjust(25), repr(np.linalg.norm(hx_num)).rjust(25), repr(np.linalg.norm(hxa-hx_num)).rjust(25), repr(np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa)).rjust(25), repr(np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa) < tol_ElecDipole_Z).center(12)
        print str('H_y:').rjust(4) , repr(np.linalg.norm(hya)).rjust(25), repr(np.linalg.norm(hy_num)).rjust(25), repr(np.linalg.norm(hya-hy_num)).rjust(25), repr(np.linalg.norm(hya-hy_num)/np.linalg.norm(hya)).rjust(25), repr(np.linalg.norm(hya-hy_num)/np.linalg.norm(hya) < tol_ElecDipole_Z).center(12)
        print str('H_z:').rjust(4) , repr(np.linalg.norm(hza)).rjust(25), repr(np.linalg.norm(hz_num)).rjust(25), repr(np.linalg.norm(hza-hz_num)).rjust(25), str('').rjust(25)                                             , repr(np.linalg.norm(hza-hz_num) < tol_NumErrZero).center(12)
        print
        self.assertTrue(np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for Hx do not agree.')
        self.assertTrue(np.linalg.norm(hya-hy_num)/np.linalg.norm(hya) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for Hy do not agree.')
        self.assertTrue(np.linalg.norm(hza-hz_num) < tol_NumErrZero, msg='Analytic and numeric solutions for Hz do not agree.')

        # Plot Tx and Rx locations on mesh
        if plotIt:

            x_Hx = self.XYZ_Ex[:, 0]
            x_Hy = self.XYZ_Ey[:, 0]
            x_Hz = self.XYZ_Ez[:, 0]

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fz[0], self.src_loc_Fz[2], 'ro', ms=8)
            ax.plot(self.XYZ_Ex[:, 0], self.XYZ_Ex[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Ey[:, 0], self.XYZ_Ey[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Ez[:, 0], self.XYZ_Ez[:, 2], 'k.', ms=6)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

            # Plot H
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, hx_num.real, hxa.real, title='H_x Real')
            plotLine_num_ana(ax[1], x, hx_num.imag, hxa.imag, title='H_x Imag')

            plotLine_num_ana(ax[2], x, hy_num.real, hya.real, title='H_y Real')
            plotLine_num_ana(ax[3], x, hy_num.imag, hya.imag, title='H_y Imag')

            plotLine_num_ana(ax[4], x, hz_num.real, hza.real, title='H_z Real')
            plotLine_num_ana(ax[5], x, hz_num.imag, hza.imag, title='H_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

    def test_3DMesh_Z_ElecDipoleTest_B(self):
        print('Testing B components of a Z-oriented analytic harmonic electric dipole.')

        # Specify toleraces
        tol_ElecDipole_Z = 2e-2
        tol_NumErrZero = 1e-16

        # Get E which lives on cell centres
        b_numCC = self.numFields_ElecDipole_Z[self.de_z, 'b']

        # Apply data projection matrix to get E at the reciever locations
        bx_num, by_num, bz_num = self.Pccx*b_numCC, self.Pccy*b_numCC, self.Pccz*b_numCC

        # Get analytic solution
        bxa, bya, bza = EM.Analytics.FDEMDipolarfields.B_from_ElectricDipoleWholeSpace(self.XYZ_CC, self.src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        bxa, bya, bza = Utils.mkvc(bxa, 2), Utils.mkvc(bya, 2), Utils.mkvc(bza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('B_x:').rjust(4) , repr(np.linalg.norm(bxa)).rjust(25), repr(np.linalg.norm(bx_num)).rjust(25), repr(np.linalg.norm(bxa-bx_num)).rjust(25), repr(np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa)).rjust(25), repr(np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa) < tol_ElecDipole_Z).center(12)
        print str('B_y:').rjust(4) , repr(np.linalg.norm(bya)).rjust(25), repr(np.linalg.norm(by_num)).rjust(25), repr(np.linalg.norm(bya-by_num)).rjust(25), repr(np.linalg.norm(bya-by_num)/np.linalg.norm(bya)).rjust(25), repr(np.linalg.norm(bya-by_num)/np.linalg.norm(bya) < tol_ElecDipole_Z).center(12)
        print str('B_z:').rjust(4) , repr(np.linalg.norm(bza)).rjust(25), repr(np.linalg.norm(bz_num)).rjust(25), repr(np.linalg.norm(bza-bz_num)).rjust(25), str('').rjust(25)                                             , repr(np.linalg.norm(bza-bz_num) < tol_NumErrZero).center(12)
        print
        self.assertTrue(np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for Bx do not agree.')
        self.assertTrue(np.linalg.norm(bya-by_num)/np.linalg.norm(bya) < tol_ElecDipole_Z, msg='Analytic and numeric solutions for By do not agree.')
        self.assertTrue(np.linalg.norm(bza-bz_num) < tol_NumErrZero, msg='Analytic and numeric solutions for Bz do not agree.')


        # Plot Tx and Rx locations on mesh
        if plotIt:

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fz[0], self.src_loc_Fz[2], 'ro', ms=8)
            ax.plot(self.XYZ_CC[:, 0], self.XYZ_CC[:, 2], 'k.', ms=8)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)


            # Plot E
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, bx_num.real, bxa.real, title='B_x Real')
            plotLine_num_ana(ax[1], x, bx_num.imag, bxa.imag, title='B_x Imag')

            plotLine_num_ana(ax[2], x, by_num.real, bya.real, title='B_y Real')
            plotLine_num_ana(ax[3], x, by_num.imag, bya.imag, title='B_y Imag')

            plotLine_num_ana(ax[4], x, bz_num.real, bza.real, title='B_z Real')
            plotLine_num_ana(ax[5], x, bz_num.imag, bza.imag, title='B_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

class X_MaDipoleTest_3DMesh(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        print('Testing a X-oriented analytic harmonic magnetic dipole against the numerical solution on a 3D-tesnsor mesh.')

        # Define model parameters
        self.sigmaback = SIGMABACK
        # mu = mu_0*(1+kappa)
        self.kappa = KAPPA

        # Create 3D mesh
        self.mesh = setUpMesh()

        # Set source parameters
        self.freq = 500.
        src_loc = np.r_[0., 0., -35]
        src_loc_CCInd = Utils.closestPoints(self.mesh, src_loc, 'CC')
        self.src_loc_CC = self.mesh.gridCC[src_loc_CCInd,:]
        self.src_loc_CC = self.src_loc_CC[0]

        # Compute skin depth
        skdpth = 500. / np.sqrt(self.sigmaback * self.freq)

        # make sure mesh is big enough
        # print('skin depth =', skdpth)
        # self.assertTrue(self.mesh.hx.sum() > skdpth*2.)
        # self.assertTrue(self.mesh.hy.sum() > skdpth*2.)
        # self.assertTrue(self.mesh.hz.sum() > skdpth*2.)

        # Create wholespace models
        SigmaBack = self.sigmaback*np.ones((self.mesh.nC))
        MuBack = (mu_0*(1 + self.kappa))*np.ones((self.mesh.nC))

        # Define reciever locations
        xlim = 40. # x locations from -50 to 50
        xInd = np.where(np.abs(self.mesh.vectorCCx) < xlim)
        x = self.mesh.vectorCCx[xInd[0]]
        y = 10.
        z = 30.

        # where we choose to measure
        XYZ = Utils.ndgrid(x, np.r_[y], np.r_[z])

        # Cell centred recievers
        XYZ_CCInd = Utils.closestPoints(self.mesh, XYZ, 'CC')
        self.XYZ_CC = self.mesh.gridCC[XYZ_CCInd,:]

        # Edge recievers
        XYZ_ExInd = Utils.closestPoints(self.mesh, XYZ, 'Ex')
        self.XYZ_Ex = self.mesh.gridEx[XYZ_ExInd,:]
        XYZ_EyInd = Utils.closestPoints(self.mesh, XYZ, 'Ey')
        self.XYZ_Ey = self.mesh.gridEy[XYZ_EyInd,:]
        XYZ_EzInd = Utils.closestPoints(self.mesh, XYZ, 'Ez')
        self.XYZ_Ez = self.mesh.gridEz[XYZ_EzInd,:]

        # Face recievers
        XYZ_FxInd = Utils.closestPoints(self.mesh, XYZ, 'Fx')
        self.XYZ_Fx = self.mesh.gridFx[XYZ_FxInd,:]
        XYZ_FyInd = Utils.closestPoints(self.mesh, XYZ, 'Fy')
        self.XYZ_Fy = self.mesh.gridFy[XYZ_FyInd,:]
        XYZ_FzInd = Utils.closestPoints(self.mesh, XYZ, 'Fz')
        self.XYZ_Fz = self.mesh.gridFz[XYZ_FzInd,:]

        # Form data interpolation matrices
        Pcc = self.mesh.getInterpolationMat(self.XYZ_CC, 'CC')
        Zero = sp.csr_matrix(Pcc.shape)
        self.Pccx, self.Pccy, self.Pccz = sp.hstack([Pcc, Zero, Zero]), sp.hstack([Zero, Pcc, Zero]), sp.hstack([Zero, Zero, Pcc])

        self.Pex, self.Pey, self.Pez = self.mesh.getInterpolationMat(self.XYZ_Ex, 'Ex'), self.mesh.getInterpolationMat(self.XYZ_Ey, 'Ey'), self.mesh.getInterpolationMat(self.XYZ_Ez, 'Ez')
        self.Pfx, self.Pfy, self.Pfz = self.mesh.getInterpolationMat(self.XYZ_Fx, 'Fx'), self.mesh.getInterpolationMat(self.XYZ_Fy, 'Fy'), self.mesh.getInterpolationMat(self.XYZ_Fz, 'Fz')

        # Define the source
        # Search over x-faces to find face nearest src_loc
        s_ind = Utils.closestPoints(self.mesh, self.src_loc_CC, 'Fx')
        dm = np.zeros(self.mesh.nF, dtype=complex)
        dm[s_ind] = (-1j*(2*np.pi*self.freq)*(mu_0*(1 + self.kappa)))/self.mesh.hx.min()
        self.dm_x = [EM.FDEM.Src.RawVec_m([], self.freq, dm/self.mesh.area)]

        self.src_loc_Fx = self.mesh.gridFx[Utils.closestPoints(self.mesh, self.src_loc_CC, 'Fx'),:]
        self.src_loc_Fx = self.src_loc_Fx[0]

        # Create survey and problem object
        survey = EM.FDEM.Survey(self.dm_x)
        mapping = [('sigma', Maps.IdentityMap(self.mesh)), ('mu', Maps.IdentityMap(self.mesh))]
        problem = EM.FDEM.Problem3D_b(self.mesh, mapping=mapping)

        # Pair problem and survey
        problem.pair(survey)
        problem.Solver = Solver

        # Solve forward problem
        self.numFields_MagDipole_X = problem.fields(np.r_[SigmaBack, MuBack])

        # setUp_Done = True

    def test_3DMesh_X_MagDipoleTest_E(self):
        print('Testing E components of a X-oriented analytic harmonic magnetic dipole.')

        # Specify toleraces
        tol_MagDipole_X = 2e-2
        tol_NumErrZero = 1e-16

        # Get E which lives on edges
        e_numE = self.numFields_MagDipole_X[self.dm_x, 'e']

        # Apply data projection matrix to get E at the reciever locations
        ex_num, ey_num, ez_num = self.Pex*e_numE, self.Pey*e_numE, self.Pez*e_numE

        # Get analytic solution
        exa, _ , _  = EM.Analytics.FDEMDipolarfields.E_from_MagneticDipoleWholeSpace(self.XYZ_Ex, self.src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
        _ , eya , _ = EM.Analytics.FDEMDipolarfields.E_from_MagneticDipoleWholeSpace(self.XYZ_Ey, self.src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
        _ , _ , eza = EM.Analytics.FDEMDipolarfields.E_from_MagneticDipoleWholeSpace(self.XYZ_Ez, self.src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
        exa, eya, eza = Utils.mkvc(exa, 2), Utils.mkvc(eya, 2), Utils.mkvc(eza, 2)


        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('E_x:').rjust(4) , repr(np.linalg.norm(exa)).rjust(25), repr(np.linalg.norm(ex_num)).rjust(25), repr(np.linalg.norm(exa-ex_num)).rjust(25), str('').rjust(25)                                             , repr(np.linalg.norm(exa-ex_num) < tol_NumErrZero).center(12)
        print str('E_y:').rjust(4) , repr(np.linalg.norm(eya)).rjust(25), repr(np.linalg.norm(ey_num)).rjust(25), repr(np.linalg.norm(eya-ey_num)).rjust(25), repr(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya)).rjust(25), repr(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya) < tol_MagDipole_X).center(12)
        print str('E_z:').rjust(4) , repr(np.linalg.norm(eza)).rjust(25), repr(np.linalg.norm(ez_num)).rjust(25), repr(np.linalg.norm(eza-ez_num)).rjust(25), repr(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza)).rjust(25), repr(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza) < tol_MagDipole_X).center(12)
        print
        self.assertTrue(np.linalg.norm(exa-ex_num) < tol_NumErrZero, msg='Analytic and numeric solutions for Ex do not agree.')
        self.assertTrue(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya) < tol_MagDipole_X, msg='Analytic and numeric solutions for Ey do not agree.')
        self.assertTrue(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza) < tol_MagDipole_X, msg='Analytic and numeric solutions for Ez do not agree.')

        # Plot Tx and Rx locations on mesh
        if plotIt:

            x_Ex = self.XYZ_Ex[:, 0]
            x_Ey = self.XYZ_Ey[:, 0]
            x_Ez = self.XYZ_Ez[:, 0]

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fx[0], self.src_loc_Fx[2], 'ro', ms=8)
            ax.plot(self.XYZ_Ex[:, 0], self.XYZ_Ex[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Ey[:, 0], self.XYZ_Ey[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Ez[:, 0], self.XYZ_Ez[:, 2], 'k.', ms=6)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

            # Plot E
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, ex_num.real, exa.real, title='E_x Real')
            plotLine_num_ana(ax[1], x, ex_num.imag, exa.imag, title='E_x Imag')

            plotLine_num_ana(ax[2], x, ey_num.real, eya.real, title='E_y Real')
            plotLine_num_ana(ax[3], x, ey_num.imag, eya.imag, title='E_y Imag')

            plotLine_num_ana(ax[4], x, ez_num.real, eza.real, title='E_z Real')
            plotLine_num_ana(ax[5], x, ez_num.imag, eza.imag, title='E_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

    def test_3DMesh_X_MagDipoleTest_J(self):
        print('Testing J components of a X-oriented analytic harmonic magnetic dipole.')
        # Specify toleraces
        tol_MagDipole_X = 2e-2
        tol_NumErrZero = 1e-16

        # Get J which lives on faces
        j_numCC = self.numFields_MagDipole_X[self.dm_x, 'j']

        # Apply data projection matrix to get J at the reciever locations
        jx_num, jy_num, jz_num = self.Pccx*j_numCC, self.Pccy*j_numCC, self.Pccz*j_numCC

        # Get analytic solution
        jxa, jya, jza = EM.Analytics.FDEMDipolarfields.J_from_MagneticDipoleWholeSpace(self.XYZ_CC, self.src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
        jxa, jya, jza = Utils.mkvc(jxa, 2), Utils.mkvc(jya, 2), Utils.mkvc(jza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('J_x:').rjust(4) , repr(np.linalg.norm(jxa)).rjust(25), repr(np.linalg.norm(jx_num)).rjust(25), repr(np.linalg.norm(jxa-jx_num)).rjust(25), str('').rjust(25)                                             , repr(np.linalg.norm(jxa-jx_num) < tol_NumErrZero).center(12)
        print str('J_y:').rjust(4) , repr(np.linalg.norm(jya)).rjust(25), repr(np.linalg.norm(jy_num)).rjust(25), repr(np.linalg.norm(jya-jy_num)).rjust(25), repr(np.linalg.norm(jya-jy_num)/np.linalg.norm(jya)).rjust(25), repr(np.linalg.norm(jya-jy_num)/np.linalg.norm(jya) < tol_MagDipole_X).center(12)
        print str('J_z:').rjust(4) , repr(np.linalg.norm(jza)).rjust(25), repr(np.linalg.norm(jz_num)).rjust(25), repr(np.linalg.norm(jza-jz_num)).rjust(25), repr(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza)).rjust(25), repr(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza) < tol_MagDipole_X).center(12)
        print
        self.assertTrue(np.linalg.norm(jxa-jx_num) < tol_NumErrZero, msg='Analytic and numeric solutions for Jx do not agree.')
        self.assertTrue(np.linalg.norm(jya-jy_num)/np.linalg.norm(jya) < tol_MagDipole_X, msg='Analytic and numeric solutions for Jy do not agree.')
        self.assertTrue(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza) < tol_MagDipole_X, msg='Analytic and numeric solutions for Jz do not agree.')

        # Plot Tx and Rx locations on mesY
        if plotIt:

            x = self.XYZ_CC[:, 0]

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fx[0], self.src_loc_Fx[2], 'ro', ms=8)
            ax.plot(self.XYZ_CC[:, 0], self.XYZ_CC[:, 2], 'k.', ms=6)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

            # Plot J
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, jx_num.real, jxa.real, title='J_x Real')
            plotLine_num_ana(ax[1], x, jx_num.imag, jxa.imag, title='J_x Imag')

            plotLine_num_ana(ax[2], x, jy_num.real, jya.real, title='J_y Real')
            plotLine_num_ana(ax[3], x, jy_num.imag, jya.imag, title='J_y Imag')

            plotLine_num_ana(ax[4], x, jz_num.real, jza.real, title='J_z Real')
            plotLine_num_ana(ax[5], x, jz_num.imag, jza.imag, title='J_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

    def test_3DMesh_X_MagDipoleTest_H(self):
        print('Testing H components of a X-oriented analytic harmonic magnetic dipole.')
        # Specify toleraces
        tol_MagDipole_X = 4e-2
        tol_NumErrZero = 1e-16

        # Get H which lives on edges
        h_numCC = self.numFields_MagDipole_X[self.dm_x, 'h']

        # Apply data projection matrix to get J at the reciever locations
        hx_num, hy_num, hz_num = self.Pccx*h_numCC, self.Pccy*h_numCC, self.Pccz*h_numCC

        # Get analytic solution
        hxa, hya, hza = EM.Analytics.FDEMDipolarfields.H_from_MagneticDipoleWholeSpace(self.XYZ_CC, self.src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
        hxa, hya, hza = Utils.mkvc(hxa, 2), Utils.mkvc(hya, 2), Utils.mkvc(hza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('H_x:').rjust(4) , repr(np.linalg.norm(hxa)).rjust(25), repr(np.linalg.norm(hx_num)).rjust(25), repr(np.linalg.norm(hxa-hx_num)).rjust(25), repr(np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa)).rjust(25), repr(np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa) < tol_MagDipole_X).center(12)
        print str('H_y:').rjust(4) , repr(np.linalg.norm(hya)).rjust(25), repr(np.linalg.norm(hy_num)).rjust(25), repr(np.linalg.norm(hya-hy_num)).rjust(25), repr(np.linalg.norm(hya-hy_num)/np.linalg.norm(hya)).rjust(25), repr(np.linalg.norm(hya-hy_num)/np.linalg.norm(hya) < tol_MagDipole_X).center(12)
        print str('H_z:').rjust(4) , repr(np.linalg.norm(hza)).rjust(25), repr(np.linalg.norm(hz_num)).rjust(25), repr(np.linalg.norm(hza-hz_num)).rjust(25), repr(np.linalg.norm(hza-hz_num)/np.linalg.norm(hza)).rjust(25), repr(np.linalg.norm(hza-hz_num)/np.linalg.norm(hza) < tol_MagDipole_X).center(12)
        print
        self.assertTrue(np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa) < tol_MagDipole_X, msg='Analytic and numeric solutions for Hx do not agree.')
        self.assertTrue(np.linalg.norm(hya-hy_num)/np.linalg.norm(hya) < tol_MagDipole_X, msg='Analytic and numeric solutions for Hy do not agree.')
        self.assertTrue(np.linalg.norm(hza-hz_num)/np.linalg.norm(hza) < tol_MagDipole_X, msg='Analytic and numeric solutions for Hz do not agree.')

        # Plot Tx and Rx locations on mesh
        if plotIt:

            x = self.XYZ_CC[:, 0]

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fx[0], self.src_loc_Fx[2], 'ro', ms=8)
            ax.plot(self.XYZ_CC[:, 0], self.XYZ_CC[:, 2], 'k.', ms=6)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)


            # Plot J
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, hx_num.real, hxa.real, title='H_x Real')
            plotLine_num_ana(ax[1], x, hx_num.imag, hxa.imag, title='H_x Imag')

            plotLine_num_ana(ax[2], x, hy_num.real, hya.real, title='H_y Real')
            plotLine_num_ana(ax[3], x, hy_num.imag, hya.imag, title='H_y Imag')

            plotLine_num_ana(ax[4], x, hz_num.real, hza.real, title='H_z Real')
            plotLine_num_ana(ax[5], x, hz_num.imag, hza.imag, title='H_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

    def test_3DMesh_X_MagDipoleTest_B(self):
        print('Testing B components of a X-oriented analytic harmonic magnetic dipole.')

        # Specify toleraces
        tol_MagDipole_X = 4e-2
        tol_NumErrZero = 1e-16

        # Get E which lives on cell centres
        b_numF = self.numFields_MagDipole_X[self.dm_x, 'b']

        # Apply data projection matrix to get E at the reciever locations
        bx_num, by_num, bz_num = self.Pfx*b_numF, self.Pfy*b_numF, self.Pfz*b_numF

        # Get analytic solution
        bxa, _ , _  = EM.Analytics.FDEMDipolarfields.B_from_MagneticDipoleWholeSpace(self.XYZ_Fx, self.src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
        _ , bya , _ = EM.Analytics.FDEMDipolarfields.B_from_MagneticDipoleWholeSpace(self.XYZ_Fy, self.src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
        _ , _ , bza = EM.Analytics.FDEMDipolarfields.B_from_MagneticDipoleWholeSpace(self.XYZ_Fz, self.src_loc_Fx, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='X',kappa= self.kappa)
        bxa, bya, bza = Utils.mkvc(bxa, 2), Utils.mkvc(bya, 2), Utils.mkvc(bza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('B_x:').rjust(4) , repr(np.linalg.norm(bxa)).rjust(25), repr(np.linalg.norm(bx_num)).rjust(25), repr(np.linalg.norm(bxa-bx_num)).rjust(25), repr(np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa)).rjust(25), repr(np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa) < tol_MagDipole_X).center(12)
        print str('B_y:').rjust(4) , repr(np.linalg.norm(bya)).rjust(25), repr(np.linalg.norm(by_num)).rjust(25), repr(np.linalg.norm(bya-by_num)).rjust(25), repr(np.linalg.norm(bya-by_num)/np.linalg.norm(bya)).rjust(25), repr(np.linalg.norm(bya-by_num)/np.linalg.norm(bya) < tol_MagDipole_X).center(12)
        print str('B_z:').rjust(4) , repr(np.linalg.norm(bza)).rjust(25), repr(np.linalg.norm(bz_num)).rjust(25), repr(np.linalg.norm(bza-bz_num)).rjust(25), repr(np.linalg.norm(bza-bz_num)/np.linalg.norm(bza)).rjust(25), repr(np.linalg.norm(bza-bz_num)/np.linalg.norm(bza) < tol_MagDipole_X).center(12)
        print
        self.assertTrue(np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa) < tol_MagDipole_X, msg='Analytic and numeric solutions for Bx do not agree.')
        self.assertTrue(np.linalg.norm(bya-by_num)/np.linalg.norm(bya) < tol_MagDipole_X, msg='Analytic and numeric solutions for By do not agree.')
        self.assertTrue(np.linalg.norm(bza-bz_num)/np.linalg.norm(bza) < tol_MagDipole_X, msg='Analytic and numeric solutions for Bz do not agree.')


        # Plot Tx and Rx locations on mesh
        if plotIt:

            x_Bx = self.XYZ_Fx[:, 0]
            x_By = self.XYZ_Fy[:, 0]
            x_Bz = self.XYZ_Fz[:, 0]

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fx[0], self.src_loc_Fx[2], 'ro', ms=8)
            ax.plot(self.XYZ_Fx[:, 0], self.XYZ_Fx[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Fy[:, 0], self.XYZ_Fy[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Fz[:, 0], self.XYZ_Fz[:, 2], 'k.', ms=6)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

            # Plot E
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))

            plotLine_num_ana(ax[0], x_Bx, bx_num.real, bxa.real,
                             title='B_x Real')
            plotLine_num_ana(ax[1], x_Bx, bx_num.imag, bxa.imag,
                             title='B_x Imag')

            plotLine_num_ana(ax[2], x_By, by_num.real, bya.real,
                             title='B_y Real')
            plotLine_num_ana(ax[3], x_By, by_num.imag, bya.imag,
                             title='B_y Imag')

            plotLine_num_ana(ax[4], x_Bz, bz_num.real, bza.real,
                             title='B_z Real')
            plotLine_num_ana(ax[5], x_Bz, bz_num.imag, bza.imag,
                             title='B_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

class Y_MaDipoleTest_3DMesh(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        print('Testing a Y-oriented analytic harmonic magnetic dipole against the numerical solution on a 3D-tesnsor mesh.')

        # Define model parameters
        self.sigmaback = SIGMABACK
        # mu = mu_0*(1+kappa)
        self.kappa = KAPPA

        # Create 3D mesh
        self.mesh = setUpMesh()

        # Set source parameters
        self.freq = 500.
        src_loc = np.r_[0., 0., -35]
        src_loc_CCInd = Utils.closestPoints(self.mesh, src_loc, 'CC')
        self.src_loc_CC = self.mesh.gridCC[src_loc_CCInd,:]
        self.src_loc_CC = self.src_loc_CC[0]

        # Compute skin depth
        skdpth = 500. / np.sqrt(self.sigmaback * self.freq)

        # make sure mesh is big enough
        # print('skin depth =', skdpth)
        # self.assertTrue(self.mesh.hx.sum() > skdpth*2.)
        # self.assertTrue(self.mesh.hy.sum() > skdpth*2.)
        # self.assertTrue(self.mesh.hz.sum() > skdpth*2.)

        # Create wholespace models
        SigmaBack = self.sigmaback*np.ones((self.mesh.nC))
        MuBack = (mu_0*(1 + self.kappa))*np.ones((self.mesh.nC))

        # Define reciever locations
        xlim = 40. # x locations from -50 to 50
        xInd = np.where(np.abs(self.mesh.vectorCCx) < xlim)
        x = self.mesh.vectorCCx[xInd[0]]
        y = 10.
        z = 30.

        # where we choose to measure
        XYZ = Utils.ndgrid(x, np.r_[y], np.r_[z])

        # Cell centred recievers
        XYZ_CCInd = Utils.closestPoints(self.mesh, XYZ, 'CC')
        self.XYZ_CC = self.mesh.gridCC[XYZ_CCInd,:]

        # Edge recievers
        XYZ_ExInd = Utils.closestPoints(self.mesh, XYZ, 'Ex')
        self.XYZ_Ex = self.mesh.gridEx[XYZ_ExInd,:]
        XYZ_EyInd = Utils.closestPoints(self.mesh, XYZ, 'Ey')
        self.XYZ_Ey = self.mesh.gridEy[XYZ_EyInd,:]
        XYZ_EzInd = Utils.closestPoints(self.mesh, XYZ, 'Ez')
        self.XYZ_Ez = self.mesh.gridEz[XYZ_EzInd,:]

        # Face recievers
        XYZ_FxInd = Utils.closestPoints(self.mesh, XYZ, 'Fx')
        self.XYZ_Fx = self.mesh.gridFx[XYZ_FxInd,:]
        XYZ_FyInd = Utils.closestPoints(self.mesh, XYZ, 'Fy')
        self.XYZ_Fy = self.mesh.gridFy[XYZ_FyInd,:]
        XYZ_FzInd = Utils.closestPoints(self.mesh, XYZ, 'Fz')
        self.XYZ_Fz = self.mesh.gridFz[XYZ_FzInd,:]

        # Form data interpolation matrices
        Pcc = self.mesh.getInterpolationMat(self.XYZ_CC, 'CC')
        Zero = sp.csr_matrix(Pcc.shape)
        self.Pccx, self.Pccy, self.Pccz = sp.hstack([Pcc, Zero, Zero]), sp.hstack([Zero, Pcc, Zero]), sp.hstack([Zero, Zero, Pcc])

        self.Pex, self.Pey, self.Pez = self.mesh.getInterpolationMat(self.XYZ_Ex, 'Ex'), self.mesh.getInterpolationMat(self.XYZ_Ey, 'Ey'), self.mesh.getInterpolationMat(self.XYZ_Ez, 'Ez')
        self.Pfx, self.Pfy, self.Pfz = self.mesh.getInterpolationMat(self.XYZ_Fx, 'Fx'), self.mesh.getInterpolationMat(self.XYZ_Fy, 'Fy'), self.mesh.getInterpolationMat(self.XYZ_Fz, 'Fz')

        # Define the source
        # Search over x-faces to find face nearest src_loc
        s_ind = Utils.closestPoints(self.mesh, self.src_loc_CC, 'Fy') + self.mesh.nFx
        dm = np.zeros(self.mesh.nF, dtype=complex)
        dm[s_ind] = (-1j*(2*np.pi*self.freq)*(mu_0*(1 + self.kappa)))/self.mesh.hy.min()
        self.dm_y = [EM.FDEM.Src.RawVec_m([], self.freq, dm/self.mesh.area)]

        self.src_loc_Fy = self.mesh.gridFy[Utils.closestPoints(self.mesh, self.src_loc_CC, 'Fy'),:]
        self.src_loc_Fy = self.src_loc_Fy[0]

        # Create survey and problem object
        survey = EM.FDEM.Survey(self.dm_y)
        mapping = [('sigma', Maps.IdentityMap(self.mesh)), ('mu', Maps.IdentityMap(self.mesh))]
        problem = EM.FDEM.Problem3D_b(self.mesh, mapping=mapping)

        # Pair problem and survey
        problem.pair(survey)
        problem.Solver = Solver

        # Solve forward problem
        self.numFields_MagDipole_Y = problem.fields(np.r_[SigmaBack, MuBack])

        # setUp_Done = True

    def test_3DMesh_Y_MagDipoleTest_E(self):
        print('Testing E components of a Y-oriented analytic harmonic magnetic dipole.')

        # Specify toleraces
        tol_MagDipole_Y = 2e-2
        tol_NumErrZero = 1e-16

        # Get E which lives on edges
        e_numE = self.numFields_MagDipole_Y[self.dm_y, 'e']

        # Apply data projection matrix to get E at the reciever locations
        ex_num, ey_num, ez_num = self.Pex*e_numE, self.Pey*e_numE, self.Pez*e_numE

        # Get analytic solution
        exa, _ , _  = EM.Analytics.FDEMDipolarfields.E_from_MagneticDipoleWholeSpace(self.XYZ_Ex, self.src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
        _ , eya , _ = EM.Analytics.FDEMDipolarfields.E_from_MagneticDipoleWholeSpace(self.XYZ_Ey, self.src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
        _ , _ , eza = EM.Analytics.FDEMDipolarfields.E_from_MagneticDipoleWholeSpace(self.XYZ_Ez, self.src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
        exa, eya, eza = Utils.mkvc(exa, 2), Utils.mkvc(eya, 2), Utils.mkvc(eza, 2)


        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('E_x:').rjust(4) , repr(np.linalg.norm(exa)).rjust(25), repr(np.linalg.norm(ex_num)).rjust(25), repr(np.linalg.norm(exa-ex_num)).rjust(25), repr(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa)).rjust(25), repr(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa) < tol_MagDipole_Y).center(12)
        print str('E_y:').rjust(4) , repr(np.linalg.norm(eya)).rjust(25), repr(np.linalg.norm(ey_num)).rjust(25), repr(np.linalg.norm(eya-ey_num)).rjust(25), str('').rjust(25)                                             , repr(np.linalg.norm(eya-ey_num) < tol_NumErrZero).center(12)
        print str('E_z:').rjust(4) , repr(np.linalg.norm(eza)).rjust(25), repr(np.linalg.norm(ez_num)).rjust(25), repr(np.linalg.norm(eza-ez_num)).rjust(25), repr(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza)).rjust(25), repr(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza) < tol_MagDipole_Y).center(12)
        print
        self.assertTrue(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa) < tol_MagDipole_Y, msg='Analytic and numeric solutions for Ex do not agree.')
        self.assertTrue(np.linalg.norm(eya-ey_num) < tol_NumErrZero, msg='Analytic and numeric solutions for Ey do not agree.')
        self.assertTrue(np.linalg.norm(eza-ez_num)/np.linalg.norm(eza) < tol_MagDipole_Y, msg='Analytic and numeric solutions for Ez do not agree.')

        # Plot Tx and Rx locations on mesh
        if plotIt:

            x_Ex = self.XYZ_Ex[:, 0]
            x_Ey = self.XYZ_Ey[:, 0]
            x_Ez = self.XYZ_Ez[:, 0]

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fy[0], self.src_loc_Fy[2], 'ro', ms=8)
            ax.plot(self.XYZ_Ex[:, 0], self.XYZ_Ex[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Ey[:, 0], self.XYZ_Ey[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Ez[:, 0], self.XYZ_Ez[:, 2], 'k.', ms=6)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

            # Plot E
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, ex_num.real, exa.real, title='E_x Real')
            plotLine_num_ana(ax[1], x, ex_num.imag, exa.imag, title='E_x Imag')

            plotLine_num_ana(ax[2], x, ey_num.real, eya.real, title='E_y Real')
            plotLine_num_ana(ax[3], x, ey_num.imag, eya.imag, title='E_y Imag')

            plotLine_num_ana(ax[4], x, ez_num.real, eza.real, title='E_z Real')
            plotLine_num_ana(ax[5], x, ez_num.imag, eza.imag, title='E_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

    def test_3DMesh_Y_MagDipoleTest_J(self):
        print('Testing J components of a Y-oriented analytic harmonic magnetic dipole.')
        # Specify toleraces
        tol_MagDipole_Y = 2e-2
        tol_NumErrZero = 1e-16

        # Get J which lives on faces
        j_numCC = self.numFields_MagDipole_Y[self.dm_y, 'j']

        # Apply data projection matrix to get J at the reciever locations
        jx_num, jy_num, jz_num = self.Pccx*j_numCC, self.Pccy*j_numCC, self.Pccz*j_numCC

        # Get analytic solution
        jxa, jya, jza = EM.Analytics.FDEMDipolarfields.J_from_MagneticDipoleWholeSpace(self.XYZ_CC, self.src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
        jxa, jya, jza = Utils.mkvc(jxa, 2), Utils.mkvc(jya, 2), Utils.mkvc(jza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('J_x:').rjust(4) , repr(np.linalg.norm(jxa)).rjust(25), repr(np.linalg.norm(jx_num)).rjust(25), repr(np.linalg.norm(jxa-jx_num)).rjust(25), repr(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa)).rjust(25), repr(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa) < tol_MagDipole_Y).center(12)
        print str('J_y:').rjust(4) , repr(np.linalg.norm(jya)).rjust(25), repr(np.linalg.norm(jy_num)).rjust(25), repr(np.linalg.norm(jya-jy_num)).rjust(25), str('').rjust(25)                                             , repr(np.linalg.norm(jya-jy_num) < tol_NumErrZero).center(12)
        print str('J_z:').rjust(4) , repr(np.linalg.norm(jza)).rjust(25), repr(np.linalg.norm(jz_num)).rjust(25), repr(np.linalg.norm(jza-jz_num)).rjust(25), repr(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza)).rjust(25), repr(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza) < tol_MagDipole_Y).center(12)
        print
        self.assertTrue(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa) < tol_MagDipole_Y, msg='Analytic and numeric solutions for Jx do not agree.')
        self.assertTrue(np.linalg.norm(jya-jy_num) < tol_NumErrZero, msg='Analytic and numeric solutions for Jy do not agree.')
        self.assertTrue(np.linalg.norm(jza-jz_num)/np.linalg.norm(jza) < tol_MagDipole_Y, msg='Analytic and numeric solutions for Jz do not agree.')

        # Plot Tx and Rx locations on mesY
        if plotIt:

            x = self.XYZ_CC[:, 0]

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fy[0], self.src_loc_Fy[2], 'ro', ms=8)
            ax.plot(self.XYZ_CC[:, 0], self.XYZ_CC[:, 2], 'k.', ms=6)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

            # Plot J
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, jx_num.real, jxa.real, title='J_x Real')
            plotLine_num_ana(ax[1], x, jx_num.imag, jxa.imag, title='J_x Imag')

            plotLine_num_ana(ax[2], x, jy_num.real, jya.real, title='J_y Real')
            plotLine_num_ana(ax[3], x, jy_num.imag, jya.imag, title='J_y Imag')

            plotLine_num_ana(ax[4], x, jz_num.real, jza.real, title='J_z Real')
            plotLine_num_ana(ax[5], x, jz_num.imag, jza.imag, title='J_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

    def test_3DMesh_Y_MagDipoleTest_H(self):
        print('Testing H components of a Y-oriented analytic harmonic magnetic dipole.')
        # Specify toleraces
        tol_MagDipole_Y = 4e-2
        tol_NumErrZero = 1e-16

        # Get H which lives on edges
        h_numCC = self.numFields_MagDipole_Y[self.dm_y, 'h']

        # Apply data projection matrix to get J at the reciever locations
        hx_num, hy_num, hz_num = self.Pccx*h_numCC, self.Pccy*h_numCC, self.Pccz*h_numCC

        # Get analytic solution
        hxa, hya, hza = EM.Analytics.FDEMDipolarfields.H_from_MagneticDipoleWholeSpace(self.XYZ_CC, self.src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
        hxa, hya, hza = Utils.mkvc(hxa, 2), Utils.mkvc(hya, 2), Utils.mkvc(hza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('H_x:').rjust(4) , repr(np.linalg.norm(hxa)).rjust(25), repr(np.linalg.norm(hx_num)).rjust(25), repr(np.linalg.norm(hxa-hx_num)).rjust(25), repr(np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa)).rjust(25), repr(np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa) < tol_MagDipole_Y).center(12)
        print str('H_y:').rjust(4) , repr(np.linalg.norm(hya)).rjust(25), repr(np.linalg.norm(hy_num)).rjust(25), repr(np.linalg.norm(hya-hy_num)).rjust(25), repr(np.linalg.norm(hya-hy_num)/np.linalg.norm(hya)).rjust(25), repr(np.linalg.norm(hya-hy_num)/np.linalg.norm(hya) < tol_MagDipole_Y).center(12)
        print str('H_z:').rjust(4) , repr(np.linalg.norm(hza)).rjust(25), repr(np.linalg.norm(hz_num)).rjust(25), repr(np.linalg.norm(hza-hz_num)).rjust(25), repr(np.linalg.norm(hza-hz_num)/np.linalg.norm(hza)).rjust(25), repr(np.linalg.norm(hza-hz_num)/np.linalg.norm(hza) < tol_MagDipole_Y).center(12)
        print
        self.assertTrue(np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa) < tol_MagDipole_Y, msg='Analytic and numeric solutions for Hx do not agree.')
        self.assertTrue(np.linalg.norm(hya-hy_num)/np.linalg.norm(hya) < tol_MagDipole_Y, msg='Analytic and numeric solutions for Hy do not agree.')
        self.assertTrue(np.linalg.norm(hza-hz_num)/np.linalg.norm(hza) < tol_MagDipole_Y, msg='Analytic and numeric solutions for Hz do not agree.')

        # Plot Tx and Rx locations on mesh
        if plotIt:

            x = self.XYZ_CC[:, 0]

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fy[0], self.src_loc_Fy[2], 'ro', ms=8)
            ax.plot(self.XYZ_CC[:, 0], self.XYZ_CC[:, 2], 'k.', ms=6)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)


            # Plot H
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, hx_num.real, hxa.real, title='H_x Real')
            plotLine_num_ana(ax[1], x, hx_num.imag, hxa.imag, title='H_x Imag')

            plotLine_num_ana(ax[2], x, hy_num.real, hya.real, title='H_y Real')
            plotLine_num_ana(ax[3], x, hy_num.imag, hya.imag, title='H_y Imag')

            plotLine_num_ana(ax[4], x, hz_num.real, hza.real, title='H_z Real')
            plotLine_num_ana(ax[5], x, hz_num.imag, hza.imag, title='H_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

    def test_3DMesh_Y_MagDipoleTest_B(self):
        print('Testing B components of a Y-oriented analytic harmonic magnetic dipole.')

        # Specify toleraces
        tol_MagDipole_Y = 4e-2
        tol_NumErrZero = 1e-16

        # Get E which lives on cell centres
        b_numF = self.numFields_MagDipole_Y[self.dm_y, 'b']

        # Apply data projection matrix to get E at the reciever locations
        bx_num, by_num, bz_num = self.Pfx*b_numF, self.Pfy*b_numF, self.Pfz*b_numF

        # Get analytic solution
        bxa, _ , _  = EM.Analytics.FDEMDipolarfields.B_from_MagneticDipoleWholeSpace(self.XYZ_Fx, self.src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
        _ , bya , _ = EM.Analytics.FDEMDipolarfields.B_from_MagneticDipoleWholeSpace(self.XYZ_Fy, self.src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
        _ , _ , bza = EM.Analytics.FDEMDipolarfields.B_from_MagneticDipoleWholeSpace(self.XYZ_Fz, self.src_loc_Fy, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Y',kappa= self.kappa)
        bxa, bya, bza = Utils.mkvc(bxa, 2), Utils.mkvc(bya, 2), Utils.mkvc(bza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('B_x:').rjust(4) , repr(np.linalg.norm(bxa)).rjust(25), repr(np.linalg.norm(bx_num)).rjust(25), repr(np.linalg.norm(bxa-bx_num)).rjust(25), repr(np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa)).rjust(25), repr(np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa) < tol_MagDipole_Y).center(12)
        print str('B_y:').rjust(4) , repr(np.linalg.norm(bya)).rjust(25), repr(np.linalg.norm(by_num)).rjust(25), repr(np.linalg.norm(bya-by_num)).rjust(25), repr(np.linalg.norm(bya-by_num)/np.linalg.norm(bya)).rjust(25), repr(np.linalg.norm(bya-by_num)/np.linalg.norm(bya) < tol_MagDipole_Y).center(12)
        print str('B_z:').rjust(4) , repr(np.linalg.norm(bza)).rjust(25), repr(np.linalg.norm(bz_num)).rjust(25), repr(np.linalg.norm(bza-bz_num)).rjust(25), repr(np.linalg.norm(bza-bz_num)/np.linalg.norm(bza)).rjust(25), repr(np.linalg.norm(bza-bz_num)/np.linalg.norm(bza) < tol_MagDipole_Y).center(12)
        print
        self.assertTrue(np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa) < tol_MagDipole_Y, msg='Analytic and numeric solutions for Bx do not agree.')
        self.assertTrue(np.linalg.norm(bya-by_num)/np.linalg.norm(bya) < tol_MagDipole_Y, msg='Analytic and numeric solutions for By do not agree.')
        self.assertTrue(np.linalg.norm(bza-bz_num)/np.linalg.norm(bza) < tol_MagDipole_Y, msg='Analytic and numeric solutions for Bz do not agree.')


        # Plot Tx and Rx locations on mesh
        if plotIt:

            x_Bx = self.XYZ_Fx[:, 0]
            x_By = self.XYZ_Fy[:, 0]
            x_Bz = self.XYZ_Fz[:, 0]

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fy[0], self.src_loc_Fy[2], 'ro', ms=8)
            ax.plot(self.XYZ_Fx[:, 0], self.XYZ_Fx[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Fy[:, 0], self.XYZ_Fy[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Fz[:, 0], self.XYZ_Fz[:, 2], 'k.', ms=6)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

            # Plot E
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))

            plotLine_num_ana(ax[0], x_Bx, bx_num.real, bxa.real,
                             title='B_x Real')
            plotLine_num_ana(ax[1], x_Bx, bx_num.imag, bxa.imag,
                             title='B_x Imag')

            plotLine_num_ana(ax[2], x_By, by_num.real, bya.real,
                             title='B_y Real')
            plotLine_num_ana(ax[3], x_By, by_num.imag, bya.imag,
                             title='B_y Imag')

            plotLine_num_ana(ax[4], x_Bz, bz_num.real, bza.real,
                             title='B_z Real')
            plotLine_num_ana(ax[5], x_Bz, bz_num.imag, bza.imag,
                             title='B_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

class Z_MaDipoleTest_3DMesh(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        print('Testing a Z-oriented analytic harmonic magnetic dipole against the numerical solution on a 3D-tesnsor mesh.')

        # Define model parameters
        self.sigmaback = SIGMABACK
        # mu = mu_0*(1+kappa)
        self.kappa = KAPPA

        # Create 3D mesh
        self.mesh = setUpMesh()

        # Set source parameters
        self.freq = 500.
        src_loc = np.r_[0., 0., -35]
        src_loc_CCInd = Utils.closestPoints(self.mesh, src_loc, 'CC')
        self.src_loc_CC = self.mesh.gridCC[src_loc_CCInd,:]
        self.src_loc_CC = self.src_loc_CC[0]

        # Compute skin depth
        skdpth = 500. / np.sqrt(self.sigmaback * self.freq)

        # make sure mesh is big enough
        # print('skin depth =', skdpth)
        # self.assertTrue(self.mesh.hx.sum() > skdpth*2.)
        # self.assertTrue(self.mesh.hy.sum() > skdpth*2.)
        # self.assertTrue(self.mesh.hz.sum() > skdpth*2.)

        # Create wholespace models
        SigmaBack = self.sigmaback*np.ones((self.mesh.nC))
        MuBack = (mu_0*(1 + self.kappa))*np.ones((self.mesh.nC))

        # Define reciever locations
        xlim = 40. # x locations from -50 to 50
        xInd = np.where(np.abs(self.mesh.vectorCCx) < xlim)
        x = self.mesh.vectorCCx[xInd[0]]
        y = 10.
        z = 30.

        # where we choose to measure
        XYZ = Utils.ndgrid(x, np.r_[y], np.r_[z])

        # Cell centred recievers
        XYZ_CCInd = Utils.closestPoints(self.mesh, XYZ, 'CC')
        self.XYZ_CC = self.mesh.gridCC[XYZ_CCInd,:]

        # Edge recievers
        XYZ_ExInd = Utils.closestPoints(self.mesh, XYZ, 'Ex')
        self.XYZ_Ex = self.mesh.gridEx[XYZ_ExInd,:]
        XYZ_EyInd = Utils.closestPoints(self.mesh, XYZ, 'Ey')
        self.XYZ_Ey = self.mesh.gridEy[XYZ_EyInd,:]
        XYZ_EzInd = Utils.closestPoints(self.mesh, XYZ, 'Ez')
        self.XYZ_Ez = self.mesh.gridEz[XYZ_EzInd,:]

        # Face recievers
        XYZ_FxInd = Utils.closestPoints(self.mesh, XYZ, 'Fx')
        self.XYZ_Fx = self.mesh.gridFx[XYZ_FxInd,:]
        XYZ_FyInd = Utils.closestPoints(self.mesh, XYZ, 'Fy')
        self.XYZ_Fy = self.mesh.gridFy[XYZ_FyInd,:]
        XYZ_FzInd = Utils.closestPoints(self.mesh, XYZ, 'Fz')
        self.XYZ_Fz = self.mesh.gridFz[XYZ_FzInd,:]

        # Form data interpolation matrices
        Pcc = self.mesh.getInterpolationMat(self.XYZ_CC, 'CC')
        Zero = sp.csr_matrix(Pcc.shape)
        self.Pccx, self.Pccy, self.Pccz = sp.hstack([Pcc, Zero, Zero]), sp.hstack([Zero, Pcc, Zero]), sp.hstack([Zero, Zero, Pcc])

        self.Pex, self.Pey, self.Pez = self.mesh.getInterpolationMat(self.XYZ_Ex, 'Ex'), self.mesh.getInterpolationMat(self.XYZ_Ey, 'Ey'), self.mesh.getInterpolationMat(self.XYZ_Ez, 'Ez')
        self.Pfx, self.Pfy, self.Pfz = self.mesh.getInterpolationMat(self.XYZ_Fx, 'Fx'), self.mesh.getInterpolationMat(self.XYZ_Fy, 'Fy'), self.mesh.getInterpolationMat(self.XYZ_Fz, 'Fz')

        # Define the source
        # Search over x-faces to find face nearest src_loc
        s_ind = Utils.closestPoints(self.mesh, self.src_loc_CC, 'Fz') + self.mesh.nFx + self.mesh.nFy
        dm = np.zeros(self.mesh.nF, dtype=complex)
        dm[s_ind] = (-1j*(2*np.pi*self.freq)*(mu_0*(1 + self.kappa)))/self.mesh.hz.min()
        self.dm_z = [EM.FDEM.Src.RawVec_m([], self.freq, dm/self.mesh.area)]

        self.src_loc_Fz = self.mesh.gridFz[Utils.closestPoints(self.mesh, self.src_loc_CC, 'Fz'),:]
        self.src_loc_Fz = self.src_loc_Fz[0]

        # Create survey and problem object
        survey = EM.FDEM.Survey(self.dm_z)
        mapping = [('sigma', Maps.IdentityMap(self.mesh)), ('mu', Maps.IdentityMap(self.mesh))]
        problem = EM.FDEM.Problem3D_b(self.mesh, mapping=mapping)

        # Pair problem and survey
        problem.pair(survey)
        problem.Solver = Solver

        # Solve forward problem
        self.numFields_MagDipole_Z = problem.fields(np.r_[SigmaBack, MuBack])

        # setUp_Done = True

    def test_3DMesh_Z_MagDipoleTest_E(self):
        print('Testing E components of a Z-oriented analytic harmonic magnetic dipole.')

        # Specify toleraces
        tol_MagDipole_Z = 2e-2
        tol_NumErrZero = 1e-16

        # Get E which lives on edges
        e_numE = self.numFields_MagDipole_Z[self.dm_z, 'e']

        # Apply data projection matrix to get E at the reciever locations
        ex_num, ey_num, ez_num = self.Pex*e_numE, self.Pey*e_numE, self.Pez*e_numE

        # Get analytic solution
        exa, _ , _  = EM.Analytics.FDEMDipolarfields.E_from_MagneticDipoleWholeSpace(self.XYZ_Ex, self.src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        _ , eya , _ = EM.Analytics.FDEMDipolarfields.E_from_MagneticDipoleWholeSpace(self.XYZ_Ey, self.src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        _ , _ , eza = EM.Analytics.FDEMDipolarfields.E_from_MagneticDipoleWholeSpace(self.XYZ_Ez, self.src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        exa, eya, eza = Utils.mkvc(exa, 2), Utils.mkvc(eya, 2), Utils.mkvc(eza, 2)


        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('E_x:').rjust(4) , repr(np.linalg.norm(exa)).rjust(25), repr(np.linalg.norm(ex_num)).rjust(25), repr(np.linalg.norm(exa-ex_num)).rjust(25), repr(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa)).rjust(25), repr(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa) < tol_MagDipole_Z).center(12)
        print str('E_y:').rjust(4) , repr(np.linalg.norm(eya)).rjust(25), repr(np.linalg.norm(ey_num)).rjust(25), repr(np.linalg.norm(eya-ey_num)).rjust(25), repr(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya)).rjust(25), repr(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya) < tol_MagDipole_Z).center(12)
        print str('E_z:').rjust(4) , repr(np.linalg.norm(eza)).rjust(25), repr(np.linalg.norm(ez_num)).rjust(25), repr(np.linalg.norm(eza-ez_num)).rjust(25), str('').rjust(25)                                             , repr(np.linalg.norm(eza-ez_num) < tol_NumErrZero).center(12)
        print
        self.assertTrue(np.linalg.norm(exa-ex_num)/np.linalg.norm(exa) < tol_MagDipole_Z, msg='Analytic and numeric solutions for Ex do not agree.')
        self.assertTrue(np.linalg.norm(eya-ey_num)/np.linalg.norm(eya) < tol_MagDipole_Z, msg='Analytic and numeric solutions for Ey do not agree.')
        self.assertTrue(np.linalg.norm(eza-ez_num) < tol_NumErrZero, msg='Analytic and numeric solutions for Ez do not agree.')

        # Plot Tx and Rx locations on mesh
        if plotIt:

            x_Ex = self.XYZ_Ex[:, 0]
            x_Ey = self.XYZ_Ey[:, 0]
            x_Ez = self.XYZ_Ez[:, 0]

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fz[0], self.src_loc_Fz[2], 'ro', ms=8)
            ax.plot(self.XYZ_Ex[:, 0], self.XYZ_Ex[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Ey[:, 0], self.XYZ_Ey[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Ez[:, 0], self.XYZ_Ez[:, 2], 'k.', ms=6)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

            # Plot E
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, ex_num.real, exa.real, title='E_x Real')
            plotLine_num_ana(ax[1], x, ex_num.imag, exa.imag, title='E_x Imag')

            plotLine_num_ana(ax[2], x, ey_num.real, eya.real, title='E_y Real')
            plotLine_num_ana(ax[3], x, ey_num.imag, eya.imag, title='E_y Imag')

            plotLine_num_ana(ax[4], x, ez_num.real, eza.real, title='E_z Real')
            plotLine_num_ana(ax[5], x, ez_num.imag, eza.imag, title='E_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

    def test_3DMesh_Z_MagDipoleTest_J(self):
        print('Testing J components of a Z-oriented analytic harmonic magnetic dipole.')
        # Specify toleraces
        tol_MagDipole_Z = 2e-2
        tol_NumErrZero = 1e-16

        # Get J which lives on faces
        j_numCC = self.numFields_MagDipole_Z[self.dm_z, 'j']

        # Apply data projection matrix to get J at the reciever locations
        jx_num, jy_num, jz_num = self.Pccx*j_numCC, self.Pccy*j_numCC, self.Pccz*j_numCC

        # Get analytic solution
        jxa, jya, jza = EM.Analytics.FDEMDipolarfields.J_from_MagneticDipoleWholeSpace(self.XYZ_CC, self.src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        jxa, jya, jza = Utils.mkvc(jxa, 2), Utils.mkvc(jya, 2), Utils.mkvc(jza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('J_x:').rjust(4) , repr(np.linalg.norm(jxa)).rjust(25), repr(np.linalg.norm(jx_num)).rjust(25), repr(np.linalg.norm(jxa-jx_num)).rjust(25), repr(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa)).rjust(25), repr(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa) < tol_MagDipole_Z).center(12)
        print str('J_y:').rjust(4) , repr(np.linalg.norm(jya)).rjust(25), repr(np.linalg.norm(jy_num)).rjust(25), repr(np.linalg.norm(jya-jy_num)).rjust(25), repr(np.linalg.norm(jya-jy_num)/np.linalg.norm(jya)).rjust(25), repr(np.linalg.norm(jya-jy_num)/np.linalg.norm(jya) < tol_MagDipole_Z).center(12)
        print str('J_z:').rjust(4) , repr(np.linalg.norm(jza)).rjust(25), repr(np.linalg.norm(jz_num)).rjust(25), repr(np.linalg.norm(jza-jz_num)).rjust(25), str('').rjust(25)                                             , repr(np.linalg.norm(jza-jz_num) < tol_NumErrZero).center(12)
        print
        self.assertTrue(np.linalg.norm(jxa-jx_num)/np.linalg.norm(jxa) < tol_MagDipole_Z, msg='Analytic and numeric solutions for Jx do not agree.')
        self.assertTrue(np.linalg.norm(jya-jy_num)/np.linalg.norm(jya) < tol_MagDipole_Z, msg='Analytic and numeric solutions for Jy do not agree.')
        self.assertTrue(np.linalg.norm(jza-jz_num) < tol_NumErrZero, msg='Analytic and numeric solutions for Jz do not agree.')

        # Plot Tx and Rx locations on mesY
        if plotIt:

            x = self.XYZ_CC[:, 0]

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fz[0], self.src_loc_Fz[2], 'ro', ms=8)
            ax.plot(self.XYZ_CC[:, 0], self.XYZ_CC[:, 2], 'k.', ms=6)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

            # Plot J
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, jx_num.real, jxa.real, title='J_x Real')
            plotLine_num_ana(ax[1], x, jx_num.imag, jxa.imag, title='J_x Imag')

            plotLine_num_ana(ax[2], x, jy_num.real, jya.real, title='J_y Real')
            plotLine_num_ana(ax[3], x, jy_num.imag, jya.imag, title='J_y Imag')

            plotLine_num_ana(ax[4], x, jz_num.real, jza.real, title='J_z Real')
            plotLine_num_ana(ax[5], x, jz_num.imag, jza.imag, title='J_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

    def test_3DMesh_Z_MagDipoleTest_H(self):
        print('Testing H components of a Z-oriented analytic harmonic magnetic dipole.')
        # Specify toleraces
        tol_MagDipole_Z = 2e-2
        tol_NumErrZero = 1e-16

        # Get H which lives on edges
        h_numCC = self.numFields_MagDipole_Z[self.dm_z, 'h']

        # Apply data projection matrix to get J at the reciever locations
        hx_num, hy_num, hz_num = self.Pccx*h_numCC, self.Pccy*h_numCC, self.Pccz*h_numCC

        # Get analytic solution
        hxa, hya, hza = EM.Analytics.FDEMDipolarfields.H_from_MagneticDipoleWholeSpace(self.XYZ_CC, self.src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        hxa, hya, hza = Utils.mkvc(hxa, 2), Utils.mkvc(hya, 2), Utils.mkvc(hza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('H_x:').rjust(4) , repr(np.linalg.norm(hxa)).rjust(25), repr(np.linalg.norm(hx_num)).rjust(25), repr(np.linalg.norm(hxa-hx_num)).rjust(25), repr(np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa)).rjust(25), repr(np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa) < tol_MagDipole_Z).center(12)
        print str('H_y:').rjust(4) , repr(np.linalg.norm(hya)).rjust(25), repr(np.linalg.norm(hy_num)).rjust(25), repr(np.linalg.norm(hya-hy_num)).rjust(25), repr(np.linalg.norm(hya-hy_num)/np.linalg.norm(hya)).rjust(25), repr(np.linalg.norm(hya-hy_num)/np.linalg.norm(hya) < tol_MagDipole_Z).center(12)
        print str('H_z:').rjust(4) , repr(np.linalg.norm(hza)).rjust(25), repr(np.linalg.norm(hz_num)).rjust(25), repr(np.linalg.norm(hza-hz_num)).rjust(25), repr(np.linalg.norm(hza-hz_num)/np.linalg.norm(hza)).rjust(25), repr(np.linalg.norm(hza-hz_num)/np.linalg.norm(hza) < tol_MagDipole_Z).center(12)
        print
        self.assertTrue(np.linalg.norm(hxa-hx_num)/np.linalg.norm(hxa) < tol_MagDipole_Z, msg='Analytic and numeric solutions for Hx do not agree.')
        self.assertTrue(np.linalg.norm(hya-hy_num)/np.linalg.norm(hya) < tol_MagDipole_Z, msg='Analytic and numeric solutions for Hy do not agree.')
        self.assertTrue(np.linalg.norm(hza-hz_num)/np.linalg.norm(hza) < tol_MagDipole_Z, msg='Analytic and numeric solutions for Hz do not agree.')

        # Plot Tx and Rx locations on mesh
        if plotIt:

            x = self.XYZ_CC[:, 0]

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fz[0], self.src_loc_Fz[2], 'ro', ms=8)
            ax.plot(self.XYZ_CC[:, 0], self.XYZ_CC[:, 2], 'k.', ms=6)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)


            # Plot J
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax = Utils.mkvc(ax)

            x = self.XYZ_CC[:, 0]

            plotLine_num_ana(ax[0], x, hx_num.real, hxa.real, title='H_x Real')
            plotLine_num_ana(ax[1], x, hx_num.imag, hxa.imag, title='H_x Imag')

            plotLine_num_ana(ax[2], x, hy_num.real, hya.real, title='H_y Real')
            plotLine_num_ana(ax[3], x, hy_num.imag, hya.imag, title='H_y Imag')

            plotLine_num_ana(ax[4], x, hz_num.real, hza.real, title='H_z Real')
            plotLine_num_ana(ax[5], x, hz_num.imag, hza.imag, title='H_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()

    def test_3DMesh_Z_MagDipoleTest_B(self):
        print('Testing B components of a Z-oriented analytic harmonic magnetic dipole.')

        # Specify toleraces
        tol_MagDipole_Z = 2e-2
        tol_NumErrZero = 1e-16

        # Get E which lives on cell centres
        b_numF = self.numFields_MagDipole_Z[self.dm_z, 'b']

        # Apply data projection matrix to get E at the reciever locations
        bx_num, by_num, bz_num = self.Pfx*b_numF, self.Pfy*b_numF, self.Pfz*b_numF

        # Get analytic solution
        bxa, _ , _  = EM.Analytics.FDEMDipolarfields.B_from_MagneticDipoleWholeSpace(self.XYZ_Fx, self.src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        _ , bya , _ = EM.Analytics.FDEMDipolarfields.B_from_MagneticDipoleWholeSpace(self.XYZ_Fy, self.src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        _ , _ , bza = EM.Analytics.FDEMDipolarfields.B_from_MagneticDipoleWholeSpace(self.XYZ_Fz, self.src_loc_Fz, self.sigmaback, Utils.mkvc(np.array(self.freq)),orientation='Z',kappa= self.kappa)
        bxa, bya, bza = Utils.mkvc(bxa, 2), Utils.mkvc(bya, 2), Utils.mkvc(bza, 2)

        print str('Comp').center(4), str('Ana').center(25)              , str('Num').center(25)                 , str('Num - Ana').center(25)               , str('(Num - Ana)/Ana').center(25)                             , str('Pass Status').center(12)
        print str('B_x:').rjust(4) , repr(np.linalg.norm(bxa)).rjust(25), repr(np.linalg.norm(bx_num)).rjust(25), repr(np.linalg.norm(bxa-bx_num)).rjust(25), repr(np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa)).rjust(25), repr(np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa) < tol_MagDipole_Z).center(12)
        print str('B_y:').rjust(4) , repr(np.linalg.norm(bya)).rjust(25), repr(np.linalg.norm(by_num)).rjust(25), repr(np.linalg.norm(bya-by_num)).rjust(25), repr(np.linalg.norm(bya-by_num)/np.linalg.norm(bya)).rjust(25), repr(np.linalg.norm(bya-by_num)/np.linalg.norm(bya) < tol_MagDipole_Z).center(12)
        print str('B_z:').rjust(4) , repr(np.linalg.norm(bza)).rjust(25), repr(np.linalg.norm(bz_num)).rjust(25), repr(np.linalg.norm(bza-bz_num)).rjust(25), repr(np.linalg.norm(bza-bz_num)/np.linalg.norm(bza)).rjust(25), repr(np.linalg.norm(bza-bz_num)/np.linalg.norm(bza) < tol_MagDipole_Z).center(12)
        print
        self.assertTrue(np.linalg.norm(bxa-bx_num)/np.linalg.norm(bxa) < tol_MagDipole_Z, msg='Analytic and numeric solutions for Bx do not agree.')
        self.assertTrue(np.linalg.norm(bya-by_num)/np.linalg.norm(bya) < tol_MagDipole_Z, msg='Analytic and numeric solutions for By do not agree.')
        self.assertTrue(np.linalg.norm(bza-bz_num)/np.linalg.norm(bza) < tol_MagDipole_Z, msg='Analytic and numeric solutions for Bz do not agree.')


        # Plot Tx and Rx locations on mesh
        if plotIt:

            x_Bx = self.XYZ_Fx[:, 0]
            x_By = self.XYZ_Fy[:, 0]
            x_Bz = self.XYZ_Fz[:, 0]

            # Plot Tx and Rx locations on mesh
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.plot(self.src_loc_Fz[0], self.src_loc_Fz[2], 'ro', ms=8)
            ax.plot(self.XYZ_Fx[:, 0], self.XYZ_Fx[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Fy[:, 0], self.XYZ_Fy[:, 2], 'k.', ms=6)
            ax.plot(self.XYZ_Fz[:, 0], self.XYZ_Fz[:, 2], 'k.', ms=6)
            self.mesh.plotSlice(np.zeros(self.mesh.nC)*np.nan, grid=True, normal="Y", ax = ax)

            # Plot Br
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))

            plotLine_num_ana(ax[0], x_Bx, bx_num.real, bxa.real,
                             title='B_x Real')
            plotLine_num_ana(ax[1], x_Bx, bx_num.imag, bxa.imag,
                             title='B_x Imag')

            plotLine_num_ana(ax[2], x_By, by_num.real, bya.real,
                             title='B_y Real')
            plotLine_num_ana(ax[3], x_By, by_num.imag, bya.imag,
                             title='B_y Imag')

            plotLine_num_ana(ax[4], x_Bz, bz_num.real, bza.real,
                             title='B_z Real')
            plotLine_num_ana(ax[5], x_Bz, bz_num.imag, bza.imag,
                             title='B_z Imag')

            plt.legend(['Num', 'Ana'], bbox_to_anchor=(1.5, 0.5))
            plt.tight_layout()
            plt.show()



if __name__ == '__main__':
    unittest.main()
