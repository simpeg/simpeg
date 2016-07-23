import unittest
from SimPEG import EM, Mesh, Utils, np, Maps
# import sys
from scipy.constants import mu_0


plotIt = False
tol_EBdipole = 1e-2

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

    def test_CylMesh_HarmonicElecDipoleTests(self):
        print('Testing various componemts of the analytic harmonic electric dipole against a numerical solution on a cylindrical mesh')

        # Define the source
        # Search over z-faces to find face nearest src_loc then add nFx to get to global face index [nFx][nFy = 0][nFz]
        s_ind = Utils.closestPoints(mesh, src_loc, 'Fz') + mesh.nFx
        de = np.zeros(mesh.nF, dtype=complex)
        de[s_ind] = 1./csz
        de_p = [EM.FDEM.Src.RawVec_e([], freq, de/mesh.area)]

        # Pair the problem and survey
        survey = EM.FDEM.Survey(de_p)

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

        rlim = [20., 500.]
        r = mesh.vectorCCx[np.argmin(np.abs(mesh.vectorCCx-rlim[0])):np.argmin(np.abs(mesh.vectorCCx-rlim[1]))]
        z = 100.

        # where we choose to measure
        XYZ = Utils.ndgrid(r, np.r_[0.], np.r_[z])

        Pf = mesh.getInterpolationMat(XYZ, 'CC')
        Zero = sp.csr_matrix(Pf.shape)
        Pfx, Pfz = sp.hstack([Pf, Zero]), sp.hstack([Zero, Pf])

        # J lives on faces
        j_num = numFields_ElecDipole[de_p, 'j']

        Rho = Utils.sdiag(1./SigmaBack)
        Rho = sp.block_diag([Rho, Rho])
        # E lives on cell centres
        e_num = numFields_ElecDipole[de_p, 'e']
        e_numTest = Rho*mesh.aveF2CCV*j_num

        # H lives on edges
        h_num = numFields_ElecDipole[de_p, 'h']
        h_num = mesh.aveE2CCV*h_num
        # B lives on cell centers
        b_num = numFields_ElecDipole[de_p, 'b']


        ex, ez = Pfx*e_num, Pfz*e_num
        # Since we are evaluating along the plane y=0 the b_theta == b_y in cartesian coordiantes
        b_theta = Pf*b_num
        h_theta = Pf*h_num

        # get analytic solution
        # E_from_ElectricDipoleWholeSpace
        # J_from_ElectricDipoleWholeSpace
        # H_from_ElectricDipoleWholeSpace
        # B_from_ElectricDipoleWholeSpace
        # A_from_ElectricDipoleWholeSpace

        exa, eya, eza = EM.Analytics.FDEM.E_from_ElectricDipoleWholeSpace(XYZ, src_loc, sigmaback, freq,orientation='Z',kappa= kappa)
        exa, eya, eza = Utils.mkvc(exa,2), Utils.mkvc(eya,2), Utils.mkvc(eza,2)



        # bxa, bya, bza = EM.Analytics.FDEM.MagneticDipoleWholeSpace(XYZ, src_loc, sigmaback, freq,orientation='Z',mu= mur*mu_0)
        # bxa, bya, bza = Utils.mkvc(bxa,2), Utils.mkvc(bya,2), Utils.mkvc(bza,2)

        print ' comp,       anayltic,       numeric,       num - ana,       (num - ana)/ana'
        print '  ex:', np.linalg.norm(exa), np.linalg.norm(ex), np.linalg.norm(exa-ex), np.linalg.norm(exa-ex)/np.linalg.norm(exa)
        print '  ez:', np.linalg.norm(eza), np.linalg.norm(ez), np.linalg.norm(eza-ez), np.linalg.norm(eza-ez)/np.linalg.norm(eza)

        print '  bx:', np.linalg.norm(bxa), np.linalg.norm(bx), np.linalg.norm(bxa-bx), np.linalg.norm(bxa-bx)/np.linalg.norm(bxa)
        print '  bz:', np.linalg.norm(bza), np.linalg.norm(bz), np.linalg.norm(bza-bz), np.linalg.norm(bza-bz)/np.linalg.norm(bza)

        if plotIt:
            # Edipole
            plt.subplot(221)
            plt.plot(r, ex.real, 'o', r, exa.real, linewidth=2)
            plt.grid(which='both')
            plt.title('Ex Real')
            plt.xlabel('r (m)')

            plt.subplot(222)
            plt.plot(r,ex.imag,'o',r,exa.imag,linewidth=2)
            plt.grid(which='both')
            plt.title('Ex Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('r (m)')

            plt.subplot(223)
            plt.plot(r,ez.real,'o',r,eza.real,linewidth=2)
            plt.grid(which='both')
            plt.title('Ez Real')
            plt.xlabel('r (m)')

            plt.subplot(224)
            plt.plot(r,ez.imag,'o',r,eza.imag,linewidth=2)
            plt.grid(which='both')
            plt.title('Ez Imag')
            plt.xlabel('r (m)')

            plt.tight_layout()

            # Bdipole
            plt.subplot(221)
            plt.plot(r,bx.real,'o',r,bxa.real,linewidth=2)
            plt.grid(which='both')
            plt.title('Bx Real')
            plt.xlabel('r (m)')

            plt.subplot(222)
            plt.plot(r,bx.imag,'o',r,bxa.imag,linewidth=2)
            plt.grid(which='both')
            plt.title('Bx Imag')
            plt.legend(['Num','Ana'],bbox_to_anchor=(1.5,0.5))
            plt.xlabel('r (m)')

            plt.subplot(223)
            plt.plot(r,bz.real,'o',r,bza.real,linewidth=2)
            plt.grid(which='both')
            plt.title('Bz Real')
            plt.xlabel('r (m)')

            plt.subplot(224)
            plt.plot(r,bz.imag,'o',r,bza.imag,linewidth=2)
            plt.grid(which='both')
            plt.title('Bz Imag')
            plt.xlabel('r (m)')

            plt.tight_layout()

        self.assertTrue(np.linalg.norm(exa-ex)/np.linalg.norm(exa) < tol_EBdipole)
        self.assertTrue(np.linalg.norm(eza-ez)/np.linalg.norm(eza) < tol_EBdipole)

        self.assertTrue(np.linalg.norm(bxa-bx)/np.linalg.norm(bxa) < tol_EBdipole)
        self.assertTrue(np.linalg.norm(bza-bz)/np.linalg.norm(bza) < tol_EBdipole)



if __name__ == '__main__':
    unittest.main()
