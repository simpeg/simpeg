from __future__ import print_function
import unittest
from SimPEG import (Directives, Maps,
                    InvProblem, Optimization, DataMisfit,
                    Inversion, Utils, Regularization, Mesh)
from discretize.utils import meshutils
import SimPEG.PF as PF
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from SimPEG.Utils import mkvc


class AmpProblemTest(unittest.TestCase):

    def setUp(self):
        # We will assume a vertical inducing field
        H0 = (50000., 90., 0.)

        # The magnetization is set along a different direction (induced + remanence)
        M = np.array([90., 0.])

        # Block with an effective susceptibility
        chi_e = 0.05

        # Create grid of points for topography
        # Lets create a simple Gaussian topo and set the active cells
        [xx, yy] = np.meshgrid(np.linspace(-200, 200, 50), np.linspace(-200, 200, 50))
        b = 100
        A = 50
        zz = A*np.exp(-0.5*((xx/b)**2. + (yy/b)**2.))
        topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

        # Create and array of observation points
        xr = np.linspace(-100., 100., 20)
        yr = np.linspace(-100., 100., 20)
        X, Y = np.meshgrid(xr, yr)
        Z = A*np.exp(-0.5*((X/b)**2. + (Y/b)**2.)) + 5

        # Create a MAGsurvey
        xyzLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
        Rx = PF.BaseMag.RxObs(xyzLoc)
        srcField = PF.BaseMag.SrcField([Rx], param=H0)
        survey = PF.BaseMag.LinearSurvey(srcField)

        # Create a mesh
        h = [5, 5, 5]
        padDist = np.ones((3, 2)) * 100
        nCpad = [4, 4, 2]

        self.mesh = meshutils.mesh_builder_xyz(
            xyzLoc, h, padding_distance=padDist,
            mesh_type='TREE',
        )

        self.mesh = meshutils.refine_tree_xyz(
            self.mesh, topo, method='surface',
            octree_levels=nCpad,
            octree_levels_padding=nCpad,
            finalize=True,
        )
        # Define an active cells from topo
        actv = Utils.surface2ind_topo(self.mesh, topo)
        nC = int(actv.sum())

        # Convert the inclination declination to vector in Cartesian
        M_xyz = Utils.matutils.dip_azimuth2cartesian(np.ones(nC)*M[0], np.ones(nC)*M[1])

        # Get the indicies of the magnetized block
        ind = Utils.ModelBuilder.getIndicesBlock(
            np.r_[-20, -20, -15], np.r_[20, 20, 20],
            self.mesh.gridCC,
        )[0]

        # Assign magnetization value, inducing field strength will
        # be applied in by the :class:`SimPEG.PF.Magnetics` problem
        model = np.zeros(self.mesh.nC)
        model[ind] = chi_e

        # Remove air cells
        self.model = model[actv]

        # Create active map to go from reduce set to full
        self.actvPlot = Maps.InjectActiveCells(self.mesh, actv, np.nan)

        # Creat reduced identity map
        idenMap = Maps.IdentityMap(nP=nC)

        # Create the forward model operator
        prob = PF.Magnetics.MagneticIntegral(
            self.mesh, M=M_xyz, chiMap=idenMap, actInd=actv
        )

        # Pair the survey and problem
        survey.pair(prob)

        # Compute some data and add some random noise
        data = prob.fields(self.model)

        # Split the data in components
        nD = xyzLoc.shape[0]

        std = 5  # nT
        data += np.random.randn(nD)*std
        wd = np.ones(nD)*std

        # Assigne data and uncertainties to the survey
        survey.dobs = data
        survey.std = wd

        ######################################################################
        # Equivalent Source

        # Get the active cells for equivalent source is the top only
        surf = Utils.modelutils.surface_layer_index(self.mesh, topo)

        # Get the layer of cells directyl below topo
        nC = np.count_nonzero(surf)  # Number of active cells

        # Create active map to go from reduce set to full
        surfMap = Maps.InjectActiveCells(self.mesh, surf, np.nan)

        # Create identity map
        idenMap = Maps.IdentityMap(nP=nC)

        # Create static map
        prob = PF.Magnetics.MagneticIntegral(
                self.mesh, chiMap=idenMap, actInd=surf,
                parallelized=False, equiSourceLayer=True)

        prob.solverOpts['accuracyTol'] = 1e-4

        # Pair the survey and problem
        if survey.ispaired:
            survey.unpair()
        survey.pair(prob)

        # Create a regularization function, in this case l2l2
        reg = Regularization.Sparse(
            self.mesh, indActive=surf, mapping=Maps.IdentityMap(nP=nC),
            scaledIRLS=False
        )
        reg.mref = np.zeros(nC)

        # Specify how the optimization will proceed,
        # set susceptibility bounds to inf
        opt = Optimization.ProjectedGNCG(maxIter=20, lower=-np.inf,
                                         upper=np.inf, maxIterLS=20,
                                         maxIterCG=20, tolCG=1e-3)

        # Define misfit function (obs-calc)
        dmis = DataMisfit.l2_DataMisfit(survey)
        dmis.W = 1./survey.std

        # Create the default L2 inverse problem from the above objects
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

        # Specify how the initial beta is found
        betaest = Directives.BetaEstimate_ByEig()

        # Target misfit to stop the inversion,
        # try to fit as much as possible of the signal,
        # we don't want to lose anything
        IRLS = Directives.Update_IRLS(f_min_change=1e-3, minGNiter=1,
                                      beta_tol=1e-1)
        update_Jacobi = Directives.UpdatePreconditioner()
        # Put all the parts together
        inv = Inversion.BaseInversion(
            invProb, directiveList=[betaest, IRLS, update_Jacobi]
        )

        # Run the equivalent source inversion
        mstart = np.ones(nC)*1e-4
        mrec = inv.run(mstart)

        # Won't store the sensitivity and output 'xyz' data.
        prob.forwardOnly = True
        prob.rx_type = 'xyz'
        prob._G = None
        prob.modelType = 'amplitude'
        prob.model = mrec
        pred = prob.fields(mrec)

        bx = pred[:nD]
        by = pred[nD:2*nD]
        bz = pred[2*nD:]

        bAmp = (bx**2. + by**2. + bz**2.)**0.5

        # AMPLITUDE INVERSION
        # Create active map to go from reduce space to full
        actvMap = Maps.InjectActiveCells(self.mesh, actv, -100)
        nC = int(actv.sum())

        # Create identity map
        idenMap = Maps.IdentityMap(nP=nC)

        self.mstart = np.ones(nC)*1e-4

        # Create the forward model operator
        prob = PF.Magnetics.MagneticIntegral(
            self.mesh, chiMap=idenMap, actInd=actv,
            modelType='amplitude', rx_type='xyz'
        )
        prob.model = self.mstart
        # Change the survey to xyz components
        surveyAmp = PF.BaseMag.LinearSurvey(survey.srcField)

        # Pair the survey and problem
        surveyAmp.pair(prob)
        # Create a regularization function, in this case l2l2
        wr = np.sum(prob.G**2., axis=0)**0.5
        wr /= np.max(wr)
        # Re-set the observations to |B|
        surveyAmp.dobs = bAmp
        surveyAmp.std = wd

        # Create a sparse regularization
        reg = Regularization.Sparse(self.mesh, indActive=actv, mapping=idenMap)
        reg.norms = np.c_[0, 0, 0, 0]
        reg.mref = np.zeros(nC)
        reg.cell_weights = wr
        # Data misfit function
        dmis = DataMisfit.l2_DataMisfit(surveyAmp)
        dmis.W = 1./surveyAmp.std

        # Add directives to the inversion
        opt = Optimization.ProjectedGNCG(maxIter=30, lower=0., upper=1.,
                                         maxIterLS=20, maxIterCG=20,
                                         tolCG=1e-3)

        invProb = InvProblem.BaseInvProblem(dmis, reg, opt)

        # Here is the list of directives
        betaest = Directives.BetaEstimate_ByEig()

        # Specify the sparse norms
        IRLS = Directives.Update_IRLS(f_min_change=1e-3,
                                      minGNiter=1, coolingRate=1,
                                      betaSearch=False)

        # The sensitivity weights are update between each iteration.
        update_SensWeight = Directives.UpdateSensitivityWeights()
        update_Jacobi = Directives.UpdatePreconditioner()

        # Put all together
        self.inv = Inversion.BaseInversion(
            invProb, directiveList=[
                betaest, IRLS, update_SensWeight, update_Jacobi
                ]
        )


    def test_mag_inverse(self):

        # Run the inversion
        mrec_Amp = self.inv.run(self.mstart)

        residual = (
            np.linalg.norm(mrec_Amp-self.model) /
            np.linalg.norm(self.model)
        )
        # print(residual)
        # import matplotlib.pyplot as plt

        # # Plot the amplitude model
        # plt.figure()
        # ax = plt.subplot(2, 1, 1)
        # im = self.mesh.plotSlice(self.actvPlot*self.model,
        #  ax=ax, normal='Y', ind=66,
        #     pcolorOpts={"vmin":0., "vmax":0.01}
        # )
        # plt.colorbar(im[0])
        # ax.set_xlim([-200, 200])
        # ax.set_ylim([-100, 75])
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # plt.gca().set_aspect('equal', adjustable='box')

        # ax = plt.subplot(2, 1, 2)
        # im = self.mesh.plotSlice(self.actvPlot*mrec_Amp, ax=ax, normal='Y', ind=66,
        #     pcolorOpts={"vmin":0., "vmax":0.01}
        # )
        # plt.colorbar(im[0])
        # ax.set_xlim([-200, 200])
        # ax.set_ylim([-100, 75])
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # plt.gca().set_aspect('equal', adjustable='box')

        # plt.show()
        self.assertTrue(residual < 1.)


if __name__ == '__main__':
    unittest.main()
