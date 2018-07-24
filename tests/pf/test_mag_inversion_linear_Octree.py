from __future__ import print_function
import unittest
from SimPEG import (Mesh, Directives, Maps,
                    InvProblem, Optimization, DataMisfit,
                    Inversion, Utils, Regularization)

import SimPEG.PF as PF
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


class MagInvLinProblemTest(unittest.TestCase):

    def setUp(self):

        np.random.seed(0)

        # First we need to define the direction of the inducing field
        # As a simple case, we pick a vertical inducing field of magnitude
        # 50,000nT.
        # From old convention, field orientation is given as an
        # azimuth from North (positive clockwise)
        # and dip from the horizontal (positive downward).
        H0 = (50000., 90., 0.)

        # Assume all induced so the magnetization M is in the same direction
        M = np.array([90, 0])

        # Create a mesh
        h = [5, 5, 5]
        padDist = np.ones((3, 2)) * 100

        # Create grid of points for topography
        # Lets create a simple Gaussian topo and set the active cells
        [xx, yy] = np.meshgrid(
            np.linspace(-200, 200, 50),
            np.linspace(-200, 200, 50)
        )

        b = 100
        A = 50
        zz = A*np.exp(-0.5*((xx/b)**2. + (yy/b)**2.))

        # We would usually load a topofile
        topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]

        # Create and array of observation points
        xr = np.linspace(-100., 100., 20)
        yr = np.linspace(-100., 100., 20)
        X, Y = np.meshgrid(xr, yr)
        Z = A*np.exp(-0.5*((X/b)**2. + (Y/b)**2.)) + 5

        # Create a MAGsurvey
        xyzLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
        rxLoc = PF.BaseMag.RxObs(xyzLoc)
        srcField = PF.BaseMag.SrcField([rxLoc], param=H0)
        survey = PF.BaseMag.LinearSurvey(srcField)

        # Create a topography
        self.mesh = Utils.modelutils.meshBuilder(topo, h, padDist,
                                                 meshType='TREE',
                                                 verticalAlignment='center')

        # Refine the mesh around topography
        self.mesh = Utils.modelutils.refineTree(
            self.mesh, topo, dtype='surface', nCpad=[0, 3, 1], finalize=False
        )

        # Refine around obs points
        self.mesh = Utils.modelutils.refineTree(
            self.mesh, xyzLoc, dtype='surface', nCpad=[4, 0, 0], finalize=True
        )

        # Define an active cells from topo
        actv = Utils.surface2ind_topo(self.mesh, topo)
        nC = int(actv.sum())

        # We can now create a susceptibility model and generate data
        # Lets start with a simple block in half-space
        self.model = Utils.ModelBuilder.addBlock(
            self.mesh.gridCC, np.zeros(self.mesh.nC),
            np.r_[-40, -40, -50], np.r_[40, 40, 0], 0.05
        )[actv]

        # Create active map to go from reduce set to full
        self.actvMap = Maps.InjectActiveCells(self.mesh, actv, np.nan)

        # Creat reduced identity map
        idenMap = Maps.IdentityMap(nP=nC)

        # Create the forward model operator
        prob = PF.Magnetics.MagneticIntegral(
            self.mesh, chiMap=idenMap, actInd=actv
        )

        # Pair the survey and problem
        survey.pair(prob)

        # Compute linear forward operator and compute some data
        data = prob.fields(self.model)

        # Add noise and uncertainties (1nT)
        data += np.random.randn(len(data))
        wd = np.ones(len(data))*1.

        survey.dobs = data
        survey.std = wd

        # Create sensitivity weights from our linear forward operator
        rxLoc = survey.srcField.rxList[0].locs
        wr = np.zeros(prob.G.shape[1])
        for ii in range(survey.nD):
            wr += (prob.G[ii, :]/survey.std[ii])**2.

        # wr = (wr/np.max(wr))
        wr = wr**0.5

        # Create a regularization
        reg = Regularization.Sparse(self.mesh, indActive=actv, mapping=idenMap)
        reg.norms = np.c_[0, 0, 0, 0]
        reg.cell_weights = wr
        reg.mref = np.zeros(nC)

        # Data misfit function
        dmis = DataMisfit.l2_DataMisfit(survey)
        dmis.W = 1./survey.std

        # Add directives to the inversion
        opt = Optimization.ProjectedGNCG(
            maxIter=30, lower=0., upper=10.,
            maxIterLS=20, maxIterCG=20, tolCG=1e-4
        )

        invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
        betaest = Directives.BetaEstimate_ByEig()

        # Here is where the norms are applied
        # Use pick a treshold parameter empirically based on the distribution of
        #  model parameters
        IRLS = Directives.Update_IRLS(
            f_min_change=1e-3, maxIRLSiter=30, beta_tol=5e-1
        )
        update_Jacobi = Directives.UpdatePreconditioner()

        # saveOuput = Directives.SaveOutputEveryIteration()
        # saveModel.fileName = work_dir + out_dir + 'ModelSus'
        self.inv = Inversion.BaseInversion(
            invProb,
            directiveList=[betaest, IRLS, update_Jacobi]
        )

    def test_mag_inverse(self):

        # Run the inversion
        mrec = self.inv.run(self.model)

        residual = np.linalg.norm(mrec-self.model) / np.linalg.norm(self.model)
        # print(residual)

        # plt.figure()
        # ax = plt.subplot(1, 2, 1)
        # midx = 65
        # self.mesh.plotSlice(self.actvMap*mrec, ax=ax, normal='Y', ind=midx,
        #                grid=True, clim=(0, 0.02))

        # ax = plt.subplot(1, 2, 2)
        # self.mesh.plotSlice(self.actvMap*self.model, ax=ax, normal='Y', ind=midx,
        #                grid=True, clim=(0, 0.02))
        # plt.show()

        self.assertTrue(residual < 0.1)
        # self.assertTrue(residual < 0.05)


if __name__ == '__main__':
    unittest.main()
