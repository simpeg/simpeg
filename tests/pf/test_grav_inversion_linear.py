from __future__ import print_function
import unittest
import numpy as np
from SimPEG import Mesh
from SimPEG import Utils
from SimPEG import Maps
from SimPEG import Regularization
from SimPEG import DataMisfit
from SimPEG import Optimization
from SimPEG import InvProblem
from SimPEG import Directives
from SimPEG import Inversion
import matplotlib.pyplot as plt
import SimPEG.PF as PF

np.random.seed(43)


class GravInvLinProblemTest(unittest.TestCase):

    def setUp(self):

        ndv = -100
        # Create a mesh
        dx = 5.

        hxind = [(dx, 5, -1.3), (dx, 5), (dx, 5, 1.3)]
        hyind = [(dx, 5, -1.3), (dx, 5), (dx, 5, 1.3)]
        hzind = [(dx, 5, -1.3), (dx, 6)]

        mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC')

        # Get index of the center
        midx = int(mesh.nCx/2)
        midy = int(mesh.nCy/2)

        # Lets create a simple Gaussian topo and set the active cells
        [xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
        zz = -np.exp((xx**2 + yy**2) / 75**2) + mesh.vectorNz[-1]

        # Go from topo to actv cells
        topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]
        actv = Utils.surface2ind_topo(mesh, topo, 'N')
        actv = np.asarray([inds for inds, elem in enumerate(actv, 1)
                          if elem], dtype=int) - 1

        # Create active map to go from reduce space to full
        actvMap = Maps.InjectActiveCells(mesh, actv, -100)
        nC = len(actv)

        # Create and array of observation points
        xr = np.linspace(-20., 20., 20)
        yr = np.linspace(-20., 20., 20)
        X, Y = np.meshgrid(xr, yr)

        # Move the observation points 5m above the topo
        Z = -np.exp((X**2 + Y**2) / 75**2) + mesh.vectorNz[-1] + 5.

        # Create a MAGsurvey
        locXYZ = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
        rxLoc = PF.BaseGrav.RxObs(locXYZ)
        srcField = PF.BaseGrav.SrcField([rxLoc])
        survey = PF.BaseGrav.LinearSurvey(srcField)

        # We can now create a density model and generate data
        # Here a simple block in half-space
        model = np.zeros((mesh.nCx, mesh.nCy, mesh.nCz))
        model[(midx-2):(midx+2), (midy-2):(midy+2), -6:-2] = 0.5
        model = Utils.mkvc(model)
        self.model = model[actv]

        # Create active map to go from reduce set to full
        self.actvMap = Maps.InjectActiveCells(mesh, actv, ndv)

        # Create reduced identity map
        idenMap = Maps.IdentityMap(nP=nC)

        # Create the forward model operator
        prob = PF.Gravity.GravityIntegral(
            mesh,
            rhoMap=idenMap,
            actInd=actv
        )

        # Pair the survey and problem
        survey.pair(prob)

        # Compute linear forward operator and compute some data
        d = prob.fields(self.model)

        # Add noise and uncertainties (1nT)
        data = d + np.random.randn(len(d))*0.001
        wd = np.ones(len(data))*.001

        survey.dobs = data
        survey.std = wd

        # PF.Gravity.plot_obs_2D(survey.srcField.rxList[0].locs, d=data)

        # Create sensitivity weights from our linear forward operator
        wr = PF.Magnetics.get_dist_wgt(mesh, locXYZ, actv, 2., 2.)
        wr = np.sum(prob.G**2., axis=0)**0.5
        wr = (wr/np.max(wr))

        # Create a regularization
        reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
        reg.cell_weights = wr
        reg.norms = [0, 0, 0, 0]
        # reg.eps_p, reg.eps_q = 5e-2, 1e-2

        # Data misfit function
        dmis = DataMisfit.l2_DataMisfit(survey)
        dmis.W = 1/wd

        # Add directives to the inversion
        opt = Optimization.ProjectedGNCG(maxIter=100, lower=-1., upper=1.,
                                         maxIterLS=20, maxIterCG=10,
                                         tolCG=1e-3)
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=1e+8)

        # Here is where the norms are applied
        IRLS = Directives.Update_IRLS(f_min_change=1e-3,
                                      minGNiter=1)
        update_Jacobi = Directives.UpdatePreconditioner(mapping=idenMap)

        self.inv = Inversion.BaseInversion(invProb,
                                           directiveList=[IRLS,
                                                          update_Jacobi])
        self.mesh = mesh
    def test_mag_inverse(self):

        # Run the inversion
        mesh= self.mesh
        model = self.model
        mrec = self.inv.run(self.model)
        residual = np.linalg.norm(mrec-self.model) / np.linalg.norm(self.model)
        print(residual)
        midx = int(self.mesh.nCx/2)
        midy = int(self.mesh.nCy/2)
        # self.assertTrue(residual < 0.05)
        ypanel = midx

        zpanel = -7
        m_l2 = self.actvMap * self.inv.invProb.l2model
        m_l2[m_l2 == -100] = np.nan

        m_lp = self.actvMap * mrec
        m_lp[m_lp == -100] = np.nan

        m_true = self.actvMap * model
        m_true[m_true == -100] = np.nan

        vmin, vmax = mrec.min(), mrec.max()

        # Plot the data
        # PF.Gravity.plot_obs_2D(rxLoc, d=data)

        plt.figure()

        # Plot L2 model
        ax = plt.subplot(321)
        mesh.plotSlice(m_l2, ax=ax, normal='Z', ind=zpanel,
                       grid=True, clim=(vmin, vmax))
        plt.plot(([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 ([mesh.vectorCCy[ypanel], mesh.vectorCCy[ypanel]]), color='w')
        plt.title('Plan l2-model.')
        plt.gca().set_aspect('equal')
        plt.ylabel('y')
        ax.xaxis.set_visible(False)
        plt.gca().set_aspect('equal', adjustable='box')

        # Vertica section
        ax = plt.subplot(322)
        mesh.plotSlice(m_l2, ax=ax, normal='Y', ind=midx,
                       grid=True, clim=(vmin, vmax))
        plt.plot(([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 ([mesh.vectorCCz[zpanel], mesh.vectorCCz[zpanel]]), color='w')
        plt.title('E-W l2-model.')
        plt.gca().set_aspect('equal')
        ax.xaxis.set_visible(False)
        plt.ylabel('z')
        plt.gca().set_aspect('equal', adjustable='box')

        # Plot Lp model
        ax = plt.subplot(323)
        mesh.plotSlice(m_lp, ax=ax, normal='Z', ind=zpanel,
                       grid=True, clim=(vmin, vmax))
        plt.plot(([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 ([mesh.vectorCCy[ypanel], mesh.vectorCCy[ypanel]]), color='w')
        plt.title('Plan lp-model.')
        plt.gca().set_aspect('equal')
        ax.xaxis.set_visible(False)
        plt.ylabel('y')
        plt.gca().set_aspect('equal', adjustable='box')

        # Vertical section
        ax = plt.subplot(324)
        mesh.plotSlice(m_lp, ax=ax, normal='Y', ind=midx,
                       grid=True, clim=(vmin, vmax))
        plt.plot(([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 ([mesh.vectorCCz[zpanel], mesh.vectorCCz[zpanel]]), color='w')
        plt.title('E-W lp-model.')
        plt.gca().set_aspect('equal')
        ax.xaxis.set_visible(False)
        plt.ylabel('z')
        plt.gca().set_aspect('equal', adjustable='box')

        # Plot True model
        ax = plt.subplot(325)
        mesh.plotSlice(m_true, ax=ax, normal='Z', ind=zpanel,
                       grid=True, clim=(vmin, vmax))
        plt.plot(([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 ([mesh.vectorCCy[ypanel], mesh.vectorCCy[ypanel]]), color='w')
        plt.title('Plan true model.')
        plt.gca().set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal', adjustable='box')

        # Vertical section
        ax = plt.subplot(326)
        mesh.plotSlice(m_true, ax=ax, normal='Y', ind=midx,
                       grid=True, clim=(vmin, vmax))
        plt.plot(([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 ([mesh.vectorCCz[zpanel], mesh.vectorCCz[zpanel]]), color='w')
        plt.title('E-W true model.')
        plt.gca().set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

if __name__ == '__main__':
    unittest.main()
