from __future__ import print_function
import unittest
from SimPEG import (directives, maps,
                    inverse_problem, optimization, data_misfit,
                    inversion, utils, regularization)

import discretize
import numpy as np
from SimPEG.potential_fields import magnetics as mag
from scipy.interpolate import NearestNDInterpolator

from SimPEG.utils import mkvc
import shutil


class MVIProblemTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        H0 = (50000., 90., 0.)

        # The magnetization is set along a different
        # direction (induced + remanence)
        M = np.array([45., 90.])

        # Create grid of points for topography
        # Lets create a simple Gaussian topo
        # and set the active cells
        [xx, yy] = np.meshgrid(
            np.linspace(-200, 200, 50),
            np.linspace(-200, 200, 50)
        )
        b = 100
        A = 50
        zz = A*np.exp(-0.5*((xx/b)**2. + (yy/b)**2.))

        # We would usually load a topofile
        topo = np.c_[utils.mkvc(xx), utils.mkvc(yy), utils.mkvc(zz)]

        # Create and array of observation points
        xr = np.linspace(-100., 100., 20)
        yr = np.linspace(-100., 100., 20)
        X, Y = np.meshgrid(xr, yr)
        Z = A*np.exp(-0.5*((X/b)**2. + (Y/b)**2.)) + 5

        # Create a MAGsurvey
        xyzLoc = np.c_[utils.mkvc(X.T), utils.mkvc(Y.T), utils.mkvc(Z.T)]
        rxLoc = mag.point_receiver(xyzLoc)
        srcField = mag.SourceField([rxLoc], parameters=H0)
        survey = mag.MagneticSurvey(srcField)

        # Create a mesh
        h = [5, 5, 5]
        padDist = np.ones((3, 2)) * 100
        nCpad = [2, 4, 2]

        # Get extent of points
        limx = np.r_[topo[:, 0].max(), topo[:, 0].min()]
        limy = np.r_[topo[:, 1].max(), topo[:, 1].min()]
        limz = np.r_[topo[:, 2].max(), topo[:, 2].min()]

        # Get center of the mesh
        midX = np.mean(limx)
        midY = np.mean(limy)
        midZ = np.mean(limz)

        nCx = int(limx[0]-limx[1]) / h[0]
        nCy = int(limy[0]-limy[1]) / h[1]
        nCz = int(limz[0]-limz[1]+int(np.min(np.r_[nCx, nCy])/3)) / h[2]
        # Figure out full extent required from input
        extent = np.max(np.r_[nCx * h[0] + padDist[0, :].sum(),
                              nCy * h[1] + padDist[1, :].sum(),
                              nCz * h[2] + padDist[2, :].sum()])

        maxLevel = int(np.log2(extent/h[0]))+1

        # Number of cells at the small octree level
        nCx, nCy, nCz = 2**(maxLevel), 2**(maxLevel), 2**(maxLevel)

        # Define the mesh and origin
        # For now cubic cells
        mesh = discretize.TreeMesh([np.ones(nCx)*h[0],
                              np.ones(nCx)*h[1],
                              np.ones(nCx)*h[2]])

        # Set origin
        mesh.x0 = np.r_[
            -nCx*h[0]/2.+midX,
            -nCy*h[1]/2.+midY,
            -nCz*h[2]/2.+midZ
        ]

        # Refine the mesh around topography
        # Get extent of points
        F = NearestNDInterpolator(topo[:, :2], topo[:, 2])
        zOffset = 0
        # Cycle through the first 3 octree levels
        for ii in range(3):

            dx = mesh.hx.min()*2**ii

            nCx = int((limx[0]-limx[1]) / dx)
            nCy = int((limy[0]-limy[1]) / dx)

            # Create a grid at the octree level in xy
            CCx, CCy = np.meshgrid(
                np.linspace(limx[1], limx[0], nCx),
                np.linspace(limy[1], limy[0], nCy)
            )

            z = F(mkvc(CCx), mkvc(CCy))

            # level means number of layers in current OcTree level
            for level in range(int(nCpad[ii])):

                mesh.insert_cells(
                    np.c_[
                        mkvc(CCx),
                        mkvc(CCy),
                        z-zOffset
                    ], np.ones_like(z)*maxLevel-ii,
                    finalize=False
                )

                zOffset += dx

        mesh.finalize()
        self.mesh = mesh
        # Define an active cells from topo
        actv = utils.surface2ind_topo(mesh, topo)
        nC = int(actv.sum())

        model = np.zeros((mesh.nC, 3))

        # Convert the inclination declination to vector in Cartesian
        M_xyz = utils.matutils.dip_azimuth2cartesian(M[0], M[1])

        # Get the indicies of the magnetized block
        ind = utils.ModelBuilder.getIndicesBlock(
            np.r_[-20, -20, -10], np.r_[20, 20, 25],
            mesh.gridCC,
        )[0]

        # Assign magnetization values
        model[ind, :] = np.kron(
            np.ones((ind.shape[0], 1)), M_xyz*0.05
        )

        # Remove air cells
        self.model = model[actv, :]

        # Create active map to go from reduce set to full
        self.actvMap = maps.InjectActiveCells(mesh, actv, np.nan)

        # Creat reduced identity map
        idenMap = maps.IdentityMap(nP=nC*3)

        # Create the forward model operator
        sim = mag.IntegralSimulation(
            self.mesh,
            survey=survey,
            modelType='vector',
            chiMap=idenMap,
            actInd=actv,
            store_sensitivities='ram'
        )
        self.sim = sim

        # Compute some data and add some random noise
        data = sim.make_synthetic_data(
            utils.mkvc(self.model),
                standard_deviation=0.0, noise_floor=5.0, add_noise=True
            )

        # Create an projection matrix for plotting later
        actvPlot = maps.InjectActiveCells(mesh, actv, np.nan)

        # This Mapping connects the regularizations for the three-component
        # vector model
        wires = maps.Wires(('p', nC), ('s', nC), ('t', nC))

        # Create sensitivity weights from our linear forward operator
        # so that all cells get equal chance to contribute to the solution
        wr = sim.getJtJdiag(utils.mkvc(self.model))**0.5
        wr /= np.max(wr)

        # Create three regularization for the different components
        # of magnetization
        reg_p = regularization.Sparse(mesh, indActive=actv, mapping=wires.p)
        reg_p.mref = np.zeros(3*nC)
        reg_p.cell_weights = (wires.p * wr)

        reg_s = regularization.Sparse(mesh, indActive=actv, mapping=wires.s)
        reg_s.mref = np.zeros(3*nC)
        reg_s.cell_weights = (wires.s * wr)

        reg_t = regularization.Sparse(mesh, indActive=actv, mapping=wires.t)
        reg_t.mref = np.zeros(3*nC)
        reg_t.cell_weights = (wires.t * wr)

        reg = reg_p + reg_s + reg_t
        reg.mref = np.zeros(3*nC)

        # Data misfit function
        dmis = data_misfit.L2DataMisfit(simulation=sim, data=data)
        #dmis.W = 1./survey.std

        # Add directives to the inversion
        opt = optimization.ProjectedGNCG(maxIter=30, lower=-10, upper=10.,
                                         maxIterLS=20, maxIterCG=20, tolCG=1e-4)

        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)

        # A list of directive to control the inverson
        betaest = directives.BetaEstimate_ByEig()

        # Here is where the norms are applied
        # Use pick a treshold parameter empirically based on the distribution of
        #  model parameters
        IRLS = directives.Update_IRLS(
            f_min_change=1e-3, max_irls_iterations=0, beta_tol=5e-1
        )

        # Pre-conditioner
        update_Jacobi = directives.UpdatePreconditioner()

        inv = inversion.BaseInversion(invProb,
                                      directiveList=[IRLS, update_Jacobi, betaest])

        # Run the inversion
        m0 = np.ones(3*nC) * 1e-4  # Starting model
        mrec_MVIC = inv.run(m0)

        sim.chiMap = maps.SphericalSystem(nP=nC*3)
        self.mstart = sim.chiMap.inverse(mrec_MVIC)
        dmis.simulation.model = self.mstart
        beta = invProb.beta


        # Create a block diagonal regularization
        wires = maps.Wires(('amp', nC), ('theta', nC), ('phi', nC))

        # Create a Combo Regularization
        # Regularize the amplitude of the vectors
        reg_a = regularization.Sparse(mesh, indActive=actv,
                                      mapping=wires.amp)
        reg_a.norms = np.c_[0., 0., 0., 0.]  # Sparse on the model and its gradients
        reg_a.mref = np.zeros(3*nC)

        # Regularize the vertical angle of the vectors
        reg_t = regularization.Sparse(mesh, indActive=actv,
                                      mapping=wires.theta)
        reg_t.alpha_s = 0.  # No reference angle
        reg_t.space = 'spherical'
        reg_t.norms = np.c_[2., 0., 0., 0.]  # Only norm on gradients used

        # Regularize the horizontal angle of the vectors
        reg_p = regularization.Sparse(mesh, indActive=actv,
                                      mapping=wires.phi)
        reg_p.alpha_s = 0.  # No reference angle
        reg_p.space = 'spherical'
        reg_p.norms = np.c_[2., 0., 0., 0.]  # Only norm on gradients used

        reg = reg_a + reg_t + reg_p
        reg.mref = np.zeros(3*nC)

        Lbound = np.kron(np.asarray([0, -np.inf, -np.inf]), np.ones(nC))
        Ubound = np.kron(np.asarray([10, np.inf, np.inf]), np.ones(nC))

        # Add directives to the inversion
        opt = optimization.ProjectedGNCG(maxIter=20,
                                         lower=Lbound,
                                         upper=Ubound,
                                         maxIterLS=20,
                                         maxIterCG=30,
                                         tolCG=1e-3,
                                         stepOffBoundsFact=1e-3,
                                         )
        opt.approxHinv = None

        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=beta)

        # Here is where the norms are applied
        IRLS = directives.Update_IRLS(f_min_change=1e-4, max_irls_iterations=20,
                                      minGNiter=1, beta_tol=0.5,
                                      coolingRate=1, coolEps_q=True,
                                      beta_search=False)

        # Special directive specific to the mag amplitude problem. The sensitivity
        # weights are update between each iteration.
        ProjSpherical = directives.ProjectSphericalBounds()
        update_SensWeight = directives.UpdateSensitivityWeights()
        update_Jacobi = directives.UpdatePreconditioner()

        self.inv = inversion.BaseInversion(
            invProb,
            directiveList=[
                ProjSpherical, IRLS, update_SensWeight, update_Jacobi
            ]
        )

    def test_mag_inverse(self):

        # Run the inversion
        mrec_MVI_S = self.inv.run(self.mstart)

        nC = int(mrec_MVI_S.shape[0]/3)
        vec_xyz = utils.matutils.spherical2cartesian(
                mrec_MVI_S.reshape((nC, 3), order='F')).reshape((nC, 3), order='F')

        residual = np.linalg.norm(vec_xyz-self.model) / np.linalg.norm(self.model)
        # print(residual)
        # import matplotlib.pyplot as plt

        # mrec = np.sum(vec_xyz**2., axis=1)**0.5
        # plt.figure()
        # ax = plt.subplot(1, 2, 1)
        # midx = 65
        # self.mesh.plotSlice(self.actvMap*mrec, ax=ax, normal='Y', ind=midx,
        #                grid=True, clim=(0, 0.03))
        # ax.set_xlim(self.mesh.gridCC[:, 0].min(), self.mesh.gridCC[:, 0].max())
        # ax.set_ylim(self.mesh.gridCC[:, 2].min(), self.mesh.gridCC[:, 2].max())

        # plt.show()
        self.assertLess(residual, 0.25)
        # self.assertTrue(residual < 0.05)

    def tearDown(self):
        # Clean up the working directory
        if self.sim.store_sensitivities == 'disk':
            shutil.rmtree(self.sim.sensitivity_path)


if __name__ == '__main__':
    unittest.main()
