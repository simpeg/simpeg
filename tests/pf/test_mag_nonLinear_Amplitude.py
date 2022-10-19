from __future__ import print_function
import numpy as np
from SimPEG import (
    data,
    data_misfit,
    directives,
    maps,
    inverse_problem,
    optimization,
    inversion,
    regularization,
)

from SimPEG.potential_fields import magnetics
from SimPEG import utils
from SimPEG.utils import mkvc
from discretize.utils import mesh_builder_xyz, refine_tree_xyz
import unittest
import shutil


class AmpProblemTest(unittest.TestCase):
    def setUp(self):
        # We will assume a vertical inducing field
        H0 = (50000.0, 90.0, 0.0)

        # The magnetization is set along a different direction (induced + remanence)
        M = np.array([45.0, 90.0])

        # Block with an effective susceptibility
        chi_e = 0.05

        # Create grid of points for topography
        # Lets create a simple Gaussian topo and set the active cells
        [xx, yy] = np.meshgrid(np.linspace(-200, 200, 50), np.linspace(-200, 200, 50))
        b = 100
        A = 50
        zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))
        topo = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

        # Create an array of observation points
        xr = np.linspace(-100.0, 100.0, 20)
        yr = np.linspace(-100.0, 100.0, 20)
        X, Y = np.meshgrid(xr, yr)
        Z = A * np.exp(-0.5 * ((X / b) ** 2.0 + (Y / b) ** 2.0)) + 10

        # Create a MAGsurvey
        rxLoc = np.c_[mkvc(X.T), mkvc(Y.T), mkvc(Z.T)]
        receiver_list = magnetics.receivers.Point(rxLoc)
        srcField = magnetics.sources.SourceField(
            receiver_list=[receiver_list], parameters=H0
        )
        survey = magnetics.survey.Survey(srcField)

        ###############################################################################
        # Inversion Mesh

        # Create a mesh
        h = [5, 5, 5]
        padDist = np.ones((3, 2)) * 100

        mesh = mesh_builder_xyz(
            rxLoc, h, padding_distance=padDist, depth_core=100, mesh_type="tree"
        )
        mesh = refine_tree_xyz(
            mesh, topo, method="surface", octree_levels=[4, 4], finalize=True
        )

        # Define an active cells from topo
        actv = utils.surface2ind_topo(mesh, topo)
        nC = int(actv.sum())

        # Convert the inclination declination to vector in Cartesian
        M_xyz = utils.mat_utils.dip_azimuth2cartesian(
            np.ones(nC) * M[0], np.ones(nC) * M[1]
        )

        # Get the indicies of the magnetized block
        ind = utils.model_builder.getIndicesBlock(
            np.r_[-20, -20, -10],
            np.r_[20, 20, 25],
            mesh.gridCC,
        )[0]

        # Assign magnetization value, inducing field strength will
        # be applied in by the :class:`SimPEG.PF.Magnetics` problem
        model = np.zeros(mesh.nC)
        model[ind] = chi_e

        # Remove air cells
        model = model[actv]

        # Creat reduced identity map
        idenMap = maps.IdentityMap(nP=nC)

        # Create the forward model operator
        simulation = magnetics.Simulation3DIntegral(
            survey=survey,
            mesh=mesh,
            chiMap=idenMap,
            actInd=actv,
            store_sensitivities="forward_only",
        )
        simulation.M = M_xyz

        # Compute some data and add some random noise
        synthetic_data = simulation.dpred(model)

        # Split the data in components
        nD = rxLoc.shape[0]

        std = 5  # nT
        synthetic_data += np.random.randn(nD) * std
        wd = np.ones(nD) * std

        # Assigne data and uncertainties to the survey
        data_object = data.Data(survey, dobs=synthetic_data, noise_floor=wd)

        ######################################################################
        # Equivalent Source

        # Get the active cells for equivalent source is the top only
        surf = utils.model_utils.surface_layer_index(mesh, topo)
        nC = np.count_nonzero(surf)  # Number of active cells
        mstart = np.ones(nC) * 1e-4

        # Create active map to go from reduce set to full
        surfMap = maps.InjectActiveCells(mesh, surf, np.nan)

        # Create identity map
        idenMap = maps.IdentityMap(nP=nC)

        # Create static map
        simulation = magnetics.simulation.Simulation3DIntegral(
            mesh=mesh,
            survey=survey,
            chiMap=idenMap,
            actInd=surf,
            store_sensitivities="ram",
        )
        simulation.model = mstart

        # Create a regularization function, in this case l2l2
        reg = regularization.Sparse(
            mesh, indActive=surf, mapping=maps.IdentityMap(nP=nC), alpha_z=0
        )
        reg.mref = np.zeros(nC)

        # Specify how the optimization will proceed, set susceptibility bounds to inf
        opt = optimization.ProjectedGNCG(
            maxIter=10,
            lower=-np.inf,
            upper=np.inf,
            maxIterLS=5,
            maxIterCG=5,
            tolCG=1e-3,
        )

        # Define misfit function (obs-calc)
        dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_object)

        # Create the default L2 inverse problem from the above objects
        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)

        # Specify how the initial beta is found
        betaest = directives.BetaEstimate_ByEig(beta0_ratio=2)

        # Target misfit to stop the inversion,
        # try to fit as much as possible of the signal, we don't want to lose anything
        IRLS = directives.Update_IRLS(
            f_min_change=1e-3, minGNiter=1, beta_tol=1e-1, max_irls_iterations=5
        )
        update_Jacobi = directives.UpdatePreconditioner()
        # Put all the parts together
        inv = inversion.BaseInversion(
            invProb, directiveList=[betaest, IRLS, update_Jacobi]
        )

        # Run the equivalent source inversion
        print("Solving for Equivalent Source")
        mrec = inv.run(mstart)

        ########################################################
        # Forward Amplitude Data
        # ----------------------
        #
        # Now that we have an equialent source layer, we can forward model alh three
        # components of the field and add them up: :math:`|B| = \sqrt{( Bx^2 + Bx^2 + Bx^2 )}`
        #

        receiver_list = magnetics.receivers.Point(rxLoc, components=["bx", "by", "bz"])
        srcField = magnetics.sources.SourceField(
            receiver_list=[receiver_list], parameters=H0
        )
        surveyAmp = magnetics.survey.Survey(srcField)

        simulation = magnetics.simulation.Simulation3DIntegral(
            mesh=mesh,
            survey=surveyAmp,
            chiMap=idenMap,
            actInd=surf,
            is_amplitude_data=True,
            store_sensitivities="forward_only",
        )

        bAmp = simulation.fields(mrec)

        ######################################################################
        # Amplitude Inversion
        # -------------------
        #
        # Now that we have amplitude data, we can invert for an effective
        # susceptibility. This is a non-linear inversion.
        #

        # Create active map to go from reduce space to full
        actvMap = maps.InjectActiveCells(mesh, actv, -100)
        nC = int(actv.sum())

        # Create identity map
        idenMap = maps.IdentityMap(nP=nC)

        mstart = np.ones(nC) * 1e-4

        # Create the forward model operator
        simulation = magnetics.simulation.Simulation3DIntegral(
            survey=surveyAmp,
            mesh=mesh,
            chiMap=idenMap,
            actInd=actv,
            is_amplitude_data=True,
        )

        data_obj = data.Data(survey, dobs=bAmp, noise_floor=wd)

        # Create a sparse regularization
        reg = regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
        reg.norms = [1, 0, 0, 0]
        reg.mref = np.zeros(nC)

        # Data misfit function
        dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_obj)

        # Add directives to the inversion
        opt = optimization.ProjectedGNCG(
            maxIter=10, lower=0.0, upper=1.0, maxIterLS=5, maxIterCG=5, tolCG=1e-3
        )

        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)

        # Here is the list of directives
        betaest = directives.BetaEstimate_ByEig(beta0_ratio=1)

        # Specify the sparse norms
        IRLS = directives.Update_IRLS(
            max_irls_iterations=5,
            f_min_change=1e-3,
            minGNiter=1,
            coolingRate=1,
            beta_search=False,
        )

        # Special directive specific to the mag amplitude problem. The sensitivity
        # weights are update between each iteration.
        update_SensWeight = directives.UpdateSensitivityWeights()
        update_Jacobi = directives.UpdatePreconditioner()

        # Put all together
        self.inv = inversion.BaseInversion(
            invProb, directiveList=[update_SensWeight, betaest, IRLS, update_Jacobi]
        )

        self.mstart = mstart
        self.model = model
        self.sim = simulation

    def test_mag_inverse(self):
        # Run the inversion
        mrec_Amp = self.inv.run(self.mstart)

        residual = np.linalg.norm(mrec_Amp - self.model) / np.linalg.norm(self.model)
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
        self.assertTrue(residual < 1.0)

    def tearDown(self):
        # Clean up the working directory
        if self.sim.store_sensitivities == "disk":
            shutil.rmtree(self.sim.sensitivity_path)


if __name__ == "__main__":
    unittest.main()
