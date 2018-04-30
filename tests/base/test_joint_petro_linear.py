from __future__ import print_function

import unittest

from SimPEG import (
    Mesh, Problem, Survey, Maps, Utils, DataMisfit,
    Regularization, Optimization, InvProblem,
    Directives, Inversion)
import numpy as np

np.random.seed(1)


class JointInversionTest(unittest.TestCase):

    def setUp(self):

        self.PlotIt = False

        # Mesh
        N = 100
        mesh = Mesh.TensorMesh([N])

        # Survey design parameters
        nk = 30
        jk = np.linspace(1., 60., nk)
        p = -0.25
        q = 0.25

        # Physics
        def g(k):
            return (
                np.exp(p * jk[k] * mesh.vectorCCx) *
                np.cos(np.pi * q * jk[k] * mesh.vectorCCx)
            )

        G = np.empty((nk, mesh.nC))

        for i in range(nk):
            G[i, :] = g(i)

        # Model 1st problem
        m0 = np.zeros(mesh.nC)
        m0[20:41] = np.linspace(0., 1., 21)
        m0[41:57] = np.linspace(-1, 0., 16)

        # Nonlinear relationships
        poly0 = Maps.PolynomialPetroClusterMap(coeffyx=np.r_[0., -2., 2.])
        poly1 = Maps.PolynomialPetroClusterMap(coeffyx=np.r_[-0., 3, 6, 4.])
        poly0_inverse = Maps.PolynomialPetroClusterMap(
            coeffyx=-np.r_[-0., -2., 2.])
        poly1_inverse = Maps.PolynomialPetroClusterMap(
            coeffyx=-np.r_[0., 3, 6, 4.])
        cluster_mapping = [Maps.IdentityMap(), poly0_inverse, poly1_inverse]

        # model 2nd problem
        m1 = np.zeros(100)
        m1[20:41] = 1. + (poly0 * np.vstack([m0[20:41], m1[20:41]]).T)[:, 1]
        m1[41:57] = -1. + (poly1 * np.vstack([m0[41:57], m1[41:57]]).T)[:, 1]

        model = np.vstack([m0, m1]).T
        m = Utils.mkvc(model)

        clfmapping = Utils.GaussianMixtureWithMapping(
            n_components=3, covariance_type='full', tol=1e-3,
            reg_covar=1e-3, max_iter=100, n_init=20, init_params='kmeans',
            random_state=None, warm_start=False,
            verbose=0, verbose_interval=10, cluster_mapping=cluster_mapping
        )
        clfmapping = clfmapping.fit(model)

        clfnomapping = Utils.GaussianMixture(
            n_components=3, covariance_type='full', tol=1e-3,
            reg_covar=1e-3, max_iter=100, n_init=10, init_params='kmeans',
            random_state=None, warm_start=False,
            verbose=0, verbose_interval=10,
        )
        clfnomapping = clfnomapping.fit(model)

        wires = Maps.Wires(('m1', mesh.nC), ('m2', mesh.nC))
        prob1 = Problem.LinearProblem(mesh, G=G, modelMap=wires.m1)
        survey1 = Survey.LinearSurvey()
        survey1.pair(prob1)
        survey1.makeSyntheticData(m, std=0.01)
        survey1.eps = 0.

        prob2 = Problem.LinearProblem(mesh, G=G, modelMap=wires.m2)
        survey2 = Survey.LinearSurvey()
        survey2.pair(prob2)
        survey2.makeSyntheticData(m, std=0.01)
        survey2.eps = 0.

        dmis1 = DataMisfit.l2_DataMisfit(survey1)
        dmis2 = DataMisfit.l2_DataMisfit(survey2)
        dmis = dmis1 + dmis2

        self.mesh = mesh
        self.model = m

        self.survey1 = survey1
        self.prob1 = prob1

        self.survey2 = survey2
        self.prob2 = prob2

        self.dmiscombo = dmis

        self.clfmapping = clfmapping
        self.clfnomapping = clfnomapping
        self.wires = wires

        self.minit = np.zeros_like(self.model)

        # Distance weighting
        wr1 = np.sum(prob1.getJ(self.minit)**2., axis=0)**0.5
        wr1 = wr1 / np.max(wr1)
        wr2 = np.sum(prob2.getJ(self.minit)**2., axis=0)**0.5
        wr2 = wr2 / np.max(wr2)
        wr = wr1 + wr2

        self.W = wr

    def test_joint_petro_inv_with_mapping(self):

        print("test_joint_petro_inv_with_mapping: ")
        reg_simple = Regularization.SimplePetroWithMappingRegularization(
            mesh=self.mesh,
            GMmref=self.clfmapping,
            GMmodel=self.clfmapping,
            approx_gradient=True, alpha_x=0.,
            wiresmap=self.wires,
            evaltype='approx'
        )
        reg_simple.objfcts[0].cell_weights = self.W

        opt = Optimization.ProjectedGNCG(
            maxIter=20, tolX=1e-6, maxIterCG=100, tolCG=1e-3
        )

        invProb = InvProblem.BaseInvProblem(self.dmiscombo, reg_simple, opt)

        # Directives
        Alphas = Directives.AlphasSmoothEstimate_ByEig(
            alpha0_ratio=5e-2, ninit=10, verbose=True
        )
        Scales = Directives.ScalingEstimate_ByEig(
            Chi0_ratio=.1, verbose=True, ninit=100)
        beta = Directives.BetaEstimate_ByEig(beta0_ratio=1e-5, ninit=100)
        betaIt = Directives.PetroBetaReWeighting(
            verbose=True, rateCooling=5., rateWarming=1.,
            tolerance=0.02, UpdateRate=1,
            ratio_in_cooling=False,
            progress=0.1,
            update_prior_confidence=False,
            ratio_in_gamma_cooling=False,
            alphadir_rateCooling=1.,
            kappa_rateCooling=1.,
            nu_rateCooling=1.,)
        targets = Directives.PetroTargetMisfit(
            TriggerSmall=True, TriggerTheta=False,
            verbose=True
        )
        petrodir = Directives.UpdateReference()

        # Setup Inversion
        inv = Inversion.BaseInversion(
            invProb, directiveList=[
                Alphas, Scales, beta,
                petrodir, targets,
                betaIt
            ]
        )

        self.mcluster_map = inv.run(self.minit)

        if self.PlotIt:
            import matplotlib.pyplot as plt
            import seaborn
            seaborn.set()

            fig, axes = plt.subplots(1, 4, figsize=(15, 5))
            axes = axes.reshape(4)
            left, width = .25, .5
            bottom, height = .25, .5
            right = left + width
            top = bottom + height
            axes[0].set_axis_off()
            axes[0].text(
                0.5 * (left + right), 0.5 * (bottom + top),
                ('Using true nonlinear\npetrophysics clusters'),
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20, color='black',
                transform=axes[0].transAxes
            )
            axes[1].plot(self.mesh.vectorCCx, self.wires.m1 *
                         self.mcluster_map, 'b.-', ms=5, marker='v')
            axes[1].plot(self.mesh.vectorCCx, self.wires.m1 * self.model, 'k--')
            axes[1].set_title('Problem 1')
            axes[1].legend(['Recovered Model', 'True Model'])
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Property 1')

            axes[2].plot(self.mesh.vectorCCx, self.wires.m2 *
                         self.mcluster_map, 'r.-', ms=5, marker='v')
            axes[2].plot(self.mesh.vectorCCx, self.wires.m2 * self.model, 'k--')
            axes[2].set_title('Problem 2')
            axes[2].legend(['Recovered Model', 'True Model'])
            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Property 2')

            x, y = np.mgrid[-1:1:.01, -2:2:.01]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y
            CS = axes[3].contour(x, y, np.exp(self.clfmapping.score_samples(
                pos.reshape(-1, 2)).reshape(x.shape)), 100, alpha=0.25, cmap='viridis')
            axes[3].scatter(self.wires.m1 * self.mcluster_map,
                            self.wires.m2 * self.mcluster_map, marker='v')
            axes[3].set_title('Petro Distribution')
            CS.collections[0].set_label('')
            axes[3].legend(['True Petro Distribution',
                            'Recovered model crossplot'])
            axes[3].set_xlabel('Property 1')
            axes[3].set_ylabel('Property 2')

            fig.suptitle(
                'Doodling with Mapping: one mapping per identified rock unit\n' +
                'Joint inversion of 1D Linear Problems ' +
                'with nonlinear petrophysical relationships',
                fontsize=24
            )
            plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.85)
            plt.show()

    def test_joint_petro_inv(self):

        print("test_joint_petro_inv: ")
        reg_simple = Regularization.SimplePetroRegularization(
            mesh=self.mesh,
            GMmref=self.clfnomapping,
            GMmodel=self.clfnomapping,
            approx_gradient=True, alpha_x=0.,
            wiresmap=self.wires,
            evaltype='approx'
        )
        reg_simple.objfcts[0].cell_weights = self.W
        reg_simple.gamma = np.ones(self.clfnomapping.n_components) * 1e8

        opt = Optimization.ProjectedGNCG(
            maxIter=20, tolX=1e-6, maxIterCG=100, tolCG=1e-3
        )

        invProb = InvProblem.BaseInvProblem(self.dmiscombo, reg_simple, opt)
        # Directives
        Alphas = Directives.AlphasSmoothEstimate_ByEig(
            alpha0_ratio=5e-2, ninit=10, verbose=True)
        Scales = Directives.ScalingEstimate_ByEig(
            Chi0_ratio=1., verbose=True, ninit=100)
        beta = Directives.BetaEstimate_ByEig(beta0_ratio=1e-5, ninit=100)
        betaIt = Directives.PetroBetaReWeighting(
            verbose=True, rateCooling=5., rateWarming=1.,
            tolerance=0.02, UpdateRate=1,
            ratio_in_cooling=False,
            progress=0.1,
            update_prior_confidence=False,
            ratio_in_gamma_cooling=False,
            alphadir_rateCooling=1.,
            kappa_rateCooling=1.,
            nu_rateCooling=1.,)
        targets = Directives.PetroTargetMisfit(
            TriggerSmall=True, TriggerTheta=False,
            verbose=True
        )
        petrodir = Directives.GaussianMixtureUpdateModel()

        # Setup Inversion
        inv = Inversion.BaseInversion(
            invProb, directiveList=[
                Alphas, Scales, beta,
                petrodir, targets,
                betaIt
            ]
        )

        self.mcluster_no_map = inv.run(self.minit)

        if self.PlotIt:
            import matplotlib.pyplot as plt
            import seaborn
            seaborn.set()

            fig, axes = plt.subplots(1, 4, figsize=(15, 5))
            axes = axes.reshape(4)
            left, width = .25, .5
            bottom, height = .25, .5
            right = left + width
            top = bottom + height
            axes[0].set_axis_off()
            axes[0].text(
                0.5 * (left + right), 0.5 * (bottom + top),
                ('Using true nonlinear\npetrophysics clusters'),
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20, color='black',
                transform=axes[0].transAxes
            )
            axes[0].set_axis_off()
            axes[0].text(
                0.5 * (left + right), 0.5 * (bottom + top),
                ('Using linear\npetrophysics clusters'),
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20, color='black',
                transform=axes[0].transAxes
            )

            axes[1].plot(self.mesh.vectorCCx, self.wires.m1 *
                         self.mcluster_no_map, 'b.-', ms=5, marker='v')
            axes[1].plot(self.mesh.vectorCCx, self.wires.m1 * self.model, 'k--')
            axes[1].set_title('Problem 1')
            axes[1].legend(['Recovered Model', 'True Model'])
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Property 1')

            axes[2].plot(self.mesh.vectorCCx, self.wires.m2 *
                         self.mcluster_no_map, 'r.-', ms=5, marker='v')
            axes[2].plot(self.mesh.vectorCCx, self.wires.m2 * self.model, 'k--')
            axes[2].set_title('Problem 2')
            axes[2].legend(['Recovered Model', 'True Model'])
            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Property 2')

            x, y = np.mgrid[-1:1:.01, -2:2:.01]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y
            CSF = axes[3].contour(x, y, np.exp(self.clfmapping.score_samples(
                pos.reshape(-1, 2)).reshape(x.shape)), 100, alpha=0.5)  # , cmap='viridis')
            CS = axes[3].contour(x, y, np.exp(self.clfnomapping.score_samples(
                pos.reshape(-1, 2)).reshape(x.shape)), 500, alpha=0.25, cmap='viridis')
            axes[3].scatter(self.wires.m1 * self.mcluster_no_map,
                            self.wires.m2 * self.mcluster_no_map, marker='v')
            axes[3].set_title('Petro Distribution')
            CSF.collections[0].set_label('')
            CS.collections[0].set_label('')
            axes[3].legend(
                [
                    'True Petro Distribution',
                    'Modeled Petro Distribution',
                    'Recovered model crossplot'
                ]
            )
            axes[3].set_xlabel('Property 1')
            axes[3].set_ylabel('Property 2')

            plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.85)
            plt.show()

if __name__ == '__main__':
    unittest.main()
