from __future__ import print_function

import unittest

from SimPEG import (
    Mesh, Problem, Survey, Maps, Utils, DataMisfit,
    Regularization, Optimization, InvProblem,
    Directives, Inversion)
import numpy as np

np.random.seed(518936)

class JointInversionTest(unittest.TestCase):

    def setUp(self):
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
            reg_covar=1e-3, max_iter=100, n_init=10, init_params='kmeans',
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
            alpha0_ratio=5e-2, ninit=10, verbose=True)
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

        mcluster_map = inv.run(self.minit)

    def test_joint_petro_inv(self):

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
            alpha0_ratio=5e-2, ninit=10, verbose=True)
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

        mcluster_no_map = inv.run(self.minit)

if __name__ == '__main__':
    unittest.main()
