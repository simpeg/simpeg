import numpy as np
import unittest
import discretize
from SimPEG.maps import Wires
from SimPEG.utils import (
    mkvc,
    WeightedGaussianMixture,
    GaussianMixtureWithPrior,
)
from scipy.stats import norm, multivariate_normal


class TestGMMs(unittest.TestCase):
    def setUp(self):

        np.random.seed(518936)

        # Create a cloud of  random points from a random gaussian mixture
        self.ndim = 2
        self.n_components = 2
        sigma = np.random.randn(self.n_components, self.ndim, self.ndim)
        sigma = np.c_[[sigma[i].dot(sigma[i].T) for i in range(sigma.shape[0])]]
        sigma[0] += np.eye(self.ndim)
        sigma[1] += np.eye(self.ndim) - 0.25 * np.eye(self.ndim).transpose((1, 0))
        self.sigma = sigma
        self.means = np.abs(np.random.randn(self.ndim, self.ndim)) * np.c_[[100.0, -100.0]]
        self.rv0 = multivariate_normal(self.means[0], self.sigma[0])
        self.rv1 = multivariate_normal(self.means[1], self.sigma[1])
        self.proportions = np.r_[0.6, 0.4]
        self.nsample = 1000
        self.s0 = self.rv0.rvs(int(self.nsample * self.proportions[0]))
        self.s1 = self.rv1.rvs(int(self.nsample * self.proportions[1]))
        self.samples = np.r_[self.s0, self.s1]
        self.model = mkvc(self.samples)
        self.mesh = discretize.TensorMesh([np.maximum(1e-1,np.random.randn(self.nsample) ** 2.0)])
        self.wires = Wires(("s0", self.mesh.nC), ("s1", self.mesh.nC))

        self.PlotIt = False

    def test_weighted_gaussian_mixture_multicomponents_multidimensions(self):
        clf = WeightedGaussianMixture(
            mesh=self.mesh,
            n_components=self.n_components,
            covariance_type="full",
            max_iter=1000,
            n_init=20,
            tol=1e-8,
            means_init=self.means,
            warm_start=True,
            precisions_init=np.linalg.inv(self.sigma),
            weights_init=self.proportions,
        )
        clf.fit(self.samples)

        checking_means = np.c_[
            np.average(self.s0, axis=0, weights=self.mesh.cell_volumes[:self.s0.shape[0]]),
            np.average(self.s1, axis=0, weights=self.mesh.cell_volumes[self.s0.shape[0]:]),
        ].T

        checking_covariances = np.r_[
            np.cov(self.s0.T, ddof=0, aweights=self.mesh.cell_volumes[:self.s0.shape[0]]),
            np.cov(self.s1.T, ddof=0, aweights=self.mesh.cell_volumes[self.s0.shape[0]:])
        ].reshape(clf.covariances_.shape)

        checking_proportions = np.r_[
            self.mesh.cell_volumes[:self.s0.shape[0]].sum(),
            self.mesh.cell_volumes[self.s0.shape[0]:].sum()
        ]
        checking_proportions /= checking_proportions.sum()

        self.assertTrue(np.all(np.isclose(clf.means_, checking_means)))
        self.assertTrue(np.all(np.isclose(clf.covariances_, checking_covariances)))
        self.assertTrue(np.all(np.isclose(clf.weights_, checking_proportions)))
        print("WeightedGaussianMixture is estimating correctly in 2D with 2 components.")

    def test_weighted_gaussian_mixture_one_component_1d(self):
        model1d = self.wires.s0 * self.model
        clf = WeightedGaussianMixture(
            mesh=self.mesh,
            n_components=1,
            covariance_type="full",
            max_iter=1000,
            n_init=10,
            tol=1e-8,
            warm_start=True,
        )
        clf.fit(model1d.reshape(-1, 1))

        cheching_mean = np.average(model1d, weights=self.mesh.cell_volumes)
        checking_covariance = np.cov(model1d, ddof=0, aweights=self.mesh.cell_volumes)

        self.assertTrue(np.isclose(clf.means_[0], cheching_mean))
        self.assertTrue(np.isclose(clf.covariances_[0], checking_covariance))
        print("WeightedGaussianMixture is estimating correctly in 1D with 1 component.")

    def test_MAP_estimate_one_component_1d(self):
        #subsample mesh and model between mle and prior
        n_samples = int(self.nsample * self.proportions.min())
        model_map = self.wires.s0 * self.model
        model_mle = model_map[:n_samples]
        model_prior = model_map[-n_samples:]
        actv = np.zeros(self.mesh.nC, dtype='bool')
        actv[:n_samples] = np.ones(n_samples, dtype='bool')

        clfref = WeightedGaussianMixture(
            mesh=self.mesh,
            actv=actv,
            n_components=1,
            covariance_type="full",
            max_iter=1000,
            n_init=10,
            tol=1e-8,
            warm_start=True,
        )
        clfref.fit(model_prior.reshape(-1,1))

        clf = GaussianMixtureWithPrior(
            gmmref=clfref,
            max_iter=1000,
            n_init=10,
            tol=1e-8,
            warm_start=True,
            nu=1,kappa=1,zeta=1,
            prior_type="full",
            update_covariances=True,
        )
        clf.fit(model_mle.reshape(-1,1))

        checking_means = np.average(
            np.r_[model_mle,model_prior],
            weights=np.r_[self.mesh.cell_volumes[actv], self.mesh.cell_volumes[actv]]
        )
        checking_covariance = np.cov(
            np.r_[model_mle,model_prior],
            ddof=0,
            aweights=np.r_[self.mesh.cell_volumes[actv], self.mesh.cell_volumes[actv]]
        )

        self.assertTrue(np.isclose(checking_covariance, clf.covariances_))
        self.assertTrue(np.isclose(checking_means, clf.means_))
        print("GaussianMixtureWithPrior is fully-MAP-estimating correctly in 1D with 1 component.")

        clfsemi = GaussianMixtureWithPrior(
            gmmref=clfref,
            max_iter=1000,
            n_init=10,
            tol=1e-8,
            warm_start=True,
            nu=1,kappa=1,zeta=1,
            prior_type="semi",
            update_covariances=True,
        )
        clfsemi.fit(model_mle.reshape(-1,1))

        checking_means_semi = np.average(
            np.r_[model_mle,model_prior],
            weights=np.r_[self.mesh.cell_volumes[actv], self.mesh.cell_volumes[actv]]
        )
        checking_covariance_semi = 0.5 * np.cov(
            model_mle,
            ddof=0,
            aweights=self.mesh.cell_volumes[actv]
        ) +  0.5 * np.cov(
            model_prior,
            ddof=0,
            aweights=self.mesh.cell_volumes[actv]
        )
        self.assertTrue(np.isclose(checking_covariance_semi, clfsemi.covariances_))
        self.assertTrue(np.isclose(checking_means_semi, clfsemi.means_))
        print("GaussianMixtureWithPrior is semi-MAP-estimating correctly in 1D with 1 component.")

    def test_MAP_estimate_multi_component_multidimensions(self):
        #prior model at three-quarter-way the means and identity covariances
        model_prior = np.random.randn(*self.samples.shape) + \
                0.9 * self.means[np.random.choice(2, size=self.nsample, p=[0.9,0.1])]

        clfref = WeightedGaussianMixture(
            mesh=self.mesh,
            n_components=self.n_components,
            covariance_type="full",
            max_iter=1000,
            n_init=10,
            tol=1e-8,
            warm_start=True,
        )
        clfref.fit(model_prior)
        clfref.order_clusters_GM_weight()

        clf = GaussianMixtureWithPrior(
            gmmref=clfref,
            max_iter=1000,
            n_init=100,
            tol=1e-10,
            nu=1,kappa=1,zeta=1,
            prior_type="semi",
            update_covariances=True,
        )
        clf.fit(self.samples)

        # This is a rough estimate of the multidimensional, multi-components means
        checking_means = np.c_[
            (clf.weights_[0] * np.average(self.s0, axis=0, weights=self.mesh.cell_volumes[:self.s0.shape[0]]) +
                clfref.weights_[0] * clfref.means_[0]) / (clf.weights_[0] + clfref.weights_[0]),
            (clf.weights_[1] * np.average(self.s1, axis=0, weights=self.mesh.cell_volumes[self.s0.shape[0]:]) +
                clfref.weights_[1] * clfref.means_[1]) / (clf.weights_[1] + clfref.weights_[1]),
        ].T
        self.assertTrue(np.all(np.isclose(checking_means, clf.means_, rtol=1e-2)))

        # This is a rough estimate of the multidimensional, multi-components covariances_
        checking_covariances = np.r_[
            (clf.weights_[0] * np.cov(self.s0.T, ddof=0, aweights=self.mesh.cell_volumes[:self.s0.shape[0]]) +
                clfref.weights_[0] * clfref.covariances_[0]) / (clf.weights_[0] + clfref.weights_[0]),
            (clf.weights_[1] * np.cov(self.s1.T, ddof=0, aweights=self.mesh.cell_volumes[self.s0.shape[0]:]) +
                clfref.weights_[1] * clfref.covariances_[1]) / (clf.weights_[1] + clfref.weights_[1])
        ].reshape(clf.covariances_.shape)
        self.assertTrue(np.all(np.isclose(checking_covariances, clf.covariances_, rtol=0.15)))

        checking_proportions = np.r_[
            self.mesh.cell_volumes[:self.s0.shape[0]].sum() + clfref.weights_[0] * self.mesh.cell_volumes.sum(),
            self.mesh.cell_volumes[self.s0.shape[0]:].sum() + + clfref.weights_[1] * self.mesh.cell_volumes.sum()
        ]
        checking_proportions /= checking_proportions.sum()
        self.assertTrue(np.all(np.isclose(checking_proportions, clf.weights_)))
        print("GaussianMixtureWithPrior is semi-MAP-estimating correctly in 2D with 2 components.")

if __name__ == "__main__":
    unittest.main()
