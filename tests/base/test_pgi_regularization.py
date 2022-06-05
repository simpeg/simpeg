import numpy as np
import unittest
import discretize
from SimPEG import regularization
from SimPEG.maps import Wires
from SimPEG.utils import (
    mkvc,
    WeightedGaussianMixture,
)
from scipy.stats import multivariate_normal
from scipy.sparse.linalg import spsolve, LinearOperator, bicgstab
from pymatsolver import SolverLU


class TestPGI(unittest.TestCase):
    def setUp(self):

        np.random.seed(518936)

        # Create a cloud of  random points from a random gaussian mixture
        self.ndim = 2
        self.n_components = 3
        sigma = np.random.randn(self.n_components, self.ndim, self.ndim)
        sigma = np.c_[[sigma[i].dot(sigma[i].T) for i in range(sigma.shape[0])]]
        sigma[0] += np.eye(self.ndim)
        sigma[1] += np.eye(self.ndim) - 0.25 * np.eye(self.ndim).transpose((1, 0))
        self.sigma = sigma
        self.means = (
            np.abs(np.random.randn(self.n_components, self.ndim))
            * np.c_[[-100, 100], [100, 1], [-100, -100]].T
        )
        self.rv_list = [
            multivariate_normal(mean, sigma)
            for i, (mean, sigma) in enumerate(zip(self.means, self.sigma))
        ]
        proportions = np.round(np.abs(np.random.rand(self.n_components)), decimals=1)
        proportions = np.abs(np.random.rand(self.n_components))
        self.proportions = proportions / proportions.sum()
        nsample = 1000
        self.samples = np.concatenate(
            [
                rv.rvs(int(nsample * prp))
                for i, (rv, prp) in enumerate(zip(self.rv_list, self.proportions))
            ]
        )
        self.nsample = self.samples.shape[0]
        self.model = mkvc(self.samples)
        self.mesh = discretize.TensorMesh(
            [np.maximum(1e-1, np.random.randn(self.nsample) ** 2.0)]
        )
        self.wires = Wires(("s0", self.mesh.nC), ("s1", self.mesh.nC))
        self.cell_weights_list = [
            np.maximum(1e-1, np.random.randn(self.mesh.nC) ** 2.0),
            np.maximum(1e-1, np.random.randn(self.mesh.nC) ** 2.0),
        ]
        self.PlotIt = False

    def test_full_covariances(self):

        print("Test Full covariances: ")
        print("=======================")
        # Fit a Gaussian Mixture
        clf = WeightedGaussianMixture(
            mesh=self.mesh,
            n_components=self.n_components,
            covariance_type="full",
            max_iter=1000,
            n_init=10,
            tol=1e-8,
            means_init=self.means,
            warm_start=True,
            precisions_init=np.linalg.inv(self.sigma),
            weights_init=self.proportions,
        )
        clf.fit(self.samples)

        # Define reg
        reg = regularization.PGI(
            self.mesh,
            clf,
            approx_gradient=True,
            alpha_x=0.0,
            alpha_s=0.0,
            wiresmap=self.wires,
            weights_list=self.cell_weights_list,
        )

        mref = mkvc(clf.means_[clf.predict(self.samples)])

        # check score value
        dm = self.model - mref
        score_approx0 = reg(self.model)
        score_approx1 = 0.5 * dm.dot(reg.deriv2(self.model, dm))
        passed_score_approx = np.allclose(score_approx0, score_approx1)
        self.assertTrue(passed_score_approx)

        reg.objfcts[0].approx_eval = False
        score = reg(self.model) - reg(mref)
        passed_score = np.allclose(score_approx0, score, rtol=1e-4)
        self.assertTrue(passed_score)

        print("scores for PGI  & Full Cov. are ok.")

        # check derivatives as an optimization on locally quadratic function
        deriv = reg.deriv(self.model)
        reg.objfcts[0].approx_gradient = False
        reg.objfcts[0].approx_hessian = False
        deriv_full = reg.deriv(self.model)
        passed_deriv1 = np.allclose(deriv, deriv_full, rtol=1e-4)
        self.assertTrue(passed_deriv1)
        print("1st derivatives for PGI & Full Cov. are ok.")

        Hinv = SolverLU(reg.deriv2(self.model))
        p = Hinv * deriv
        direction2 = np.c_[self.wires * p]
        passed_derivative = np.allclose(
            mkvc(self.samples - direction2), mkvc(mref), rtol=1e-4
        )
        self.assertTrue(passed_derivative)
        print("2nd derivatives for PGI & Full Cov. are ok.")

        if self.PlotIt:
            print("Plotting", self.PlotIt)
            import matplotlib.pyplot as plt

            xmin, xmax = ymin, ymax = self.samples.min(), self.samples.max()
            x, y = np.mgrid[xmin:xmax:0.5, ymin:ymax:0.5]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y
            rv = clf.score_samples(pos.reshape(-1, 2))
            rvm = clf.predict(pos.reshape(-1, 2))
            figfull, axfull = plt.subplots(1, 1, figsize=(16, 8))
            figfull.suptitle("Full Covariances Tests")

            axfull.contourf(x, y, rvm.reshape(x.shape), alpha=0.25, cmap="brg")
            axfull.contour(x, y, rv.reshape(x.shape), 20)
            axfull.scatter(
                self.samples[:, 0], self.samples[:, 1], color="blue", s=5.0, alpha=0.25
            )
            axfull.quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -(self.wires.s0 * deriv),
                -(self.wires.s1 * deriv),
                color="red",
                alpha=0.25,
            )
            axfull.quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -direction2[:, 0],
                -direction2[:, 1],
                color="k",
            )
            axfull.scatter(
                (self.samples - direction2)[:, 0],
                (self.samples - direction2)[:, 1],
                color="k",
                s=50.0,
            )
            axfull.set_xlabel("Property 1")
            axfull.set_ylabel("Property 2")
            axfull.set_title("PGI with W")

            plt.show()

    def test_tied_covariances(self):

        print("Test Tied covariances: ")
        print("=======================")
        # Fit a Gaussian Mixture
        clf = WeightedGaussianMixture(
            mesh=self.mesh,
            n_components=self.n_components,
            covariance_type="tied",
            max_iter=1000,
            n_init=10,
            tol=1e-8,
            means_init=self.means,
            warm_start=True,
            precisions_init=np.linalg.inv(self.sigma[0]),
            weights_init=self.proportions,
        )
        clf.fit(self.samples)

        # Define reg
        reg = regularization.PGI(
            self.mesh,
            clf,
            approx_gradient=True,
            alpha_x=0.0,
            alpha_s=0.0,
            wiresmap=self.wires,
            weights_list=self.cell_weights_list,
        )

        mref = mkvc(clf.means_[clf.predict(self.samples)])

        # check score value
        dm = self.model - mref
        score_approx0 = reg(self.model)
        score_approx1 = 0.5 * dm.dot(reg.deriv2(self.model, dm))
        passed_score_approx = np.allclose(score_approx0, score_approx1)
        self.assertTrue(passed_score_approx)
        reg.objfcts[0].approx_eval = False
        score = reg(self.model) - reg(mref)
        passed_score = np.allclose(score_approx0, score, rtol=1e-4)
        self.assertTrue(passed_score)
        print("scores for PGI & tied Cov. are ok.")

        # check derivatives as an optimization on locally quadratic function
        deriv = reg.deriv(self.model)
        reg.objfcts[0].approx_gradient = False
        reg.objfcts[0].approx_hessian = False
        deriv_full = reg.deriv(self.model)
        passed_deriv1 = np.allclose(deriv, deriv_full, rtol=1e-4)
        self.assertTrue(passed_deriv1)
        print("1st derivatives for PGI & tied Cov. are ok.")

        Hinv = SolverLU(reg.deriv2(self.model))
        p = Hinv * deriv
        direction2 = np.c_[self.wires * p]
        passed_derivative = np.allclose(
            mkvc(self.samples - direction2), mkvc(mref), rtol=1e-4
        )
        self.assertTrue(passed_derivative)
        print("2nd derivatives for PGI & tied Cov. are ok.")

        if self.PlotIt:
            import matplotlib.pyplot as plt

            xmin, xmax = ymin, ymax = self.samples.min(), self.samples.max()
            x, y = np.mgrid[xmin:xmax:0.5, ymin:ymax:0.5]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y
            rv = clf.score_samples(pos.reshape(-1, 2))
            rvm = clf.predict(pos.reshape(-1, 2))
            figtied, axtied = plt.subplots(1, 1, figsize=(16, 8))
            figtied.suptitle("Tied Covariances Tests")

            axtied.contourf(x, y, rvm.reshape(x.shape), alpha=0.25, cmap="brg")
            axtied.contour(x, y, rv.reshape(x.shape), 20)
            axtied.scatter(
                self.samples[:, 0], self.samples[:, 1], color="blue", s=5.0, alpha=0.25
            )
            axtied.quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -(self.wires.s0 * deriv),
                -(self.wires.s1 * deriv),
                color="red",
                alpha=0.25,
            )
            axtied.quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -direction2[:, 0],
                -direction2[:, 1],
                color="k",
            )
            axtied.scatter(
                (self.samples - direction2)[:, 0],
                (self.samples - direction2)[:, 1],
                color="k",
                s=50.0,
            )
            axtied.set_xlabel("Property 1")
            axtied.set_ylabel("Property 2")
            axtied.set_title("PGI with W")

            plt.show()

    def test_diag_covariances(self):

        print("Test Diagonal covariances: ")
        print("===========================")
        # Fit a Gaussian Mixture
        clf = WeightedGaussianMixture(
            mesh=self.mesh,
            n_components=self.n_components,
            covariance_type="diag",
            max_iter=1000,
            n_init=10,
            tol=1e-8,
            means_init=self.means,
            warm_start=True,
            weights_init=self.proportions,
        )
        clf.fit(self.samples)

        # Define reg
        reg = regularization.PGI(
            self.mesh,
            clf,
            approx_gradient=True,
            alpha_x=0.0,
            alpha_s=0.0,
            wiresmap=self.wires,
            weights_list=self.cell_weights_list,
        )

        mref = mkvc(clf.means_[clf.predict(self.samples)])

        # check score value
        dm = self.model - mref
        score_approx0 = reg(self.model)
        score_approx1 = 0.5 * dm.dot(reg.deriv2(self.model, dm))
        passed_score_approx = np.allclose(score_approx0, score_approx1)
        self.assertTrue(passed_score_approx)
        reg.objfcts[0].approx_eval = False
        score = reg(self.model) - reg(mref)
        passed_score = np.allclose(score_approx0, score, rtol=1e-4)
        self.assertTrue(passed_score)
        print("scores for PGI & diag Cov. are ok.")

        # check derivatives as an optimization on locally quadratic function
        deriv = reg.deriv(self.model)
        reg.objfcts[0].approx_gradient = False
        reg.objfcts[0].approx_hessian = False
        deriv_full = reg.deriv(self.model)
        passed_deriv1 = np.allclose(deriv, deriv_full, rtol=1e-4)
        self.assertTrue(passed_deriv1)
        print("1st derivatives for PGI & diag Cov. are ok.")

        Hinv = SolverLU(reg.deriv2(self.model))
        p = Hinv * deriv
        direction2 = np.c_[self.wires * p]
        passed_derivative = np.allclose(
            mkvc(self.samples - direction2), mkvc(mref), rtol=1e-4
        )
        self.assertTrue(passed_derivative)
        print("2nd derivatives for PGI & diag Cov. are ok.")

        if self.PlotIt:
            import matplotlib.pyplot as plt

            xmin, xmax = ymin, ymax = self.samples.min(), self.samples.max()
            x, y = np.mgrid[xmin:xmax:0.5, ymin:ymax:0.5]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y
            rv = clf.score_samples(pos.reshape(-1, 2))
            rvm = clf.predict(pos.reshape(-1, 2))
            figdiag, axdiag = plt.subplots(1, 1, figsize=(16, 8))
            figdiag.suptitle("Diag Covariances Tests")

            axdiag.contourf(x, y, rvm.reshape(x.shape), alpha=0.25, cmap="brg")
            axdiag.contour(x, y, rv.reshape(x.shape), 20)
            axdiag.scatter(
                self.samples[:, 0], self.samples[:, 1], color="blue", s=5.0, alpha=0.25
            )
            axdiag.quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -(self.wires.s0 * deriv),
                -(self.wires.s1 * deriv),
                color="red",
                alpha=0.25,
            )
            axdiag.quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -direction2[:, 0],
                -direction2[:, 1],
                color="k",
            )
            axdiag.scatter(
                (self.samples - direction2)[:, 0],
                (self.samples - direction2)[:, 1],
                color="k",
                s=50.0,
            )
            axdiag.set_xlabel("Property 1")
            axdiag.set_ylabel("Property 2")
            axdiag.set_title("PGI with W")

            plt.show()

    def test_spherical_covariances(self):

        print("Test Spherical covariances: ")
        print("============================")
        # Fit a Gaussian Mixture
        clf = WeightedGaussianMixture(
            mesh=self.mesh,
            n_components=self.n_components,
            covariance_type="spherical",
            max_iter=1000,
            n_init=10,
            tol=1e-8,
            means_init=self.means,
            warm_start=True,
            weights_init=self.proportions,
        )
        clf.fit(self.samples)

        # Define reg
        reg = regularization.PGI(
            self.mesh,
            clf,
            approx_gradient=True,
            alpha_x=0.0,
            alpha_s=0.0,
            wiresmap=self.wires,
            weights_list=self.cell_weights_list,
        )

        mref = mkvc(clf.means_[clf.predict(self.samples)])

        # check score value
        dm = self.model - mref
        score_approx0 = reg(self.model)
        score_approx1 = 0.5 * dm.dot(reg.deriv2(self.model, dm))
        passed_score_approx = np.allclose(score_approx0, score_approx1)
        self.assertTrue(passed_score_approx)
        reg.objfcts[0].approx_eval = False
        score = reg(self.model) - reg(mref)
        passed_score = np.allclose(score_approx0, score, rtol=1e-4)
        self.assertTrue(passed_score)
        print("scores for PGI & spherical Cov. are ok.")

        # check derivatives as an optimization on locally quadratic function
        deriv = reg.deriv(self.model)
        reg.objfcts[0].approx_gradient = False
        reg.objfcts[0].approx_hessian = False
        deriv_full = reg.deriv(self.model)
        passed_deriv1 = np.allclose(deriv, deriv_full, rtol=1e-4)
        self.assertTrue(passed_deriv1)
        print("1st derivatives for PGI & spherical Cov. are ok.")

        Hinv = SolverLU(reg.deriv2(self.model))
        p = Hinv * deriv
        direction2 = np.c_[self.wires * p]
        passed_derivative = np.allclose(
            mkvc(self.samples - direction2), mkvc(mref), rtol=1e-4
        )
        self.assertTrue(passed_derivative)
        print("2nd derivatives for PGI & spherical Cov. are ok.")

        if self.PlotIt:
            import matplotlib.pyplot as plt

            xmin, xmax = ymin, ymax = self.samples.min(), self.samples.max()
            x, y = np.mgrid[xmin:xmax:0.5, ymin:ymax:0.5]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y
            rv = clf.score_samples(pos.reshape(-1, 2))
            rvm = clf.predict(pos.reshape(-1, 2))
            figspherical, axspherical = plt.subplots(1, 1, figsize=(16, 8))
            figspherical.suptitle("Spherical Covariances Tests")

            axspherical.contourf(x, y, rvm.reshape(x.shape), alpha=0.25, cmap="brg")
            axspherical.contour(x, y, rv.reshape(x.shape), 20)
            axspherical.scatter(
                self.samples[:, 0], self.samples[:, 1], color="blue", s=5.0, alpha=0.25
            )
            axspherical.quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -(self.wires.s0 * deriv),
                -(self.wires.s1 * deriv),
                color="red",
                alpha=0.25,
            )
            axspherical.quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -direction2[:, 0],
                -direction2[:, 1],
                color="k",
            )
            axspherical.scatter(
                (self.samples - direction2)[:, 0],
                (self.samples - direction2)[:, 1],
                color="k",
                s=50.0,
            )
            axspherical.set_xlabel("Property 1")
            axspherical.set_ylabel("Property 2")
            axspherical.set_title("PGI with W")

            plt.show()


if __name__ == "__main__":
    unittest.main()
