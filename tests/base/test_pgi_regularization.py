import numpy as np
import unittest
import discretize as Mesh
from SimPEG import regularization
from SimPEG.maps import Wires
from SimPEG.utils import (
    mkvc,
    WeightedGaussianMixture,
    make_PGI_regularization,
    make_SimplePGI_regularization,
)
from scipy.stats import multivariate_normal
from scipy.sparse.linalg import LinearOperator, bicgstab
from pymatsolver import PardisoSolver


class TestPGI(unittest.TestCase):
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
        self.means = np.abs(np.random.randn(self.ndim, self.ndim)) * np.c_[[10.0, -10.0]]
        self.rv0 = multivariate_normal(self.means[0], self.sigma[0])
        self.rv1 = multivariate_normal(self.means[1], self.sigma[1])
        self.proportions = np.r_[0.6, 0.4]
        self.nsample = 1000
        self.s0 = self.rv0.rvs(int(self.nsample * self.proportions[0]))
        self.s1 = self.rv1.rvs(int(self.nsample * self.proportions[1]))
        self.samples = np.r_[self.s0, self.s1]
        self.model = mkvc(self.samples)
        self.mesh = Mesh.TensorMesh([self.samples.shape[0]])
        self.wires = Wires(("s0", self.mesh.nC), ("s1", self.mesh.nC))
        self.cell_weights_list = [
            np.maximum(1e-1,np.random.randn(self.mesh.nC) ** 2.0),
            np.maximum(1e-1,np.random.randn(self.mesh.nC) ** 2.0),
        ]
        self.PlotIt = False

    def test_full_covariances(self):

        print("Test Full covariances: ")
        # Fit a Gaussian Mixture
        clf = WeightedGaussianMixture(
            mesh=self.mesh,
            n_components=self.n_components,
            covariance_type="full",
            max_iter=1000,
            n_init=100,
            tol=1e-8,
            means_init=self.means,
            warm_start=True,
            precisions_init=np.linalg.inv(self.sigma),
            weights_init=self.proportions,
        )
        clf.fit(self.samples)

        # Define reg Simple
        reg_simple = make_SimplePGI_regularization(
            mesh=self.mesh,
            gmmref=clf,
            approx_gradient=True,
            alpha_x=0.0,
            wiresmap=self.wires,
            cell_weights_list=self.cell_weights_list,
        )
        # Define reg with volumes
        reg = make_PGI_regularization(
            mesh=self.mesh,
            gmmref=clf,
            approx_gradient=True,
            alpha_x=0.0,
            wiresmap=self.wires,
            cell_weights_list=self.cell_weights_list,
        )
        
        mref = mkvc(clf.means_[clf.predict(self.samples)])
        
        # check score value
        score_approx0 = reg_simple(self.model)
        dm = (self.model - mref)
        score_approx1 = 0.5 * dm.dot(reg_simple.deriv2(self.model, dm))
        print(score_approx0,score_approx1)
        passed_score_approx_simple = (score_approx0 == score_approx1)
        self.assertTrue(passed_score_approx_simple)

        reg_simple.objfcts[0].approx_eval = False
        score = reg_simple(self.model) - reg_simple(mref)
        passed_score_simple = np.allclose(score_approx0, score, rtol=1e-1)
        print("scores:", score_approx0,score_approx1, score)
        self.assertTrue(passed_score_simple)
        
        print("scores for SimplePGI & Full Cov. are ok.")

        score_approx0 = reg(self.model)
        score_approx1 = 0.5 * dm.dot(reg.deriv2(self.model,dm))
        passed_score_approx = np.allclose(score_approx0, score_approx1)
        self.assertTrue(passed_score_approx)

        reg.objfcts[0].approx_eval = False
        score = reg(self.model) - reg(mref)
        passed_score = np.allclose(score_approx0, score, rtol=1e-1)
        print("scores:", score_approx0,score_approx1, score)
        self.assertTrue(passed_score)
        
        print("scores for PGI  & Full Cov. are ok.")

        # check derivatives as an optimization on locally quadratic function
        # Simple

        deriv_simple = reg_simple.deriv(self.model)
        reg_simple.objfcts[0].approx_gradient = False
        deriv_simple_full = reg_simple.deriv(self.model)
        passed_deriv1 = np.allclose(deriv_simple, deriv_simple_full, rtol=1e-1)
        self.assertTrue(passed_deriv1)
        print("1st derivatives for SimplePGI & Full Cov. are ok.")
        deriv_simple = reg_simple.deriv(self.model)
        Hinv = PardisoSolver(reg_simple.deriv2(self.model))
        p_simple = Hinv * deriv_simple
        direction2_simple = np.c_[self.wires * p_simple]
        passed_derivative_simple = np.allclose(
            mkvc(self.samples - direction2_simple), mkvc(mref), rtol=1e-1
        )
        self.assertTrue(passed_derivative_simple)
        print("2nd derivatives for SimplePGI & Full Cov. are ok.")

        # With volumes
        deriv = reg.deriv(self.model)
        reg.objfcts[0].approx_gradient = False
        deriv_full = reg.deriv(self.model)
        passed_deriv1 = np.allclose(deriv, deriv_full, rtol=1e-1)
        self.assertTrue(passed_deriv1)
        print("1st derivatives for PGI & Full Cov. are ok.")
        Hinv = PardisoSolver(reg.deriv2(self.model))
        p = Hinv * deriv
        direction2 = np.c_[self.wires * p]
        passed_derivative = np.allclose(
            mkvc(self.samples - direction2), mkvc(mref), rtol=1e-1
        )
        self.assertTrue(passed_derivative)
        print("2nd derivatives for PGI & Full Cov. are ok.")

        if self.PlotIt:
            import matplotlib.pyplot as plt

            xmin, xmax = ymin, ymax = self.samples.min(), self.samples.max()
            x, y = np.mgrid[xmin:xmax:0.01, ymin:ymax:0.01]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y
            rv = clf.score_samples(pos.reshape(-1, 2))
            rvm = clf.predict(pos.reshape(-1, 2))
            figfull, axfull = plt.subplots(1, 2, figsize=(16, 8))
            figfull.suptitle("Full Covariances Tests")
            # Simple
            axfull[0].contourf(x, y, rvm.reshape(x.shape), alpha=0.25, cmap="brg")
            axfull[0].contour(x, y, rv.reshape(x.shape), 20)
            axfull[0].scatter(
                self.s0[:, 0], self.s0[:, 1], color="blue", s=5.0, alpha=0.25
            )
            axfull[0].scatter(
                self.s1[:, 0], self.s1[:, 1], color="green", s=5.0, alpha=0.25
            )
            axfull[0].quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -(self.wires.s0 * deriv_simple),
                -(self.wires.s1 * deriv_simple),
                color="red",
                alpha=0.25,
            )
            axfull[0].quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -direction2_simple[:, 0],
                -direction2_simple[:, 1],
                color="k",
            )
            axfull[0].scatter(
                (self.samples - direction2_simple)[:, 0],
                (self.samples - direction2_simple)[:, 1],
                color="k",
                s=50.0,
            )
            axfull[0].set_xlabel("Property 1")
            axfull[0].set_ylabel("Property 2")
            axfull[0].set_title("SimplePGI")
            # With W
            axfull[1].contourf(x, y, rvm.reshape(x.shape), alpha=0.25, cmap="brg")
            axfull[1].contour(x, y, rv.reshape(x.shape), 20)
            axfull[1].scatter(
                self.s0[:, 0], self.s0[:, 1], color="blue", s=5.0, alpha=0.25
            )
            axfull[1].scatter(
                self.s1[:, 0], self.s1[:, 1], color="green", s=5.0, alpha=0.25
            )
            axfull[1].quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -(self.wires.s0 * deriv),
                -(self.wires.s1 * deriv),
                color="red",
                alpha=0.25,
            )
            axfull[1].quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -direction2[:, 0],
                -direction2[:, 1],
                color="k",
            )
            axfull[1].scatter(
                (self.samples - direction2)[:, 0],
                (self.samples - direction2)[:, 1],
                color="k",
                s=50.0,
            )
            axfull[1].set_xlabel("Property 1")
            axfull[1].set_ylabel("Property 2")
            axfull[1].set_title("PGI with W")

            plt.show()

    def test_tied_covariances(self):

        print("Test Tied covariances: ")
        # Fit a Gaussian Mixture
        clf = WeightedGaussianMixture(
            mesh=self.mesh,
            n_components=self.n_components,
            covariance_type="tied",
            max_iter=1000,
            n_init=100,
            tol=1e-8,
            means_init=self.means,
            warm_start=True,
            weights_init=self.proportions,
        )
        clf.fit(self.samples)

        # Define reg Simple
        reg_simple = make_SimplePGI_regularization(
            mesh=self.mesh,
            gmmref=clf,
            approx_gradient=True,
            alpha_x=0.0,
            wiresmap=self.wires,
            cell_weights_list=self.cell_weights_list,
        )
        # Define reg with volumes
        reg = make_PGI_regularization(
            mesh=self.mesh,
            gmmref=clf,
            approx_gradient=True,
            alpha_x=0.0,
            wiresmap=self.wires,
            cell_weights_list=self.cell_weights_list,
        )
        
        mref = mkvc(clf.means_[clf.predict(self.samples)])
        
        # check score value
        score_approx0 = reg_simple(self.model)
        dm = (self.model - mref)
        score_approx1 = 0.5 * dm.dot(reg_simple.deriv2(self.model, dm))
        passed_score_approx_simple = (score_approx0 == score_approx1)
        self.assertTrue(passed_score_approx_simple)
        reg_simple.objfcts[0].approx_eval = False
        score = reg_simple(self.model) - reg_simple(mref)
        passed_score_simple = np.allclose(score_approx0, score, rtol=1e-1)
        print("scores:", score_approx0,score_approx1, score)
        self.assertTrue(passed_score_simple)
        print("scores for SimplePGI & tied Cov. are ok.")

        score_approx0 = reg(self.model)
        score_approx1 = 0.5 * dm.dot(reg.deriv2(self.model,dm))
        passed_score_approx = np.allclose(score_approx0, score_approx1)
        self.assertTrue(passed_score_approx)
        reg.objfcts[0].approx_eval = False
        score = reg(self.model) - reg(mref)
        passed_score = np.allclose(score_approx0, score, rtol=1e-1)
        print("scores:", score_approx0,score_approx1, score)
        self.assertTrue(passed_score)
        print("scores for PGI & tied Cov. are ok.")

        # check derivatives as an optimization on locally quadratic function
        # Simple

        deriv_simple = reg_simple.deriv(self.model)
        reg_simple.objfcts[0].approx_gradient = False
        deriv_simple_full = reg_simple.deriv(self.model)
        passed_deriv1 = np.allclose(deriv_simple, deriv_simple_full, rtol=1e-1)
        self.assertTrue(passed_deriv1)
        print("1st derivatives for SimplePGI & tied Cov. are ok.")
        deriv_simple = reg_simple.deriv(self.model)
        Hinv = PardisoSolver(reg_simple.deriv2(self.model))
        p_simple = Hinv * deriv_simple
        direction2_simple = np.c_[self.wires * p_simple]
        passed_derivative_simple = np.allclose(
            mkvc(self.samples - direction2_simple), mkvc(mref), rtol=1e-1
        )
        self.assertTrue(passed_derivative_simple)
        print("2nd derivatives for SimplePGI & tied Cov. are ok.")

        # With volumes
        deriv = reg.deriv(self.model)
        reg.objfcts[0].approx_gradient = False
        deriv_full = reg.deriv(self.model)
        passed_deriv1 = np.allclose(deriv, deriv_full, rtol=1e-1)
        self.assertTrue(passed_deriv1)
        print("1st derivatives for PGI & tied Cov. are ok.")

        Hinv = PardisoSolver(reg.deriv2(self.model))
        p = Hinv * deriv
        direction2 = np.c_[self.wires * p]
        passed_derivative = np.allclose(
            mkvc(self.samples - direction2), mkvc(mref), rtol=1e-1
        )
        self.assertTrue(passed_derivative)
        print("2nd derivatives for PGI & tied Cov. are ok.")

        if self.PlotIt:
            import matplotlib.pyplot as plt

            xmin, xmax = ymin, ymax = self.samples.min(), self.samples.max()
            x, y = np.mgrid[xmin:xmax:0.01, ymin:ymax:0.01]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y
            rv = clf.score_samples(pos.reshape(-1, 2))
            rvm = clf.predict(pos.reshape(-1, 2))
            figtied, axtied = plt.subplots(1, 2, figsize=(16, 8))
            figtied.suptitle("Tied Covariances Tests")
            # Simple
            axtied[0].contourf(x, y, rvm.reshape(x.shape), alpha=0.25, cmap="brg")
            axtied[0].contour(x, y, rv.reshape(x.shape), 20)
            axtied[0].scatter(
                self.s0[:, 0], self.s0[:, 1], color="blue", s=5.0, alpha=0.25
            )
            axtied[0].scatter(
                self.s1[:, 0], self.s1[:, 1], color="green", s=5.0, alpha=0.25
            )
            axtied[0].quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -(self.wires.s0 * deriv_simple),
                -(self.wires.s1 * deriv_simple),
                color="red",
                alpha=0.25,
            )
            axtied[0].quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -direction2_simple[:, 0],
                -direction2_simple[:, 1],
                color="k",
            )
            axtied[0].scatter(
                (self.samples - direction2_simple)[:, 0],
                (self.samples - direction2_simple)[:, 1],
                color="k",
                s=50.0,
            )
            axtied[0].set_xlabel("Property 1")
            axtied[0].set_ylabel("Property 2")
            axtied[0].set_title("SimplePGI")
            # With W
            axtied[1].contourf(x, y, rvm.reshape(x.shape), alpha=0.25, cmap="brg")
            axtied[1].contour(x, y, rv.reshape(x.shape), 20)
            axtied[1].scatter(
                self.s0[:, 0], self.s0[:, 1], color="blue", s=5.0, alpha=0.25
            )
            axtied[1].scatter(
                self.s1[:, 0], self.s1[:, 1], color="green", s=5.0, alpha=0.25
            )
            axtied[1].quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -(self.wires.s0 * deriv),
                -(self.wires.s1 * deriv),
                color="red",
                alpha=0.25,
            )
            axtied[1].quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -direction2[:, 0],
                -direction2[:, 1],
                color="k",
            )
            axtied[1].scatter(
                (self.samples - direction2)[:, 0],
                (self.samples - direction2)[:, 1],
                color="k",
                s=50.0,
            )
            axtied[1].set_xlabel("Property 1")
            axtied[1].set_ylabel("Property 2")
            axtied[1].set_title("PGI with W")

            plt.show()

    def test_diag_covariances(self):

        print("Test Diagonal covariances: ")
        # Fit a Gaussian Mixture
        clf = WeightedGaussianMixture(
            mesh=self.mesh,
            n_components=self.n_components,
            covariance_type="diag",
            max_iter=1000,
            n_init=100,
            tol=1e-8,
            means_init=self.means,
            warm_start=True,
            weights_init=self.proportions,
        )
        clf.fit(self.samples)

        # Define reg Simple
        reg_simple = make_SimplePGI_regularization(
            mesh=self.mesh,
            gmmref=clf,
            approx_gradient=True,
            alpha_x=0.0,
            wiresmap=self.wires,
            cell_weights_list=self.cell_weights_list,
        )
        # Define reg with volumes
        reg = make_PGI_regularization(
            mesh=self.mesh,
            gmmref=clf,
            approx_gradient=True,
            alpha_x=0.0,
            wiresmap=self.wires,
            cell_weights_list=self.cell_weights_list,
        )
        
        mref = mkvc(clf.means_[clf.predict(self.samples)])
        
        # check score value
        score_approx0 = reg_simple(self.model)
        dm = (self.model - mref)
        score_approx1 = 0.5 * dm.dot(reg_simple.deriv2(self.model, dm))
        passed_score_approx_simple = (score_approx0 == score_approx1)
        self.assertTrue(passed_score_approx_simple)
        reg_simple.objfcts[0].approx_eval = False
        score = reg_simple(self.model) - reg_simple(mref)
        passed_score_simple = np.allclose(score_approx0, score, rtol=1e-1)
        print("scores:", score_approx0,score_approx1, score)
        self.assertTrue(passed_score_simple)
        print("scores for SimplePGI & diag Cov. are ok.")

        score_approx0 = reg(self.model)
        score_approx1 = 0.5 * dm.dot(reg.deriv2(self.model,dm))
        passed_score_approx = np.allclose(score_approx0, score_approx1)
        self.assertTrue(passed_score_approx)
        reg.objfcts[0].approx_eval = False
        score = reg(self.model) - reg(mref)
        passed_score = np.allclose(score_approx0, score, rtol=1e-1)
        print("scores:", score_approx0,score_approx1, score)
        self.assertTrue(passed_score)
        print("scores for PGI & diag Cov. are ok.")

        # check derivatives as an optimization on locally quadratic function
        # Simple

        deriv_simple = reg_simple.deriv(self.model)
        reg_simple.objfcts[0].approx_gradient = False
        deriv_simple_full = reg_simple.deriv(self.model)
        passed_deriv1 = np.allclose(deriv_simple, deriv_simple_full, rtol=1e-1)
        self.assertTrue(passed_deriv1)
        print("1st derivatives for SimplePGI & diag Cov. are ok.")
        deriv_simple = reg_simple.deriv(self.model)
        Hinv = PardisoSolver(reg_simple.deriv2(self.model))
        p_simple = Hinv * deriv_simple
        direction2_simple = np.c_[self.wires * p_simple]
        passed_derivative_simple = np.allclose(
            mkvc(self.samples - direction2_simple), mkvc(mref), rtol=1e-1
        )
        self.assertTrue(passed_derivative_simple)
        print("2nd derivatives for SimplePGI & diag Cov. are ok.")

        # With volumes
        deriv = reg.deriv(self.model)
        reg.objfcts[0].approx_gradient = False
        deriv_full = reg.deriv(self.model)
        passed_deriv1 = np.allclose(deriv, deriv_full, rtol=1e-1)
        self.assertTrue(passed_deriv1)
        print("1st derivatives for PGI & diag Cov. are ok.")
        Hinv = PardisoSolver(reg.deriv2(self.model))
        p = Hinv * deriv
        direction2 = np.c_[self.wires * p]
        passed_derivative = np.allclose(
            mkvc(self.samples - direction2), mkvc(mref), rtol=1e-1
        )
        self.assertTrue(passed_derivative)
        print("2nd derivatives for PGI & diag Cov. are ok.")

        if self.PlotIt:
            import matplotlib.pyplot as plt

            xmin, xmax = ymin, ymax = self.samples.min(), self.samples.max()
            x, y = np.mgrid[xmin:xmax:0.01, ymin:ymax:0.01]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y
            rv = clf.score_samples(pos.reshape(-1, 2))
            rvm = clf.predict(pos.reshape(-1, 2))
            figdiag, axdiag = plt.subplots(1, 2, figsize=(16, 8))
            figdiag.suptitle("Diag Covariances Tests")
            # Simple
            axdiag[0].contourf(x, y, rvm.reshape(x.shape), alpha=0.25, cmap="brg")
            axdiag[0].contour(x, y, rv.reshape(x.shape), 20)
            axdiag[0].scatter(
                self.s0[:, 0], self.s0[:, 1], color="blue", s=5.0, alpha=0.25
            )
            axdiag[0].scatter(
                self.s1[:, 0], self.s1[:, 1], color="green", s=5.0, alpha=0.25
            )
            axdiag[0].quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -(self.wires.s0 * deriv_simple),
                -(self.wires.s1 * deriv_simple),
                color="red",
                alpha=0.25,
            )
            axdiag[0].quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -direction2_simple[:, 0],
                -direction2_simple[:, 1],
                color="k",
            )
            axdiag[0].scatter(
                (self.samples - direction2_simple)[:, 0],
                (self.samples - direction2_simple)[:, 1],
                color="k",
                s=50.0,
            )
            axdiag[0].set_xlabel("Property 1")
            axdiag[0].set_ylabel("Property 2")
            axdiag[0].set_title("SimplePGI")
            # With W
            axdiag[1].contourf(x, y, rvm.reshape(x.shape), alpha=0.25, cmap="brg")
            axdiag[1].contour(x, y, rv.reshape(x.shape), 20)
            axdiag[1].scatter(
                self.s0[:, 0], self.s0[:, 1], color="blue", s=5.0, alpha=0.25
            )
            axdiag[1].scatter(
                self.s1[:, 0], self.s1[:, 1], color="green", s=5.0, alpha=0.25
            )
            axdiag[1].quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -(self.wires.s0 * deriv),
                -(self.wires.s1 * deriv),
                color="red",
                alpha=0.25,
            )
            axdiag[1].quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -direction2[:, 0],
                -direction2[:, 1],
                color="k",
            )
            axdiag[1].scatter(
                (self.samples - direction2)[:, 0],
                (self.samples - direction2)[:, 1],
                color="k",
                s=50.0,
            )
            axdiag[1].set_xlabel("Property 1")
            axdiag[1].set_ylabel("Property 2")
            axdiag[1].set_title("PGI with W")

            plt.show()

    def test_spherical_covariances(self):

        print("Test Spherical covariances: ")
        # Fit a Gaussian Mixture
        clf = WeightedGaussianMixture(
            mesh=self.mesh,
            n_components=self.n_components,
            covariance_type="spherical",
            max_iter=1000,
            n_init=100,
            tol=1e-8,
            means_init=self.means,
            warm_start=True,
            weights_init=self.proportions,
        )
        clf.fit(self.samples)

        # Define reg Simple
        reg_simple = make_SimplePGI_regularization(
            mesh=self.mesh,
            gmmref=clf,
            approx_gradient=True,
            alpha_x=0.0,
            wiresmap=self.wires,
            cell_weights_list=self.cell_weights_list,
        )
        # Define reg with volumes
        reg = make_PGI_regularization(
            mesh=self.mesh,
            gmmref=clf,
            approx_gradient=True,
            alpha_x=0.0,
            wiresmap=self.wires,
            cell_weights_list=self.cell_weights_list,
        )
        
        mref = mkvc(clf.means_[clf.predict(self.samples)])
        
        # check score value
        score_approx0 = reg_simple(self.model)
        dm = self.model - mref
        score_approx1 = 0.5 * dm.dot(reg_simple.deriv2(self.model, dm))
        passed_score_approx_simple = (score_approx0 == score_approx1)
        self.assertTrue(passed_score_approx_simple)
        reg_simple.objfcts[0].approx_eval = False
        score = reg_simple(self.model) - reg_simple(mref)
        passed_score_simple = np.allclose(score_approx0, score, rtol=1e-1)
        print("scores:", score_approx0,score_approx1, score)
        self.assertTrue(passed_score_simple)
        print("scores for SimplePGI & spherical Cov. are ok.")

        score_approx0 = reg(self.model)
        score_approx1 = 0.5 * dm.dot(reg.deriv2(self.model,dm))
        passed_score_approx = np.allclose(score_approx0, score_approx1)
        self.assertTrue(passed_score_approx)
        reg.objfcts[0].approx_eval = False
        score = reg(self.model) - reg(mref)
        passed_score = np.allclose(score_approx0, score, rtol=1e-1)
        print("scores:", score_approx0,score_approx1, score)
        self.assertTrue(passed_score)
        print("scores for PGI & spherical Cov. are ok.")

        # check derivatives as an optimization on locally quadratic function
        # Simple

        deriv_simple = reg_simple.deriv(self.model)
        reg_simple.objfcts[0].approx_gradient = False
        deriv_simple_full = reg_simple.deriv(self.model)
        passed_deriv1 = np.allclose(deriv_simple, deriv_simple_full, rtol=1e-1)
        self.assertTrue(passed_deriv1)
        print("1st derivatives for SimplePGI & spherical Cov. are ok.")
        deriv_simple = reg_simple.deriv(self.model)
        Hinv = PardisoSolver(reg_simple.deriv2(self.model))
        p_simple = Hinv * deriv_simple
        direction2_simple = np.c_[self.wires * p_simple]
        passed_derivative_simple = np.allclose(
            mkvc(self.samples - direction2_simple), mkvc(mref), rtol=1e-1
        )
        self.assertTrue(passed_derivative_simple)
        print("2nd derivatives for SimplePGI & spherical Cov. are ok.")

        # With volumes
        deriv = reg.deriv(self.model)
        reg.objfcts[0].approx_gradient = False
        deriv_full = reg.deriv(self.model)
        passed_deriv1 = np.allclose(deriv, deriv_full, rtol=1e-1)
        self.assertTrue(passed_deriv1)
        print("1st derivatives for PGI & spherical Cov. are ok.")
  
        Hinv = PardisoSolver(reg.deriv2(self.model))
        p = Hinv * deriv
        direction2 = np.c_[self.wires * p]
        passed_derivative = np.allclose(
            mkvc(self.samples - direction2), mkvc(mref), rtol=1e-1
        )
        self.assertTrue(passed_derivative)
        print("2nd derivatives for PGI & spherical Cov. are ok.")

        if self.PlotIt:
            import matplotlib.pyplot as plt

            xmin, xmax = ymin, ymax = self.samples.min(), self.samples.max()
            x, y = np.mgrid[xmin:xmax:0.01, ymin:ymax:0.01]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y
            rv = clf.score_samples(pos.reshape(-1, 2))
            rvm = clf.predict(pos.reshape(-1, 2))
            figspherical, axspherical = plt.subplots(1, 2, figsize=(16, 8))
            figspherical.suptitle("Spherical Covariances Tests")
            # Simple
            axspherical[0].contourf(x, y, rvm.reshape(x.shape), alpha=0.25, cmap="brg")
            axspherical[0].contour(x, y, rv.reshape(x.shape), 20)
            axspherical[0].scatter(
                self.s0[:, 0], self.s0[:, 1], color="blue", s=5.0, alpha=0.25
            )
            axspherical[0].scatter(
                self.s1[:, 0], self.s1[:, 1], color="green", s=5.0, alpha=0.25
            )
            axspherical[0].quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -(self.wires.s0 * deriv_simple),
                -(self.wires.s1 * deriv_simple),
                color="red",
                alpha=0.25,
            )
            axspherical[0].quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -direction2_simple[:, 0],
                -direction2_simple[:, 1],
                color="k",
            )
            axspherical[0].scatter(
                (self.samples - direction2_simple)[:, 0],
                (self.samples - direction2_simple)[:, 1],
                color="k",
                s=50.0,
            )
            axspherical[0].set_xlabel("Property 1")
            axspherical[0].set_ylabel("Property 2")
            axspherical[0].set_title("SimplePGI")
            # With W
            axspherical[1].contourf(x, y, rvm.reshape(x.shape), alpha=0.25, cmap="brg")
            axspherical[1].contour(x, y, rv.reshape(x.shape), 20)
            axspherical[1].scatter(
                self.s0[:, 0], self.s0[:, 1], color="blue", s=5.0, alpha=0.25
            )
            axspherical[1].scatter(
                self.s1[:, 0], self.s1[:, 1], color="green", s=5.0, alpha=0.25
            )
            axspherical[1].quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -(self.wires.s0 * deriv),
                -(self.wires.s1 * deriv),
                color="red",
                alpha=0.25,
            )
            axspherical[1].quiver(
                self.samples[:, 0],
                self.samples[:, 1],
                -direction2[:, 0],
                -direction2[:, 1],
                color="k",
            )
            axspherical[1].scatter(
                (self.samples - direction2)[:, 0],
                (self.samples - direction2)[:, 1],
                color="k",
                s=50.0,
            )
            axspherical[1].set_xlabel("Property 1")
            axspherical[1].set_ylabel("Property 2")
            axspherical[1].set_title("PGI with W")

            plt.show()


if __name__ == "__main__":
    unittest.main()
