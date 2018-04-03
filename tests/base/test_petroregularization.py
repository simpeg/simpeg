from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import unittest
from SimPEG import Mesh, Maps, Regularization, Utils, Tests, ObjectiveFunction
from scipy.stats import multivariate_normal
from scipy.sparse.linalg import dsolve, spsolve, LinearOperator, bicgstab
import inspect


class TestPetroRegularization(unittest.TestCase):

    def setUp(self):

        np.random.seed(518936)

        # Create a cloud of  random points from a random gaussian mixture
        self.ndim = 2
        self.n_components = 2
        sigma = np.random.randn(self.n_components, self.ndim, self.ndim)
        sigma = np.c_[[sigma[i].dot(sigma[i].T)
                       for i in range(sigma.shape[0])]]
        sigma[0] += np.eye(self.ndim)
        sigma[1] += np.eye(self.ndim) - 0.25 * \
            np.eye(self.ndim).transpose((1, 0))
        self.sigma = sigma
        self.means = np.abs(np.random.randn(
            self.ndim, self.ndim)) * np.c_[[5., -5.]]
        self.rv0 = multivariate_normal(self.means[0], self.sigma[0])
        self.rv1 = multivariate_normal(self.means[1], self.sigma[1])
        self.proportions = np.r_[0.6, 0.4]
        self.nsample = 1000.
        self.s0 = self.rv0.rvs(int(self.nsample * self.proportions[0]))
        self.s1 = self.rv1.rvs(int(self.nsample * self.proportions[1]))
        self.samples = np.r_[self.s0, self.s1]
        self.reference = np.r_[np.ones_like(
            self.s0) * self.means[0], np.ones_like(self.s1) * self.means[1]]
        self.mesh = Mesh.TensorMesh([self.samples.shape[0]])
        self.wires = Maps.Wires(('s0', self.mesh.nC), ('s1', self.mesh.nC))

        self.PlotIt = True

    def test_full_covariances(self):
        # Fit a Gaussian Mixture
        clf = Utils.GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            max_iter=1000, n_init=10,
            means_init=self.means,
            warm_start=True,
            precisions_init=np.linalg.inv(self.sigma),
            weights_init=self.proportions)
        clf.fit(self.samples)

        # Define reg Simple
        reg_simple = Regularization.SimplePetroRegularization(
            mesh=self.mesh,
            GMmref=clf,
            approx_gradient=True, alpha_x=0.,
            wiresmap=self.wires,
            evaltype='approx'
        )
        # Define reg with volumes
        reg = Regularization.PetroRegularization(
            mesh=self.mesh,
            GMmref=clf,
            approx_gradient=True, alpha_x=0.,
            wiresmap=self.wires,
            evaltype='approx'
        )

        # check score value
        score_approx = reg_simple(Utils.mkvc(self.samples))
        reg_simple.objfcts[0].evaltype = 'full'
        score = reg_simple(Utils.mkvc(self.samples))
        passed_score_simple = np.allclose(score_approx, score, rtol=1e-1)
        self.assertTrue(passed_score_simple)
        print('scores for SimplePetro are ok')

        score_approx = reg(Utils.mkvc(self.samples))
        reg.objfcts[0].evaltype = 'full'
        score = reg(Utils.mkvc(self.samples))
        passed_score = np.allclose(score_approx, score, rtol=1e-1)
        self.assertTrue(passed_score)
        print('scores for Petro are ok')

        # check derivatives as an optimization on locally quadratic function
        # Simple

        reference = clf.means_[clf.predict(self.samples)]

        deriv_simple = reg_simple.deriv(Utils.mkvc(self.samples))
        reg.approx_gradient = False
        deriv_simple_full = reg_simple.deriv(Utils.mkvc(self.samples))
        passed_deriv1 = np.allclose(deriv_simple, deriv_simple_full, rtol=1e-1)
        self.assertTrue(passed_deriv1)
        print('1st derivatives for Simple  ro are ok')
        deriv_simple = reg_simple.deriv(Utils.mkvc(self.samples))
        Hessian_simple = lambda x: reg_simple.deriv2(
            Utils.mkvc(self.samples), x)
        HV_simple = LinearOperator(
            [len(self.samples) * self.ndim, len(self.samples) * self.ndim],
            matvec=Hessian_simple,
            rmatvec=Hessian_simple
        )
        p_simple = bicgstab(HV_simple, deriv_simple)
        direction2_simple = np.c_[self.wires * p_simple[0]]
        passed_derivative_simple = np.allclose(
            Utils.mkvc(self.samples - direction2_simple),
            Utils.mkvc(reference), rtol=1e-1
        )
        self.assertTrue(passed_derivative_simple)
        print('derivatives for SimplePetro are ok')

        # With volumes
        deriv = reg.deriv(Utils.mkvc(self.samples))
        Hessian = lambda x: reg.deriv2(Utils.mkvc(self.samples), x)
        HV = LinearOperator(
            [len(self.samples) * self.ndim, len(self.samples) * self.ndim],
            matvec=Hessian,
            rmatvec=Hessian
        )
        p = bicgstab(HV, deriv)
        direction2 = np.c_[self.wires * p[0]]
        passed_derivative = np.allclose(
            Utils.mkvc(self.samples - direction2),
            Utils.mkvc(reference), rtol=1e-1
        )
        self.assertTrue(passed_derivative)
        print('derivatives for Petro are ok')

        if self.PlotIt:
            import matplotlib.pyplot as plt
            xmin, xmax = ymin, ymax = self.samples.min(), self.samples.max()
            x, y = np.mgrid[xmin:xmax:.01, ymin:ymax:.01]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y
            rv = clf.score_samples(pos.reshape(-1, 2))
            rvm = clf.predict(pos.reshape(-1, 2))
            figfull, axfull = plt.subplots(1, 2, figsize=(16, 8))
            figfull.suptitle('Full Covariances Tests')
            # Simple
            axfull[0].contourf(x, y, rvm.reshape(
                x.shape), alpha=0.25, cmap='brg')
            axfull[0].contour(x, y, rv.reshape(x.shape), 20)
            axfull[0].scatter(
                self.s0[:, 0], self.s0[:, 1],
                color='blue', s=5., alpha=0.25
            )
            axfull[0].scatter(
                self.s1[:, 0], self.s1[:, 1],
                color='green', s=5., alpha=0.25
            )
            axfull[0].quiver(
                self.samples[:, 0], self.samples[:, 1], -
                (self.wires.s0 * deriv_simple),
                -(self.wires.s1 * deriv_simple), color='red', alpha=0.25
            )
            axfull[0].quiver(
                self.samples[:, 0], self.samples[:, 1],
                -direction2_simple[:, 0], -direction2_simple[:, 1],
                color='k'
            )
            axfull[0].scatter(
                (self.samples - direction2_simple)[:, 0],
                (self.samples - direction2_simple)[:, 1],
                color='k',
                s=50.)
            axfull[0].set_xlabel('Property 1')
            axfull[0].set_ylabel('Property 2')
            axfull[0].set_title('SimplePetro')
            # With W
            axfull[1].contourf(x, y, rvm.reshape(
                x.shape), alpha=0.25, cmap='brg')
            axfull[1].contour(x, y, rv.reshape(x.shape), 20)
            axfull[1].scatter(
                self.s0[:, 0], self.s0[:, 1],
                color='blue', s=5., alpha=0.25
            )
            axfull[1].scatter(
                self.s1[:, 0], self.s1[:, 1],
                color='green', s=5., alpha=0.25
            )
            axfull[1].quiver(
                self.samples[:, 0], self.samples[
                    :, 1], -(self.wires.s0 * deriv),
                -(self.wires.s1 * deriv), color='red', alpha=0.25
            )
            axfull[1].quiver(
                self.samples[:, 0], self.samples[:, 1],
                -direction2[:, 0], -direction2[:, 1],
                color='k'
            )
            axfull[1].scatter(
                (self.samples - direction2)[:, 0],
                (self.samples - direction2)[:, 1],
                color='k',
                s=50.)
            axfull[1].set_xlabel('Property 1')
            axfull[1].set_ylabel('Property 2')
            axfull[1].set_title('Petro with W')

            plt.show()

    def test_tied_covariances(self):
        # Fit a Gaussian Mixture
        clf = Utils.GaussianMixture(
            n_components=self.n_components,
            covariance_type='tied',
            max_iter=1000, n_init=10,
            means_init=self.means,
            warm_start=True,
            weights_init=self.proportions)
        clf.fit(self.samples)

        # Define reg Simple
        reg_simple = Regularization.SimplePetroRegularization(
            mesh=self.mesh,
            GMmref=clf,
            approx_gradient=True, alpha_x=0.,
            wiresmap=self.wires,
            evaltype='approx'
        )
        # Define reg with volumes
        reg = Regularization.PetroRegularization(
            mesh=self.mesh,
            GMmref=clf,
            approx_gradient=True, alpha_x=0.,
            wiresmap=self.wires,
            evaltype='approx'
        )

        # check score value
        score_approx = reg_simple(Utils.mkvc(self.samples))
        reg_simple.objfcts[0].evaltype = 'full'
        score = reg_simple(Utils.mkvc(self.samples))
        passed_score_simple = np.allclose(score_approx, score, rtol=1e-1)
        self.assertTrue(passed_score_simple)
        print('scores for SimplePetro are ok')

        score_approx = reg(Utils.mkvc(self.samples))
        reg.objfcts[0].evaltype = 'full'
        score = reg(Utils.mkvc(self.samples))
        passed_score = np.allclose(score_approx, score, rtol=1e-1)
        self.assertTrue(passed_score)
        print('scores for Petro are ok')

        # check derivatives as an optimization on locally quadratic function
        # Simple

        reference = clf.means_[clf.predict(self.samples)]

        deriv_simple = reg_simple.deriv(Utils.mkvc(self.samples))
        Hessian_simple = lambda x: reg_simple.deriv2(
            Utils.mkvc(self.samples), x)
        HV_simple = LinearOperator(
            [len(self.samples) * self.ndim, len(self.samples) * self.ndim],
            matvec=Hessian_simple,
            rmatvec=Hessian_simple
        )
        p_simple = bicgstab(HV_simple, deriv_simple)
        direction2_simple = np.c_[self.wires * p_simple[0]]
        passed_derivative_simple = np.allclose(
            Utils.mkvc(self.samples - direction2_simple),
            Utils.mkvc(reference), rtol=1e-1
        )
        self.assertTrue(passed_derivative_simple)
        print('derivatives for SimplePetro are ok')

        # With volumes
        deriv = reg.deriv(Utils.mkvc(self.samples))
        Hessian = lambda x: reg.deriv2(Utils.mkvc(self.samples), x)
        HV = LinearOperator(
            [len(self.samples) * self.ndim, len(self.samples) * self.ndim],
            matvec=Hessian,
            rmatvec=Hessian
        )
        p = bicgstab(HV, deriv)
        direction2 = np.c_[self.wires * p[0]]
        passed_derivative = np.allclose(
            Utils.mkvc(self.samples - direction2),
            Utils.mkvc(reference), rtol=1e-1
        )
        self.assertTrue(passed_derivative)
        print('derivatives for Petro are ok')

        if self.PlotIt:
            import matplotlib.pyplot as plt
            xmin, xmax = ymin, ymax = self.samples.min(), self.samples.max()
            x, y = np.mgrid[xmin:xmax:.01, ymin:ymax:.01]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y
            rv = clf.score_samples(pos.reshape(-1, 2))
            rvm = clf.predict(pos.reshape(-1, 2))
            figtied, axtied = plt.subplots(1, 2, figsize=(16, 8))
            figtied.suptitle('Tied Covariances Tests')
            # Simple
            axtied[0].contourf(x, y, rvm.reshape(
                x.shape), alpha=0.25, cmap='brg')
            axtied[0].contour(x, y, rv.reshape(x.shape), 20)
            axtied[0].scatter(
                self.s0[:, 0], self.s0[:, 1],
                color='blue', s=5., alpha=0.25
            )
            axtied[0].scatter(
                self.s1[:, 0], self.s1[:, 1],
                color='green', s=5., alpha=0.25
            )
            axtied[0].quiver(
                self.samples[:, 0], self.samples[:, 1], -
                (self.wires.s0 * deriv_simple),
                -(self.wires.s1 * deriv_simple), color='red', alpha=0.25
            )
            axtied[0].quiver(
                self.samples[:, 0], self.samples[:, 1],
                -direction2_simple[:, 0], -direction2_simple[:, 1],
                color='k'
            )
            axtied[0].scatter(
                (self.samples - direction2_simple)[:, 0],
                (self.samples - direction2_simple)[:, 1],
                color='k',
                s=50.)
            axtied[0].set_xlabel('Property 1')
            axtied[0].set_ylabel('Property 2')
            axtied[0].set_title('SimplePetro')
            # With W
            axtied[1].contourf(x, y, rvm.reshape(
                x.shape), alpha=0.25, cmap='brg')
            axtied[1].contour(x, y, rv.reshape(x.shape), 20)
            axtied[1].scatter(
                self.s0[:, 0], self.s0[:, 1],
                color='blue', s=5., alpha=0.25
            )
            axtied[1].scatter(
                self.s1[:, 0], self.s1[:, 1],
                color='green', s=5., alpha=0.25
            )
            axtied[1].quiver(
                self.samples[:, 0], self.samples[
                    :, 1], -(self.wires.s0 * deriv),
                -(self.wires.s1 * deriv), color='red', alpha=0.25
            )
            axtied[1].quiver(
                self.samples[:, 0], self.samples[:, 1],
                -direction2[:, 0], -direction2[:, 1],
                color='k'
            )
            axtied[1].scatter(
                (self.samples - direction2)[:, 0],
                (self.samples - direction2)[:, 1],
                color='k',
                s=50.)
            axtied[1].set_xlabel('Property 1')
            axtied[1].set_ylabel('Property 2')
            axtied[1].set_title('Petro with W')

            plt.show()

    def test_diag_covariances(self):
        # Fit a Gaussian Mixture
        clf = Utils.GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',
            max_iter=1000, n_init=10,
            means_init=self.means,
            warm_start=True,
            weights_init=self.proportions)
        clf.fit(self.samples)

        # Define reg Simple
        reg_simple = Regularization.SimplePetroRegularization(
            mesh=self.mesh,
            GMmref=clf,
            approx_gradient=True, alpha_x=0.,
            wiresmap=self.wires,
            evaltype='approx'
        )
        # Define reg with volumes
        reg = Regularization.PetroRegularization(
            mesh=self.mesh,
            GMmref=clf,
            approx_gradient=True, alpha_x=0.,
            wiresmap=self.wires,
            evaltype='approx'
        )

        # check score value
        score_approx = reg_simple(Utils.mkvc(self.samples))
        reg_simple.objfcts[0].evaltype = 'full'
        score = reg_simple(Utils.mkvc(self.samples))
        passed_score_simple = np.allclose(score_approx, score, rtol=1e-1)
        self.assertTrue(passed_score_simple)
        print('scores for SimplePetro are ok')

        score_approx = reg(Utils.mkvc(self.samples))
        reg.objfcts[0].evaltype = 'full'
        score = reg(Utils.mkvc(self.samples))
        passed_score = np.allclose(score_approx, score, rtol=1e-1)
        self.assertTrue(passed_score)
        print('scores for Petro are ok')

        # check derivatives as an optimization on locally quadratic function
        # Simple

        reference = clf.means_[clf.predict(self.samples)]

        deriv_simple = reg_simple.deriv(Utils.mkvc(self.samples))
        Hessian_simple = lambda x: reg_simple.deriv2(
            Utils.mkvc(self.samples), x)
        HV_simple = LinearOperator(
            [len(self.samples) * self.ndim, len(self.samples) * self.ndim],
            matvec=Hessian_simple,
            rmatvec=Hessian_simple
        )
        p_simple = bicgstab(HV_simple, deriv_simple)
        direction2_simple = np.c_[self.wires * p_simple[0]]
        passed_derivative_simple = np.allclose(
            Utils.mkvc(self.samples - direction2_simple),
            Utils.mkvc(reference), rtol=1e-1
        )
        self.assertTrue(passed_derivative_simple)
        print('derivatives for SimplePetro are ok')

        # With volumes
        deriv = reg.deriv(Utils.mkvc(self.samples))
        Hessian = lambda x: reg.deriv2(Utils.mkvc(self.samples), x)
        HV = LinearOperator(
            [len(self.samples) * self.ndim, len(self.samples) * self.ndim],
            matvec=Hessian,
            rmatvec=Hessian
        )
        p = bicgstab(HV, deriv)
        direction2 = np.c_[self.wires * p[0]]
        passed_derivative = np.allclose(
            Utils.mkvc(self.samples - direction2),
            Utils.mkvc(reference), rtol=1e-1
        )
        self.assertTrue(passed_derivative)
        print('derivatives for Petro are ok')

        if self.PlotIt:
            import matplotlib.pyplot as plt
            xmin, xmax = ymin, ymax = self.samples.min(), self.samples.max()
            x, y = np.mgrid[xmin:xmax:.01, ymin:ymax:.01]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y
            rv = clf.score_samples(pos.reshape(-1, 2))
            rvm = clf.predict(pos.reshape(-1, 2))
            figdiag, axdiag = plt.subplots(1, 2, figsize=(16, 8))
            figdiag.suptitle('Diag Covariances Tests')
            # Simple
            axdiag[0].contourf(x, y, rvm.reshape(
                x.shape), alpha=0.25, cmap='brg')
            axdiag[0].contour(x, y, rv.reshape(x.shape), 20)
            axdiag[0].scatter(
                self.s0[:, 0], self.s0[:, 1],
                color='blue', s=5., alpha=0.25
            )
            axdiag[0].scatter(
                self.s1[:, 0], self.s1[:, 1],
                color='green', s=5., alpha=0.25
            )
            axdiag[0].quiver(
                self.samples[:, 0], self.samples[:, 1], -
                (self.wires.s0 * deriv_simple),
                -(self.wires.s1 * deriv_simple), color='red', alpha=0.25
            )
            axdiag[0].quiver(
                self.samples[:, 0], self.samples[:, 1],
                -direction2_simple[:, 0], -direction2_simple[:, 1],
                color='k'
            )
            axdiag[0].scatter(
                (self.samples - direction2_simple)[:, 0],
                (self.samples - direction2_simple)[:, 1],
                color='k',
                s=50.)
            axdiag[0].set_xlabel('Property 1')
            axdiag[0].set_ylabel('Property 2')
            axdiag[0].set_title('SimplePetro')
            # With W
            axdiag[1].contourf(x, y, rvm.reshape(
                x.shape), alpha=0.25, cmap='brg')
            axdiag[1].contour(x, y, rv.reshape(x.shape), 20)
            axdiag[1].scatter(
                self.s0[:, 0], self.s0[:, 1],
                color='blue', s=5., alpha=0.25
            )
            axdiag[1].scatter(
                self.s1[:, 0], self.s1[:, 1],
                color='green', s=5., alpha=0.25
            )
            axdiag[1].quiver(
                self.samples[:, 0], self.samples[
                    :, 1], -(self.wires.s0 * deriv),
                -(self.wires.s1 * deriv), color='red', alpha=0.25
            )
            axdiag[1].quiver(
                self.samples[:, 0], self.samples[:, 1],
                -direction2[:, 0], -direction2[:, 1],
                color='k'
            )
            axdiag[1].scatter(
                (self.samples - direction2)[:, 0],
                (self.samples - direction2)[:, 1],
                color='k',
                s=50.)
            axdiag[1].set_xlabel('Property 1')
            axdiag[1].set_ylabel('Property 2')
            axdiag[1].set_title('Petro with W')

            plt.show()

    def test_spherical_covariances(self):
        # Fit a Gaussian Mixture
        clf = Utils.GaussianMixture(
            n_components=self.n_components,
            covariance_type='spherical',
            max_iter=1000, n_init=10,
            means_init=self.means,
            warm_start=True,
            weights_init=self.proportions)
        clf.fit(self.samples)

        # Define reg Simple
        reg_simple = Regularization.SimplePetroRegularization(
            mesh=self.mesh,
            GMmref=clf,
            approx_gradient=True, alpha_x=0.,
            wiresmap=self.wires,
            evaltype='approx'
        )
        # Define reg with volumes
        reg = Regularization.PetroRegularization(
            mesh=self.mesh,
            GMmref=clf,
            approx_gradient=True, alpha_x=0.,
            wiresmap=self.wires,
            evaltype='approx'
        )

        # check score value
        score_approx = reg_simple(Utils.mkvc(self.samples))
        reg_simple.objfcts[0].evaltype = 'full'
        score = reg_simple(Utils.mkvc(self.samples))
        passed_score_simple = np.allclose(score_approx, score, rtol=1e-1)
        self.assertTrue(passed_score_simple)
        print('scores for SimplePetro are ok')

        score_approx = reg(Utils.mkvc(self.samples))
        reg.objfcts[0].evaltype = 'full'
        score = reg(Utils.mkvc(self.samples))
        passed_score = np.allclose(score_approx, score, rtol=1e-1)
        self.assertTrue(passed_score)
        print('scores for Petro are ok')

        # check derivatives as an optimization on locally quadratic function
        # Simple

        reference = clf.means_[clf.predict(self.samples)]

        deriv_simple = reg_simple.deriv(Utils.mkvc(self.samples))
        Hessian_simple = lambda x: reg_simple.deriv2(
            Utils.mkvc(self.samples), x)
        HV_simple = LinearOperator(
            [len(self.samples) * self.ndim, len(self.samples) * self.ndim],
            matvec=Hessian_simple,
            rmatvec=Hessian_simple
        )
        p_simple = bicgstab(HV_simple, deriv_simple)
        direction2_simple = np.c_[self.wires * p_simple[0]]
        passed_derivative_simple = np.allclose(
            Utils.mkvc(self.samples - direction2_simple),
            Utils.mkvc(reference), rtol=1e-1
        )
        self.assertTrue(passed_derivative_simple)
        print('derivatives for SimplePetro are ok')

        # With volumes
        deriv = reg.deriv(Utils.mkvc(self.samples))
        Hessian = lambda x: reg.deriv2(Utils.mkvc(self.samples), x)
        HV = LinearOperator(
            [len(self.samples) * self.ndim, len(self.samples) * self.ndim],
            matvec=Hessian,
            rmatvec=Hessian
        )
        p = bicgstab(HV, deriv)
        direction2 = np.c_[self.wires * p[0]]
        passed_derivative = np.allclose(
            Utils.mkvc(self.samples - direction2),
            Utils.mkvc(reference), rtol=1e-1
        )
        self.assertTrue(passed_derivative)
        print('derivatives for Petro are ok')

        if self.PlotIt:
            import matplotlib.pyplot as plt
            xmin, xmax = ymin, ymax = self.samples.min(), self.samples.max()
            x, y = np.mgrid[xmin:xmax:.01, ymin:ymax:.01]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y
            rv = clf.score_samples(pos.reshape(-1, 2))
            rvm = clf.predict(pos.reshape(-1, 2))
            figspherical, axspherical = plt.subplots(1, 2, figsize=(16, 8))
            figspherical.suptitle('Spherical Covariances Tests')
            # Simple
            axspherical[0].contourf(x, y, rvm.reshape(
                x.shape), alpha=0.25, cmap='brg')
            axspherical[0].contour(x, y, rv.reshape(x.shape), 20)
            axspherical[0].scatter(
                self.s0[:, 0], self.s0[:, 1],
                color='blue', s=5., alpha=0.25
            )
            axspherical[0].scatter(
                self.s1[:, 0], self.s1[:, 1],
                color='green', s=5., alpha=0.25
            )
            axspherical[0].quiver(
                self.samples[:, 0], self.samples[:, 1], -
                (self.wires.s0 * deriv_simple),
                -(self.wires.s1 * deriv_simple), color='red', alpha=0.25
            )
            axspherical[0].quiver(
                self.samples[:, 0], self.samples[:, 1],
                -direction2_simple[:, 0], -direction2_simple[:, 1],
                color='k'
            )
            axspherical[0].scatter(
                (self.samples - direction2_simple)[:, 0],
                (self.samples - direction2_simple)[:, 1],
                color='k',
                s=50.)
            axspherical[0].set_xlabel('Property 1')
            axspherical[0].set_ylabel('Property 2')
            axspherical[0].set_title('SimplePetro')
            # With W
            axspherical[1].contourf(x, y, rvm.reshape(
                x.shape), alpha=0.25, cmap='brg')
            axspherical[1].contour(x, y, rv.reshape(x.shape), 20)
            axspherical[1].scatter(
                self.s0[:, 0], self.s0[:, 1],
                color='blue', s=5., alpha=0.25
            )
            axspherical[1].scatter(
                self.s1[:, 0], self.s1[:, 1],
                color='green', s=5., alpha=0.25
            )
            axspherical[1].quiver(
                self.samples[:, 0], self.samples[
                    :, 1], -(self.wires.s0 * deriv),
                -(self.wires.s1 * deriv), color='red', alpha=0.25
            )
            axspherical[1].quiver(
                self.samples[:, 0], self.samples[:, 1],
                -direction2[:, 0], -direction2[:, 1],
                color='k'
            )
            axspherical[1].scatter(
                (self.samples - direction2)[:, 0],
                (self.samples - direction2)[:, 1],
                color='k',
                s=50.)
            axspherical[1].set_xlabel('Property 1')
            axspherical[1].set_ylabel('Property 2')
            axspherical[1].set_title('Petro with W')

            plt.show()

if __name__ == '__main__':
    unittest.main()
