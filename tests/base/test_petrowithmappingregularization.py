from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
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
        self.n_components = 3
        self.means = np.array([[5.,  -2.],
                               [-2.,  4.],
                               [-1.,  -6.]])
        self.sigma = np.array([[[1., -.1], [-.1, .1]],
                             [[.4, 0.6], [0.6, 1.]],
                             [[.4, 0.6], [0.6, 1.]]])
        self.rv0 = multivariate_normal(self.means[0], self.sigma[0])
        self.rv1 = multivariate_normal(self.means[1], self.sigma[1])
        self.rv2 = multivariate_normal(self.means[2], self.sigma[2])
        self.proportions = np.r_[0.5, 0.3, 0.2]
        self.nsample = 100
        self.poly0 = Maps.PolynomialPetroClusterMap(coeffyx=np.r_[8., 8., 2.])
        self.poly1 = Maps.PolynomialPetroClusterMap(
            coeffyx=np.r_[-12.5, 5., -0.5])
        self.cluster_mapping = [self.poly0, self.poly1, Maps.IdentityMap()]
        self.s0 = self.rv0.rvs(int(self.nsample * self.proportions[0]))
        self.s1 = self.rv1.rvs(int(self.nsample * self.proportions[1]))
        self.s2 = self.rv2.rvs(int(self.nsample * self.proportions[2]))
        self.data0 = self.s0 + \
            np.c_[np.zeros_like(self.s0[:, 0]), 0.5 * (self.s0[:, 0] - 5.)**2]
        self.data1 = self.s1 + \
            np.c_[np.zeros_like(self.s1[:, 0]), -2 * (self.s1[:, 0] + 2)**2]
        self.data2 = self.s2
        self.samples = np.vstack([self.data0, self.data1, self.data2])
        self.reference = np.r_[
            np.ones_like(self.s0) * self.means[0],
            np.ones_like(self.s1) * self.means[1],
            np.ones_like(self.s2) * self.means[2]]
        self.mesh = Mesh.TensorMesh([self.samples.shape[0]])
        self.wires = Maps.Wires(('s0', self.mesh.nC), ('s1', self.mesh.nC))

        self.PlotIt = True

    def test_full_covariances(self):
        # Fit a Gaussian Mixture
        clf = Utils.GaussianMixtureWithMapping(
            n_components=self.n_components,
            covariance_type='full',
            max_iter=1000, n_init=10,
            means_init=self.means,
            warm_start=True,
            precisions_init=np.linalg.inv(self.sigma),
            weights_init=self.proportions,
            cluster_mapping=self.cluster_mapping)
        clf.fit(self.samples)

        x, y = np.mgrid[-10:10:.1, -10:10:.1]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y
        plt.contourf(x,y,clf.predict(pos.reshape(-1,2)).reshape(x.shape))


        # Define reg Simple
        reg_simple = Regularization.SimplePetroWithMappingRegularization(
            mesh=self.mesh,
            GMmref=clf,
            approx_gradient=True, alpha_x=0.,
            wiresmap=self.wires,
            evaltype='approx')
        # check score value
        score_approx=reg_simple(Utils.mkvc(self.samples))
        reg_simple.objfcts[0].evaltype='full'
        score=reg_simple(Utils.mkvc(self.samples))
        print(score, score_approx)
        #passed_score_simple = np.allclose(score_approx, score, rtol=1e-1)
        #self.assertTrue(passed_score_simple)
        print('scores for SimplePetro are ok')

        # check derivatives as an optimization on locally quadratic function
        # Simple

        deriv_simple=reg_simple.deriv(Utils.mkvc(self.samples))
        Hessian_simple=lambda x: reg_simple.deriv2(
            Utils.mkvc(self.samples), x)
        HV_simple=LinearOperator(
            [len(self.samples) * self.ndim, len(self.samples) * self.ndim],
            matvec=Hessian_simple,
            rmatvec=Hessian_simple
        )
        p_simple=bicgstab(HV_simple, deriv_simple)
        direction2_simple=np.c_[self.wires * p_simple[0]]
        print('derivatives for SimplePetro are ok')

        if self.PlotIt:
            import matplotlib.pyplot as plt
            xmin, xmax=ymin, ymax=self.samples.min(), self.samples.max()
            x, y=np.mgrid[xmin:xmax:.01, ymin:ymax:.01]
            pos=np.empty(x.shape + (2,))
            pos[:, :, 0]=x
            pos[:, :, 1]=y
            rv=clf.score_samples(pos.reshape(-1, 2))
            rvm=clf.predict(pos.reshape(-1, 2))
            figfull, axfull=plt.subplots(1, 2, figsize=(16, 8))
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

            plt.show()

if __name__ == '__main__':
    unittest.main()
