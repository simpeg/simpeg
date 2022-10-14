import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import linalg
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.mixture._gaussian_mixture import (
    _compute_precision_cholesky,
    _compute_log_det_cholesky,
    _estimate_gaussian_covariances_full,
    _estimate_gaussian_covariances_diag,
    _estimate_gaussian_covariances_spherical,
    _check_means,
    _check_precisions,
    _check_shape,
)
from sklearn.mixture._base import check_random_state, ConvergenceWarning
import warnings
from SimPEG.maps import IdentityMap


###############################################################################
# Disclaimer: the following classes built upon the GaussianMixture class      #
# from Scikit-Learn. New functionalitie are added, as well as modifications to#
# existing functions, to serve the purposes pursued within SimPEG.            #
# This use is allowed by the Scikit-Learn licensing (BSD-3-Clause License)    #
# and we are grateful for their contributions to the open-source community.   #                                                   #
###############################################################################


class WeightedGaussianMixture(GaussianMixture):
    """
    Weighted Gaussian mixture class

    This class upon the GaussianMixture class from Scikit-Learn.
    Two main modifications:

        1. Each sample/observation is given a weight, the volume of the
        corresponding discretize.BaseMesh cell, when fitting the
        Gaussian Mixture Model (GMM). More volume gives more importance, ensuing a
        mesh-free evaluation of the clusters of the geophysical model.

        2. When set manually, the proportions can be set either globally (normal behavior)
        or cell-by-cell (improvements)

    Disclaimer: this class built upon the GaussianMixture class from Scikit-Learn.
    New functionalitie are added, as well as modifications to
    existing functions, to serve the purposes pursued within SimPEG.
    This use is allowed by the Scikit-Learn licensing (BSD-3-Clause License)
    and we are grateful for their contributions to the open-source community.

    Addtional parameters to provide, compared to sklearn.mixture.gaussian_mixture:

    Parameters
    ----------
    n_components : int
        Number of components
    mesh : discretize.BaseMesh
        ``TensorMesh`` or ``QuadTree`` or Octree) mesh: the volume
        of the cells give each sample/observations its weight in the fitting proces
    actv : numpy.ndarry, optional
        Active indexes
    """

    def __init__(
        self,
        n_components,
        mesh,
        actv=None,
        covariance_type="full",
        init_params="kmeans",
        max_iter=100,
        means_init=None,
        n_init=10,
        precisions_init=None,
        random_state=None,
        reg_covar=1e-06,
        tol=0.001,
        verbose=0,
        verbose_interval=10,
        warm_start=False,
        weights_init=None,
        # **kwargs
    ):
        self.mesh = mesh
        self.actv = actv
        if self.actv is None:
            self.cell_volumes = self.mesh.cell_volumes
        else:
            self.cell_volumes = self.mesh.cell_volumes[self.actv]

        super(WeightedGaussianMixture, self).__init__(
            covariance_type=covariance_type,
            init_params=init_params,
            max_iter=max_iter,
            means_init=means_init,
            n_components=n_components,
            n_init=n_init,
            precisions_init=precisions_init,
            random_state=random_state,
            reg_covar=reg_covar,
            tol=tol,
            verbose=verbose,
            verbose_interval=verbose_interval,
            warm_start=warm_start,
            weights_init=weights_init,
            # **kwargs
        )
        # setKwargs(self, **kwargs)

    def compute_clusters_precisions(self):
        """Compute and set the precisions matrices and their Cholesky decomposition.

        Use this function after setting covariances manually.
        """
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )
        if self.covariance_type == "full":
            self.precisions_ = np.empty(self.precisions_cholesky_.shape)
            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)

        elif self.covariance_type == "tied":
            self.precisions_ = np.dot(
                self.precisions_cholesky_, self.precisions_cholesky_.T
            )
        else:
            self.precisions_ = self.precisions_cholesky_ ** 2

    def compute_clusters_covariances(self):
        """Compute the precisions matrices and their Cholesky decomposition.

        Use this function after setting precisions matrices manually.
        """
        self.covariances_cholesky_ = _compute_precision_cholesky(
            self.precisions_, self.covariance_type
        )
        if self.covariance_type == "full":
            self.covariances_ = np.empty(self.covariances_cholesky_.shape)
            for k, cov_chol in enumerate(self.covariances_cholesky_):
                self.covariances_[k] = np.dot(cov_chol, cov_chol.T)

        elif self.covariance_type == "tied":
            self.covariances_ = np.dot(
                self.covariances_cholesky_, self.covariances_cholesky_.T
            )
        else:
            self.covariances_ = self.covariances_cholesky_ ** 2

        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

    def order_clusters_GM_weight(self, outputindex=False):
        """Order clusters by decreasing weights

        PARAMETERS
        ----------
        outputindex : bool, default: ``True``
            If ``True``, return the sorting index

        RETURN
        ------
        np.ndarray
            Sorting index
        """
        if self.weights_.ndim == 1:
            indx = np.argsort(self.weights_, axis=0)[::-1]
            self.weights_ = self.weights_[indx].reshape(self.weights_.shape)
        else:
            indx = np.argsort(self.weights_.sum(axis=0), axis=0)[::-1]
            self.weights_ = self.weights_[:, indx].reshape(self.weights_.shape)

        self.means_ = self.means_[indx].reshape(self.means_.shape)

        if self.covariance_type == "tied":
            pass
        else:
            self.precisions_ = self.precisions_[indx].reshape(self.precisions_.shape)
            self.covariances_ = self.covariances_[indx].reshape(self.covariances_.shape)

        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

        if outputindex:
            return indx

    def _check_weights(self, weights, n_components, n_samples):
        """
        [modified from Scikit-Learn.mixture.gaussian_mixture]
        Check the user provided 'weights'.
        Parameters
        ----------
        weights : array-like, shape (n_components,) or (n_samples, n_components)
            The proportions of components of each mixture.
        n_components : int
            Number of components.

        Returns
        -------
        weights : array, shape (n_components,)
        """

        if len(weights.shape) == 2:
            weights = check_array(
                weights, dtype=[np.float64, np.float32], ensure_2d=True
            )
            _check_shape(weights, (n_samples, n_components), "weights")
        else:
            weights = check_array(
                weights, dtype=[np.float64, np.float32], ensure_2d=False
            )
            _check_shape(weights, (n_components,), "weights")

        # check range
        if any(np.less(weights, 0.0)) or any(np.greater(weights, 1.0)):
            raise ValueError(
                "The parameter 'weights' should be in the range "
                "[0, 1], but got max value %.5f, min value %.5f"
                % (np.min(weights), np.max(weights))
            )

        # check normalization
        if not np.allclose(np.abs(1.0 - np.sum(weights.T, axis=0)), 0.0):
            raise ValueError(
                "The parameter 'weights' should be normalized, "
                "but got sum(weights) = %.5f" % np.sum(weights)
            )

        return weights

    def _check_parameters(self, X):
        """
        [modified from Scikit-Learn.mixture.gaussian_mixture]
        Check the Gaussian mixture parameters are well defined.
        """
        n_samples, n_features = X.shape
        if self.covariance_type not in ["spherical", "tied", "diag", "full"]:
            raise ValueError(
                "Invalid value for 'covariance_type': %s "
                "'covariance_type' should be in "
                "['spherical', 'tied', 'diag', 'full']" % self.covariance_type
            )

        if self.weights_init is not None:
            self.weights_init = self._check_weights(
                self.weights_init,
                self.n_components,
                n_samples,
            )

        if self.means_init is not None:
            self.means_init = _check_means(
                self.means_init, self.n_components, n_features
            )

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(
                self.precisions_init,
                self.covariance_type,
                self.n_components,
                n_features,
            )

    def _initialize_parameters(self, X, random_state):
        """
        [modified from Scikit-Learn.mixture._base]
        Initialize the model parameters.
        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)
        random_state : RandomState
            A random number generator instance.
        """
        n_samples, _ = X.shape

        if self.init_params == "kmeans":
            resp = np.zeros((n_samples, self.n_components))
            label = (
                KMeans(
                    n_clusters=self.n_components, n_init=1, random_state=random_state
                )
                .fit(X, sample_weight=self.cell_volumes)
                .labels_
            )
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == "random":
            resp = random_state.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError(
                "Unimplemented initialization method '%s'" % self.init_params
            )

        self._initialize(X, resp)

    def _m_step(self, X, log_resp):
        """
        [modified from Scikit-Learn.mixture.gaussian_mixture]
        M step.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        Volume = np.mean(self.cell_volumes)
        weights, self.means_, self.covariances_ = self._estimate_gaussian_parameters(
            X, self.mesh, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        weights /= n_samples * Volume
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

        if len(self.weights_.shape) == 1:
            self.weights_ = weights

    def _estimate_gaussian_covariances_tied(self, resp, X, nk, means, reg_covar):
        """
        [modified from Scikit-Learn.mixture.gaussian_mixture]
        Estimate the tied covariance matrix.
        Parameters
        ----------
        resp : array-like, shape (n_samples, n_components)
        X : array-like, shape (n_samples, n_features)
        nk : array-like, shape (n_components,)
        means : array-like, shape (n_components, n_features)
        reg_covar : float
        Returns
        -------
        covariance : array, shape (n_features, n_features)
        The tied covariance matrix of the components.
        """
        avg_X2 = np.dot(self.cell_volumes * X.T, X)
        avg_means2 = np.dot(nk * means.T, means)
        covariance = avg_X2 - avg_means2
        covariance /= nk.sum()
        covariance.flat[:: len(covariance) + 1] += reg_covar
        return covariance

    def _estimate_gaussian_parameters(self, X, mesh, resp, reg_covar, covariance_type):
        """
        [modified from Scikit-Learn.mixture.gaussian_mixture]
        Estimate the Gaussian distribution parameters.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data array.
        resp : array-like, shape (n_samples, n_components)
            The responsibilities for each data sample in X.
        reg_covar : float
            The regularization added to the diagonal of the covariance matrices.
        covariance_type : {'full', 'tied', 'diag', 'spherical'}
            The type of precision matrices.
        Returns
        -------
        nk : array-like, shape (n_components,)
            The numbers of data samples in the current components.
        means : array-like, shape (n_components, n_features)
            The centers of the current components.
        covariances : array-like
            The covariance matrix of the current components.
            The shape depends of the covariance_type.
        """
        respVol = self.cell_volumes.reshape(-1, 1) * resp
        nk = respVol.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        means = np.dot(respVol.T, X) / nk[:, np.newaxis]
        covariances = {
            "full": _estimate_gaussian_covariances_full,
            "tied": self._estimate_gaussian_covariances_tied,
            "diag": _estimate_gaussian_covariances_diag,
            "spherical": _estimate_gaussian_covariances_spherical,
        }[covariance_type](respVol, X, nk, means, reg_covar)
        return nk, means, covariances

    def _e_step(self, X):
        """
        [modified from Scikit-Learn.mixture.gaussian_mixture]
        E step.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X
        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.average(log_prob_norm, weights=self.cell_volumes), log_resp

    def score(self, X, y=None):
        """Compute the per-sample average log-likelihood

        [modified from Scikit-Learn.mixture.gaussian_mixture]
        Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : (n_samples, n_dimensions) array-like
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        y : ``None``
            Placeholder variable

        Returns
        -------
        float
            Log likelihood of the Gaussian mixture given X.
        """
        return np.average(self.score_samples(X), weights=self.cell_volumes)

    def _estimate_log_gaussian_prob_with_sensW(
        self, X, sensW, means, precisions_chol, covariance_type
    ):
        """
        [New function, modified from Scikit-Learn.mixture.gaussian_mixture._estimate_log_gaussian_prob]
        Estimate the log Gaussian probability with depth or sensitivity weighting.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        means : array-like, shape (n_components, n_features)
        sensW: array-like, Sensitvity or Depth Weighting, shape(n_samples, n_features)
        precisions_chol : array-like,
            Cholesky decompositions of the precision matrices.
            'full' : shape of (n_components, n_features, n_features)
            'tied' : shape of (n_features, n_features)
            'diag' : shape of (n_components, n_features)
            'spherical' : shape of (n_components,)
        covariance_type : {'full', 'tied', 'diag', 'spherical'}
        Returns
        -------
        log_prob : array, shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape
        n_components, _ = means.shape
        # det(precision_chol) is half of det(precision)
        log_det = _compute_log_det_cholesky(
            precisions_chol, covariance_type, n_features
        )

        if covariance_type == "full":
            log_prob = np.empty((n_samples, n_components))
            for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
                y = np.dot(X * sensW, prec_chol) - np.dot(mu * sensW, prec_chol)
                log_prob[:, k] = np.sum(np.square(y), axis=1)

        elif covariance_type == "tied":
            log_prob = np.empty((n_samples, n_components))
            for k, mu in enumerate(means):
                y = np.dot(X * sensW, precisions_chol) - np.dot(
                    mu * sensW, precisions_chol
                )
                log_prob[:, k] = np.sum(np.square(y), axis=1)

        else:
            log_prob = np.empty((n_samples, n_components))
            for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
                prec_chol_mat = np.eye(n_features) * prec_chol
                y = np.dot(X * sensW, prec_chol_mat) - np.dot(mu * sensW, prec_chol_mat)
                log_prob[:, k] = np.sum(np.square(y), axis=1)

        return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det

    def _estimate_log_prob_with_sensW(self, X, sensW):
        """
        [New function, modified from Scikit-Learn.mixture.gaussian_mixture._estimate_log_prob]
        """
        return self._estimate_log_gaussian_prob_with_sensW(
            X, sensW, self.means_, self.precisions_cholesky_, self.covariance_type
        )

    def _estimate_weighted_log_prob_with_sensW(self, X, sensW):
        """
        [New function, modified from Scikit-Learn.mixture.gaussian_mixture._estimate_weighted_log_prob]
        Estimate the weighted log-probabilities, log P(X | Z) + log weights.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        return (
            self._estimate_log_prob_with_sensW(X, sensW) + self._estimate_log_weights()
        )

    def score_samples_with_sensW(self, X, sensW):
        """Compute the weighted log probabilities for each sample.

        [New function, modified from Scikit-Learn.mixture.gaussian_mixture.score_samples]
        Compute the weighted log probabilities for each sample.

        Parameters
        ----------
        X : (n_samples, n_features) array_like
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        sensW : (n_samples) array_like
            Sensitivity weights

        Returns
        -------
        (n_samples) numpy.array
            Log probabilities of each data point in X.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)

        return logsumexp(self._estimate_weighted_log_prob_with_sensW(X, sensW), axis=1)

    def plot_pdf(
        self,
        ax=None,
        flag2d=False,
        x_component=None,
        y_component=None,
        padding=0.2,
        plotting_precision=100,
        plot_membership=False,
        contour_opts={},
        level_opts={},
    ):
        """
        Utils to plot the marginal PDFs of a GMM, either in 1D or 2D (1 or 2 physical properties at the time).

        Parameters
        ----------
        ax : matplotlib.Axes, optional
            Matplotliv axes object. Need to be a 3-array if flag2d is True
        flag2d : bool, default: ``False``
            Flag to either plot a 1D or 2D probability distributions
        x_component : int, optional
            Physical property to plot on the X-axis, as ordered in the GMM.
        y_component : int, optional
            Physical property to plot on the Y-axis, as ordered in the GMM
        padding : float, default: 0.2
            How much relative padding around the petrophysical means for the 1D and 2D plots
        plotting_precision : int, default: 100
            Number of divisions for the 1D and 2D plots
        plot_membership : bool, default: ``False``
            Plot the membership rather than the probability
        contour_opts : dict
            Modify the plotting options of the contour plot (in 1D and 2D)
        level_opts : dict
            Modify the plotting options of the level plot (in 1D and 2D)

        Returns
        -------
        matplotlib.Axes
            Axes including the plot
        """

        plotting_precision = int(plotting_precision)

        if x_component is None:
            x_component = 0
            if y_component is None:
                if flag2d and self.means_.shape[1] > 1:
                    y_component = 1

        if (not (x_component is None)) and (not (y_component is None)):
            flag2d = True

        if ax is None:
            if flag2d:
                fig = plt.figure(figsize=(10, 10))
                ax0 = plt.subplot2grid((4, 4), (3, 1), colspan=3)
                ax1 = plt.subplot2grid((4, 4), (0, 1), colspan=3, rowspan=3)
                ax2 = plt.subplot2grid((4, 4), (0, 0), rowspan=3)
                ax = [ax0, ax1, ax2]
            else:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax = np.r_[ax]

        # deal with the various possible shapes of covariances
        if self.covariance_type == "tied":
            covariances = np.r_[
                [self.covariances_ for i in range(self.n_components)]
            ].reshape(self.n_components, self.n_features_in_, self.n_features_in_)
        elif self.covariance_type == "diag" or self.covariance_type == "spherical":
            covariances = np.r_[
                [
                    self.covariances_[i] * np.eye(self.n_features_in_)
                    for i in range(self.n_components)
                ]
            ].reshape(self.n_components, self.n_features_in_, self.n_features_in_)
        else:
            covariances = self.covariances_

        dx = padding * (
            self.means_[:, x_component].max() - self.means_[:, x_component].min()
        )
        xmin, xmax = (
            self.means_[:, x_component].min() - dx,
            self.means_[:, x_component].max() + dx,
        )

        # create a sklearn.clustering.GaussianMixture for plotting (no influence from mesh and local weights)
        meansx = self.means_[:, x_component].reshape(self.n_components, 1)
        covx = covariances[:, [x_component]][:, :, [x_component]]
        if len(self.weights_.shape) == 2:
            weights = self.weights_.sum(axis=0)
            weights /= weights.sum()
        else:
            weights = self.weights_

        clfx = GaussianMixture(
            n_components=self.n_components,
            means_init=meansx,
            n_init=1,
            max_iter=2,
            tol=np.inf,
        )
        # random fit, we set values after.
        clfx.fit(np.random.randn(10, 1))
        clfx.means_ = meansx
        clfx.covariances_ = covx
        clfx.precisions_cholesky_ = _compute_precision_cholesky(
            clfx.covariances_, clfx.covariance_type
        )
        clfx.weights_ = weights

        xplot = np.linspace(xmin, xmax, plotting_precision)[:, np.newaxis]
        if plot_membership:
            rvx = clfx.predict(xplot)
            labelx = "membership"
        else:
            rvx = np.exp(clfx.score_samples(xplot))
            labelx = "1D Probability\nDensity\nDistribution"

        ax[0].set_xlim(xmin, xmax)
        ax[0].plot(
            xplot,
            rvx,
            linewidth=3.0,
            label=labelx,
            c="k",
        )
        ax[0].legend()
        ax[0].set_xlabel("Physical property {}".format(x_component))
        ax[0].set_ylabel("Probability Density values")

        if flag2d:

            dy = padding * (
                self.means_[:, y_component].max() - self.means_[:, y_component].min()
            )
            ymin, ymax = (
                self.means_[:, y_component].min() - dy,
                self.means_[:, y_component].max() + dy,
            )

            # create a sklearn.clustering.GaussianMixture for plotting (no influence from mesh and local weights)
            meansy = self.means_[:, y_component].reshape(self.n_components, 1)
            covy = covariances[:, [y_component]][:, :, [y_component]]

            clfy = GaussianMixture(
                n_components=self.n_components,
                means_init=meansy,
                n_init=1,
                max_iter=2,
                tol=np.inf,
            )
            # random fit, we set values after.
            clfy.fit(np.random.randn(10, 1))
            clfy.means_ = meansy
            clfy.covariances_ = covy
            clfy.precisions_cholesky_ = _compute_precision_cholesky(
                clfy.covariances_, clfy.covariance_type
            )
            clfy.weights_ = weights

            # 1d y-plot
            yplot = np.linspace(ymin, ymax, plotting_precision)[:, np.newaxis]
            if plot_membership:
                rvy = clfy.predict(yplot)
                labely = "membership"
            else:
                rvy = np.exp(clfy.score_samples(yplot))
                labely = "1D Probability\nDensity\nDistribution"
            ax[2].plot(rvy, yplot, linewidth=3.0, c="k", label=labely)
            ax[2].set_ylabel("Physical property {}".format(y_component))
            ax[2].set_ylim(ymin, ymax)
            ax[2].legend()

            # 2d plot
            mean2d = self.means_[:, [x_component, y_component]]
            cov2d = covariances[:, [x_component, y_component]][
                :, :, [x_component, y_component]
            ]
            clf2d = GaussianMixture(
                n_components=self.n_components,
                means_init=mean2d,
                n_init=1,
                max_iter=2,
                tol=np.inf,
            )
            # random fit, we set values after.
            clf2d.fit(np.random.randn(10, 2))
            clf2d.means_ = mean2d
            clf2d.covariances_ = cov2d
            clf2d.precisions_cholesky_ = _compute_precision_cholesky(
                clf2d.covariances_, clf2d.covariance_type
            )
            clf2d.weights_ = weights

            x, y = np.mgrid[
                xmin : xmax : (xmax - xmin) / plotting_precision,
                ymin : ymax : (ymax - ymin) / plotting_precision,
            ]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y

            if plot_membership:
                rv2d = clf2d.predict(pos.reshape(-1, 2))
                labely = "membership"
            else:
                rv2d = clf2d.score_samples(pos.reshape(-1, 2))
                labely = "2D Probability Density Distribution"

            contour_opts = {"levels": 10, "cmap": "viridis", **contour_opts}
            surf = ax[1].contourf(x, y, rv2d.reshape(x.shape), **contour_opts)

            level_opts = {
                "levels": 10,
                "colors": "k",
                "linewidths": 1.0,
                "linestyles": "dashdot",
                **level_opts,
            }

            ax[1].contour(x, y, rv2d.reshape(x.shape), **level_opts)
            ax[1].scatter(
                meansx,
                meansy,
                label="Petrophysical means",
                cmap="inferno_r",
                c=np.linspace(0, self.n_components, self.n_components),
                marker="v",
                edgecolors="k",
            )

            axbar = inset_axes(
                ax[1],
                width="40%",
                height="3%",
                loc="upper right",
                borderpad=1,
            )
            cbpetro = plt.colorbar(surf, cax=axbar, orientation="horizontal")
            cbpetro.set_ticks([rv2d.min(), rv2d.max()])
            cbpetro.set_ticklabels(["Low", "High"])
            cbpetro.set_label(labely)
            cbpetro.outline.set_edgecolor("k")

            ax[1].set_xlim(xmin, xmax)
            ax[1].set_ylim(ymin, ymax)
            ax[1].legend(loc=3)
            ax[1].set_ylabel("")
            ax[1].set_xlabel("")

        return ax


class GaussianMixtureWithPrior(WeightedGaussianMixture):
    """
    This class built upon the WeightedGaussianMixture, which itself built upon from
    the mixture.gaussian_mixture.GaussianMixture class from Scikit-Learn.

    In addition to weights samples/observations by the cells volume of the mesh,
    this class uses a posterior approach to fit the GMM parameters. This means
    it takes prior parameters, passed through `WeightedGaussianMixture` gmmref.
    The prior distribution for each parameters (proportions, means, covariances) is
    defined through a conjugate or semi-conjugate approach (prior_type), to the choice of the user.
    See Astic & Oldenburg 2019: A framework for petrophysically and geologically
    guided geophysical inversion (https://doi.org/10.1093/gji/ggz389) for more information.

    Disclaimer: this class built upon the GaussianMixture class from Scikit-Learn.
    New functionalitie are added, as well as modifications to
    existing functions, to serve the purposes pursued within SimPEG.
    This use is allowed by the Scikit-Learn licensing (BSD-3-Clause License)
    and we are grateful for their contributions to the open-source community.

    Addtional parameters to provide, compared to `WeightedGaussianMixture`:

    Parameters
    ----------
    kappa : numpy.ndarray
        strength of the confidence in the prior means
    nu : numpy.ndarry
        strength of the confidence in the prior covariances
    zeta : numpy.ndarry
        strength of the confidence in the prior proportions
    prior_type : str
        Choose from one of the following:

            - "semi": semi-conjugate prior, the means and covariances priors are indepedent
            - "full": conjugate prior, the means and covariances priors are inter-depedent

    update_covariances : bool
        Choose from two options:

        - ``True``: semi or conjugate prior by averaging the covariances
        - ``False``: alternative (not conjugate) prior: average the precisions instead

    fixed_membership : numpy.ndarray of int, optional
        A 2d numpy.ndarray to fix the membership to a chosen lithology of particular cells.
        The first column contains the numeric index of the cells, the second column the respective lithology index.
        Shape is (index of the fixed cell, lithology index) fixed_membership:
    """

    def __init__(
        self,
        gmmref,
        kappa=0.0,
        nu=0.0,
        zeta=0.0,
        prior_type="semi",  # semi or full
        update_covariances=True,
        fixed_membership=None,
        init_params="kmeans",
        max_iter=100,
        means_init=None,
        n_init=10,
        precisions_init=None,
        random_state=None,
        reg_covar=1e-06,
        tol=0.001,
        verbose=0,
        verbose_interval=10,
        warm_start=False,
        weights_init=None,
        # **kwargs
    ):
        self.mesh = gmmref.mesh
        self.n_components = gmmref.n_components
        self.gmmref = gmmref
        self.covariance_type = gmmref.covariance_type
        self.kappa = kappa * np.ones((self.n_components, gmmref.means_.shape[1]))
        self.nu = nu * np.ones(self.n_components)
        self.zeta = zeta * np.ones_like(self.gmmref.weights_)
        self.prior_type = prior_type
        self.update_covariances = update_covariances
        self.fixed_membership = fixed_membership

        super(GaussianMixtureWithPrior, self).__init__(
            covariance_type=self.covariance_type,
            mesh=self.mesh,
            actv=self.gmmref.actv,
            init_params=init_params,
            max_iter=max_iter,
            means_init=means_init,
            n_components=self.n_components,
            n_init=n_init,
            precisions_init=precisions_init,
            random_state=random_state,
            reg_covar=reg_covar,
            tol=tol,
            verbose=verbose,
            verbose_interval=verbose_interval,
            warm_start=warm_start,
            weights_init=weights_init,
            # **kwargs
        )
        # setKwargs(self, **kwargs)

    def order_cluster(self, outputindex=False):
        """Order cluster

        Arrange the clusters of gmm in the same order as those of gmmref,
        based on their relative similarities and priorizing first the most proeminent
        clusters (highest proportions)

        Parameters
        ----------
        outputindex : bool, default: ``False``
            If ``True``, return the ordering index for the clusters of GMM.

        Returns
        -------
        numpy.ndarray of int
            Returns the ordering index for the clusters of GMM if
            *outputindex* is ``True``
        """
        self.order_clusters_GM_weight()

        idx_ref = np.ones(len(self.gmmref.means_), dtype=bool)

        indx = []

        for i in range(self.n_components):
            dis = self._estimate_log_prob(
                self.gmmref.means_[idx_ref].reshape(
                    [-1] + [d for d in self.gmmref.means_.shape[1:]]
                )
            )
            id_dis = dis.argmax(axis=0)[i]
            idrefmean = np.where(
                np.all(
                    self.gmmref.means_ == self.gmmref.means_[idx_ref][id_dis], axis=1
                )
            )[0][0]
            indx.append(idrefmean)
            idx_ref[idrefmean] = False

        self.means_ = self.means_[indx].reshape(self.means_.shape)

        if self.weights_.ndim == 1:
            self.weights_ = self.weights_[indx].reshape(self.weights_.shape)
        else:
            self.weights_ = self.weights_[:, indx].reshape(self.weights_.shape)

        if self.covariance_type == "tied":
            pass
        else:
            self.precisions_ = self.precisions_[indx].reshape(self.precisions_.shape)
            self.covariances_ = self.covariances_[indx].reshape(self.covariances_.shape)

        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

        if outputindex:
            return indx

    def update_gmm_with_priors(self, debug=False):
        """Update GMM with priors

        This method, inserted in the fit function, modify the Maximum Likelyhood Estimation (MLE)
        of the GMM's parameters to a Maximum A Posteriori (MAP) estimation.

        Parameters
        ----------
        debug : bool, default: ``False``
            Print debug statments
        """

        self.compute_clusters_precisions()
        self.order_cluster()

        if debug:
            print("before update means: ", self.means_)
            print("before update weights: ", self.weights_)
            print("before update precisions: ", self.precisions_)

        if self.weights_.ndim == 1:
            weights_ = self.weights_
        else:
            weights_ = (np.c_[self.cell_volumes] * self.weights_).sum(axis=0) / (
                np.c_[self.cell_volumes] * self.weights_
            ).sum()

        if self.gmmref.weights_.ndim == 1:
            ref_weights_ = self.gmmref.weights_
        else:
            ref_weights_ = (np.c_[self.gmmref.cell_volumes] * self.gmmref.weights_).sum(
                axis=0
            ) / (np.c_[self.gmmref.cell_volumes] * self.gmmref.weights_).sum()

        for k in range(self.n_components):
            if self.prior_type == "full":
                smu = (self.kappa[k] * weights_[k]) * (
                    (self.gmmref.means_[k] - self.means_[k]) ** 2.0
                )
                smu /= self.kappa[k] + weights_[k]
                smu *= 1.0 / (weights_[k] + ref_weights_[k] * self.nu[k])

            self.means_[k] = (1.0 / (weights_[k] + ref_weights_[k] * self.kappa[k])) * (
                weights_[k] * self.means_[k]
                + ref_weights_[k] * self.kappa[k] * self.gmmref.means_[k]
            )

            if self.gmmref.covariance_type == "tied":
                pass
            elif self.update_covariances:
                self.covariances_[k] = (
                    1.0 / (weights_[k] + ref_weights_[k] * self.nu[k])
                ) * (
                    weights_[k] * self.covariances_[k]
                    + ref_weights_[k] * self.nu[k] * self.gmmref.covariances_[k]
                )

                if self.prior_type == "full":
                    self.covariances_[k] += smu

            else:
                self.precisions_[k] = (
                    1.0 / (weights_[k] + ref_weights_[k] * self.nu[k])
                ) * (
                    weights_[k] * self.precisions_[k]
                    + ref_weights_[k] * self.nu[k] * self.gmmref.precisions_[k]
                )

        self.weights_ = (
            (self.weights_ + self.zeta * self.gmmref.weights_).T
            * (1.0 / (1.0 + np.sum((self.zeta * self.gmmref.weights_).T, axis=0)))
        ).T

        if self.gmmref.covariance_type == "tied":
            if self.update_covariances:
                self.covariances_ = (1.0 / (1.0 + np.sum(ref_weights_ * self.nu))) * (
                    self.covariances_
                    + np.sum(ref_weights_ * self.nu) * self.gmmref.covariances_
                )
                self.compute_clusters_precisions()
            else:
                self.precisions_ = (1.0 / (1.0 + np.sum(ref_weights_ * self.nu))) * (
                    self.precisions_
                    + np.sum(ref_weights_ * self.nu) * self.gmmref.precisions_
                )
                self.compute_clusters_covariances()
        elif self.update_covariances:
            self.compute_clusters_precisions()
        else:
            self.compute_clusters_covariances()

        if debug:
            print("after update means: ", self.means_)
            print("after update weights: ", self.weights_)
            print("after update precisions: ", self.precisions_)

    def fit(self, X, y=None, debug=False):
        """Estimate model parameters with the EM algorithm.

        [modified from Scikit-Learn for Maximum A Posteriori estimates (MAP)]
        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for ``max_iter``
        times until the change of likelihood or lower bound is less than
        ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
        If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
        initialization is performed upon the first call. Upon consecutive
        calls, training starts where it left off.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        y : Ignored
            Not used, present for API consistency by convention.
        debug : bool, default: ``False``
            If ``True``, print debug statements

        Returns
        -------
        self : object
            The fitted mixture.
        """
        self.fit_predict(X, y, debug)
        return self

    def fit_predict(self, X, y=None, debug=False):
        """Estimate model parameters using X and predict the labels for X.

        [modified from Scikit-Learn for Maximum A Posteriori estimates (MAP)]
        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        Parameters
        ----------
        X : (n_samples, n_features) array_like
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        y : Ignored
            Not used, present for API consistency by convention.
        debug : bool, default: ``False``
            If ``True``, print debug statements

        Returns
        -------
        (n_samples) array
            Component labels.
        """
        if self.verbose:
            print("modified from scikit-learn")

        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)

            lower_bound = -np.inf if do_init else self.lower_bound_

            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound

                log_prob_norm, log_resp = self._e_step(X)

                if self.fixed_membership is not None:
                    # force responsibilities
                    aux = -(np.inf) * np.ones(
                        (self.fixed_membership.shape[0], self.n_components)
                    )
                    aux[np.arange(len(aux)), self.fixed_membership[:, 1]] = 0.0
                    log_resp[self.fixed_membership[:, 0]] = aux

                self._m_step(X, log_resp)
                self.update_gmm_with_priors(debug=debug)

                if self.fixed_membership is not None and self.weights_.ndim == 2:
                    # force local weights
                    aux = np.zeros((self.fixed_membership.shape[0], self.n_components))
                    aux[np.arange(len(aux)), self.fixed_membership[:, 1]] = 1
                    self.weights_[self.fixed_membership[:, 0]] = aux

                lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)

                change = lower_bound - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end(lower_bound)

            if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = n_iter

        if not self.converged_:
            warnings.warn(
                "Initialization %d did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data." % (init + 1),
                ConvergenceWarning,
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        return self


class GaussianMixtureWithNonlinearRelationships(WeightedGaussianMixture):
    """Gaussian mixture class for non-linear relationships.

    This class built upon the WeightedGaussianMixture, which itself built upon from
    the mixture.gaussian_mixture.GaussianMixture class from Scikit-Learn.

    In addition to weights samples/observations by the cells volume of the mesh,
    this class can be given specified nonlinear relationships between physical properties.
    (polynomial etc.) Those nonlinear relationships are given in the form of a
    list of mapping (cluster_mapping argument). Functions are added and modified
    to fill that purpose, in particular the `fit` and  `samples` functions.

    Disclaimer: this class built upon the GaussianMixture class from Scikit-Learn.
    New functionalitie are added, as well as modifications to
    existing functions, to serve the purposes pursued within SimPEG.
    This use is allowed by the Scikit-Learn licensing (BSD-3-Clause License)
    and we are grateful for their contributions to the open-source community.

    Addtional parameters to provide, compared to `WeightedGaussianMixture`:

    Parameters
    ----------
    cluster_mapping : (n_components) list
        List of mapping describing a nonlinear relationships between physical properties; one per cluster/unit.
    """

    def __init__(
        self,
        mesh,
        n_components=1,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
        cluster_mapping=None,
    ):

        if cluster_mapping is None:
            self.cluster_mapping = [IdentityMap() for i in range(n_components)]
        else:
            self.cluster_mapping = cluster_mapping

        super(GaussianMixtureWithNonlinearRelationships, self).__init__(
            mesh=mesh,
            covariance_type=covariance_type,
            init_params=init_params,
            max_iter=max_iter,
            means_init=means_init,
            n_components=n_components,
            n_init=n_init,
            precisions_init=precisions_init,
            random_state=random_state,
            reg_covar=reg_covar,
            tol=tol,
            verbose=verbose,
            verbose_interval=verbose_interval,
            warm_start=warm_start,
            weights_init=weights_init,
            # **kwargs
        )

    def _initialize(self, X, resp):
        """
        [modified from Scikit-Learn.mixture.gaussian_mixture]
        Initialization of the Gaussian mixture parameters.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        resp : array-like, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances = self._estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type
        )
        weights /= n_samples

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type
            )
        elif self.covariance_type == "full":
            self.precisions_cholesky_ = np.array(
                [
                    linalg.cholesky(prec_init, lower=True)
                    for prec_init in self.precisions_init
                ]
            )
        elif self.covariance_type == "tied":
            self.precisions_cholesky_ = linalg.cholesky(
                self.precisions_init, lower=True
            )
        else:
            self.precisions_cholesky_ = self.precisions_init

    @classmethod
    def _estimate_log_gaussian_prob(
        self, X, means, precisions_chol, covariance_type, cluster_mapping
    ):
        """
        [modified from Scikit-Learn.mixture.gaussian_mixture]
        Estimate the log Gaussian probability.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        means : array-like, shape (n_components, n_features)
        precisions_chol : array-like
            Cholesky decompositions of the precision matrices.
            'full' : shape of (n_components, n_features, n_features)
            'tied' : shape of (n_features, n_features)
            'diag' : shape of (n_components, n_features)
            'spherical' : shape of (n_components,)
        covariance_type : {'full', 'tied', 'diag', 'spherical'}
        Returns
        -------
        log_prob : array, shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape
        n_components, _ = means.shape
        # det(precision_chol) is half of det(precision)
        log_det = _compute_log_det_cholesky(
            precisions_chol, covariance_type, n_features
        )

        if covariance_type == "full":
            log_prob = np.empty((n_samples, n_components))
            for k, (mu, prec_chol, mapping) in enumerate(
                zip(means, precisions_chol, cluster_mapping)
            ):
                y = np.dot(mapping * X, prec_chol) - np.dot(mu, prec_chol)
                log_prob[:, k] = np.sum(np.square(y), axis=1)

        elif covariance_type == "tied":
            log_prob = np.empty((n_samples, n_components))
            for k, (mu, mapping) in enumerate(zip(means, cluster_mapping)):
                y = np.dot(mapping * X, precisions_chol) - np.dot(mu, precisions_chol)
                log_prob[:, k] = np.sum(np.square(y), axis=1)

        elif covariance_type == "diag" or covariance_type == "spherical":
            log_prob = np.empty((n_samples, n_components))
            for k, (mu, prec_chol, mapping) in enumerate(
                zip(means, precisions_chol, cluster_mapping)
            ):
                y = np.dot(mapping * X, prec_chol * np.eye(n_features)) - np.dot(
                    mu, prec_chol * np.eye(n_features)
                )
                log_prob[:, k] = np.sum(np.square(y), axis=1)

        return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det

    def _estimate_log_prob(self, X):
        """
        [modified from Scikit-Learn.mixture.gaussian_mixture]
        """
        return self._estimate_log_gaussian_prob(
            X,
            self.means_,
            self.precisions_cholesky_,
            self.covariance_type,
            self.cluster_mapping,
        )

    def _estimate_gaussian_parameters(self, X, resp, reg_covar, covariance_type):
        """
        [modified from Scikit-Learn.mixture.gaussian_mixture]
        """
        respVol = self.cell_volumes.reshape(-1, 1) * resp
        nk = respVol.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        # stupid lazy piece of junk code to get the shapes right
        means = np.dot(respVol.T, X) / nk[:, np.newaxis]
        covariances = {
            "full": _estimate_gaussian_covariances_full,
            "tied": self._estimate_gaussian_covariances_tied,
            "diag": _estimate_gaussian_covariances_diag,
            "spherical": _estimate_gaussian_covariances_spherical,
        }[covariance_type](respVol, X, nk, means, reg_covar)
        # The actual calculation
        for k in range(means.shape[0]):
            means[k] = (
                np.dot(respVol.T, self.cluster_mapping[k] * X) / nk[:, np.newaxis]
            )[k]
        for k in range(means.shape[0]):
            covariances[k] = (
                {
                    "full": _estimate_gaussian_covariances_full,
                    "tied": self._estimate_gaussian_covariances_tied,
                    "diag": _estimate_gaussian_covariances_diag,
                    "spherical": _estimate_gaussian_covariances_spherical,
                }[covariance_type](
                    respVol, self.cluster_mapping[k] * X, nk, means, reg_covar
                )
            )[k]
        return nk, means, covariances

    # TODOs: Still not working because of inverse mapping not implemented
    def sample(self, n_samples=1):
        """
        [modified from Scikit-Learn.mixture.gaussian_mixture]
        Generate random samples from the fitted Gaussian distribution
        with nonlinear relationships.

        Parameters
        ----------
        n_samples : int, default: `
            Number of samples to generate.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample
        y : array, shape (nsamples,)
            Component labels

        """
        check_is_fitted(self)

        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (self.n_components)
            )

        _, n_features = self.means_.shape
        rng = check_random_state(self.random_state)
        n_samples_comp = rng.multinomial(n_samples, self.weights_)

        if self.covariance_type == "full":
            X = np.vstack(
                [
                    rng.multivariate_normal(mean, covariance, int(sample))
                    for (mean, covariance, sample) in zip(
                        self.means_, self.covariances_, n_samples_comp
                    )
                ]
            )
        elif self.covariance_type == "tied":
            X = np.vstack(
                [
                    rng.multivariate_normal(mean, self.covariances_, int(sample))
                    for (mean, sample) in zip(self.means_, n_samples_comp)
                ]
            )
        else:
            X = np.vstack(
                [
                    mean + rng.randn(sample, n_features) * np.sqrt(covariance)
                    for (mean, covariance, sample) in zip(
                        self.means_, self.covariances_, n_samples_comp
                    )
                ]
            )

        y = np.concatenate(
            [j * np.ones(sample, dtype=int) for j, sample in enumerate(n_samples_comp)]
        )
        X = np.vstack(
            [
                self.cluster_mapping[y[i]].inverse(X[i].reshape(-1, n_features))
                for i in range(len(X))
            ]
        )

        return (X, y)

    def _m_step(self, X, log_resp):
        """
        [modified from Scikit-Learn.mixture.gaussian_mixture]
        M step.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        (
            self.weights_,
            self.means_,
            self.covariances_,
        ) = self._estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )


class GaussianMixtureWithNonlinearRelationshipsWithPrior(GaussianMixtureWithPrior):
    """Gaussian mixture class for non-linear relationships with priors.

    This class built upon the `GaussianMixtureWithPrior`, which itself built upon from
    the `WeightedGaussianMixture`, built up from the
    mixture.gaussian_mixture.GaussianMixture class from Scikit-Learn.

    In addition to weights samples/observations by the cells volume of the mesh
    (from `WeightedGaussianMixture`), and nonlinear relationships for each cluster
    (from `GaussianMixtureWithNonlinearRelationships`), this class uses a
    posterior approach to fit the GMM parameters (from `GaussianMixtureWithPrior`).
    It takes prior parameters, passed through `WeightedGaussianMixture` gmmref.
    The prior distribution for each parameters (proportions, means, covariances) is
    defined through a conjugate or semi-conjugate approach (prior_type), to the choice of the user.
    See Astic & Oldenburg 2019: A framework for petrophysically and geologically
    guided geophysical inversion (https://doi.org/10.1093/gji/ggz389) for more information.

    Disclaimer: this class built upon the GaussianMixture class from Scikit-Learn.
    New functionalitie are added, as well as modifications to
    existing functions, to serve the purposes pursued within SimPEG.
    This use is allowed by the Scikit-Learn licensing (BSD-3-Clause License)
    and we are grateful for their contributions to the open-source community.

    Addtional parameters to provide, compared to `GaussianMixtureWithPrior`:

    Parameters
    ----------
    cluster_mapping : (n_components) list
        List of mapping describing a nonlinear relationships between physical properties; one per cluster/unit.

    """

    def __init__(
        self,
        gmmref,
        kappa=0.0,
        nu=0.0,
        zeta=0.0,
        prior_type="semi",  # semi or conjugate
        cluster_mapping=None,
        init_params="kmeans",
        max_iter=100,
        means_init=None,
        n_init=10,
        precisions_init=None,
        random_state=None,
        reg_covar=1e-06,
        tol=0.001,
        verbose=0,
        verbose_interval=10,
        warm_start=False,
        weights_init=None,
        update_covariances=True,
        fixed_membership=None,
    ):

        if cluster_mapping is None:
            self.cluster_mapping = gmmref.cluster_mapping
        else:
            self.cluster_mapping = cluster_mapping

        super(GaussianMixtureWithNonlinearRelationshipsWithPrior, self).__init__(
            gmmref=gmmref,
            kappa=kappa,
            nu=nu,
            zeta=zeta,
            init_params=init_params,
            max_iter=max_iter,
            means_init=means_init,
            n_init=n_init,
            precisions_init=precisions_init,
            random_state=random_state,
            reg_covar=reg_covar,
            tol=tol,
            verbose=verbose,
            verbose_interval=verbose_interval,
            warm_start=warm_start,
            weights_init=weights_init,
            update_covariances=update_covariances,
            fixed_membership=fixed_membership
            # **kwargs
        )

    def _initialize(self, X, resp):
        """
        [modified from Scikit-Learn.mixture.gaussian_mixture]
        Initialization of the Gaussian mixture parameters.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        resp : array-like, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances = self._estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type
        )
        weights /= n_samples

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type
            )
        elif self.covariance_type == "full":
            self.precisions_cholesky_ = np.array(
                [
                    linalg.cholesky(prec_init, lower=True)
                    for prec_init in self.precisions_init
                ]
            )
        elif self.covariance_type == "tied":
            self.precisions_cholesky_ = linalg.cholesky(
                self.precisions_init, lower=True
            )
        else:
            self.precisions_cholesky_ = self.precisions_init

    @classmethod
    def _estimate_log_gaussian_prob(
        self, X, means, precisions_chol, covariance_type, cluster_mapping
    ):
        """
        [modified from Scikit-Learn.mixture.gaussian_mixture]
        Estimate the log Gaussian probability.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        means : array-like, shape (n_components, n_features)
        precisions_chol : array-like
            Cholesky decompositions of the precision matrices.
            'full' : shape of (n_components, n_features, n_features)
            'tied' : shape of (n_features, n_features)
            'diag' : shape of (n_components, n_features)
            'spherical' : shape of (n_components,)
        covariance_type : {'full', 'tied', 'diag', 'spherical'}
        cluster_mapping: list of mapping of length (n_components,)
        Returns
        -------
        log_prob : array, shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape
        n_components, _ = means.shape
        # det(precision_chol) is half of det(precision)
        log_det = _compute_log_det_cholesky(
            precisions_chol, covariance_type, n_features
        )

        if covariance_type == "full":
            log_prob = np.empty((n_samples, n_components))
            for k, (mu, prec_chol, mapping) in enumerate(
                zip(means, precisions_chol, cluster_mapping)
            ):
                y = np.dot(mapping * X, prec_chol) - np.dot(mu, prec_chol)
                log_prob[:, k] = np.sum(np.square(y), axis=1)

        elif covariance_type == "tied":
            log_prob = np.empty((n_samples, n_components))
            for k, (mu, mapping) in enumerate(zip(means, cluster_mapping)):
                y = np.dot(mapping * X, precisions_chol) - np.dot(mu, precisions_chol)
                log_prob[:, k] = np.sum(np.square(y), axis=1)

        elif covariance_type == "diag" or covariance_type == "spherical":
            log_prob = np.empty((n_samples, n_components))
            precisions = precisions_chol ** 2
            for k, (mu, prec_chol, mapping) in enumerate(
                zip(means, precisions_chol, cluster_mapping)
            ):
                y = np.dot(mapping * X, prec_chol * np.eye(n_features)) - np.dot(
                    mu, prec_chol * np.eye(n_features)
                )
                log_prob[:, k] = np.sum(np.square(y), axis=1)

        return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det

    def _estimate_log_prob(self, X):
        return self._estimate_log_gaussian_prob(
            X,
            self.means_,
            self.precisions_cholesky_,
            self.covariance_type,
            self.cluster_mapping,
        )

    def _estimate_gaussian_parameters(self, X, resp, reg_covar, covariance_type):
        """
        [modified from Scikit-Learn.mixture.gaussian_mixture]
        """
        respVol = self.cell_volumes.reshape(-1, 1) * resp
        nk = respVol.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        # stupid lazy piece of junk code to get the shapes right
        means = np.dot(respVol.T, X) / nk[:, np.newaxis]
        covariances = {
            "full": _estimate_gaussian_covariances_full,
            "tied": self._estimate_gaussian_covariances_tied,
            "diag": _estimate_gaussian_covariances_diag,
            "spherical": _estimate_gaussian_covariances_spherical,
        }[covariance_type](respVol, X, nk, means, reg_covar)
        # The actual calculation
        for k in range(means.shape[0]):
            means[k] = (
                np.dot(respVol.T, self.cluster_mapping[k] * X) / nk[:, np.newaxis]
            )[k]
        for k in range(means.shape[0]):
            covariances[k] = (
                {
                    "full": _estimate_gaussian_covariances_full,
                    "tied": self._estimate_gaussian_covariances_tied,
                    "diag": _estimate_gaussian_covariances_diag,
                    "spherical": _estimate_gaussian_covariances_spherical,
                }[covariance_type](
                    respVol, self.cluster_mapping[k] * X, nk, means, reg_covar
                )
            )[k]
        return nk, means, covariances

    def _m_step(self, X, log_resp):
        """
        [modified from Scikit-Learn.mixture.gaussian_mixture]
        M step.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        (
            self.weights_,
            self.means_,
            self.covariances_,
        ) = self._estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )
