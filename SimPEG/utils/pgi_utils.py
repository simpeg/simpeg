import numpy as np
import copy
from scipy.stats import multivariate_normal
from scipy import spatial, linalg
from scipy.special import logsumexp
from scipy.sparse import diags
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.utils import check_array
from sklearn.mixture._gaussian_mixture import (
    _compute_precision_cholesky,
    _compute_log_det_cholesky,
    _estimate_gaussian_covariances_full,
    _estimate_gaussian_covariances_tied,
    _estimate_gaussian_covariances_diag,
    _estimate_gaussian_covariances_spherical,
    _check_means,
    _check_precisions,
    _check_shape,
)
from sklearn.mixture._base import _check_X, check_random_state, ConvergenceWarning
import warnings
from .mat_utils import mkvc
from ..maps import IdentityMap


def ComputeDistances(a, b):

    if a.ndim == 1:
        x = mkvc(a, numDims=2)
    else:
        x = a
    if b.ndim == 1:
        y = mkvc(b, numDims=2)
    else:
        y = b

    n, d = x.shape
    t, d1 = y.shape

    if not d == d1:
        raise Exception("vectors must have same number of columns")

    sq_dis = (
        np.dot((x ** 2.0), np.ones([d, t]))
        + np.dot(np.ones([n, d]), (y ** 2.0).T)
        - 2.0 * np.dot(x, y.T)
    )

    idx = np.argmin(sq_dis, axis=1)

    return sq_dis ** 0.5, idx


def order_clusters_GM_weight(gmm, outputindex=False):
    """
    order cluster by increasing mean for Gaussian Mixture scikit object
    """
    if gmm.weights_.ndim == 1:
        indx = np.argsort(gmm.weights_, axis=0)[::-1]
        gmm.weights_ = gmm.weights_[indx].reshape(gmm.weights_.shape)

    else:
        indx = np.argsort(gmm.weights_.sum(axis=0), axis=0)[::-1]
        gmm.weights_ = gmm.weights_[:, indx].reshape(gmm.weights_.shape)
    gmm.means_ = gmm.means_[indx].reshape(gmm.means_.shape)
    if gmm.covariance_type == "tied":
        pass
    else:
        gmm.precisions_ = gmm.precisions_[indx].reshape(gmm.precisions_.shape)
        gmm.covariances_ = gmm.covariances_[indx].reshape(gmm.covariances_.shape)
    gmm.precisions_cholesky_ = _compute_precision_cholesky(
        gmm.covariances_, gmm.covariance_type
    )

    if outputindex:
        return indx


def order_clusters_GM_mean(gmm, outputindex=False):
    """
    order cluster by increasing mean for Gaussian Mixture scikit object
    """

    indx = np.argsort(gmm.means_, axis=0)[::-1]
    gmm.means_ = gmm.means_[indx].reshape(gmm.means_.shape)
    if gmm.weights_.ndim == 1:
        gmm.weights_ = gmm.weights_[indx].reshape(gmm.weights_.shape)
    else:
        gmm.weights_ = gmm.weights_[:, indx].reshape(gmm.weights_.shape)

    if gmm.covariance_type == "tied":
        pass
    else:
        gmm.precisions_ = gmm.precisions_[indx].reshape(gmm.precisions_.shape)
        gmm.covariances_ = gmm.covariances_[indx].reshape(gmm.covariances_.shape)
    gmm.precisions_cholesky_ = _compute_precision_cholesky(
        gmm.covariances_, gmm.covariance_type
    )

    if outputindex:
        return indx


def order_cluster(gmm, gmmref, outputindex=False):
    order_clusters_GM_weight(gmm)

    idx_ref = np.ones(len(gmmref.means_), dtype=bool)

    indx = []

    for i in range(gmm.n_components):
        dis = gmm._estimate_log_prob(
            gmmref.means_[idx_ref].reshape([-1] + [d for d in gmmref.means_.shape[1:]])
        )
        id_dis = dis.argmax(axis=0)[i]
        idrefmean = np.where(
            np.all(gmmref.means_ == gmmref.means_[idx_ref][id_dis], axis=1)
        )[0][0]
        indx.append(idrefmean)
        idx_ref[idrefmean] = False

    gmm.means_ = gmm.means_[indx].reshape(gmm.means_.shape)

    if gmm.weights_.ndim == 1:
        gmm.weights_ = gmm.weights_[indx].reshape(gmm.weights_.shape)
    else:
        gmm.weights_ = gmm.weights_[:, indx].reshape(gmm.weights_.shape)

    if gmm.covariance_type == "tied":
        pass
    else:
        gmm.precisions_ = gmm.precisions_[indx].reshape(gmm.precisions_.shape)
        gmm.covariances_ = gmm.covariances_[indx].reshape(gmm.covariances_.shape)
    gmm.precisions_cholesky_ = _compute_precision_cholesky(
        gmm.covariances_, gmm.covariance_type
    )

    if outputindex:
        return indx


def compute_clusters_precision(gmm):
    gmm.precisions_cholesky_ = _compute_precision_cholesky(
        gmm.covariances_, gmm.covariance_type
    )
    if gmm.covariance_type == "full":
        gmm.precisions_ = np.empty(gmm.precisions_cholesky_.shape)
        for k, prec_chol in enumerate(gmm.precisions_cholesky_):
            gmm.precisions_[k] = np.dot(prec_chol, prec_chol.T)

    elif gmm.covariance_type == "tied":
        gmm.precisions_ = np.dot(gmm.precisions_cholesky_, gmm.precisions_cholesky_.T)
    else:
        gmm.precisions_ = gmm.precisions_cholesky_ ** 2


def computeCovariance(gmm):
    if gmm.covariance_type == "full":
        gmm.covariances_ = np.empty(gmm.covariances_cholesky_.shape)
        for k, cov_chol in enumerate(gmm.covariances_cholesky_):
            gmm.covariances_[k] = np.dot(cov_chol, cov_chol.T)

    elif gmm.covariance_type == "tied":
        gmm.covariances_ = np.dot(
            gmm.covariances_cholesky_, gmm.covariances_cholesky_.T
        )
    else:
        gmm.covariances_ = gmm.covariances_cholesky_ ** 2


def ComputeConstantTerm(gmm):
    cste = 0.0
    d = gmm.means_[0].shape[0]
    for i in range(gmm.n_components):
        if gmm.covariance_type == "tied":
            cste += gmm.weights_[i] * (
                (1.0 / 2.0)
                * np.log(((2.0 * np.pi) ** d) * np.linalg.det(gmm.covariances_))
                - np.log(gmm.weights_[i])
            )
        elif gmm.covariance_type == "diag" or gmm.covariance_type == "spherical":
            cste += gmm.weights_[i] * (
                (1.0 / 2.0)
                * np.log(
                    ((2.0 * np.pi) ** d)
                    * np.linalg.det(gmm.covariances_[i] * np.eye(gmm.means_.shape[1]))
                )
                - np.log(gmm.weights_[i])
            )
        else:
            cste += gmm.weights_[i] * (
                (1.0 / 2.0)
                * np.log(((2.0 * np.pi) ** d) * np.linalg.det(gmm.covariances_[i]))
                - np.log(gmm.weights_[i])
            )
    return cste


def UpdateGaussianMixtureModel(
    gmm,
    gmmref,
    zeta,
    nu,
    kappa,
    verbose=False,
    update_covariances=False,
    prior_type="semi",
):

    compute_clusters_precision(gmm)
    order_cluster(gmm, gmmref)

    if verbose:
        print("before update means: ", gmm.means_)
        print("before update weights: ", gmm.weights_)
        print("before update precisions: ", gmm.precisions_)

    if gmm.weights_.ndim == 1:
        weights_ = gmm.weights_
    else:
        weights_ = (np.c_[gmm.vol] * gmm.weights_).sum(axis=0) / (
            np.c_[gmm.vol] * gmm.weights_
        ).sum()

    if gmmref.weights_.ndim == 1:
        ref_weights_ = gmmref.weights_
    else:
        ref_weights_ = (np.c_[gmmref.vol] * gmmref.weights_).sum(axis=0) / (
            np.c_[gmmref.vol] * gmmref.weights_
        ).sum()

    for k in range(gmm.n_components):
        if prior_type == "full":
            smu = (kappa[k] * weights_[k]) * ((gmmref.means_[k] - gmm.means_[k]) ** 2.0)
            smu /= kappa[k] + weights_[k]
            smu *= 1.0 / (weights_[k] + ref_weights_[k] * nu[k])

        gmm.means_[k] = (1.0 / (weights_[k] + ref_weights_[k] * kappa[k])) * (
            weights_[k] * gmm.means_[k] + ref_weights_[k] * kappa[k] * gmmref.means_[k]
        )

        if gmmref.covariance_type == "tied":
            pass
        elif update_covariances:
            gmm.covariances_[k] = (1.0 / (weights_[k] + ref_weights_[k] * nu[k])) * (
                weights_[k] * gmm.covariances_[k]
                + ref_weights_[k] * nu[k] * gmmref.covariances_[k]
            )

            if prior_type == "full":
                gmm.covariances_[k] += smu
                print("full", smu)

        else:
            gmm.precisions_[k] = (1.0 / (weights_[k] + ref_weights_[k] * nu[k])) * (
                weights_[k] * gmm.precisions_[k]
                + ref_weights_[k] * nu[k] * gmmref.precisions_[k]
            )

    gmm.weights_ = (
        (gmm.weights_ + zeta * gmmref.weights_).T
        * (1.0 / (1.0 + np.sum((zeta * gmmref.weights_).T, axis=0)))
    ).T

    if gmmref.covariance_type == "tied":
        if update_covariances:
            gmm.covariances_ = (1.0 / (1.0 + np.sum(ref_weights_ * nu))) * (
                gmm.covariances_ + np.sum(ref_weights_ * nu) * gmmref.covariances_
            )
            gmm.precisions_cholesky_ = _compute_precision_cholesky(
                gmm.covariances_, gmm.covariance_type
            )
            compute_clusters_precision(gmm)
        else:
            gmm.precisions_ = (1.0 / (1.0 + np.sum(ref_weights_ * nu))) * (
                gmm.precisions_ + np.sum(ref_weights_ * nu) * gmmref.precisions_
            )
            gmm.covariances_cholesky_ = _compute_precision_cholesky(
                gmm.precisions_, gmm.covariance_type
            )
            computeCovariance(gmm)
            gmm.precisions_cholesky_ = _compute_precision_cholesky(
                gmm.covariances_, gmm.covariance_type
            )
    elif update_covariances:
        gmm.precisions_cholesky_ = _compute_precision_cholesky(
            gmm.covariances_, gmm.covariance_type
        )
        compute_clusters_precision(gmm)
    else:
        gmm.covariances_cholesky_ = _compute_precision_cholesky(
            gmm.precisions_, gmm.covariance_type
        )
        computeCovariance(gmm)
        gmm.precisions_cholesky_ = _compute_precision_cholesky(
            gmm.covariances_, gmm.covariance_type
        )

    if verbose:
        print("after update means: ", gmm.means_)
        print("after update weights: ", gmm.weights_)
        print("after update precisions: ", gmm.precisions_)


class FuzzyGaussianMixtureWithPrior(GaussianMixture):
    def __init__(
        self,
        gmmref,
        kappa=0.0,
        nu=0.0,
        zeta=1.0,
        fuzzyness=2.0,
        GMinit="auto",
        init_params="kmeans",
        max_iter=100,
        means_init=None,
        n_components=3,
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

        self.fuzzyness = fuzzyness
        self.gmmref = gmmref
        self.covariance_type = gmmref.covariance_type
        self.kappa = np.ones(n_components) * kappa
        self.nu = np.ones(n_components) * nu
        self.zeta = np.ones(n_components) * zeta
        self.GMinit = GMinit

        super(FuzzyGaussianMixtureWithPrior, self).__init__(
            covariance_type=self.covariance_type,
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

    def FitFuzzyWithConjugatePrior(self, X, **kwargs):
        """
        beta is the same size as components
        """
        n_data, n_features = X.shape
        n_components = self.gmmref.n_components

        if self.GMinit == None:
            km = KMeans(n_clusters=n_components)
            km.fit(X)
            winit = (
                np.r_[[np.sum(km.labels_ == i) for i in range(n_components)]]
                / float(n_data)
            ).reshape(-1, 1)
            precision_init = np.r_[
                [np.diag(np.ones(n_features)) for i in range(n_components)]
            ]
            self.means_ = km.cluster_centers_
            self.weights_ = mkvc(winit)
            self.precisions_ = precision_init
            self.covariances_ = precision_init
            self.precisions_cholesky_ = _compute_precision_cholesky(
                self.covariances_, self.covariance_type
            )
        elif self.GMinit == "auto":
            self.fit(X)
        else:
            self.means_ = copy.deepcopy(self.GMinit.means_)
            self.weights_ = copy.deepcopy(self.GMinit.weights_)
            self.precisions_ = copy.deepcopy(self.GMinit.precisions_)
            self.covariances_ = copy.deepcopy(self.GMinit.covariances_)
            self.precisions_cholesky_ = copy.deepcopy(self.GMinit.precisions_cholesky_)

        # Order clusters by increasing mean TODO: what happened with several properties
        # idx = order_cluster(self,self.gmmref,outputindex = True)
        zeta = self.zeta
        kappa = self.kappa
        nu = self.nu

        # Init Membership
        # E step
        logW = np.log(np.ones((n_data, n_components)) * self.weights_)
        # print(logW)
        change = np.inf
        it = 0

        while it < self.max_iter and change > self.tol:

            change = 0.0

            logP = np.zeros((n_data, n_components))

            if self.gmmref.covariance_type == "full":
                for k in range(n_components):
                    logP[:, k] = (
                        mkvc(
                            multivariate_normal(
                                self.means_[k],
                                (self.covariances_[k]) * (self.fuzzyness - 1.0),
                            ).logpdf(X)
                        )
                        + np.log(self.fuzzyness - 1) / 2.0
                    )
            elif self.gmmref.covariance_type == "tied":
                raise Exception("Implementation in progress")
            else:
                raise Exception("Spherical is not implemented yet")
            logWP = logW + logP
            log_r = logWP - logsumexp(logWP, axis=1, keepdims=True)
            # print(np.sum(np.exp(log_r),axis=0))
            r = np.exp(self.fuzzyness * log_r)
            sumr = np.exp(logsumexp(self.fuzzyness * log_r))
            # M step
            for k in range(n_components):

                # total Membership of the cluster
                rk = np.exp(logsumexp(self.fuzzyness * log_r[:, k]))
                # print(rk)
                if rk != 0:
                    # Update cluster center
                    muk = (
                        (1.0 / (rk + kappa[k])) * (np.sum(diags(r[:, k]) * X, axis=0))
                    ) + ((kappa[k] / (rk + kappa[k])) * self.gmmref.means_[k])
                    if self.means_[k] != 0.0:
                        change = np.maximum(
                            np.abs((self.means_[k] - muk) / self.means_[k]).max(),
                            change,
                        )
                    else:
                        change = np.maximum(
                            np.abs((self.means_[k] - muk)).max(), change
                        )
                    self.means_[k] = muk

                if rk != 0:
                    # Update cluster covariance
                    if self.gmmref.covariance_type == "full":
                        xmean = (np.sum(diags(r[:, k]) * X, axis=0)) / rk
                        Sk0 = (nu[k] + n_features + 2.0) * self.gmmref.covariances_[k]
                        Sx = np.dot((X - xmean).T, (diags(r[:, k]) * (X - xmean)))
                        Smu = ((kappa[k] * rk) / (rk + kappa[k])) * np.dot(
                            (muk - xmean).T, (muk - xmean)
                        )
                        covk = (Sk0 + Sx + Smu) / (nu[k] + rk + n_features + 2.0)
                    elif self.gmmref.covariance_type == "tied":
                        Stot = (nu[0] + n_features + 2.0) * self.gmmref.covariances_
                        for k in range(n_components):
                            xmean = (np.sum(diags(r[:, k]) * X, axis=0)) / rk
                            Sx = np.dot((X - xmean).T, (diags(r[:, k]) * (X - xmean)))
                            Smu = ((kappa[k] * rk) / (rk + kappa[k])) * np.dot(
                                (muk - xmean).T, (muk - xmean)
                            )
                            Stot = Stot + Sx + Smu
                        covk = (Stot) / (nu[0] + n_data + n_features + 2.0)

                # Regularize
                covid = self.reg_covar * np.eye(n_features)
                idx = np.abs(covk) < self.reg_covar
                # Set Off-diag to 0
                covk[idx] = 0.0
                # Set On-diag to reg_covar
                covk = covk + covid * idx

                if self.gmmref.covariance_type == "full":
                    change = np.maximum(
                        np.abs(
                            (self.covariances_[k] - covk) / self.covariances_[k]
                        ).max(),
                        change,
                    )
                    self.covariances_[k] = covk
                    # print('cov: ',covk,change)
                elif self.covariance_type == "tied":
                    self.covariances_ = covk
                    change = np.maximum(
                        np.abs((self.covariances_ - covk) / self.covariances_).max(),
                        change,
                    )
                # Update cluster Proportion
                thetak = (rk + zeta[k] - 1.0) / (sumr + np.sum(zeta) - n_components)
                # Real Derichlet distribution
                if self.weights_[k] != 0.0:
                    change = np.maximum(
                        np.abs((self.weights_[k] - thetak) / self.weights_[k]).max(),
                        change,
                    )
                else:
                    change = np.maximum(
                        np.abs((self.weights_[k] - thetak)).max(), change
                    )
                self.weights_[k] = thetak
                # print('weights: ',thetak,change)

            self.precisions_cholesky_ = _compute_precision_cholesky(
                self.covariances_, self.covariance_type
            )
            compute_clusters_precision(self)
            if self.verbose:
                print("iteration: ", it)
                print("Maximum relative change done to parameters: ", change)
            it += +1
        self.n_iter_ = it


class WeightedGaussianMixture(GaussianMixture):
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
        update_covariances=False,
        fixed_membership=None,
        # **kwargs
    ):
        self.mesh = mesh
        self.actv = actv
        if self.actv is None:
            self.vol = self.mesh.vol
        else:
            self.vol = self.mesh.vol[self.actv]

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

    def _check_weights(self, weights, n_components, n_samples):
        """Check the user provided 'weights'.
        Parameters
        ----------
        weights : array-like, shape (n_components,)
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
        if np.less(weights, 0.0).any() or (np.greater(weights, 1.0)).any():
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
        """Check the Gaussian mixture parameters are well defined."""
        n_samples, n_features = X.shape
        if self.covariance_type not in ["spherical", "tied", "diag", "full"]:
            raise ValueError(
                "Invalid value for 'covariance_type': %s "
                "'covariance_type' should be in "
                "['spherical', 'tied', 'diag', 'full']" % self.covariance_type
            )

        if self.weights_init is not None:
            self.weights_init = self._check_weights(
                self.weights_init, self.n_components, n_samples,
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
        """Initialize the model parameters.
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
                .fit(X, sample_weight=self.vol)
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
        """M step.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        Volume = np.mean(self.vol)
        weights, self.means_, self.covariances_ = self._estimate_gaussian_parameters(
            X, self.mesh, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        weights /= n_samples * Volume
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

        if len(self.weights_.shape) == 1:
            self.weights_ = weights

    def _estimate_gaussian_parameters(self, X, mesh, resp, reg_covar, covariance_type):
        """Estimate the Gaussian distribution parameters.
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
        respVol = self.vol.reshape(-1, 1) * resp
        nk = respVol.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        means = np.dot(respVol.T, X) / nk[:, np.newaxis]
        covariances = {
            "full": _estimate_gaussian_covariances_full,
            "tied": _estimate_gaussian_covariances_tied,
            "diag": _estimate_gaussian_covariances_diag,
            "spherical": _estimate_gaussian_covariances_spherical,
        }[covariance_type](respVol, X, nk, means, reg_covar)
        return nk, means, covariances

    def _e_step(self, X):
        """E step.
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
        return np.average(log_prob_norm, weights=self.vol), log_resp

    def score(self, X, y=None):
        """Compute the per-sample average log-likelihood of the given data X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        log_likelihood : float
            Log likelihood of the Gaussian mixture given X.
        """
        return np.average(self.score_samples(X), weights=self.vol)


class GaussianMixtureWithPrior(WeightedGaussianMixture):
    def __init__(
        self,
        gmmref,
        kappa=0.0,
        nu=0.0,
        zeta=0.0,
        prior_type="semi",  # semi or full
        update_covariances=False,
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

    def fit(self, X, y=None):
        """
        MODIFIED FROM SCIKIT-LEARN FOR MAP ESTIMATE WITH PRIOR FOR EACH CLUSTER
        Estimate model parameters with the EM algorithm.
        The method fit the model `n_init` times and set the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a `ConvergenceWarning` is raised.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        self
        """
        if self.verbose:
            print("modified from scikit-learn")

        X = _check_X(X, self.n_components)
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)
                self.lower_bound_ = -np.infty

            for n_iter in range(self.max_iter):
                prev_lower_bound = self.lower_bound_

                log_prob_norm, log_resp = self._e_step(X)

                if self.fixed_membership is not None:
                    # force responsibilities
                    aux = -(np.inf) * np.ones(
                        (self.fixed_membership.shape[0], self.n_components)
                    )
                    aux[np.arange(len(aux)), self.fixed_membership[:, 1]] = 0.0
                    log_resp[self.fixed_membership[:, 0]] = aux

                self._m_step(X, log_resp)

                UpdateGaussianMixtureModel(
                    self,
                    self.gmmref,
                    zeta=self.zeta,
                    nu=self.nu,
                    kappa=self.kappa,
                    verbose=self.verbose,
                    update_covariances=self.update_covariances,
                    prior_type=self.prior_type,
                )

                if self.fixed_membership is not None and self.weights_.ndim == 2:
                    # force local weights
                    aux = np.zeros((self.fixed_membership.shape[0], self.n_components))
                    aux[np.arange(len(aux)), self.fixed_membership[:, 1]] = 1
                    self.weights_[self.fixed_membership[:, 0]] = aux

                self.lower_bound_ = self._compute_lower_bound(log_resp, log_prob_norm)

                change = self.lower_bound_ - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end(self.lower_bound_)

            if self.lower_bound_ > max_lower_bound:
                max_lower_bound = self.lower_bound_
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
        self.last_step_change = change

        return self


class GaussianMixtureMarkovRandomField(GaussianMixtureWithPrior):
    def __init__(
        self,
        gmmref,
        kappa=0.0,
        nu=0.0,
        zeta=0.0,
        kdtree=None,
        indexneighbors=None,
        prior_type="semi",  # semi or full
        update_covariances=False,
        fixed_membership=None,
        T=12.0,
        kneighbors=0,
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
        anisotropy=None,
        index_anisotropy=None,  # Dictionary with anisotropy and index
        index_kdtree=None,  # List of KDtree
        # **kwargs
    ):

        super(GaussianMixtureMarkovRandomField, self).__init__(
            gmmref=gmmref,
            kappa=kappa,
            nu=nu,
            zeta=zeta,
            prior_type=prior_type,
            update_covariances=update_covariances,
            fixed_membership=fixed_membership,
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
            # **kwargs
        )
        # setKwargs(self, **kwargs)
        self.kneighbors = kneighbors
        self.T = T

        self.anisotropy = anisotropy
        if self.mesh.gridCC.ndim == 1:
            xyz = np.c_[self.mesh.gridCC]
        elif self.anisotropy is not None:
            xyz = self.anisotropy.dot(self.mesh.gridCC.T).T
        else:
            xyz = self.mesh.gridCC
        if self.actv is None:
            self.xyz = xyz
        else:
            self.xyz = xyz[self.actv]
        if kdtree is None:
            print("Computing KDTree, it may take several minutes.")
            self.kdtree = spatial.KDTree(self.xyz)
        else:
            self.kdtree = kdtree
        if indexneighbors is None:
            print("Computing neighbors, it may take several minutes.")
            _, self.indexneighbors = self.kdtree.query(self.xyz, k=self.kneighbors + 1)
        else:
            self.indexneighbors = indexneighbors

        self.indexpoint = copy.deepcopy(self.indexneighbors)
        self.index_anisotropy = index_anisotropy
        self.index_kdtree = index_kdtree
        if self.index_anisotropy is not None and self.mesh.gridCC.ndim != 1:

            self.unitxyz = []
            for i, anis in enumerate(self.index_anisotropy["anisotropy"]):
                self.unitxyz.append((anis).dot(self.xyz.T).T)

            if self.index_kdtree is None:
                self.index_kdtree = []
                print(
                    "Computing rock unit specific KDTree, it may take several minutes."
                )
                for i, anis in enumerate(self.index_anisotropy["anisotropy"]):
                    self.index_kdtree.append(spatial.KDTree(self.unitxyz[i]))

            print(
                "Computing new neighbors based on rock units, it may take several minutes."
            )
            for i, unitindex in enumerate(self.index_anisotropy["index"]):
                _, self.indexpoint[unitindex] = self.index_kdtree[i].query(
                    self.unitxyz[i][unitindex], k=self.kneighbors + 1
                )

    def computeG(self, z, w):
        logG = (self.T / (2.0 * (self.kneighbors + 1))) * (
            (z[self.indexpoint] + w[self.indexpoint]).sum(axis=1)
        )
        return logG

    def _m_step(self, X, log_resp):
        """M step.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        _, self.means_, self.covariances_ = self._estimate_gaussian_parameters(
            X, self.mesh, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        # self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

        logweights = logsumexp(
            np.c_[[log_resp, self.computeG(np.exp(log_resp), self.weights_)]], axis=0
        )
        logweights = logweights - logsumexp(logweights, axis=1, keepdims=True)

        self.weights_ = np.exp(logweights)

    def _check_weights(self, weights, n_components, n_samples):
        """Check the user provided 'weights'.
        Parameters
        ----------
        weights : array-like, shape (n_components,)
            The proportions of components of each mixture.
        n_components : int
            Number of components.
        Returns
        -------
        weights : array, shape (n_components,)
        """
        weights = check_array(weights, dtype=[np.float64, np.float32], ensure_2d=True)
        _check_shape(weights, (n_samples, n_components), "weights")

        return weights

    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        n_samples, n_features = X.shape
        if self.covariance_type not in ["spherical", "tied", "diag", "full"]:
            raise ValueError(
                "Invalid value for 'covariance_type': %s "
                "'covariance_type' should be in "
                "['spherical', 'tied', 'diag', 'full']" % self.covariance_type
            )

        if self.weights_init is not None:
            self.weights_init = self._check_weights(
                self.weights_init, self.n_components, n_samples,
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

    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        resp : array-like, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances = self._estimate_gaussian_parameters(
            X, self.mesh, resp, self.reg_covar, self.covariance_type
        )
        weights /= n_samples

        self.weights_ = (
            weights * np.ones((n_samples, self.n_components))
            if self.weights_init is None
            else self.weights_init
        )
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


class GaussianMixtureWithMapping(GaussianMixture):
    def __init__(
        self,
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

        super(GaussianMixtureWithMapping, self).__init__(
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
        """Initialization of the Gaussian mixture parameters.
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
        """Estimate the log Gaussian probability.
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
        return self._estimate_log_gaussian_prob(
            X,
            self.means_,
            self.precisions_cholesky_,
            self.covariance_type,
            self.cluster_mapping,
        )

    def _estimate_gaussian_parameters(self, X, resp, reg_covar, covariance_type):

        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        # stupid lazy piece of junk code to get the shapes right
        means = np.dot(resp.T, X) / nk[:, np.newaxis]
        covariances = {
            "full": _estimate_gaussian_covariances_full,
            "tied": _estimate_gaussian_covariances_tied,
            "diag": _estimate_gaussian_covariances_diag,
            "spherical": _estimate_gaussian_covariances_spherical,
        }[covariance_type](resp, X, nk, means, reg_covar)
        # The actual calculation
        for k in range(means.shape[0]):
            means[k] = (
                np.dot(resp.T, self.cluster_mapping[k] * X) / nk[:, np.newaxis]
            )[k]
        for k in range(means.shape[0]):
            covariances[k] = (
                {
                    "full": _estimate_gaussian_covariances_full,
                    "tied": _estimate_gaussian_covariances_tied,
                    "diag": _estimate_gaussian_covariances_diag,
                    "spherical": _estimate_gaussian_covariances_spherical,
                }[covariance_type](
                    resp, self.cluster_mapping[k] * X, nk, means, reg_covar
                )
            )[k]
        return nk, means, covariances

    # TODOs: Still not working because of inverse mapping not implemented
    def sample(self, n_samples=1):
        """Generate random samples from the fitted Gaussian distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample

        y : array, shape (nsamples,)
            Component labels

        """
        self._check_is_fitted()

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
        """M step.
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


class GaussianMixtureWithMappingWithPrior(GaussianMixtureWithPrior):
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
        update_covariances=False,
        fixed_membership=None,
    ):

        if cluster_mapping is None:
            self.cluster_mapping = [IdentityMap() for i in range(n_components)]
        else:
            self.cluster_mapping = cluster_mapping

        super(GaussianMixtureWithMappingWithPrior, self).__init__(
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
        """Initialization of the Gaussian mixture parameters.
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
        """Estimate the log Gaussian probability.
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

        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        # stupid lazy piece of junk code to get the shapes right
        means = np.dot(resp.T, X) / nk[:, np.newaxis]
        covariances = {
            "full": _estimate_gaussian_covariances_full,
            "tied": _estimate_gaussian_covariances_tied,
            "diag": _estimate_gaussian_covariances_diag,
            "spherical": _estimate_gaussian_covariances_spherical,
        }[covariance_type](resp, X, nk, means, reg_covar)
        # The actual calculation
        for k in range(means.shape[0]):
            means[k] = (
                np.dot(resp.T, self.cluster_mapping[k] * X) / nk[:, np.newaxis]
            )[k]
        for k in range(means.shape[0]):
            covariances[k] = (
                {
                    "full": _estimate_gaussian_covariances_full,
                    "tied": _estimate_gaussian_covariances_tied,
                    "diag": _estimate_gaussian_covariances_diag,
                    "spherical": _estimate_gaussian_covariances_spherical,
                }[covariance_type](
                    resp, self.cluster_mapping[k] * X, nk, means, reg_covar
                )
            )[k]
        return nk, means, covariances

    def _m_step(self, X, log_resp):
        """M step.
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


def GibbsSampling_PottsDenoising(
    mesh,
    minit,
    log_univar,
    Pottmatrix,
    indActive=None,
    neighbors=8,
    norm=2,
    weighted_selection=True,
    compute_score=False,
    maxit=None,
    verbose=False,
):

    denoised = copy.deepcopy(minit)
    # Compute Tree for neighbors finding
    if mesh.dim == 1:
        GRIDCC = mkvc(mesh.gridCC, numDims=2)
    else:
        GRIDCC = mesh.gridCC
    if indActive is None:
        pass
    else:
        GRIDCC = GRIDCC[indActive]
    tree = spatial.KDTree(GRIDCC)
    n_components = log_univar.shape[1]

    if weighted_selection or compute_score:
        _, idx = tree.query(GRIDCC, k=neighbors + 1, p=norm)
        idx = idx[:, 1:]

    if weighted_selection:
        logprobnoise = -np.sum(
            np.r_[[Pottmatrix[minit[j], minit[idx[j]]] for j in range(len(minit))]],
            axis=1,
        )
        idxmin = np.where(logprobnoise == logprobnoise.min())
        logprobnoise[idxmin] = -np.inf
        probnoise = np.exp(logprobnoise - logsumexp(logprobnoise))
        choice = np.arange(len(minit))
        if maxit is None:
            maxit = int(
                (1 + len(GRIDCC) - len(idxmin[0]))
                * np.log(1 + len(GRIDCC) - len(idxmin[0]))
            )
            if verbose:
                print("max iterations: ", maxit)

    if compute_score:
        logprob_obj = []
        # Compute logprobability of the model, should increase
        unnormlogprob = np.sum(
            np.r_[[log_univar[i, denoised[i]] for i in range(len(denoised))]]
        ) + np.sum(
            np.r_[
                [
                    Pottmatrix[denoised[j], denoised[idx[j]]]
                    for j in range(len(denoised))
                ]
            ]
        )
        logprob_obj.append(unnormlogprob)

    if maxit is None:
        maxit = int((mesh.nC) * np.log(mesh.nC))
        if verbose:
            print("max iterations: ", maxit)

    for i in range(maxit):
        # select random point and neighbors
        if weighted_selection:
            j = np.random.choice(choice, p=probnoise)
            idxj = idx[j]
        else:
            j = np.random.randint(mesh.nC)
            if not weighted_selection or compute_score:
                _, idxj = tree.query(mesh.gridCC[j], k=neighbors, p=norm)

        # compute Probability
        postlogprob = np.zeros_like(log_univar[j])
        for k in range(n_components):
            postlogprob[k] = log_univar[j][k] + np.sum(
                [Pottmatrix[k, denoised[idc]] for idc in idxj]
            )
        postprobj = np.exp(postlogprob - logsumexp(postlogprob))

        denoised[j] = np.random.choice(np.arange(n_components), p=postprobj)

        if compute_score:
            # Compute logprobability of the model, should increase
            unnormlogprob = np.sum(
                np.r_[[log_univar[i, denoised[i]] for i in range(len(denoised))]]
            ) + np.sum(
                np.r_[
                    [
                        Pottmatrix[denoised[j], denoised[idx[j]]]
                        for j in range(len(denoised))
                    ]
                ]
            )
            logprob_obj.append(unnormlogprob)

        if weighted_selection:
            # Update the probability of being noisy
            logprobnoise[j] = -np.sum(np.r_[Pottmatrix[denoised[j], denoised[idx[j]]]])
            probnoise = np.exp(logprobnoise - logsumexp(logprobnoise))

    if compute_score and weighted_selection:
        return [denoised, probnoise, logprob_obj]

    elif not (compute_score or weighted_selection):
        return [denoised]

    elif compute_score:
        return [denoised, logprob_obj]

    elif weighted_selection:
        return [denoised, probnoise]


def ICM_PottsDenoising(
    mesh,
    minit,
    log_univar,
    Pottmatrix,
    indActive=None,
    neighbors=8,
    norm=2,
    weighted_selection=True,
    compute_score=False,
    maxit=None,
    verbose=True,
):

    denoised = copy.deepcopy(minit)
    # Compute Tree for neighbors finding
    if mesh.dim == 1:
        GRIDCC = mkvc(mesh.gridCC, numDims=2)
    else:
        GRIDCC = mesh.gridCC
    if indActive is None:
        pass
    else:
        GRIDCC = GRIDCC[indActive]
    tree = spatial.KDTree(GRIDCC)
    n_components = log_univar.shape[1]

    if weighted_selection or compute_score:
        _, idx = tree.query(GRIDCC, k=neighbors + 1, p=norm)
        idx = idx[:, 1:]

    if weighted_selection:
        logprobnoise = -np.sum(
            np.r_[[Pottmatrix[minit[j], minit[idx[j]]] for j in range(len(minit))]],
            axis=1,
        )
        idxmin = np.where(logprobnoise == logprobnoise.min())
        # logprobnoise[idxmin] = -np.inf
        probnoise = np.exp(logprobnoise - logsumexp(logprobnoise))
        probnoise = probnoise / np.sum(probnoise)
        choice = np.arange(len(minit))
        if maxit is None:
            maxit = int(
                (1 + len(GRIDCC) - len(idxmin[0]))
                * np.log(1 + len(GRIDCC) - len(idxmin[0]))
            )
            if verbose:
                print("max iterations: ", maxit)

    if compute_score:
        logprob_obj = []
        # Compute logprobability of the model, should increase
        unnormlogprob = np.sum(
            np.r_[[log_univar[i, denoised[i]] for i in range(len(denoised))]]
        ) + np.sum(
            np.r_[
                [
                    Pottmatrix[denoised[j], denoised[idx[j]]]
                    for j in range(len(denoised))
                ]
            ]
        )
        logprob_obj.append(unnormlogprob)

    if maxit is None:
        maxit = int((mesh.nC) * np.log(mesh.nC))
        if verbose:
            print("max iterations: ", maxit)

    for i in range(maxit):
        # select random point and neighbors
        if weighted_selection:
            j = np.random.choice(choice, p=probnoise)
            idxj = idx[j]
        else:
            j = np.random.randint(mesh.nC)
            if not weighted_selection or compute_score:
                _, idxj = tree.query(mesh.gridCC[j], k=neighbors, p=norm)

        # compute Probability
        postlogprob = np.zeros(n_components)
        for k in range(n_components):
            postlogprob[k] = log_univar[j][k] + np.sum(
                [Pottmatrix[k, denoised[idc]] for idc in idxj]
            )
        postprobj = np.exp(postlogprob - logsumexp(postlogprob))

        denoised[j] = np.argmax(postprobj)

        if compute_score:
            # Compute logprobability of the model, should increase
            unnormlogprob = np.sum(
                np.r_[[log_univar[i, denoised[i]] for i in range(len(denoised))]]
            ) + np.sum(
                np.r_[
                    [
                        Pottmatrix[denoised[j], denoised[idx[j]]]
                        for j in range(len(denoised))
                    ]
                ]
            )
            logprob_obj.append(unnormlogprob)

        if weighted_selection:
            # Update the probability of being noisy
            logprobnoise[j] = -np.sum(np.r_[Pottmatrix[denoised[j], denoised[idx[j]]]])
            probnoise = np.exp(logprobnoise - logsumexp(logprobnoise))
            probnoise = probnoise / np.sum(probnoise)

    if compute_score and weighted_selection:
        return [denoised, probnoise, logprob_obj]

    elif not (compute_score or weighted_selection):
        return [denoised]

    elif compute_score:
        return [denoised, logprob_obj]

    elif weighted_selection:
        return [denoised, probnoise]
