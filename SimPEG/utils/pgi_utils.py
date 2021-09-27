import numpy as np
import copy
from scipy.stats import multivariate_normal
from scipy import spatial, linalg
from scipy.special import logsumexp
from scipy.sparse import diags
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
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
from sklearn.mixture._base import check_random_state, ConvergenceWarning
import warnings
from .mat_utils import mkvc
from ..maps import IdentityMap, Wires
from ..regularization import (
    SimplePGI,
    Simple,
    PGI,
    Tikhonov,
    SimplePGIwithRelationships,
)


def make_SimplePGI_regularization(
    mesh,
    gmmref,
    gmm=None,
    wiresmap=None,
    maplist=None,
    cell_weights_list=None,
    approx_gradient=True,
    approx_eval=True,
    alpha_s=1.0,
    alpha_x=1.0,
    alpha_y=1.0,
    alpha_z=1.0,
    alpha_xx=0.0,
    alpha_yy=0.0,
    alpha_zz=0.0,
    **kwargs
):
    """
    Create a complete SimplePGI regularization term ComboObjectiveFunction with all
    necessary smallness and smoothness terms for any number of physical properties
    and associated mapping.

    Parameters
    ----------

    :param TensorMesh or TreeMesh mesh: TensorMesh or Treemesh object, used to weights
                        the physical properties by cell volumes when updating the
                        Gaussian Mixture Model (GMM)
    :param WeightedGaussianMixture gmmref: reference GMM.
    :param WeightedGaussianMixture gmm: Initial GMM. If not provided, gmmref is used.
    :param Wires wiresmap: Wires map to obtain the various physical properties from the model.
                        Optional for single physical property inversion. Required for multi-
                        physical properties inversion.
    :param list maplist: List of mapping for each physical property. Default is the IdentityMap for all.
    :param list cell_weights_list: list of numpy.ndarray for the cells weight to apply to each physical property.
    :param boolean approx_gradient: use the PGI least-squares approximation of the full nonlinear regularizer
                        for computing the regularizer gradient. Default is True.
    :param boolean approx_eval: use the PGI least-squares approximation of the full nonlinear regularizer
                        for computing the value of the regularizer. Default is True.
    :param float alpha_s: alpha_s multiplier for the PGI smallness.
    :param float or numpy.ndarray alpha_x: alpha_x multiplier for the 1st-derivative
                        Smoothness terms in X-direction for each physical property.
    :param float or numpy.ndarray alpha_y: alpha_y multiplier for the 1st-derivative
                        Smoothness terms in Y-direction for each physical property.
    :param float or numpy.ndarray alpha_z: alpha_z multiplier for the 1st-derivative
                        Smoothness terms in Z-direction for each physical property.
    :param float or numpy.ndarray alpha_x: alpha_x multiplier for the 2nd-derivatibe
                        Smoothness terms in X-direction for each physical property.
    :param float or numpy.ndarray alpha_y: alpha_y multiplier for the 2nd-derivatibe
                        Smoothness terms in Y-direction for each physical property.
    :param float or numpy.ndarray alpha_z: alpha_z multiplier for the 2nd-derivatibe
                        Smoothness terms in Z-direction for each physical property.


    Returns
    -------

    :param SimPEG.objective_function.ComboObjectiveFunction reg: Full regularization with simplePGIsmallness
                        and smoothness terms for all physical properties in all direction.
    """

    if wiresmap is None:
        wrmp = Wires(("m", mesh.nC))
    else:
        wrmp = wiresmap

    if maplist is None:
        mplst = [IdentityMap(mesh) for maps in wrmp.maps]
    else:
        mplst = maplist

    if cell_weights_list is None:
        clwhtlst = [Identity() for maps in wrmp.maps]
    else:
        clwhtlst = cell_weights_list

    reg = SimplePGI(
        mesh=mesh,
        gmmref=gmmref,
        gmm=gmm,
        wiresmap=wiresmap,
        maplist=maplist,
        approx_gradient=approx_gradient,
        approx_eval=approx_eval,
        alpha_s=alpha_s,
        alpha_x=0.0,
        alpha_y=0.0,
        alpha_z=0.0,
        **kwargs
    )

    if cell_weights_list is not None:
        reg.objfcts[0].cell_weights = np.hstack(clwhtlst)

    if isinstance(alpha_x, float):
        alph_x = alpha_x * np.ones(len(wrmp.maps))
    else:
        alph_x = alpha_x

    if isinstance(alpha_y, float):
        alph_y = alpha_y * np.ones(len(wrmp.maps))
    else:
        alph_y = alpha_y

    if isinstance(alpha_z, float):
        alph_z = alpha_z * np.ones(len(wrmp.maps))
    else:
        alph_z = alpha_z

    for i, (wire, maps) in enumerate(zip(wrmp.maps, mplst)):
        reg += Simple(
            mesh=mesh,
            mapping=maps * wire[1],
            alpha_s=0.0,
            alpha_x=alph_x[i],
            alpha_y=alph_y[i],
            alpha_z=alph_z[i],
            cell_weights=clwhtlst[i],
            **kwargs
        )

    return reg


def make_PGI_regularization(
    mesh,
    gmmref,
    gmm=None,
    wiresmap=None,
    maplist=None,
    cell_weights_list=None,
    approx_gradient=True,
    approx_eval=True,
    alpha_s=1.0,
    alpha_x=1.0,
    alpha_y=1.0,
    alpha_z=1.0,
    alpha_xx=0.0,
    alpha_yy=0.0,
    alpha_zz=0.0,
    **kwargs
):
    """
    Create a complete PGI regularization term ComboObjectiveFunction with all
    necessary smallness and smoothness terms for any number of physical properties
    and associated mapping.

    Parameters
    ----------

    :param TensorMesh or TreeMesh mesh: TensorMesh or Treemesh object, used to weights
                        the physical properties by cell volumes when updating the
                        Gaussian Mixture Model (GMM)
    :param WeightedGaussianMixture gmmref: reference GMM.
    :param WeightedGaussianMixture gmm: Initial GMM. If not provided, gmmref is used.
    :param Wires wiresmap: Wires map to obtain the various physical properties from the model.
                        Optional for single physical property inversion. Required for multi-
                        physical properties inversion.
    :param list maplist: List of mapping for each physical property. Default is the IdentityMap for all.
    :param list cell_weights_list: list of numpy.ndarray for the cells weight to apply to each physical property.
    :param boolean approx_gradient: use the PGI least-squares approximation of the full nonlinear regularizer
                        for computing the regularizer gradient. Default is True.
    :param boolean approx_eval: use the PGI least-squares approximation of the full nonlinear regularizer
                        for computing the value of the regularizer. Default is True.
    :param float alpha_s: alpha_s multiplier for the PGI smallness.
    :param float or numpy.ndarray alpha_x: alpha_x multiplier for the 1st-derivative
                        Smoothness terms in X-direction for each physical property.
    :param float or numpy.ndarray alpha_y: alpha_y multiplier for the 1st-derivative
                        Smoothness terms in Y-direction for each physical property.
    :param float or numpy.ndarray alpha_z: alpha_z multiplier for the 1st-derivative
                        Smoothness terms in Z-direction for each physical property.
    :param float or numpy.ndarray alpha_x: alpha_x multiplier for the 2nd-derivatibe
                        Smoothness terms in X-direction for each physical property.
    :param float or numpy.ndarray alpha_y: alpha_y multiplier for the 2nd-derivatibe
                        Smoothness terms in Y-direction for each physical property.
    :param float or numpy.ndarray alpha_z: alpha_z multiplier for the 2nd-derivatibe
                        Smoothness terms in Z-direction for each physical property.


    Returns
    -------

    :param SimPEG.objective_function.ComboObjectiveFunction reg: Full regularization with PGIsmallness
                        and smoothness terms for all physical properties in all direction.
    """

    if wiresmap is None:
        wrmp = Wires(("m", mesh.nC))
    else:
        wrmp = wiresmap

    if maplist is None:
        mplst = [IdentityMap(mesh) for maps in wrmp.maps]
    else:
        mplst = maplist

    if cell_weights_list is None:
        clwhtlst = [Identity() for maps in wrmp.maps]
    else:
        clwhtlst = cell_weights_list

    reg = PGI(
        mesh=mesh,
        gmmref=gmmref,
        gmm=gmm,
        wiresmap=wiresmap,
        maplist=maplist,
        approx_gradient=approx_gradient,
        approx_eval=approx_eval,
        alpha_s=alpha_s,
        alpha_x=0.0,
        alpha_y=0.0,
        alpha_z=0.0,
        **kwargs
    )

    if cell_weights_list is not None:
        reg.objfcts[0].cell_weights = np.hstack(clwhtlst)

    if isinstance(alpha_x, float):
        alph_x = alpha_x * np.ones(len(wrmp.maps))
    else:
        alph_x = alpha_x

    if isinstance(alpha_y, float):
        alph_y = alpha_y * np.ones(len(wrmp.maps))
    else:
        alph_y = alpha_y

    if isinstance(alpha_z, float):
        alph_z = alpha_z * np.ones(len(wrmp.maps))
    else:
        alph_z = alpha_z

    for i, (wire, maps) in enumerate(zip(wrmp.maps, mplst)):
        reg += Tikhonov(
            mesh=mesh,
            mapping=maps * wire[1],
            alpha_s=0.0,
            alpha_x=alph_x[i],
            alpha_y=alph_y[i],
            alpha_z=alph_z[i],
            cell_weights=clwhtlst[i],
            **kwargs
        )

    return reg


def make_SimplePGIwithRelationships_regularization(
    mesh,
    gmmref,
    gmm=None,
    wiresmap=None,
    maplist=None,
    cell_weights_list=None,
    approx_gradient=True,
    approx_eval=True,
    alpha_s=1.0,
    alpha_x=1.0,
    alpha_y=1.0,
    alpha_z=1.0,
    alpha_xx=0.0,
    alpha_yy=0.0,
    alpha_zz=0.0,
    **kwargs
):
    """
    Create a complete PGI, with nonlinear relationships, regularization term ComboObjectiveFunction with all
    necessary smallness and smoothness terms for any number of physical properties
    and associated mapping.

    Parameters
    ----------

    :param TensorMesh or TreeMesh mesh: TensorMesh or Treemesh object, used to weights
                        the physical properties by cell volumes when updating the
                        Gaussian Mixture Model (GMM)
    :param WeightedGaussianMixture gmmref: reference GMM.
    :param WeightedGaussianMixture gmm: Initial GMM. If not provided, gmmref is used.
    :param Wires wiresmap: Wires map to obtain the various physical properties from the model.
                        Optional for single physical property inversion. Required for multi-
                        physical properties inversion.
    :param list maplist: List of mapping for each physical property. Default is the IdentityMap for all.
    :param list cell_weights_list: list of numpy.ndarray for the cells weight to apply to each physical property.
    :param boolean approx_gradient: use the PGI least-squares approximation of the full nonlinear regularizer
                        for computing the regularizer gradient. Default is True.
    :param boolean approx_eval: use the PGI least-squares approximation of the full nonlinear regularizer
                        for computing the value of the regularizer. Default is True.
    :param float alpha_s: alpha_s multiplier for the PGI smallness.
    :param float or numpy.ndarray alpha_x: alpha_x multiplier for the 1st-derivative
                        Smoothness terms in X-direction for each physical property.
    :param float or numpy.ndarray alpha_y: alpha_y multiplier for the 1st-derivative
                        Smoothness terms in Y-direction for each physical property.
    :param float or numpy.ndarray alpha_z: alpha_z multiplier for the 1st-derivative
                        Smoothness terms in Z-direction for each physical property.
    :param float or numpy.ndarray alpha_x: alpha_x multiplier for the 2nd-derivatibe
                        Smoothness terms in X-direction for each physical property.
    :param float or numpy.ndarray alpha_y: alpha_y multiplier for the 2nd-derivatibe
                        Smoothness terms in Y-direction for each physical property.
    :param float or numpy.ndarray alpha_z: alpha_z multiplier for the 2nd-derivatibe
                        Smoothness terms in Z-direction for each physical property.


    Returns
    -------

    :param SimPEG.objective_function.ComboObjectiveFunction reg: Full regularization with
                        SimplePGIwithNonlinearRelationshipsSmallness and smoothness terms
                        for all physical properties in all direction.
    """

    if wiresmap is None:
        wrmp = Wires(("m", mesh.nC))
    else:
        wrmp = wiresmap

    if maplist is None:
        mplst = [IdentityMap(mesh) for maps in wrmp.maps]
    else:
        mplst = maplist

    if cell_weights_list is None:
        clwhtlst = [Identity() for maps in wrmp.maps]
    else:
        clwhtlst = cell_weights_list

    reg = SimplePGIwithRelationships(
        mesh=mesh,
        gmmref=gmmref,
        gmm=gmm,
        wiresmap=wiresmap,
        maplist=maplist,
        approx_gradient=approx_gradient,
        approx_eval=approx_eval,
        alpha_s=alpha_s,
        alpha_x=0.0,
        alpha_y=0.0,
        alpha_z=0.0,
        **kwargs
    )

    if cell_weights_list is not None:
        reg.objfcts[0].cell_weights = np.hstack(clwhtlst)

    if isinstance(alpha_x, float):
        alph_x = alpha_x * np.ones(len(wrmp.maps))
    else:
        alph_x = alpha_x

    if isinstance(alpha_y, float):
        alph_y = alpha_y * np.ones(len(wrmp.maps))
    else:
        alph_y = alpha_y

    if isinstance(alpha_z, float):
        alph_z = alpha_z * np.ones(len(wrmp.maps))
    else:
        alph_z = alpha_z

    for i, (wire, maps) in enumerate(zip(wrmp.maps, mplst)):
        reg += Simple(
            mesh=mesh,
            mapping=maps * wire[1],
            alpha_s=0.0,
            alpha_x=alph_x[i],
            alpha_y=alph_y[i],
            alpha_z=alph_z[i],
            cell_weights=clwhtlst[i],
            **kwargs
        )

    return reg


###############################################################################
# Disclaimer: the following classes built upon the GaussianMixture class      #
# from Scikit-Learn. New functionalitie are added, as well as modifications to#
# existing functions, to serve the purposes pursued within SimPEG.            #
# This use is allowed by the Scikit-Learn licensing (BSD-3-Clause License)    #
# and we are grateful for their contributions to the open-source community.   #                                                   #
###############################################################################


class WeightedGaussianMixture(GaussianMixture):
    """
    This class upon the GaussianMixture class from Scikit-Learn.
    Two main modifications:
        1: Each sample/observation is given a weight, the volume of the
        corresponding discretize.BaseMesh cell, when fitting the
        Gaussian Mixture Model (GMM). More volume gives more importance, ensuing a
        mesh-free evaluation of the clusters of the geophysical model.
        2: When set manually, the proportions can be set either globally (normal behavior)
        or cell-by-cell (improvements)

    Disclaimer: this class built upon the GaussianMixture class from Scikit-Learn.
    New functionalitie are added, as well as modifications to
    existing functions, to serve the purposes pursued within SimPEG.
    This use is allowed by the Scikit-Learn licensing (BSD-3-Clause License)
    and we are grateful for their contributions to the open-source community.

    Addtional parameters to provide, compared to sklearn.mixture.gaussian_mixture:

    :param discretize.BaseMesh (TensorMesh or QuadTree or Octree) mesh: the volume
        of the cells give each sample/observations its weight in the fitting proces
    :param numpy.ndarry actv: (optional) active cells index
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
        """
        Use this function after setting covariances manually.
        Compute the precisions matrices and their Cholesky decomposition.
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
        """
        Use this function after setting precisions matrices manually.
        Compute the precisions matrices and their Cholesky decomposition.
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
        """
        order clusters by decreasing weights

        PARAMETERS
        ----------
        :param boolean outputindex: if True, return the sorting index

        RETURN
        ------
        :return np.ndarray indx: sorting index
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
        weights : array-like, shape (n_components,) or (n_samples, n_components_)
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
        """
        [modified from Scikit-Learn.mixture.gaussian_mixture]
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
        """
        [modified from Scikit-Learn.mixture.gaussian_mixture]
        Compute the per-sample average log-likelihood of the given data X.
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
                prec_chol_mat = np.eye(n_components) * prec_chol
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
        """
        [New function, modified from Scikit-Learn.mixture.gaussian_mixture.score_samples]
        Compute the weighted log probabilities for each sample.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log probabilities of each data point in X.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)

        return logsumexp(self._estimate_weighted_log_prob_with_sensW(X, sensW), axis=1)


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

    :param numpy.ndarray kappa: strength of the confidence in the prior means
    :param numpy.ndarry nu: strength of the confidence in the prior covariances
    :param numpy.ndarry zeta: strength of the confidence in the prior proportions
    :param str prior_type: "semi": semi-conjugate prior, the means and covariances priors are indepedent
                           "full": conjugate prior, the means and covariances priors are inter-depedent
    :param boolean update_covariances: True: semi or conjugate prior by averaging the covariances
                                    False: alternative (not conjugate) prior: average the precisions instead
    :param numpy.ndarray, dtype=int, shape=(index of the fixed cell, lithology index) fixed_membership:
        a 2d numpy.ndarray to fix the membership to a chosen lithology of particular cells.
        The first column contains the numeric index of the cells, the second column the respective lithology index.
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
        """
        Arrange the clusters of gmm in the same order as those of gmmref,
        based on their relative similarities and priorizing first the most proeminent
        clusters (highest proportions)

        PARAMETERS
        ----------

        :param GaussianMixture gmm: Gaussian Mixture Model (GMM) to reorder.
        :param GaussianMixture gmmref: reference GMM.
        :param boolean outputindex: if True, return the ordering index for the clusters of gmm.
                                Default is False.

        RETURN
        ------

        :param numpy.ndarray indx: Optional, return the ordering index for the clusters of gmm

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
        """
        This function, inserted in the fit function, modify the Maximum Likelyhood Estimation (MLE)
        of the GMM's parameters to a Maximum A Posteriori (MAP) estimation.
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
        """
        [modified from Scikit-Learn for Maximum A Posteriori estimates (MAP)]
        Estimate model parameters with the MAP-EM algorithm.
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
                self.update_gmm_with_priors(debug=debug)

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


class GaussianMixtureWithNonlinearRelationships(WeightedGaussianMixture):
    """
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

    :param list cluster_mapping (n_components, ): list of mapping describing
        a nonlinear relationships between physical properties; one per cluster/unit.
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
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

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
    """
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

    :param list cluster_mapping (n_components, ): list of mapping describing
        a nonlinear relationships between physical properties; one per cluster/unit.

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
