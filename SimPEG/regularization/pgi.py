from __future__ import annotations

import copy
import warnings

import numpy as np
import scipy.sparse as sp

from ..maps import IdentityMap, Wires
from ..objective_function import ComboObjectiveFunction
from ..utils import (
    Identity,
    deprecate_property,
    mkvc,
    sdiag,
    timeIt,
    validate_float,
    validate_ndarray_with_shape,
)
from .base import RegularizationMesh, Smallness, WeightedLeastSquares

###############################################################################
#                                                                             #
#            Petrophysically And Geologically Guided Regularization           #
#                                                                             #
###############################################################################


# Simple Petrophysical Regularization
#####################################


class PGIsmallness(Smallness):
    r"""Smallness regularization function for petrophysically guided inversion (PGI).

    ``PGIsmallness`` is used to recover models in which the physical property values are
    consistent with petrophysical information. ``PGIsmallness`` regularization assumes that
    the statistical distribution of physical property values defining the model is characterized
    by a Gaussian mixture model (GMM). That is, the physical property values for each specified
    geological unit are characterized by a separate multivariate Gaussian distribution,
    which are summed to define the GMM. ``PGIsmallness`` is generally combined with other
    regularization classes to form a complete regularization for the inverse problem; see
    :class:`PGI`.

    ``PGIsmallness`` can be implemented to invert for a single physical property or multiple
    physical properties, each of which are defined on a linear scale (e.g. density) or a log-scale
    (e.g. electrical conductivity). If the statistical distribution(s) of physical property values
    for each property type are known, the GMM can be constructed and left static throughout the
    inversion. Otherwise, the recovered model at each iteration is used to update the GMM.
    And the updated GMM is used to constrain the recovered model for the following iteration.

    Parameters
    ----------
    gmmref : simpeg.utils.WeightedGaussianMixture
        Reference Gaussian mixture model.
    gmm : None, simpeg.utils.WeightedGaussianMixture
        Set the Gaussian mixture model used to constrain the recovered physical property model.
        Can be left static throughout the inversion or updated using the
        :class:`.directives.PGI_UpdateParameters` directive. If ``None``, the
        :class:`.directives.PGI_UpdateParameters` directive must be used to ensure there
        is a Gaussian mixture model for the inversion.
    wiresmap : None, simpeg.maps.Wires
        Mapping from the model to the model parameters of each type.
        If ``None``, we assume only a single physical property type in the inversion.
    maplist : None, list of simpeg.maps
        Ordered list of mappings from model values to physical property values;
        one for each physical property. If ``None``, we assume a single physical property type
        in the regularization and an :class:`.maps.IdentityMap` from model values to physical
        property values.
    mesh : simpeg.regularization.RegularizationMesh, discretize.base.BaseMesh
        Mesh on which the regularization is discretized. Implemented for
        ``tensor``, ``QuadTree`` or ``Octree`` meshes.
    approx_gradient : bool
        If ``True``, use the L2-approximation of the gradient by assuming
        physical property values of different types are uncorrelated.
    approx_eval : bool
        If ``True``, use the L2-approximation evaluation of the smallness term by assuming
        physical property values of different types are uncorrelated.
    approx_hessian : bool
        Approximate the Hessian of the regularization function.
    non_linear_relationship : bool
        Whether relationships in the Gaussian mixture model are non-linear.

    Notes
    -----
    For one or more physical property types (e.g. conductivity, density, susceptibility),
    the ``PGIsmallness`` regularization function (objective function) is derived by setting a
    Gaussian mixture model (GMM) as the prior within a Baysian inversion scheme.
    For a comprehensive description, see
    (`Astic, et al 2019 <https://owncloud.eoas.ubc.ca/s/TMB3Jdr8ScqSPm7/download>`__;
    `Astic et al 2020 <https://owncloud.eoas.ubc.ca/s/PAxpHQt7CGk6zT4/download>`__).

    We let :math:`\Theta` store all of the means (:math:`\boldsymbol{\mu}`), covariances
    (:math:`\boldsymbol{\Sigma}`) and proportion constants (:math:`\boldsymbol{\gamma}`)
    defining the GMM. And let :math:`\mathbf{z}^\ast` define an membership array that
    extracts the GMM parameters for the most representative rock unit within each active cell
    in the :class:`RegularizationMesh`.

    When the ``approx_eval`` property is ``True``, we assume the physical property distributions of each geologic units
    are distinct (no significant overlap of their respective physical properties distribution). The GMM probability
    density value at any each point of the physical property space can then be approximated by the locally dominant
    Gaussian distribution. In this case, the PGI regularization function (objective function) can be expressed as a
    least-square:

    .. math::
        \phi (\mathbf{m}) &= \alpha_\text{pgi}
        \big | \mathbf{W} ( \Theta , \mathbf{z}^\ast ) \, (\mathbf{m} - \mathbf{m_{ref}}(\Theta, \mathbf{z}^\ast ) \, \Big \|^2
        &+ \sum_{j=x,y,z} \alpha_j \Big \| \mathbf{W_j G_j \, m} \, \Big \|^2 \\
        &+ \sum_{j=x,y,z} \alpha_{jj} \Big \| \mathbf{W_{jj} L_j \, m} \, \Big \|^2
        \;\;\;\;\;\;\;\; \big ( \textrm{optional} \big )

    where

        - :math:`\mathbf{m}` is the model,
        - :math:`\mathbf{m_{ref}}(\Theta, \mathbf{z}^\ast )` is the reference model, and
        - :math:`\mathbf{W}(\Theta , \mathbf{z}^\ast )` is a weighting matrix.

    For the full, non-approximated PGI regularization, please refer to
    (`Astic, et al 2019 <https://owncloud.eoas.ubc.ca/s/TMB3Jdr8ScqSPm7/download>`__;
    `Astic et al 2020 <https://owncloud.eoas.ubc.ca/s/PAxpHQt7CGk6zT4/download>`__).

    When the ``approx_eval`` property is ``True``, you may also set the ``approx_gradient`` and ``approx_hessian``
    properties to ``True`` so that the least-squares approximation is used to compute the gradient, as it is making the
    same assumptions about the GMM.

    ``PGIsmallness`` regularization can be used for models consisting of one or more physical
    property types. The ordering of the physical property types within the model is defined
    using the `wiresmap`. And the mapping from model parameter values to physical property
    values is specified with `maplist`. For :math:`K` physical property types, the model is
    an array vector of the form:

    .. math::
        \mathbf{m} = \begin{bmatrix} \mathbf{m}_1 \\ \mathbf{m}_2 \\ \vdots \\ \mathbf{m}_K \end{bmatrix}

    **Constructing the Reference Model and Weighting Matrix:**

    The reference model used in the regularization function is constructed by extracting the means
    :math:`\boldsymbol{\mu}` from the GMM using the membership array :math:`\mathbf{z}^\ast`.
    We represent this vector as:

    .. math::
        \mathbf{m_{ref}} (\Theta ,{\mathbf{z}^\ast}) = \boldsymbol{\mu}_{\mathbf{z}^\ast}

    To construct the weighting matrix, :math:`\mathbf{z}^\ast` is used to extract the covariances
    :math:`\boldsymbol{\Sigma}` for each cell. And the weighting matrix is given by:

    .. math::
        \mathbf{W}(\Theta ,{\mathbf{z}^\ast } ) = \boldsymbol{\Sigma}_{\mathbf{z^\ast}}^{\frac{-1}{2}} \,
        diag \big ( \mathbf{w} \big )

    **Updating the Gaussian Mixture Model:**

    When the GMM is set using the ``gmm`` property, the GMM remains static throughout the inversion.
    When the ``gmm`` property set as ``None``, the GMM is learned and updated after every model update.
    That is, we assume the GMM defined using the ``gmmref`` property is not completely representative
    of the physical property distributions for each rock unit, and we update the all of the means
    (:math:`\boldsymbol{\mu}`), covariances (:math:`\boldsymbol{\Sigma}`) and proportion constants
    (:math:`\boldsymbol{\gamma}`) defining the GMM :math:`\Theta`. This is done by solving:

    .. math::
        \max_\Theta \; \mathcal{P}(\Theta | \mathbf{m})

    using a MAP variation of the expectation-maximization clustering algorithm introduced in
    Dempster (et al. 1977).

    **Updating the Membership Array:**

    As the model (and GMM) are updated throughout the inversion, the rock unit considered most
    indicative of the geology within each cell is updated; which is represented by the membership
    array :math:`\mathbf{z}^\ast`. W. For the current GMM with means (:math:`\boldsymbol{\mu}`),
    covariances (:math:`\boldsymbol{\Sigma}`) and proportion constants (:math:`\boldsymbol{\gamma}`),
    we solve the following for each cell:

    .. math::
        z_i^\ast = \max_n \; \gamma_{i,n} \, \mathcal{N} (\mathbf{m}_i | \boldsymbol{\mu}_n , \boldsymbol{\Sigma}_n)

    where

        - :math:`\mathbf{m_i}` are the model values for cell :math:`i`,
        - :math:`\gamma_{i,n}` is the proportion for cell :math:`i` and rock unit :math:`n`
        - :math:`\boldsymbol{\mu}_n` are the mean property values for unit :math:`n`,
        - :math:`\boldsymbol{\Sigma}_n` are the covariances for unit :math:`n`, and
        - :math:`\mathcal{N}` represent the multivariate Gaussian distribution.

    """

    _multiplier_pair = "alpha_pgi"
    _maplist = None
    _wiresmap = None

    def __init__(
        self,
        gmmref,
        gmm=None,
        wiresmap=None,
        maplist=None,
        mesh=None,
        approx_gradient=True,  # L2 approximate of the gradients
        approx_eval=True,  # L2 approximate of the value
        approx_hessian=True,
        non_linear_relationships=False,
        **kwargs,
    ):
        self.gmmref = copy.deepcopy(gmmref)
        self.gmmref.order_clusters_GM_weight()
        self.approx_gradient = approx_gradient
        self.approx_eval = approx_eval
        self.approx_hessian = approx_hessian
        self.non_linear_relationships = non_linear_relationships
        self._gmm = copy.deepcopy(gmm)
        self.wiresmap = wiresmap
        self.maplist = maplist

        if "mapping" in kwargs:
            warnings.warn(
                f"Property 'mapping' of class {type(self)} cannot be set. "
                "Defaults to IdentityMap.",
                stacklevel=2,
            )
            kwargs.pop("mapping")

        weights = kwargs.pop("weights", None)

        super().__init__(mesh=mesh, mapping=IdentityMap(nP=self.shape[0]), **kwargs)

        # Save repetitive computations (see withmapping implementation)
        self._r_first_deriv = None
        self._r_second_deriv = None

        if weights is not None:
            if isinstance(weights, (np.ndarray, list)):
                weights = {"user_weights": np.r_[weights].flatten()}
            self.set_weights(**weights)

    def set_weights(self, **weights):
        """Adds (or updates) the specified weights.

        Parameters
        ----------
        **weights : key, numpy.ndarray
            Each keyword argument is added to the weights used the regularization object.
            They can be accessed with their keyword argument.
        """
        for key, values in weights.items():
            values = validate_ndarray_with_shape("weights", values, dtype=float)

            if values.shape[0] == self.regularization_mesh.nC:
                values = np.tile(values, len(self.wiresmap.maps))

            values = validate_ndarray_with_shape(
                "weights", values, shape=(self._nC_residual,), dtype=float
            )

            self._weights[key] = values

        self._W = None

    @property
    def gmm(self):
        """Gaussian mixture model.

        If set prior to inversion, the Gaussian mixture model can be left static throughout
        the inversion, or updated using the :class:`.directives.PGI_UpdateParameters` directive.
        If this property is not set prior to inversion, the
        :class:`.directives.PGI_UpdateParameters` directive must be used to ensure there
        is a Gaussian mixture model for the inversion.

        Returns
        -------
        simpeg.utils.WeightedGaussianMixture
            Gaussian mixture model used to constrain the recovered physical property model.
        """
        if getattr(self, "_gmm", None) is None:
            self._gmm = copy.deepcopy(self.gmmref)
        return self._gmm

    @gmm.setter
    def gmm(self, gm):
        if gm is not None:
            self._gmm = copy.deepcopy(gm)

    @property
    def shape(self):
        """Number of model parameters.

        Returns
        -------
        tuple of int
            Number of model parameters.
        """
        return (self.wiresmap.nP,)

    def membership(self, m):
        """Compute and return membership array for the model provided.

        The membership array stores the index of the rock unit most representative of each cell.
        For a Gaussian mixture model containing the means and covariances for model parameter
        types (physical property types) for all rock units, this method computes the membership
        array for the model `m` provided. For a description of the membership array, see the
        *Notes*  section within the :class:`PGIsmallness` documentation.

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray of float
            The model.

        Returns
        -------
        (n_active, ) numpy.ndarray of int
            The membership array.
        """
        modellist = self.wiresmap * m
        model = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
        return self.gmm.predict(model)

    def compute_quasi_geology_model(self):
        r"""Compute and return quasi geology model.

        For each active cell in the mesh, this method returns the mean values in the Gaussian
        mixture model for the most representative rock unit, given the current model. See the
        *Notes* section for a comprehensive description.

        Returns
        -------
        (n_param, ) numpy.ndarray
            The quasi geology physical property model.

        Notes
        -----
        Consider a Gaussian mixture model (GMM) for :math:`K` physical property types and
        :math:`N` rock units. The mean model parameter values for rock unit
        :math:`n \in \{ 1, \ldots , N \}` in the GMM is represented by a vector
        :math:`\boldsymbol{\mu}_n` of length :math:`K`. For each active cell in the mesh, the
        `compute_quasi_geology_model` method computes:

        .. math::
            g_i^ = \min_{\boldsymbol{\mu}_n} \big \| \mathbf{m}_i - \boldsymbol{\mu}_n \big \|^2

        where :math:`\mathbf{m}_i` are the model parameter values for cell :math:`i` for the
        current model. The ordering of the output vector :math:`\mathbf{g}` is the same as the
        model :math:`\mathbf{m}`.
        """
        # used once mref is built
        mreflist = self.wiresmap * self.reference_model
        mrefarray = np.c_[[a for a in mreflist]].T
        return np.c_[
            [((mrefarray - mean) ** 2).sum(axis=1) for mean in self.gmm.means_]
        ].argmin(axis=0)

    @property
    def non_linear_relationships(self):
        """Whether relationships in the Gaussian mixture model are non-linear.

        Returns
        -------
        bool
            Whether relationships in the Gaussian mixture model are non-linear.
        """
        return self._non_linear_relationships

    @non_linear_relationships.setter
    def non_linear_relationships(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError(
                "Input value for 'non_linear_relationships' must be of type 'bool'. "
                f"Provided {value} of type {type(value)}."
            )
        self._non_linear_relationships = value

    @property
    def wiresmap(self):
        """Mapping from the model to the model parameters of each type.

        Returns
        -------
        simpeg.maps.Wires
            Mapping from the model to the model parameters of each type.
        """
        if getattr(self, "_wiresmap", None) is None:
            self._wiresmap = Wires(("m", self.regularization_mesh.nC))
        return self._wiresmap

    @wiresmap.setter
    def wiresmap(self, wires):
        if self._maplist is not None and len(wires.maps) != len(self._maplist):
            raise Exception(
                f"Provided 'wiresmap' should have wires the len of 'maplist' {len(self._maplist)}."
            )

        if not isinstance(wires, Wires):
            raise ValueError(f"Attribure 'wiresmap' should be of type {Wires} or None.")

        self._wiresmap = wires

    @property
    def maplist(self):
        """Ordered list of mappings from model values to physical property values.

        Returns
        -------
        list of simpeg.maps
            Ordered list of mappings from model values to physical property values;
            one for each physical property.
        """
        if getattr(self, "_maplist", None) is None:
            self._maplist = [
                IdentityMap(nP=self.regularization_mesh.nC)
                for maps in self.wiresmap.maps
            ]
        return self._maplist

    @maplist.setter
    def maplist(self, maplist):
        if self._wiresmap is not None and len(maplist) != len(self._wiresmap.maps):
            raise Exception(
                f"Provided 'maplist' should be a list of maps equal to the 'wiresmap' list of len {len(self._maplist)}."
            )

        if not isinstance(maplist, (list, type(None))):
            raise ValueError(
                f"Attribute 'maplist' should be a list of maps or None.{type(maplist)} was given."
            )

        if isinstance(maplist, list) and not all(
            isinstance(m, IdentityMap) for m in maplist
        ):
            raise ValueError(
                f"Attribute 'maplist' should be a list of maps or None.{type(maplist)} was given."
            )

        self._maplist = maplist

    @timeIt
    def __call__(self, m, external_weights=True):
        """Evaluate the regularization function for the model provided.

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model for which the function is evaluated.
        external_weights : bool
            Include custom cell weighting when evaluating the regularization function.

        Returns
        -------
        float
            The regularization function evaluated for the model provided.
        """
        if external_weights:
            W = self.W
        else:
            W = Identity()

        if getattr(self, "reference_model", None) is None:
            self.reference_model = mkvc(self.gmm.means_[self.membership(m)])

        if self.approx_eval:
            membership = self.compute_quasi_geology_model()
            dm = self.wiresmap * (m)
            dmref = self.wiresmap * (self.reference_model)
            dmm = np.c_[[a * b for a, b in zip(self.maplist, dm)]].T
            if self.non_linear_relationships:
                dmm = np.r_[
                    [
                        self.gmm.cluster_mapping[membership[i]] * dmm[i].reshape(-1, 2)
                        for i in range(dmm.shape[0])
                    ]
                ].reshape(-1, 2)

            dmmref = np.c_[[a for a in dmref]].T
            dmr = dmm - dmmref
            r0 = (W * mkvc(dmr)).reshape(dmr.shape, order="F")

            if self.gmm.covariance_type == "tied":
                r1 = np.r_[
                    [np.dot(self.gmm.precisions_, np.r_[r0[i]]) for i in range(len(r0))]
                ]
            elif (
                self.gmm.covariance_type == "diag"
                or self.gmm.covariance_type == "spherical"
            ):
                r1 = np.r_[
                    [
                        np.dot(
                            self.gmm.precisions_[membership[i]]
                            * np.eye(len(self.wiresmap.maps)),
                            np.r_[r0[i]],
                        )
                        for i in range(len(r0))
                    ]
                ]
            else:
                r1 = np.r_[
                    [
                        np.dot(self.gmm.precisions_[membership[i]], np.r_[r0[i]])
                        for i in range(len(r0))
                    ]
                ]

            return mkvc(r0).dot(mkvc(r1))

        else:
            modellist = self.wiresmap * m
            model = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T

            if self.non_linear_relationships:
                score = self.gmm.score_samples(model)
                score_vec = mkvc(np.r_[[score for maps in self.wiresmap.maps]])
                return -2 * np.sum((W.T * W) * score_vec) / len(self.wiresmap.maps)

            else:
                if external_weights and getattr(self.W, "diagonal", None) is not None:
                    sensW = np.c_[
                        [wire[1] * self.W.diagonal() for wire in self.wiresmap.maps]
                    ].T
                else:
                    sensW = np.ones_like(model)

                score = self.gmm.score_samples_with_sensW(model, sensW)
                # score_vec = mkvc(np.r_[[score for maps in self.wiresmap.maps]])
                # return -np.sum((W.T * W) * score_vec) / len(self.wiresmap.maps)
                return -2 * np.sum(score)

    @timeIt
    def deriv(self, m):
        r"""Gradient of the regularization function evaluated for the model provided.

        Where :math:`\phi (\mathbf{m})` is the discrete regularization function (objective function),
        this method evaluates and returns the derivative with respect to the model parameters;
        i.e. the gradient:

        .. math::
            \frac{\partial \phi}{\partial \mathbf{m}}

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model for which the gradient is evaluated.

        Returns
        -------
        (n_param, ) numpy.ndarray
            Gradient of the regularization function evaluated for the model provided.
        """
        if getattr(self, "reference_model", None) is None:
            self.reference_model = mkvc(self.gmm.means_[self.membership(m)])

        membership = self.compute_quasi_geology_model()
        modellist = self.wiresmap * m
        dmmodel = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
        mreflist = self.wiresmap * self.reference_model
        mD = [a.deriv(b) for a, b in zip(self.maplist, modellist)]
        mD = sp.block_diag(mD)

        if self.non_linear_relationships:
            dmmodel = np.r_[
                [
                    self.gmm.cluster_mapping[membership[i]] * dmmodel[i].reshape(-1, 2)
                    for i in range(dmmodel.shape[0])
                ]
            ].reshape(-1, 2)

        if self.approx_gradient:
            dmmref = np.c_[[a for a in mreflist]].T
            dm = dmmodel - dmmref
            r0 = (self.W * (mkvc(dm))).reshape(dm.shape, order="F")

            if self.gmm.covariance_type == "tied":
                if self.non_linear_relationships:
                    raise Exception("Not implemented")

                r = mkvc(
                    np.r_[[np.dot(self.gmm.precisions_, r0[i]) for i in range(len(r0))]]
                )
            elif (
                self.gmm.covariance_type == "diag"
                or self.gmm.covariance_type == "spherical"
            ) and not self.non_linear_relationships:
                r = mkvc(
                    np.r_[
                        [
                            np.dot(
                                self.gmm.precisions_[membership[i]]
                                * np.eye(len(self.wiresmap.maps)),
                                r0[i],
                            )
                            for i in range(len(r0))
                        ]
                    ]
                )
            else:
                if self.non_linear_relationships:
                    r = mkvc(
                        np.r_[
                            [
                                mkvc(
                                    self.gmm.cluster_mapping[membership[i]].deriv(
                                        dmmodel[i],
                                        v=np.dot(
                                            self.gmm.precisions_[membership[i]], r0[i]
                                        ),
                                    )
                                )
                                for i in range(dmmodel.shape[0])
                            ]
                        ]
                    )

                else:
                    r0 = (self.W * (mkvc(dm))).reshape(dm.shape, order="F")
                    r = mkvc(
                        np.r_[
                            [
                                np.dot(self.gmm.precisions_[membership[i]], r0[i])
                                for i in range(len(r0))
                            ]
                        ]
                    )
            return 2 * mkvc(mD.T * (self.W.T * r))

        else:
            if self.non_linear_relationships:
                raise Exception("Not implemented")

            modellist = self.wiresmap * m
            model = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T

            if getattr(self.W, "diagonal", None) is not None:
                sensW = np.c_[
                    [wire[1] * self.W.diagonal() for wire in self.wiresmap.maps]
                ].T
            else:
                sensW = np.ones_like(model)

            score = self.gmm.score_samples_with_sensW(model, sensW)
            # score = self.gmm.score_samples(model)
            score_vec = np.hstack([score for maps in self.wiresmap.maps])

            logP = np.zeros((len(model), self.gmm.n_components))
            W = []
            logP = self.gmm._estimate_log_gaussian_prob_with_sensW(
                model,
                sensW,
                self.gmm.means_,
                self.gmm.precisions_cholesky_,
                self.gmm.covariance_type,
            )
            for k in range(self.gmm.n_components):
                if self.gmm.covariance_type == "tied":
                    # logP[:, k] = mkvc(
                    #    multivariate_normal(
                    #        self.gmm.means_[k], self.gmm.covariances_
                    #    ).logpdf(model)
                    # )

                    W.append(
                        self.gmm.weights_[k]
                        * mkvc(
                            np.r_[
                                [
                                    np.dot(
                                        np.diag(sensW[i]).dot(
                                            self.gmm.precisions_.dot(np.diag(sensW[i]))
                                        ),
                                        (model[i] - self.gmm.means_[k]).T,
                                    )
                                    for i in range(len(model))
                                ]
                            ]
                        )
                    )
                elif (
                    self.gmm.covariance_type == "diag"
                    or self.gmm.covariance_type == "spherical"
                ):
                    # logP[:, k] = mkvc(
                    #    multivariate_normal(
                    #        self.gmm.means_[k],
                    #        self.gmm.covariances_[k] * np.eye(len(self.wiresmap.maps)),
                    #    ).logpdf(model)
                    # )
                    W.append(
                        self.gmm.weights_[k]
                        * mkvc(
                            np.r_[
                                [
                                    np.dot(
                                        np.diag(sensW[i]).dot(
                                            (
                                                self.gmm.precisions_[k]
                                                * np.eye(len(self.wiresmap.maps))
                                            ).dot(np.diag(sensW[i]))
                                        ),
                                        (model[i] - self.gmm.means_[k]).T,
                                    )
                                    for i in range(len(model))
                                ]
                            ]
                        )
                    )
                else:
                    # logP[:, k] = mkvc(
                    #    multivariate_normal(
                    #        self.gmm.means_[k], self.gmm.covariances_[k]
                    #    ).logpdf(model)
                    # )
                    W.append(
                        self.gmm.weights_[k]
                        * mkvc(
                            np.r_[
                                [
                                    np.dot(
                                        np.diag(sensW[i]).dot(
                                            self.gmm.precisions_[k].dot(
                                                np.diag(sensW[i])
                                            )
                                        ),
                                        (model[i] - self.gmm.means_[k]).T,
                                    )
                                    for i in range(len(model))
                                ]
                            ]
                        )
                    )
            W = np.c_[W].T
            logP = np.vstack([logP for maps in self.wiresmap.maps])
            numer = (W * np.exp(logP)).sum(axis=1)
            r = numer / (np.exp(score_vec))
            return 2 * mkvc(mD.T * r)

    @timeIt
    def deriv2(self, m, v=None):
        r"""Hessian of the regularization function evaluated for the model provided.

        Where :math:`\phi (\mathbf{m})` is the discrete regularization function (objective function),
        this method returns the second-derivative (Hessian) with respect to the model parameters:

        .. math::
            \frac{\partial^2 \phi}{\partial \mathbf{m}^2}

        or the second-derivative (Hessian) multiplied by a vector :math:`(\mathbf{v})`:

        .. math::
            \frac{\partial^2 \phi}{\partial \mathbf{m}^2} \, \mathbf{v}

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model for which the Hessian is evaluated.
        v : None, (n_param, ) numpy.ndarray (optional)
            A vector.

        Returns
        -------
        (n_param, n_param) scipy.sparse.csr_matrix | (n_param, ) numpy.ndarray
            If the input argument *v* is ``None``, the Hessian of the regularization
            function for the model provided is returned. If *v* is not ``None``,
            the Hessian multiplied by the vector provided is returned.
        """
        if getattr(self, "reference_model", None) is None:
            self.reference_model = mkvc(self.gmm.means_[self.membership(m)])

        if self.approx_hessian:
            # we approximate it with the covariance of the cluster
            # whose each point belong
            membership = self.compute_quasi_geology_model()
            modellist = self.wiresmap * m
            dmmodel = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
            mD = [a.deriv(b) for a, b in zip(self.maplist, modellist)]
            mD = sp.block_diag(mD)
            if self._r_second_deriv is None:
                if self.gmm.covariance_type == "tied":
                    if self.non_linear_relationships:
                        r = np.r_[
                            [
                                self.gmm.cluster_mapping[membership[i]].deriv(
                                    dmmodel[i],
                                    v=(
                                        self.gmm.cluster_mapping[membership[i]].deriv(
                                            dmmodel[i], v=self.gmm.precisions_
                                        )
                                    ).T,
                                )
                                for i in range(len(dmmodel))
                            ]
                        ]
                    else:
                        r = self.gmm.precisions_[np.newaxis, :, :][
                            np.zeros_like(membership)
                        ]
                elif (
                    self.gmm.covariance_type == "spherical"
                    or self.gmm.covariance_type == "diag"
                ):
                    if self.non_linear_relationships:
                        r = np.r_[
                            [
                                self.gmm.cluster_mapping[membership[i]].deriv(
                                    dmmodel[i],
                                    v=(
                                        self.gmm.cluster_mapping[membership[i]].deriv(
                                            dmmodel[i],
                                            v=self.gmm.precisions_[membership[i]]
                                            * np.eye(len(self.wiresmap.maps)),
                                        )
                                    ).T,
                                )
                                for i in range(len(dmmodel))
                            ]
                        ]
                    else:
                        r = np.r_[
                            [
                                self.gmm.precisions_[memb]
                                * np.eye(len(self.wiresmap.maps))
                                for memb in membership
                            ]
                        ]
                else:
                    if self.non_linear_relationships:
                        r = np.r_[
                            [
                                self.gmm.cluster_mapping[membership[i]].deriv(
                                    dmmodel[i],
                                    v=(
                                        self.gmm.cluster_mapping[membership[i]].deriv(
                                            dmmodel[i],
                                            v=self.gmm.precisions_[membership[i]],
                                        )
                                    ).T,
                                )
                                for i in range(len(dmmodel))
                            ]
                        ]
                    else:
                        r = self.gmm.precisions_[membership]

                self._r_second_deriv = r

            if v is not None:
                mDv = self.wiresmap * (mD * v)
                mDv = np.c_[mDv]
                r0 = (self.W * (mkvc(mDv))).reshape(mDv.shape, order="F")
                second_deriv_times_r0 = mkvc(
                    np.r_[
                        [np.dot(self._r_second_deriv[i], r0[i]) for i in range(len(r0))]
                    ]
                )
                return 2 * mkvc(mD.T * (self.W * second_deriv_times_r0))
            else:
                # Forming the Hessian by diagonal blocks
                hlist = [
                    [
                        self._r_second_deriv[:, i, j]
                        for i in range(len(self.wiresmap.maps))
                    ]
                    for j in range(len(self.wiresmap.maps))
                ]
                Hr = sp.csc_matrix((0, 0), dtype=np.float64)
                for i in range(len(self.wiresmap.maps)):
                    Hc = sp.csc_matrix((0, 0), dtype=np.float64)
                    for j in range(len(self.wiresmap.maps)):
                        Hc = sp.hstack([Hc, sdiag(hlist[i][j])])
                    Hr = sp.vstack([Hr, Hc])

                Hr = Hr.dot(self.W)

                return 2 * (mD.T * mD) * (self.W * (Hr))

        else:
            if self.non_linear_relationships:
                raise Exception("Not implemented")

            # non distinct clusters positive definite approximated Hessian
            modellist = self.wiresmap * m
            model = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T

            if getattr(self.W, "diagonal", None) is not None:
                sensW = np.c_[
                    [wire[1] * self.W.diagonal() for wire in self.wiresmap.maps]
                ].T
            else:
                sensW = np.ones_like(model)

            mD = [a.deriv(b) for a, b in zip(self.maplist, modellist)]
            mD = sp.block_diag(mD)

            score = self.gmm.score_samples_with_sensW(model, sensW)
            logP = np.zeros((len(model), self.gmm.n_components))
            W = []
            logP = self.gmm._estimate_weighted_log_prob_with_sensW(
                model,
                sensW,
            )
            for k in range(self.gmm.n_components):
                if self.gmm.covariance_type == "tied":
                    W.append(
                        [
                            np.diag(sensW[i]).dot(
                                self.gmm.precisions_.dot(np.diag(sensW[i]))
                            )
                            for i in range(len(model))
                        ]
                    )
                elif (
                    self.gmm.covariance_type == "diag"
                    or self.gmm.covariance_type == "spherical"
                ):
                    W.append(
                        [
                            np.diag(sensW[i]).dot(
                                (
                                    self.gmm.precisions_[k]
                                    * np.eye(len(self.wiresmap.maps))
                                ).dot(np.diag(sensW[i]))
                            )
                            for i in range(len(model))
                        ]
                    )
                else:
                    W.append(
                        [
                            np.diag(sensW[i]).dot(
                                self.gmm.precisions_[k].dot(np.diag(sensW[i]))
                            )
                            for i in range(len(model))
                        ]
                    )
            W = np.c_[W]

            hlist = [
                [
                    (W[:, :, i, j].T * np.exp(logP)).sum(axis=1) / np.exp(score)
                    for i in range(len(self.wiresmap.maps))
                ]
                for j in range(len(self.wiresmap.maps))
            ]

            # Forming the Hessian by diagonal blocks
            Hr = sp.csc_matrix((0, 0), dtype=np.float64)
            for i in range(len(self.wiresmap.maps)):
                Hc = sp.csc_matrix((0, 0), dtype=np.float64)
                for j in range(len(self.wiresmap.maps)):
                    Hc = sp.hstack([Hc, sdiag(hlist[i][j])])
                Hr = sp.vstack([Hr, Hc])
            Hr = 2 * (mD.T * mD) * Hr

            if v is not None:
                return Hr.dot(v)

            return Hr


class PGI(ComboObjectiveFunction):
    r"""Regularization function for petrophysically guided inversion (PGI).

    ``PGI`` is used to recover models in which 1) the physical property values are consistent
    with petrophysical information and 2) structures in the recovered model are geologically
    plausible. ``PGI`` regularization is a weighted sum of :class:`PGIsmallness`,
    :class:`SmoothnessFirstOrder` and :class:`SmoothnessSecondOrder` (optional)
    regularization functions. The PGI smallness term assumes the statistical distribution of
    physical property values defining the model is characterized
    by a Gaussian mixture model (GMM). And the smoothness terms penalize large
    spatial derivatives in the recovered model.

    ``PGI`` can be implemented to invert for a single physical property or multiple
    physical properties, each of which are defined on a linear scale (e.g. density) or a log-scale
    (e.g. electrical conductivity). If the statistical distribution(s) of physical property values
    for each property type are known, the GMM can be constructed and left static throughout the
    inversion. Otherwise, the recovered model at each iteration is used to update the GMM.
    And the updated GMM is used to constrain the recovered model for the following iteration.

    Parameters
    ----------
    mesh : simpeg.regularization.RegularizationMesh, discretize.base.BaseMesh
        Mesh on which the regularization is discretized. Implemented for
        `tensor`, `QuadTree` or `Octree` meshes.
    gmmref : simpeg.utils.WeightedGaussianMixture
        Reference Gaussian mixture model.
    gmm : None, simpeg.utils.WeightedGaussianMixture
        Set the Gaussian mixture model used to constrain the recovered physical property model.
        Can be left static throughout the inversion or updated using the
        :class:`.directives.PGI_UpdateParameters` directive. If ``None``, the
        :class:`.directives.PGI_UpdateParameters` directive must be used to ensure there
        is a Gaussian mixture model for the inversion.
    alpha_pgi : float
        Scaling constant for the PGI smallness term.
    alpha_x, alpha_y, alpha_z : float or None, optional
        Scaling constants for the first order smoothness along x, y and z, respectively.
        If set to ``None``, the scaling constant is set automatically according to the
        value of the `length_scale` parameter.
    alpha_xx, alpha_yy, alpha_zz : 0, float
        Scaling constants for the second order smoothness along x, y and z, respectively.
        If set to ``None``, the scaling constant is set automatically according to the
        length scales; see :class:`regularization.WeightedLeastSquares`.
    wiresmap : None, simpeg.maps.Wires
        Mapping from the model to the model parameters of each type.
        If ``None``, we assume only a single physical property type in the inversion.
    maplist : None, list of simpeg.maps
        Ordered list of mappings from model values to physical property values;
        one for each physical property. If ``None``, we assume a single physical property type
        in the regularization and an :class:`.maps.IdentityMap` from model values to physical
        property values.
    approx_gradient : bool
        If ``True``, use the L2-approximation of the gradient by assuming
        physical property values of different types are uncorrelated.
    approx_eval : bool
        If ``True``, use the L2-approximation evaluation of the smallness term by assuming
        physical property values of different types are uncorrelated.
    approx_hessian : bool
        Approximate the Hessian of the regularization function.
    non_linear_relationship : bool
        Whether relationships in the Gaussian mixture model are non-linear.
    reference_model_in_smooth : bool, optional
        Whether to include the reference model in the smoothness terms.

    Notes
    -----
    For one or more physical property types (e.g. conductivity, density, susceptibility),
    the ``PGI`` regularization function (objective function) is derived by using a
    Gaussian mixture model (GMM) to construct the prior within a Baysian
    inversion scheme. For a comprehensive description, see
    (`Astic, et al 2019 <https://owncloud.eoas.ubc.ca/s/TMB3Jdr8ScqSPm7/download>`__;
    `Astic et al 2020 <https://owncloud.eoas.ubc.ca/s/PAxpHQt7CGk6zT4/download>`__).

    We let :math:`\Theta` store all of the means (:math:`\boldsymbol{\mu}`), covariances
    (:math:`\boldsymbol{\Sigma}`) and proportion constants (:math:`\boldsymbol{\gamma}`)
    defining the GMM. And let :math:`\mathbf{z}^\ast` define an membership array that
    extracts the GMM parameters for the most representative rock unit within each active cell
    in the :class:`RegularizationMesh`. The regularization function (objective function) for
    ``PGI`` is given by:

    .. math::
        \phi (\mathbf{m}) &= \alpha_\text{pgi}
        \big [ \mathbf{m} - \mathbf{m_{ref}}(\Theta, \mathbf{z}^\ast ) \big ]^T
        \mathbf{W} ( \Theta , \mathbf{z}^\ast ) \,
        \big [ \mathbf{m} - \mathbf{m_{ref}}(\Theta, \mathbf{z}^\ast ) \big ] \\
        &+ \sum_{j=x,y,z} \alpha_j \Big \| \mathbf{W_j G_j \, m} \, \Big \|^2 \\
        &+ \sum_{j=x,y,z} \alpha_{jj} \Big \| \mathbf{W_{jj} L_j \, m} \, \Big \|^2
        \;\;\;\;\;\;\;\; \big ( \textrm{optional} \big )

    where

        - :math:`\mathbf{m}` is the model,
        - :math:`\mathbf{m_{ref}}(\Theta, \mathbf{z}^\ast )` is the reference model,
        - :math:`\mathbf{G_x, \, G_y, \; G_z}` are partial cell gradients operators along x, y and z,
        - :math:`\mathbf{L_x, \, L_y, \; L_z}` are second-order derivative operators with respect to x, y and z,
        - :math:`\mathbf{W}(\Theta , \mathbf{z}^\ast )` is the weighting matrix for PGI smallness, and
        - :math:`\mathbf{W_x, \, W_y, \; W_z}` are weighting matrices for smoothness terms.

    ``PGIsmallness`` regularization can be used for models consisting of one or more physical
    property types. The ordering of the physical property types within the model is defined
    using the `wiresmap`. And the mapping from model parameter values to physical property
    values is specified with `maplist`. For :math:`K` physical property types, the model is
    an array vector of the form:

    .. math::
        \mathbf{m} = \begin{bmatrix} \mathbf{m}_1 \\ \mathbf{m}_2 \\ \vdots \\ \mathbf{m}_K \end{bmatrix}

    When the ``approx_eval`` property is ``True``, we assume the physical property types have
    values that are uncorrelated. In this case, the weighting matrix is diagonal and the
    regularization function (objective function) can be expressed as:

    .. math::
        \phi (\mathbf{m}) &= \alpha_\text{pgi} \Big \| \mathbf{W}_{\! 1/2}(\Theta, \mathbf{z}^\ast ) \,
        \big [ \mathbf{m} - \mathbf{m_{ref}}(\Theta, \mathbf{z}^\ast ) \big ] \, \Big \|^2 \\
        &+ \sum_{j=x,y,z} \alpha_j \Big \| \mathbf{W_j G_j \, m} \, \Big \|^2 \\
        &+ \sum_{j=x,y,z} \alpha_{jj} \Big \| \mathbf{W_{jj} L_j \, m} \, \Big \|^2
        \;\;\;\;\;\;\;\; \big ( \textrm{optional} \big )

    When the ``approx_eval`` property is ``True``, you may also set the ``approx_gradient`` property
    to ``True`` so that the least-squares approximation is used to compute the gradient.

    **Constructing the Reference Model and Weighting Matrix:**

    The reference model used in the regularization function is constructed by extracting the means
    :math:`\boldsymbol{\mu}` from the GMM using the membership array :math:`\mathbf{z}^\ast`.
    We represent this vector as:

    .. math::
        \mathbf{m_{ref}} (\Theta ,{\mathbf{z}^\ast}) = \boldsymbol{\mu}_{\mathbf{z}^\ast}

    To construct the weighting matrix, :math:`\mathbf{z}^\ast` is used to extract the covariances
    :math:`\boldsymbol{\Sigma}` for each cell. And the weighting matrix is given by:

    .. math::
        \mathbf{W}(\Theta ,{\mathbf{z}^\ast } ) = \boldsymbol{\Sigma}_{\mathbf{z^\ast}}^{-1} \,
        diag \big ( \mathbf{v \odot w} \big )

    where :math:`\mathbf{v}` are the volumes of the active cells, and :math:`\mathbf{w}`
    are custom cell weights. When the ``approx_eval`` property is ``True``, the off-diagonal
    covariances are zero and we can use a weighting matrix of the form:

    .. math::
        \mathbf{W}_{\! 1/2}(\Theta ,{\mathbf{z}^\ast } ) = diag \Big ( \big [ \mathbf{v \odot w}
        \odot \boldsymbol{\sigma}_{\mathbf{z}^\ast}^{-2} \big ]^{1/2} \Big )

    where :math:`\boldsymbol{\sigma}_{\mathbf{z}^\ast}^2` are the variances extracted using the
    membership array :math:`\mathbf{z}^\ast`.

    **Updating the Gaussian Mixture Model:**

    When the GMM is set using the ``gmm`` property, the GMM remains static throughout the inversion.
    When the ``gmm`` property set as ``None``, the GMM is learned and updated after every model update.
    That is, we assume the GMM defined using the ``gmmref`` property is not completely representative
    of the physical property distributions for each rock unit, and we update the all of the means
    (:math:`\boldsymbol{\mu}`), covariances (:math:`\boldsymbol{\Sigma}`) and proportion constants
    (:math:`\boldsymbol{\gamma}`) defining the GMM :math:`\Theta`. This is done by solving:

    .. math::
        \max_\Theta \; \mathcal{P}(\Theta | \mathbf{m})

    using a MAP variation of the expectation-maximization clustering algorithm introduced in
    Dempster (et al. 1977).

    **Updating the Membership Array:**

    As the model (and GMM) are updated throughout the inversion, the rock unit considered most
    indicative of the geology within each cell is updated; which is represented by the membership
    array :math:`\mathbf{z}^\ast`. W. For the current GMM with means (:math:`\boldsymbol{\mu}`),
    covariances (:math:`\boldsymbol{\Sigma}`) and proportion constants (:math:`\boldsymbol{\gamma}`),
    we solve the following for each cell:

    .. math::
        z_i^\ast = \max_n \; \gamma_{i,n} \, \mathcal{N} (\mathbf{m}_i | \boldsymbol{\mu}_n , \boldsymbol{\Sigma}_n)

    where

        - :math:`\mathbf{m_i}` are the model values for cell :math:`i`,
        - :math:`\gamma_{i,n}` is the proportion for cell :math:`i` and rock unit :math:`n`
        - :math:`\boldsymbol{\mu}_n` are the mean property values for unit :math:`n`,
        - :math:`\boldsymbol{\Sigma}_n` are the covariances for unit :math:`n`, and
        - :math:`\mathcal{N}` represent the multivariate Gaussian distribution.

    """

    def __init__(
        self,
        mesh,
        gmmref,
        alpha_x=None,
        alpha_y=None,
        alpha_z=None,
        alpha_xx=0.0,
        alpha_yy=0.0,
        alpha_zz=0.0,
        gmm=None,
        wiresmap=None,
        maplist=None,
        alpha_pgi=1.0,
        approx_hessian=True,
        approx_gradient=True,
        approx_eval=True,
        weights_list=None,
        non_linear_relationships: bool = False,
        reference_model_in_smooth: bool = False,
        **kwargs,
    ):
        self._wiresmap = wiresmap
        self._maplist = maplist
        self.regularization_mesh = mesh
        self.gmmref = copy.deepcopy(gmmref)
        self.gmmref.order_clusters_GM_weight()

        objfcts = [
            PGIsmallness(
                gmmref,
                mesh=self.regularization_mesh,
                gmm=gmm,
                wiresmap=self.wiresmap,
                maplist=self.maplist,
                approx_eval=approx_eval,
                approx_gradient=approx_gradient,
                approx_hessian=approx_hessian,
                non_linear_relationships=non_linear_relationships,
                weights=weights_list,
                **kwargs,
            )
        ]

        if not isinstance(weights_list, list):
            weights_list = [weights_list] * len(self.maplist)

        for model_map, wire, weights in zip(
            self.maplist, self.wiresmap.maps, weights_list
        ):
            weights_i = {"pgi-weights": weights} if weights is not None else None
            objfcts += [
                WeightedLeastSquares(
                    alpha_s=0.0,
                    alpha_x=alpha_x,
                    alpha_y=alpha_y,
                    alpha_z=alpha_z,
                    alpha_xx=alpha_xx,
                    alpha_yy=alpha_yy,
                    alpha_zz=alpha_zz,
                    mesh=self.regularization_mesh,
                    mapping=model_map * wire[1],
                    weights=weights_i,
                    **kwargs,
                )
            ]

        super().__init__(objfcts=objfcts, unpack_on_add=False)
        self.reference_model_in_smooth = reference_model_in_smooth
        self.alpha_pgi = alpha_pgi

    @property
    def alpha_pgi(self):
        """Scaling constant for the PGI smallness term.

        Returns
        -------
        float
            Scaling constant for the PGI smallness term.
        """
        if getattr(self, "_alpha_pgi", None) is None:
            self._alpha_pgi = self.multipliers[0]
        return self._alpha_pgi

    @alpha_pgi.setter
    def alpha_pgi(self, value):
        value = validate_float("alpha_pgi", value, min_val=0.0)
        self._alpha_pgi = value
        self._multipliers[0] = value

    @property
    def gmm(self):
        """Gaussian mixture model.

        If set prior to inversion, the Gaussian mixture model can be left static throughout
        the inversion, or updated using the :class:`.directives.PGI_UpdateParameters` directive.
        If this property is not set prior to inversion, the
        :class:`.directives.PGI_UpdateParameters` directive must be used to ensure there
        is a Gaussian mixture model for the inversion.

        Returns
        -------
        None, simpeg.utils.WeightedGaussianMixture
            Gaussian mixture model.
        """
        return self.objfcts[0].gmm

    @gmm.setter
    def gmm(self, gm):
        self.objfcts[0].gmm = copy.deepcopy(gm)

    def membership(self, m):
        """Compute and return membership array for the model provided.

        The membership array stores the index of the rock unit most representative of each cell.
        For a Gaussian mixture model containing the means and covariances for model parameter
        types (physical property types) for all rock units, this method computes the membership
        array for the model `m` provided. For a description of the membership array, see the
        *Notes*  section within the :class:`PGI` documentation.

        Parameters
        ----------
        m : (n_param ) numpy.ndarray of float
            The model.

        Returns
        -------
        (n_active, ) numpy.ndarray of int
            The membership array.
        """
        return self.objfcts[0].membership(m)

    def compute_quasi_geology_model(self):
        r"""Compute and return quasi geology model.

        For each active cell in the mesh, this method returns the mean values in the Gaussian
        mixture model for the most representative rock unit, given the current model. See the
        *Notes* section for a comprehensive description.

        Returns
        -------
        (n_param ) numpy.ndarray
            The quasi geology physical property model.

        Notes
        -----
        Consider a Gaussian mixture model (GMM) for :math:`K` physical property types and
        :math:`N` rock units. The mean model parameter values for rock unit
        :math:`n \in \{ 1, \ldots , N \}` in the GMM is represented by a vector
        :math:`\boldsymbol{\mu}_n` of length :math:`K`. For each active cell in the mesh, the
        `compute_quasi_geology_model` method computes:

        .. math::
            g_i = \min_{\boldsymbol{\mu}_n} \big \| \mathbf{m}_i - \boldsymbol{\mu}_n \big \|^2

        where :math:`\mathbf{m}_i` are the model parameter values for cell :math:`i` for the
        current model. The ordering of the output vector :math:`\mathbf{g}` is the same as the
        model :math:`\mathbf{m}`.
        """
        return self.objfcts[0].compute_quasi_geology_model()

    @property
    def wiresmap(self):
        """Mapping from the model to the model parameters of each type.

        Returns
        -------
        simpeg.maps.Wires
            Mapping from the model to the model parameters of each type.
        """
        if getattr(self, "_wiresmap", None) is None:
            self._wiresmap = Wires(("m", self.regularization_mesh.nC))
        return self._wiresmap

    @property
    def maplist(self):
        """Ordered list of mappings from model values to physical property values.

        Returns
        -------
        list of simpeg.maps
            Ordered list of mappings from model values to physical property values;
            one for each physical property.
        """
        if getattr(self, "_maplist", None) is None:
            self._maplist = [
                IdentityMap(nP=self.regularization_mesh.nC)
                for maps in self.wiresmap.maps
            ]
        return self._maplist

    @property
    def regularization_mesh(self) -> RegularizationMesh:
        """Regularization mesh.

        Mesh on which the regularization is discretized. This is not the same as
        the mesh on which the simulation is defined.

        Returns
        -------
        discretize.base.RegularizationMesh
            Mesh on which the regularization is discretized.
        """
        return self._regularization_mesh

    @regularization_mesh.setter
    def regularization_mesh(self, mesh: RegularizationMesh):
        if not isinstance(mesh, RegularizationMesh):
            mesh = RegularizationMesh(mesh)

        self._regularization_mesh = mesh

    @property
    def reference_model_in_smooth(self) -> bool:
        """Whether to include the reference model in the smoothness objective functions.

        Returns
        -------
        bool
            Whether to include the reference model in the smoothness objective functions.
        """
        return self._reference_model_in_smooth

    @reference_model_in_smooth.setter
    def reference_model_in_smooth(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(
                "'reference_model_in_smooth must be of type 'bool'. "
                f"Value of type {type(value)} provided."
            )
        self._reference_model_in_smooth = value
        for fct in self.objfcts[1:]:
            if getattr(fct, "reference_model_in_smooth", None) is not None:
                fct.reference_model_in_smooth = value

    @property
    def reference_model(self) -> np.ndarray:
        """Reference model.

        Returns
        -------
        None, (n_param, ) numpy.ndarray
            Reference model. If ``None``, the reference model in the inversion is set to
            the starting model.
        """
        return self.objfcts[0].reference_model

    @reference_model.setter
    def reference_model(self, values: np.ndarray | float):
        if isinstance(values, float):
            values = np.ones(self._nC_residual) * values

        for fct in self.objfcts:
            fct.reference_model = values

    mref = deprecate_property(
        reference_model,
        "mref",
        "reference_model",
        "0.19.0",
        error=True,
    )
