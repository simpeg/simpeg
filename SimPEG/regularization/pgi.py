from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import warnings

import copy
from ..utils import (
    speye,
    sdiag,
    mkvc,
    timeIt,
    Identity,
)
from ..maps import IdentityMap, Wires
from ..objective_function import ComboObjectiveFunction
from .base import (
    BaseRegularization,
    LeastSquaresRegularization,
    RegularizationMesh,
    Small,
)

###############################################################################
#                                                                             #
#            Petrophysically And Geologically Guided Regularization           #
#                                                                             #
###############################################################################


# Simple Petrophysical Regularization
#####################################


class PGIsmallness(Small):
    """
    Smallness term for the petrophysically constrained regularization (PGI)
    with cell_weights similar to the regularization.tikhonov.SimpleSmall class.

    PARAMETERS
    ----------
    :param SimPEG.utils.WeightedGaussianMixture gmm: GMM to use
    :param SimPEG.maps.Wires wiresmap: wires mapping to the various physical properties
    :param list maplist: list of SimPEG.maps for each physical property.
    :param discretize.BaseMesh mesh: tensor, QuadTree or Octree mesh
    :param boolean approx_gradient: use the L2-approximation of the gradient, default is True
    :param boolean approx_eval: use the L2-approximation evaluation of the smallness term
    """

    _multiplier_pair = "alpha_s"
    _non_linear_relationships = False
    _maplist = None
    _wiresmap = None

    def __init__(
        self,
        gmm,
        wiresmap=None,
        maplist=None,
        mesh=None,
        approx_gradient=True,  # L2 approximate of the gradients
        approx_eval=True,  # L2 approximate of the value
        approx_hessian=True,
        non_linear_relationships=False,
        **kwargs
    ):
        self.approx_gradient = approx_gradient
        self.approx_eval = approx_eval
        self.approx_hessian = approx_hessian
        self.non_linear_relationships = non_linear_relationships
        self.gmm = gmm
        self.wiresmap = wiresmap
        self.maplist = maplist

        if "mapping" in kwargs:
            warnings.warn(f"Property 'mapping' of class {type(self)} cannot be set. Defaults to IdentityMap.")
            kwargs.pop("mapping")

        super().__init__(mesh=mesh, mapping=IdentityMap(nP=self.shape[0]), **kwargs)

        # Save repetitive computations (see withmapping implementation)
        self._r_first_deriv = None
        self._r_second_deriv = None

    def add_set_weights(self, weights: dict | np.ndarray):
        if isinstance(weights, (np.ndarray, list)):
            weights = {"user_weights": np.r_[weights].flatten()}

        if not isinstance(weights, dict):
            raise TypeError("Weights must be provided as a dictionary or numpy.ndarray.")

        for key, values in weights.items():
            self.validate_array_type("weights", values, float)

            if values.shape[0] == self.regularization_mesh.nC:
                values = np.tile(values, len(self.wiresmap.maps))

            self.validate_shape("weights", values, (self._nC_residual,))

            self.weights[key] = values

        self._W = None


    @property
    def shape(self):
        """"""
        return self.wiresmap.nP,

    def membership(self, m):
        modellist = self.wiresmap * m
        model = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
        return self.gmm.predict(model)  # mkvc(m, numDims=2))

    def compute_quasi_geology_model(self):
        # used once mref is built
        mreflist = self.wiresmap * self.mref
        mrefarray = np.c_[[a * b for a, b in zip(self.maplist, mreflist)]].T
        return np.c_[
            [((mrefarray - mean) ** 2).sum(axis=1) for mean in self.gmm.means_]
        ].argmin(axis=0)

    @property
    def non_linear_relationships(self):
        """Flag for non-linear GMM relationships"""
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
        if getattr(self, "_wiresmap", None) is None:
            self._wiresmap = Wires(("m", self.regularization_mesh.nC))
        return self._wiresmap

    @wiresmap.setter
    def wiresmap(self, wires):
        if self._maplist is not None and len(wiresmap.maps) != len(self._maplist):
            raise Exception(f"Provided 'wiresmap' should have wires the len of 'maplist' {len(self._maplist)}.")

        if not isinstance(wires, Wires):
            raise ValueError(f"Attribure 'wiresmap' should be of type {Wire} or None.")

        self._wiresmap = wires

    @property
    def maplist(self):
        if getattr(self, "_maplist", None) is None:
            self._maplist = [IdentityMap(self.regularization_mesh) for maps in self.wiresmap.maps]
        return self._maplist

    @maplist.setter
    def maplist(self, maplist):
        if self._wiresmap is not None and len(maplist) != len(self._wiresmap.maps):
            raise Exception(f"Provided 'maplist' should be a list of maps equal to the 'wiresmap' list of len {len(self._maplist)}.")

        if not isinstance(maplist, (list, type(None))):
            raise ValueError("Attribure 'maplist' should be a list of maps or None.")

        if isinstance(maplist, list) and not all(isinstance(map, IdentityMap) for map in maplist):
            raise ValueError(f"Attribure 'maplist' should be a list of maps or None.")

        self._maplist = maplist

    @timeIt
    def __call__(self, m):
        if getattr(self, "mref", None) is None:
            self.mref = mkvc(self.gmm.means_[self.membership(m)])

        if self.approx_eval:
            membership = self.compute_quasi_geology_model()
            dm = self.wiresmap * (m)
            dmref = self.wiresmap * (self.mref)
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
            # r0 = (W * mkvc(dmr)).reshape(dmr.shape, order="F")
            r0 = self.W * mkvc(dmr)

            if self.gmm.covariance_type == "tied":
                r1 = np.r_[
                    [
                        np.dot(self.gmm.precisions_, np.r_[dmr[i]])
                        for i in range(len(dmr))
                    ]
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
                            np.r_[dmr[i]],
                        )
                        for i in range(len(dmr))
                    ]
                ]
            else:
                r1 = np.r_[
                    [
                        np.dot(self.gmm.precisions_[membership[i]], np.r_[dmr[i]])
                        for i in range(len(dmr))
                    ]
                ]

            return 0.5 * r0.dot(self.W * mkvc(r1))

        else:
            modellist = self.wiresmap * m
            model = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T

            if self.non_linear_relationships:
                score = self.gmm.score_samples(model)
                score_vec = mkvc(np.r_[[score for maps in self.wiresmap.maps]])
                return -np.sum((W.T * W) * score_vec) / len(self.wiresmap.maps)

            else:
                if externalW and getattr(self.W, "diagonal", None) is not None:
                    sensW = np.c_[
                        [wire[1] * self.W.diagonal() for wire in self.wiresmap.maps]
                    ].T
                else:
                    sensW = np.ones_like(model)

                score = self.gmm.score_samples_with_sensW(model, sensW)
                # score_vec = mkvc(np.r_[[score for maps in self.wiresmap.maps]])
                # return -np.sum((W.T * W) * score_vec) / len(self.wiresmap.maps)
                return -np.sum(score)

    @timeIt
    def deriv(self, m):

        if getattr(self, "mref", None) is None:
            self.mref = mkvc(self.gmm.means_[self.membership(m)])

        membership = self.compute_quasi_geology_model()
        modellist = self.wiresmap * m
        dmmodel = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
        mreflist = self.wiresmap * self.mref
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
            # r0 = (self.W * (mkvc(dm))).reshape(dm.shape, order="F")

            if self.gmm.covariance_type == "tied":

                if self.non_linear_relationships:
                    raise Exception("Not implemented")

                r = mkvc(
                    np.r_[[np.dot(self.gmm.precisions_, dm[i]) for i in range(len(dm))]]
                )
            elif (
                    (
                        self.gmm.covariance_type == "diag"
                        or self.gmm.covariance_type == "spherical"
                    )
                and not self.non_linear_relationships
            ):
                r = self.W * mkvc(
                    np.r_[
                        [
                            np.dot(
                                self.gmm.precisions_[membership[i]]
                                * np.eye(len(self.wiresmap.maps)),
                                dm[i],
                            )
                            for i in range(len(dm))
                        ]
                    ]
                )
            else:
                if self.non_linear_relationships:
                    r = self.W * mkvc(
                        np.r_[
                            [
                                mkvc(
                                    self.gmm.cluster_mapping[membership[i]].deriv(
                                        dmmodel[i],
                                        v=np.dot(
                                            self.gmm.precisions_[membership[i]], dm[i]
                                        ),
                                    )
                                )
                                for i in range(dmmodel.shape[0])
                            ]
                        ]
                    )

                else:
                    r0 = (self.W * (mkvc(dm))).reshape(dm.shape, order="F")
                    r = self.W * mkvc(
                        np.r_[
                            [
                                np.dot(self.gmm.precisions_[membership[i]], r0[i])
                                for i in range(len(r0))
                            ]
                        ]
                    )
            return mkvc(mD.T * (self.W.T * r))

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
            return mkvc(mD.T * r)

    @timeIt
    def deriv2(self, m, v=None):

        if getattr(self, "mref", None) is None:
            self.mref = mkvc(self.gmm.means_[self.membership(m)])

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

                        r = self.gmm.precisions_[np.newaxis, :, :][np.zeros_like(membership)]
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
                                self.gmm.precisions_[memb] * np.eye(len(self.wiresmap.maps))
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
                                            dmmodel[i], v=self.gmm.precisions_[membership[i]]
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
                return mkvc(
                    mD.T
                    * (
                            (self.W.T * self.W)
                            * mkvc(
                        np.r_[
                            [
                                np.dot(self._r_second_deriv[i], mDv[i])
                                for i in range(len(mDv))
                            ]
                        ]
                    )
                    )
                )
            else:
                # Forming the Hessian by diagonal blocks
                hlist = [
                    [self._r_second_deriv[:, i, j] for i in range(len(self.wiresmap.maps))]
                    for j in range(len(self.wiresmap.maps))
                ]

                Hr = sp.csc_matrix((0, 0), dtype=np.float64)
                for i in range(len(self.wiresmap.maps)):
                    Hc = sp.csc_matrix((0, 0), dtype=np.float64)
                    for j in range(len(self.wiresmap.maps)):
                        Hc = sp.hstack([Hc, sdiag(hlist[i][j])])
                    Hr = sp.vstack([Hr, Hc])

                mDW = self.W * mD

                return (mDW.T * mDW) * Hr

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
            Hr = (mD.T * mD) * Hr

            if v is not None:
                return Hr.dot(v)

            return Hr


class PGI(ComboObjectiveFunction):
    """
    class similar to regularization.tikhonov.Simple, with a PGIsmallness.
    PARAMETERS
    ----------
    :param SimPEG.utils.WeightedGaussianMixture gmmref: refereence/prior GMM
    :param SimPEG.utils.WeightedGaussianMixture gmm: GMM to use
    :param SimPEG.maps.Wires wiresmap: wires mapping to the various physical properties
    :param list maplist: list of SimPEG.maps for each physical property.
    :param discretize.BaseMesh mesh: tensor, QuadTree or Octree mesh
    :param boolean approx_gradient: use the L2-approximation of the gradient, default is True
    :param boolean approx_eval: use the L2-approximation evaluation of the smallness term
    """

    def __init__(
        self,
        mesh,
        gmmref,
        gmm=None,
        wiresmap=None,
        maplist=None,
        approx_hessian=True,
        approx_gradient=True,
        approx_eval=True,
        weights_list=None,
        non_linear_relationships: bool = False,
        reference_model_in_smooth: bool = False,
        **kwargs
    ):
        self.gmmref = copy.deepcopy(gmmref)
        self.gmmref.order_clusters_GM_weight()
        self._gmm = copy.deepcopy(gmm)
        self._wiresmap = wiresmap
        self._maplist = maplist
        self.regularization_mesh = mesh

        objfcts = [
            PGIsmallness(
                mesh=self.regularization_mesh,
                gmm=self.gmm,
                wiresmap=self.wiresmap,
                maplist=self.maplist,
                approx_eval=approx_eval,
                approx_gradient=approx_gradient,
                approx_hessian=approx_hessian,
                non_linear_relationships=non_linear_relationships,
                weights=weights_list,
                **kwargs
            )
        ]

        if not isinstance(weights_list, list):
            weights_list = [weights_list] * len(self.maplist)

        for map, wire, weights in zip(self.maplist, self.wiresmap.maps, weights_list):
            objfcts += [
                LeastSquaresRegularization(
                    mesh=self.regularization_mesh,
                    mapping=map * wire[1],
                    weights=weights,
                    **kwargs
                )
            ]

        super(PGI, self).__init__(objfcts=objfcts)
        self.reference_model_in_smooth = reference_model_in_smooth

    @property
    def gmm(self):
        if getattr(self, "_gmm", None) is None:
            self._gmm = copy.deepcopy(self.gmmref)
        return self._gmm

    @gmm.setter
    def gmm(self, gm):
        if gm is not None:
            self._gmm = copy.deepcopy(gm)
        self.objfcts[0].gmm = self.gmm

    def membership(self, m):
        return self.objfcts[0].membership(m)

    def compute_quasi_geology_model(self):
        return self.objfcts[0].compute_quasi_geology_model()

    @property
    def wiresmap(self):
        if getattr(self, "_wiresmap", None) is None:
            self._wiresmap = Wires(("m", self.regularization_mesh.nC))
        return self._wiresmap

    @property
    def maplist(self):
        if getattr(self, "_maplist", None) is None:
            self._maplist = [IdentityMap(self.regularization_mesh) for maps in self.wiresmap.maps]
        return self._maplist

    @property
    def regularization_mesh(self) -> RegularizationMesh:
        """Regularization mesh"""
        return self._regularization_mesh

    @regularization_mesh.setter
    def regularization_mesh(self, mesh: RegularizationMesh):
        if not isinstance(mesh, RegularizationMesh):
            mesh = RegularizationMesh(mesh)

        self._regularization_mesh = mesh

    @property
    def reference_model_in_smooth(self) -> bool:
        """
        Use the reference model in the model gradient penalties.
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