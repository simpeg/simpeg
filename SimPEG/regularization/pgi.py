import numpy as np
import scipy.sparse as sp
import warnings
import properties
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import copy
from ..utils import (
    speye,
    sdiag,
    mkvc,
    timeIt,
    Identity,
    Zero,
    coterminal,
)
from ..maps import IdentityMap, Wires
from .. import props
from .base import (
    BaseRegularization,
    SimpleComboRegularization,
    BaseComboRegularization,
)
from .tikhonov import *


###############################################################################
#                                                                             #
#            Petrophysically And Geologically Guided Regularization           #
#                                                                             #
###############################################################################


# Simple Petrophysical Regularization
#####################################


class SimplePGIsmallness(BaseRegularization):
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

    def __init__(
        self,
        gmm,
        wiresmap=None,
        maplist=None,
        mesh=None,
        approx_gradient=True,  # L2 approximate of the gradients
        approx_eval=True,  # L2 approximate of the value
        approx_hessian=True,
        **kwargs
    ):

        self.approx_gradient = approx_gradient
        self.approx_eval = approx_eval
        self.approx_hessian = approx_hessian

        super(SimplePGIsmallness, self).__init__(mesh=mesh, **kwargs)
        self.gmm = gmm
        self.wiresmap = wiresmap
        self.maplist = maplist

        # Save repetitive computations (see withmapping implementation)
        self._r_first_deriv = None
        self._r_second_deriv = None

    @property
    def W(self):
        """
        Weighting matrix
        Need to change the size to match self.wiresmap.maps * mesh.nC
        """
        if self.cell_weights is not None:
            if len(self.cell_weights) == self.wiresmap.nP:
                return sdiag(np.sqrt(self.cell_weights))
            else:
                return sp.kron(
                    speye(len(self.wiresmap.maps)), sdiag(np.sqrt(self.cell_weights))
                )
        else:
            return Identity()

    @properties.validator("cell_weights")
    def _validate_cell_weights(self, change):
        if change["value"] is not None:
            if self._nC_residual != "*":
                if (len(change["value"]) != self._nC_residual) and (
                    len(change["value"]) != len(self.wiresmap.maps) * self._nC_residual
                ):
                    raise Exception(
                        "cell_weights must be length {} or {} not {}".format(
                            self._nC_residual,
                            len(self.wiresmap.maps) * self._nC_residual,
                            len(change["value"]),
                        )
                    )

    def membership(self, m):
        modellist = self.wiresmap * m
        model = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
        return self.gmm.predict(model)  # mkvc(m, numDims=2))

    def compute_quasi_geology_model(self):
        #used once mref is built
        mreflist = self.wiresmap * self.mref
        mrefarray = np.c_[[a * b for a, b in zip(self.maplist, mreflist)]].T
        return np.c_[[((mrefarray - mean)**2).sum(axis=1) for mean in self.gmm.means_]].argmin(axis=0)


    @timeIt
    def __call__(self, m, externalW=True):

        if externalW:
            W = self.W
        else:
            W = Identity()

        if getattr(self, "mref", None) is None:
            self.mref = mkvc(self.gmm.means_[self.membership(m)])

        if self.approx_eval:
            membership = self.compute_quasi_geology_model()
            dm = self.wiresmap * (m)
            dmref = self.wiresmap * (self.mref)
            dmm = np.c_[[a * b for a, b in zip(self.maplist, dm)]].T
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

            return 0.5 * mkvc(r0).dot(mkvc(r1))

        else:
            modellist = self.wiresmap * m
            model = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T

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
        mreflist = self.wiresmap * self.mref
        mD = [a.deriv(b) for a, b in zip(self.maplist, modellist)]
        mD = sp.block_diag(mD)

        if self.approx_gradient:
            dmmodel = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
            dmmref = np.c_[[a for a in mreflist]].T
            dm = dmmodel - dmmref
            r0 = (self.W * (mkvc(dm))).reshape(dm.shape, order="F")

            if self.gmm.covariance_type == "tied":
                r = mkvc(
                    np.r_[[np.dot(self.gmm.precisions_, r0[i]) for i in range(len(r0))]]
                )
            elif (
                self.gmm.covariance_type == "diag"
                or self.gmm.covariance_type == "spherical"
            ):
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
                r = mkvc(
                    np.r_[
                        [
                            np.dot(self.gmm.precisions_[membership[i]], r0[i])
                            for i in range(len(r0))
                        ]
                    ]
                )
            return mkvc(mD.T * (self.W * r))

        else:
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
            mD = [a.deriv(b) for a, b in zip(self.maplist, modellist)]
            mD = sp.block_diag(mD)

            if self.gmm.covariance_type == "tied":
                r = self.gmm.precisions_[np.newaxis, :, :][np.zeros_like(membership)]
            elif (
                self.gmm.covariance_type == "spherical"
                or self.gmm.covariance_type == "diag"
            ):
                r = np.r_[
                    [
                        self.gmm.precisions_[memb] * np.eye(len(self.wiresmap.maps))
                        for memb in membership
                    ]
                ]
            else:
                r = self.gmm.precisions_[membership]

            if v is not None:
                mDv = self.wiresmap * (mD * v)
                mDv = np.c_[mDv]
                r0 = (self.W * (mkvc(mDv))).reshape(mDv.shape, order="F")
                return mkvc(
                    mD.T
                    * (
                        self.W
                        * (mkvc(np.r_[[np.dot(r[i], r0[i]) for i in range(len(r0))]]))
                    )
                )
            else:
                # Forming the Hessian by diagonal blocks
                hlist = [
                    [r[:, i, j] for i in range(len(self.wiresmap.maps))]
                    for j in range(len(self.wiresmap.maps))
                ]
                Hr = sp.csc_matrix((0, 0), dtype=np.float64)
                for i in range(len(self.wiresmap.maps)):
                    Hc = sp.csc_matrix((0, 0), dtype=np.float64)
                    for j in range(len(self.wiresmap.maps)):
                        Hc = sp.hstack([Hc, sdiag(hlist[i][j])])
                    Hr = sp.vstack([Hr, Hc])

                Hr = Hr.dot(self.W)

                return (mD.T * mD) * (self.W * (Hr))

        else:
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
            else:
                return Hr


class SimplePGI(SimpleComboRegularization):
    """
    class similar to regularization.tikhonov.Simple, with a SimplePGIsmallness.
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
        alpha_s=1.0,
        alpha_x=1.0,
        alpha_y=1.0,
        alpha_z=1.0,
        alpha_xx=0.0,
        alpha_yy=0.0,
        alpha_zz=0.0,
        **kwargs
    ):
        self.gmmref = copy.deepcopy(gmmref)
        self.gmmref.order_clusters_GM_weight()
        self._gmm = copy.deepcopy(gmm)
        self._wiresmap = wiresmap
        self._maplist = maplist
        self._mesh = mesh
        self.mesh = mesh
        self._approx_hessian = approx_hessian
        self._approx_gradient = approx_gradient
        self._approx_eval = approx_eval
        self.mapping = IdentityMap(mesh, nP=self.wiresmap.nP)

        objfcts = [
            SimplePGIsmallness(
                mesh=mesh,
                gmm=self.gmm,
                wiresmap=self.wiresmap,
                maplist=self.maplist,
                approx_hessian=approx_hessian,
                approx_gradient=approx_gradient,
                approx_eval=approx_eval,
                mapping=self.mapping,
                **kwargs
            )
        ]
        # objfcts += [
        #     SimpleSmoothDeriv(
        #         mesh=mesh, orientation="x", mapping=maps * wire[1], **kwargs
        #     )
        #     for wire, maps in zip(self._wiresmap.maps, self._maplist)
        # ]
        # objfcts += [
        #     SmoothDeriv2(mesh=mesh, orientation="x", mapping=maps * wire[1], **kwargs)
        #     for wire, maps in zip(self._wiresmap.maps, self._maplist)
        # ]
        #
        # if mesh.dim > 1:
        #     objfcts += [
        #         SimpleSmoothDeriv(
        #             mesh=mesh, orientation="y", mapping=maps * wire[1], **kwargs
        #         )
        #         for wire, maps in zip(self._wiresmap.maps, self._maplist)
        #     ]
        #     objfcts += [
        #         SmoothDeriv2(
        #             mesh=mesh, orientation="y", mapping=maps * wire[1], **kwargs
        #         )
        #         for wire, maps in zip(self._wiresmap.maps, self._maplist)
        #     ]
        #
        # if mesh.dim > 2:
        #     objfcts += [
        #         SimpleSmoothDeriv(
        #             mesh=mesh, orientation="z", mapping=maps * wire[1], **kwargs
        #         )
        #         for wire, maps in zip(self._wiresmap.maps, self._maplist)
        #     ]
        #     objfcts += [
        #         SmoothDeriv2(
        #             mesh=mesh, orientation="z", mapping=maps * wire[1], **kwargs
        #         )
        #         for wire, maps in zip(self._wiresmap.maps, self._maplist)
        #     ]

        super(SimplePGI, self).__init__(
            mesh=mesh,
            alpha_s=alpha_s,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
            alpha_z=alpha_z,
            alpha_xx=alpha_xx,
            alpha_yy=alpha_yy,
            alpha_zz=alpha_zz,
            objfcts=objfcts,
            **kwargs
        )

    # Properties
    alpha_s = props.Float("PGI smallness multiplier")

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
            self._wiresmap = Wires(("m", self._mesh.nC))
        return self._wiresmap

    @wiresmap.setter
    def wiresmap(self, wm):
        if wm is not None:
            self._wiresmap = wm
        self.objfcts[0].wiresmap = self.wiresmap

    @property
    def maplist(self):
        if getattr(self, "_maplist", None) is None:
            self._maplist = [IdentityMap(self._mesh) for maps in self.wiresmap.maps]
        return self._maplist

    @maplist.setter
    def maplist(self, mp):
        if mp is not None:
            self._maplist = mp
        self.objfcts[0].maplist = self.maplist

    @property
    def approx_gradient(self):
        if getattr(self, "_approx_gradient", None) is None:
            self._approx_gradient = True
        return self._approx_gradient

    @approx_gradient.setter
    def approx_gradient(self, ap):
        if ap is not None:
            self._approx_gradient = ap
        self.objfcts[0].approx_gradient = self.approx_gradient

    @property
    def approx_hessian(self):
        if getattr(self, "_approx_hessian", None) is None:
            self._approx_hessian = True
        return self._approx_hessian

    @approx_hessian.setter
    def approx_hessian(self, ap):
        if ap is not None:
            self._approx_hessian = ap
        self.objfcts[0].approx_hessian = self.approx_hessian

    @property
    def approx_eval(self):
        if getattr(self, "_approx_eval", None) is None:
            self._approx_eval = True
        return self._approx_eval

    @approx_eval.setter
    def approx_eval(self, ap):
        if ap is not None:
            self._approx_eval = ap
        self.objfcts[0].approx_eval = self.approx_eval


class PGIsmallness(SimplePGIsmallness):
    """
    Smallness term for the petrophysically constrained regularization (PGI) with
    cell_weights similar to the ones used in regularization.tikhonov.Tikhonov

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

    def __init__(
        self,
        gmm,
        wiresmap=None,
        maplist=None,
        mesh=None,
        approx_hessian=True,
        approx_gradient=True,
        approx_eval=True,
        **kwargs
    ):

        super(PGIsmallness, self).__init__(
            gmm=gmm,
            wiresmap=wiresmap,
            maplist=maplist,
            mesh=mesh,
            approx_hessian=approx_hessian,
            approx_gradient=approx_gradient,
            approx_eval=approx_eval,
            **kwargs
        )

    @property
    def W(self):
        """
        Weighting matrix
        Need to change the size to match self.wiresmap.maps * mesh.nC
        """
        if self.cell_weights is not None:
            if len(self.cell_weights) == self.wiresmap.nP:
                return sdiag(np.sqrt(self.cell_weights))
            else:
                return sp.kron(
                    speye(len(self.wiresmap.maps)), sdiag(np.sqrt(self.cell_weights))
                )

        if self.cell_weights is not None:
            if len(self.cell_weights) == self.wiresmap.nP:
                return (
                    sp.kron(
                        speye(len(self.wiresmap.maps)),
                        sdiag(np.sqrt(self.regmesh.vol)),
                    )
                    * sdiag(np.sqrt(self.cell_weights))
                )
            else:
                return sp.kron(
                    speye(len(self.wiresmap.maps)),
                    sdiag(np.sqrt(self.regmesh.vol)),
                ) * sp.kron(
                    speye(len(self.wiresmap.maps)), sdiag(np.sqrt(self.cell_weights))
                )
        else:
            return sp.kron(
                speye(len(self.wiresmap.maps)),
                sdiag(np.sqrt(self.regmesh.vol)),
            )


class PGI(SimpleComboRegularization):
    """
    class similar to regularization.tikhonov.Simple, with a SimplePGIsmallness.
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
        alpha_s=1.0,
        alpha_x=1.0,
        alpha_y=1.0,
        alpha_z=1.0,
        alpha_xx=0.0,
        alpha_yy=0.0,
        alpha_zz=0.0,
        **kwargs
    ):
        self.gmmref = copy.deepcopy(gmmref)
        self.gmmref.order_clusters_GM_weight()
        self._gmm = copy.deepcopy(gmm)
        self._wiresmap = wiresmap
        self._maplist = maplist
        self._mesh = mesh
        self.mesh = mesh
        self._approx_hessian = approx_hessian
        self._approx_gradient = approx_gradient
        self._approx_eval = approx_eval
        self.mapping = IdentityMap(mesh, nP=self.wiresmap.nP)

        objfcts = [
            PGIsmallness(
                mesh=mesh,
                gmm=self.gmm,
                wiresmap=self.wiresmap,
                maplist=self.maplist,
                approx_gradient=approx_gradient,
                approx_eval=approx_eval,
                mapping=self.mapping,
                **kwargs
            )
        ]
        # objfcts += [
        #     SmoothDeriv(mesh=mesh, orientation="x", mapping=maps * wire[1], **kwargs)
        #     for wire, maps in zip(self._wiresmap.maps, self._maplist)
        # ]
        # objfcts += [
        #     SmoothDeriv2(mesh=mesh, orientation="x", mapping=maps * wire[1], **kwargs)
        #     for wire, maps in zip(self._wiresmap.maps, self._maplist)
        # ]
        #
        # for key in kwargs.keys():
        #     print("kwargs key: ", key)
        #
        # if mesh.dim > 1:
        #     objfcts += [
        #         SmoothDeriv(
        #             mesh=mesh, orientation="y", mapping=maps * wire[1], **kwargs
        #         )
        #         for wire, maps in zip(self._wiresmap.maps, self._maplist)
        #     ]
        #     objfcts += [
        #         SmoothDeriv2(
        #             mesh=mesh, orientation="y", mapping=maps * wire[1], **kwargs
        #         )
        #         for wire, maps in zip(self._wiresmap.maps, self._maplist)
        #     ]
        #
        # if mesh.dim > 2:
        #     objfcts += [
        #         SmoothDeriv(
        #             mesh=mesh, orientation="z", mapping=maps * wire[1], **kwargs
        #         )
        #         for wire, maps in zip(self._wiresmap.maps, self._maplist)
        #     ]
        #     objfcts += [
        #         SmoothDeriv2(
        #             mesh=mesh, orientation="z", mapping=maps * wire[1], **kwargs
        #         )
        #         for wire, maps in zip(self._wiresmap.maps, self._maplist)
        #     ]

        super(PGI, self).__init__(
            mesh=mesh,
            alpha_s=alpha_s,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
            alpha_z=alpha_z,
            alpha_xx=alpha_xx,
            alpha_yy=alpha_yy,
            alpha_zz=alpha_zz,
            objfcts=objfcts,
            **kwargs
        )

    # Properties
    alpha_s = props.Float("PGI smallness multiplier")

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
            self._wiresmap = Wires(("m", self._mesh.nC))
        return self._wiresmap

    @wiresmap.setter
    def wiresmap(self, wm):
        if wm is not None:
            self._wiresmap = wm
        self.objfcts[0].wiresmap = self.wiresmap

    @property
    def maplist(self):
        if getattr(self, "_maplist", None) is None:
            self._maplist = [IdentityMap(self._mesh) for maps in self.wiresmap.maps]
        return self._maplist

    @maplist.setter
    def maplist(self, mp):
        if mp is not None:
            self._maplist = mp
        self.objfcts[0].maplist = self.maplist

    @property
    def approx_gradient(self):
        if getattr(self, "_approx_gradient", None) is None:
            self._approx_gradient = True
        return self._approx_gradient

    @property
    def approx_hessian(self):
        if getattr(self, "_approx_hessian", None) is None:
            self._approx_hessian = True
        return self._approx_hessian

    @approx_hessian.setter
    def approx_hessian(self, ap):
        if ap is not None:
            self._approx_hessian = ap
        self.objfcts[0].approx_hessian = self.approx_hessian

    @approx_gradient.setter
    def approx_gradient(self, ap):
        if ap is not None:
            self._approx_gradient = ap
        self.objfcts[0].approx_gradient = self.approx_gradient

    @property
    def approx_eval(self):
        if getattr(self, "_approx_eval", None) is None:
            self._approx_eval = True
        return self._approx_eval

    @approx_eval.setter
    def approx_eval(self, ap):
        if ap is not None:
            self._approx_eval = ap
        self.objfcts[0].approx_eval = self.approx_eval


class SimplePGIwithNonlinearRelationshipsSmallness(BaseRegularization):
    """
    Smallness term for the petrophysically constrained regularization (PGI) with
    nonlinear relationships between physical properties and cells_weight s
    imilar to the ones used in regularization.tikhonov.Simple.

    PARAMETERS
    ----------
    :param SimPEG.utils.GaussianMixtureWithNonlinearRelationships gmm: GMM to use
    :param SimPEG.maps.Wires wiresmap: wires mapping to the various physical properties
    :param list maplist: list of SimPEG.maps for each physical property.
    :param discretize.BaseMesh mesh: tensor, QuadTree or Octree mesh
    :param boolean approx_gradient: use the L2-approximation of the gradient, default is True
    :param boolean approx_eval: use the L2-approximation evaluation of the smallness term
    """

    _multiplier_pair = "alpha_s"

    def __init__(
        self,
        gmm,
        wiresmap=None,
        maplist=None,
        mesh=None,
        approx_gradient=True,
        approx_eval=True,
        **kwargs
    ):

        self.approx_gradient = approx_gradient
        self.approx_eval = approx_eval

        super(SimplePGIwithNonlinearRelationshipsSmallness, self).__init__(
            mesh=mesh, **kwargs
        )
        self.gmm = gmm
        self.wiresmap = wiresmap
        self.maplist = maplist

        # storing the numpy.polynomial derivatives computations (somewhat long)
        self._r_first_deriv = None
        self._r_second_deriv = None

    @property
    def W(self):
        """
        Weighting matrix
        Need to change the size to match self.wiresmap.maps * mesh.nC
        """
        if self.cell_weights is not None:
            if len(self.cell_weights) == self.wiresmap.nP:
                return sdiag(np.sqrt(self.cell_weights))
            else:
                return sp.kron(
                    speye(len(self.wiresmap.maps)), sdiag(np.sqrt(self.cell_weights))
                )
        else:
            return Identity()

    @properties.validator("cell_weights")
    def _validate_cell_weights(self, change):
        if change["value"] is not None:
            if self._nC_residual != "*":
                if (len(change["value"]) != self._nC_residual) and (
                    len(change["value"]) != len(self.wiresmap.maps) * self._nC_residual
                ):
                    raise Exception(
                        "cell_weights must be length {} or {} not {}".format(
                            self._nC_residual,
                            len(self.wiresmap.maps) * self._nC_residual,
                            len(change["value"]),
                        )
                    )

    def membership(self, m):
        modellist = self.wiresmap * m
        model = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
        return self.gmm.predict(model)

    def compute_quasi_geology_model(self):
        #used once mref is built
        mreflist = self.wiresmap * self.mref
        mrefarray = np.c_[[a * b for a, b in zip(self.maplist, mreflist)]].T
        return np.c_[[((mrefarray - mean)**2).sum(axis=1) for mean in self.gmm.means_]].argmin(axis=0)


    @timeIt
    def __call__(self, m, externalW=True):

        if externalW:
            W = self.W
        else:
            W = Identity()

        if getattr(self, "mref", None) is None:
            self.mref = mkvc(self.gmm.means_[self.membership(m)])

        if self.approx_eval:
            membership = self.compute_quasi_geology_model()
            dm = self.wiresmap * (m)
            dmref = self.wiresmap * (self.mref)
            dmm = np.c_[[a * b for a, b in zip(self.maplist, dm)]].T
            dmm = np.r_[
                [
                    self.gmm.cluster_mapping[membership[i]] * dmm[i].reshape(-1, 2)
                    for i in range(dmm.shape[0])
                ]
            ].reshape(-1, 2)
            dmmref = np.c_[[a for a in dmref]].T
            dmr = dmm - dmmref
            r0 = W * mkvc(dmr)

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

            return 0.5 * r0.dot(W * mkvc(r1))

        else:
            modellist = self.wiresmap * m
            model = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
            score = self.gmm.score_samples(model)
            score_vec = mkvc(np.r_[[score for maps in self.wiresmap.maps]])
            return -np.sum((W.T * W) * score_vec) / len(self.wiresmap.maps)

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

        if self.approx_gradient:
            dmm = np.r_[
                [
                    self.gmm.cluster_mapping[membership[i]] * dmmodel[i].reshape(-1, 2)
                    for i in range(dmmodel.shape[0])
                ]
            ].reshape(-1, 2)
            dmmref = np.c_[[a for a in mreflist]].T
            dm = dmm - dmmref

            if self.gmm.covariance_type == "tied":
                raise Exception("Not implemented")
            else:
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
            return mkvc(mD.T * (self.W.T * r))

        else:
            raise Exception("Not implemented")

    @timeIt
    def deriv2(self, m, v=None):

        if getattr(self, "mref", None) is None:
            self.mref = mkvc(self.gmm.means_[self.membership(m)])

        # For a positive definite Hessian,
        # we approximate it with the covariance of the cluster
        # whose each point belong
        membership = self.compute_quasi_geology_model()
        modellist = self.wiresmap * m
        dmmodel = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
        mD = [a.deriv(b) for a, b in zip(self.maplist, modellist)]
        mD = sp.block_diag(mD)

        if self._r_second_deriv is None:
            if self.gmm.covariance_type == "tied":
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
            elif (
                self.gmm.covariance_type == "spherical"
                or self.gmm.covariance_type == "diag"
            ):
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


class SimplePGIwithRelationships(SimpleComboRegularization):
    """
    class similar to regularization.tikhonov.Simple, with a
    SimplePGIwithNonlinearRelationshipsSmallness.

    PARAMETERS
    ----------
    :param SimPEG.utils.GaussianMixtureWithNonlinearRelationships gmmref: refereence/prior GMM
    :param SimPEG.utils.GaussianMixtureWithNonlinearRelationships gmm: GMM to use
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
        self.gmmref = copy.deepcopy(gmmref)
        self.gmmref.order_clusters_GM_weight()
        self._gmm = copy.deepcopy(gmm)
        self._wiresmap = wiresmap
        self._maplist = maplist
        self._mesh = mesh
        self.mesh = mesh
        self._approx_gradient = approx_gradient
        self._approx_eval = approx_eval
        self.mapping = IdentityMap(mesh, nP=self.wiresmap.nP)

        objfcts = [
            SimplePGIwithNonlinearRelationshipsSmallness(
                mesh=mesh,
                gmm=self.gmm,
                wiresmap=self.wiresmap,
                maplist=self.maplist,
                approx_gradient=approx_gradient,
                approx_eval=approx_eval,
                mapping=self.mapping,
                **kwargs
            )
        ]
        objfcts += [
            SimpleSmoothDeriv(
                mesh=mesh, orientation="x", mapping=maps * wire[1], **kwargs
            )
            for wire, maps in zip(self._wiresmap.maps, self._maplist)
        ]
        objfcts += [
            SmoothDeriv2(mesh=mesh, orientation="x", mapping=maps * wire[1], **kwargs)
            for wire, maps in zip(self._wiresmap.maps, self._maplist)
        ]

        if mesh.dim > 1:
            objfcts += [
                SimpleSmoothDeriv(
                    mesh=mesh, orientation="y", mapping=maps * wire[1], **kwargs
                )
                for wire, maps in zip(self._wiresmap.maps, self._maplist)
            ]
            objfcts += [
                SmoothDeriv2(
                    mesh=mesh, orientation="y", mapping=maps * wire[1], **kwargs
                )
                for wire, maps in zip(self._wiresmap.maps, self._maplist)
            ]

        if mesh.dim > 2:
            objfcts += [
                SimpleSmoothDeriv(
                    mesh=mesh, orientation="z", mapping=maps * wire[1], **kwargs
                )
                for wire, maps in zip(self._wiresmap.maps, self._maplist)
            ]
            objfcts += [
                SmoothDeriv2(
                    mesh=mesh, orientation="z", mapping=maps * wire[1], **kwargs
                )
                for wire, maps in zip(self._wiresmap.maps, self._maplist)
            ]

        super(SimplePGIwithRelationships, self).__init__(
            mesh=mesh,
            alpha_s=alpha_s,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
            alpha_z=alpha_z,
            alpha_xx=alpha_xx,
            alpha_yy=alpha_yy,
            alpha_zz=alpha_zz,
            objfcts=objfcts,
            **kwargs
        )

    # Properties
    alpha_s = props.Float("PGI smallness multiplier")

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

    # @classmethod
    def membership(self, m):
        return self.objfcts[0].membership(m)

    def compute_quasi_geology_model(self):
        return self.objfcts[0].compute_quasi_geology_model()

    @property
    def wiresmap(self):
        if getattr(self, "_wiresmap", None) is None:
            self._wiresmap = Wires(("m", self._mesh.nC))
        return self._wiresmap

    @wiresmap.setter
    def wiresmap(self, wm):
        if wm is not None:
            self._wiresmap = wm
        self.objfcts[0].wiresmap = self.wiresmap

    @property
    def maplist(self):
        if getattr(self, "_maplist", None) is None:
            self._maplist = [IdentityMap(self._mesh) for maps in self.wiresmap.maps]
        return self._maplist

    @maplist.setter
    def maplist(self, mp):
        if mp is not None:
            self._maplist = mp
        self.objfcts[0].maplist = self.maplist

    @property
    def approx_gradient(self):
        if getattr(self, "_approx_gradient", None) is None:
            self._approx_gradient = True
        return self._approx_gradient

    @approx_gradient.setter
    def approx_gradient(self, ap):
        if ap is not None:
            self._approx_gradient = ap
        self.objfcts[0].approx_gradient = self.approx_gradient
