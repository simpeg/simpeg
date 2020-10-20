from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import warnings
import properties
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import copy
from ..utils import (
    speye, setKwargs, sdiag, mkvc, timeIt,
    Identity, Zero, order_clusters_GM_weight,
    ComputeConstantTerm, coterminal
)
from ..maps import IdentityMap, Wires
#from .. import objective_function
from .. import props

from .base import (
    BaseRegularization,
    SimpleComboRegularization,
    BaseComboRegularization,
)
from .tikhonov import *

__all__ = [
    'SimpleSmall', 'SimpleSmoothDeriv', 'Simple',
    'Small', 'SmoothDeriv', 'SmoothDeriv2', 'Tikhonov',
    'SparseSmall', 'SparseDeriv', 'Sparse',
]


###############################################################################
#                                                                             #
#                 Petrophysically-Constrained Regularization                  #
#                                                                             #
###############################################################################


# Simple Petrophysical Regularization
#####################################


class SimplePetroSmallness(BaseRegularization):
    """
    Smallness term for the petrophysically constrained regularization
    """

    _multiplier_pair = 'alpha_s'

    def __init__(self, GMmodel, wiresmap=None,
                 maplist=None, mesh=None,
                 approx_gradient=True,
                 evaltype='approx',
                 **kwargs):

        self.approx_gradient = approx_gradient
        self.evaltype = evaltype

        super(SimplePetroSmallness, self).__init__(
            mesh=mesh, **kwargs
        )
        self.GMmodel = GMmodel
        self.wiresmap = wiresmap
        self.maplist = maplist

        # TODO: Save repetitive computations (see withmapping implementation)
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
                    speye(len(self.wiresmap.maps)),
                    sdiag(np.sqrt(self.cell_weights))
                )
        else:
            return Identity()

    @properties.validator('cell_weights')
    def _validate_cell_weights(self, change):
        if change['value'] is not None:
            if self._nC_residual != '*':
                if (len(change['value']) != self._nC_residual) and (len(change['value']) != len(self.wiresmap.maps) * self._nC_residual):
                    raise Exception(
                        'cell_weights must be length {} or {} not {}'.format(
                            self._nC_residual,
                            len(self.wiresmap.maps) * self._nC_residual,
                            len(change['value'])
                        )
                    )

    def membership(self, m):
        modellist = self.wiresmap * m
        model = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
        return self.GMmodel.predict(model)  # mkvc(m, numDims=2))

    @timeIt
    def __call__(self, m, externalW=True):

        if externalW:
            W = self.W
        else:
            W = Identity()

        if getattr(self, 'mref', None) is None:
            self.mref = mkvc(self.GMmodel.means_[self.membership(m)])

        if self.evaltype == 'approx':
            membership = self.membership(self.mref)
            dm = self.wiresmap * (m)
            dmref = self.wiresmap * (self.mref)
            dmm = np.c_[[a * b for a, b in zip(self.maplist, dm)]].T
            dmmref = np.c_[[a for a in dmref]].T
            dmr = dmm - dmmref
            r0 = (W * mkvc(dmr)).reshape(dmr.shape, order='F')

            if self.GMmodel.covariance_type == 'tied':
                r1 = np.r_[[np.dot(self.GMmodel.precisions_, np.r_[r0[i]])
                            for i in range(len(r0))]]
            elif self.GMmodel.covariance_type == 'diag' or self.GMmodel.covariance_type == 'spherical':
                r1 = np.r_[[np.dot(self.GMmodel.precisions_[membership[i]] * np.eye(len(self.wiresmap.maps)),
                                   np.r_[r0[i]]) for i in range(len(r0))]]
            else:
                r1 = np.r_[[np.dot(self.GMmodel.precisions_[membership[i]],
                                   np.r_[r0[i]]) for i in range(len(r0))]]

            return 0.5 * mkvc(r0).dot(mkvc(r1))

        elif self.evaltype == 'full':
            modellist = self.wiresmap * m
            model = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
            score = self.GMmodel.score_samples(
                model) + ComputeConstantTerm(self.GMmodel)
            score_vec = mkvc(
                np.r_[[score for maps in self.wiresmap.maps]])
            return -np.sum((W.T * W) * score_vec) / len(self.wiresmap.maps)

    @timeIt
    def deriv(self, m):

        if getattr(self, 'mref', None) is None:
            self.mref = mkvc(self.GMmodel.means_[self.membership(m)])

        membership = self.membership(self.mref)
        modellist = self.wiresmap * m
        mreflist = self.wiresmap * self.mref
        mD = [a.deriv(b) for a, b in zip(self.maplist, modellist)]
        mD = sp.block_diag(mD)

        if self.approx_gradient == True:
            dmmodel = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
            dmmref = np.c_[[a for a in mreflist]].T
            dm = dmmodel - dmmref
            r0 = (self.W * (mkvc(dm))).reshape(dm.shape, order='F')

            if self.GMmodel.covariance_type == 'tied':
                r = mkvc(
                    np.r_[[np.dot(self.GMmodel.precisions_, r0[i]) for i in range(len(r0))]])
            elif self.GMmodel.covariance_type == 'diag' or self.GMmodel.covariance_type == 'spherical':
                r = mkvc(np.r_[[np.dot(self.GMmodel.precisions_[
                               membership[i]] * np.eye(len(self.wiresmap.maps)), r0[i]) for i in range(len(r0))]])
            else:
                r = mkvc(np.r_[[np.dot(self.GMmodel.precisions_[
                               membership[i]], r0[i]) for i in range(len(r0))]])
            return mkvc(mD.T * (self.W * r))

        else:
            modellist = self.wiresmap * m
            model = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
            score = self.GMmodel.score_samples(model)
            score_vec = np.hstack([score for maps in self.wiresmap.maps])
            logP = np.zeros((len(model), self.GMmodel.n_components))
            W = []
            for k in range(self.GMmodel.n_components):
                if self.GMmodel.covariance_type == 'tied':
                    logP[:, k] = mkvc(multivariate_normal(
                        self.GMmodel.means_[k], self.GMmodel.covariances_).logpdf(model))
                    W.append(self.GMmodel.weights_[k] * mkvc(np.r_[[np.dot(
                        self.GMmodel.precisions_, (model[i] - self.GMmodel.means_[k]).T)for i in range(len(model))]]))
                elif self.GMmodel.covariance_type == 'diag' or self.GMmodel.covariance_type == 'spherical':
                    logP[:, k] = mkvc(multivariate_normal(
                        self.GMmodel.means_[k], self.GMmodel.covariances_[k] * np.eye(len(self.wiresmap.maps))).logpdf(model))
                    W.append(self.GMmodel.weights_[k] * mkvc(np.r_[[np.dot(self.GMmodel.precisions_[
                             k] * np.eye(len(self.wiresmap.maps)), (model[i] - self.GMmodel.means_[k]).T)for i in range(len(model))]]))
                else:
                    logP[:, k] = mkvc(multivariate_normal(
                        self.GMmodel.means_[k], self.GMmodel.covariances_[k]).logpdf(model))
                    W.append(self.GMmodel.weights_[k] * mkvc(np.r_[[np.dot(self.GMmodel.precisions_[
                             k], (model[i] - self.GMmodel.means_[k]).T)for i in range(len(model))]]))
            W = (np.c_[W].T)
            logP = np.vstack([logP for maps in self.wiresmap.maps])
            numer, sign = logsumexp(logP, axis=1, b=W, return_sign=True)
            logderiv = numer - score_vec
            r = sign * np.exp(logderiv)
            return mkvc(mD.T * (self.W.T * (self.W * r)))

    @timeIt
    def deriv2(self, m, v=None):

        if getattr(self, 'mref', None) is None:
            self.mref = mkvc(self.GMmodel.means_[self.membership(m)])

        # For a positive definite Hessian,
        # we approximate it with the covariance of the cluster
        # whose each point belong
        membership = self.membership(self.mref)
        modellist = self.wiresmap * m
        mD = [a.deriv(b) for a, b in zip(self.maplist, modellist)]
        mD = sp.block_diag(mD)

        if self.GMmodel.covariance_type == 'tied':
            r = self.GMmodel.precisions_[np.newaxis, :, :][
                np.zeros_like(membership)]
        elif self.GMmodel.covariance_type == 'spherical' or self.GMmodel.covariance_type == 'diag':
            r = np.r_[[self.GMmodel.precisions_[memb] *
                       np.eye(len(self.wiresmap.maps)) for memb in membership]]
        else:
            r = self.GMmodel.precisions_[membership]

        if v is not None:
            mDv = self.wiresmap * (mD * v)
            mDv = np.c_[mDv]
            r0 = (self.W * (mkvc(mDv))).reshape(mDv.shape, order='F')
            return mkvc(mD.T * (self.W * (
                                mkvc(np.r_[[np.dot(r[i], r0[i]) for i in range(len(r0))]]))))
        else:
            # Forming the Hessian by diagonal blocks
            hlist = [[r[:, i, j]
                      for i in range(len(self.wiresmap.maps))]
                     for j in range(len(self.wiresmap.maps))]

            Hr = sp.csc_matrix((0, 0), dtype=np.float64)
            for i in range(len(self.wiresmap.maps)):
                Hc = sp.csc_matrix((0, 0), dtype=np.float64)
                for j in range(len(self.wiresmap.maps)):
                    Hc = sp.hstack([Hc, sdiag(hlist[i][j])])
                Hr = sp.vstack([Hr, Hc])

            Hr = Hr.dot(self.W)
            #mDW = self.W * mD

            return (mD.T * mD) * (self.W * (Hr))


class SimplePetroRegularization(SimpleComboRegularization):

    def __init__(
        self, mesh, GMmref, GMmodel=None,
        wiresmap=None, maplist=None, approx_gradient=True,
        evaltype='approx',
        alpha_s=1.0, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
        alpha_xx=0., alpha_yy=0., alpha_zz=0.,
        **kwargs
    ):
        self.GMmref = copy.deepcopy(GMmref)
        order_clusters_GM_weight(self.GMmref)
        self._GMmodel = copy.deepcopy(GMmodel)
        self._wiresmap = wiresmap
        self._maplist = maplist
        self._mesh = mesh
        self.mesh = mesh
        self._approx_gradient = approx_gradient
        self._evaltype = evaltype
        self.mapping = IdentityMap(mesh, nP=self.wiresmap.nP)

        objfcts = [
            SimplePetroSmallness(mesh=mesh, GMmodel=self.GMmodel, wiresmap=self.wiresmap,
                                 maplist=self.maplist, approx_gradient=approx_gradient,
                                 evaltype=evaltype,
                                 mapping=self.mapping, **kwargs)
        ]
        objfcts += [
            SimpleSmoothDeriv(mesh=mesh, orientation='x',
                              mapping=maps * wire[1], **kwargs)
            for wire, maps in zip(self._wiresmap.maps, self._maplist)]
        objfcts += [
            SmoothDeriv2(mesh=mesh, orientation='x',
                         mapping=maps * wire[1], **kwargs)
            for wire, maps in zip(self._wiresmap.maps, self._maplist)]

        if mesh.dim > 1:
            objfcts += [
                SimpleSmoothDeriv(mesh=mesh, orientation='y',
                                  mapping=maps * wire[1], **kwargs)
                for wire, maps in zip(self._wiresmap.maps, self._maplist)]
            objfcts += [
                SmoothDeriv2(mesh=mesh, orientation='y',
                             mapping=maps * wire[1], **kwargs)
                for wire, maps in zip(self._wiresmap.maps, self._maplist)]

        if mesh.dim > 2:
            objfcts += [
                SimpleSmoothDeriv(mesh=mesh, orientation='z',
                                  mapping=maps * wire[1], **kwargs)
                for wire, maps in zip(self._wiresmap.maps, self._maplist)]
            objfcts += [
                SmoothDeriv2(mesh=mesh, orientation='z',
                             mapping=maps * wire[1], **kwargs)
                for wire, maps in zip(self._wiresmap.maps, self._maplist)]

        super(SimplePetroRegularization, self).__init__(
            mesh=mesh,
            alpha_s=alpha_s, alpha_x=alpha_x, alpha_y=alpha_y, alpha_z=alpha_z,
            alpha_xx=alpha_xx, alpha_yy=alpha_yy, alpha_zz=alpha_zz,
            objfcts=objfcts, **kwargs)

        #setKwargs(self, **kwargs)

    # Properties
    alpha_s = props.Float("PetroPhysics weights")

    @property
    def GMmodel(self):
        if getattr(self, '_GMmodel', None) is None:
            self._GMmodel = copy.deepcopy(self.GMmref)
        return self._GMmodel

    @GMmodel.setter
    def GMmodel(self, gm):
        if gm is not None:
            self._GMmodel = copy.deepcopy(gm)
        self.objfcts[0].GMmodel = self.GMmodel

    def membership(self, m):
        return self.objfcts[0].membership(m)

    @property
    def wiresmap(self):
        if getattr(self, '_wiresmap', None) is None:
            self._wiresmap = Wires(('m', self._mesh.nC))
        return self._wiresmap

    @wiresmap.setter
    def wiresmap(self, wm):
        if wm is not None:
            self._wiresmap = wm
        self.objfcts[0].wiresmap = self.wiresmap

    @property
    def maplist(self):
        if getattr(self, '_maplist', None) is None:
            self._maplist = [IdentityMap(
                self._mesh) for maps in self.wiresmap.maps]
        return self._maplist

    @maplist.setter
    def maplist(self, mp):
        if mp is not None:
            self._maplist = mp
        self.objfcts[0].maplist = self.maplist

    @property
    def approx_gradient(self):
        if getattr(self, '_approx_gradient', None) is None:
            self._approx_gradient = True
        return self._approx_gradient

    @approx_gradient.setter
    def approx_gradient(self, ap):
        if ap is not None:
            self._approx_gradient = ap
        self.objfcts[0].approx_gradient = self.approx_gradient


class PetroSmallness(SimplePetroSmallness):
    """
    Smallness term for the petrophysically constrained regularization
    """

    _multiplier_pair = 'alpha_s'

    def __init__(self, GMmodel, wiresmap=None,
                 maplist=None, mesh=None,
                 approx_gradient=True,
                 evaltype='approx',
                 **kwargs):

        super(PetroSmallness, self).__init__(
            GMmodel=GMmodel, wiresmap=wiresmap,
            maplist=maplist, mesh=mesh,
            approx_gradient=True,
            evaltype=evaltype,
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
                return sp.kron(
                    speye(len(self.wiresmap.maps)),
                    sdiag(np.sqrt(self.regmesh.vol[self.indActive]))
                ) * sdiag(np.sqrt(self.cell_weights))
            else:
                return sp.kron(
                    speye(len(self.wiresmap.maps)),
                    sdiag(np.sqrt(self.regmesh.vol[self.indActive]))
                ) * sp.kron(
                    speye(len(self.wiresmap.maps)),
                    sdiag(np.sqrt(self.cell_weights))
                )
        else:
            return sp.kron(
                speye(len(self.wiresmap.maps)),
                sdiag(np.sqrt(self.regmesh.vol[self.indActive]))
            )

    # @properties.validator('cell_weights')
    # def _validate_cell_weights(self, change):
    #     if change['value'] is not None:
    #         if self._nC_residual != '*':
    #             if (len(change['value']) != self._nC_residual) and (len(change['value']) != len(self.wiresmap.maps) * self._nC_residual):
    #                 raise Exception(
    #                     'cell_weights must be length {} or {} not {}'.format(
    #                         self._nC_residual,
    #                         len(self.wiresmap.maps) * self._nC_residual,
    #                         len(change['value'])
    #                     )
    #                 )

    # def membership(self, m):
    #     modellist = self.wiresmap * m
    #     model = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
    #     return self.GMmodel.predict(model)  # mkvc(m, numDims=2))

    # @timeIt
    # def __call__(self, m, externalW=True):

    #     if externalW:
    #         W = self.W
    #     else:
    #         W = Identity()

    #     if getattr(self, 'mref', None) is None:
    #         self.mref = mkvc(self.GMmodel.means_[self.membership(m)])

    #     if self.evaltype == 'approx':
    #         membership = self.membership(self.mref)
    #         dm = self.wiresmap * (m)
    #         dmref = self.wiresmap * (self.mref)
    #         dmm = np.c_[[a * b for a, b in zip(self.maplist, dm)]].T
    #         dmmref = np.c_[[a for a in dmref]].T
    #         dmr = dmm - dmmref
    #         r0 = W * mkvc(dmr)

    #         if self.GMmodel.covariance_type == 'tied':
    #             r1 = np.r_[[np.dot(self.GMmodel.precisions_, np.r_[dmr[i]])
    #                         for i in range(len(dmr))]]
    #         elif self.GMmodel.covariance_type == 'diag' or self.GMmodel.covariance_type == 'spherical':
    #             r1 = np.r_[[np.dot(self.GMmodel.precisions_[membership[i]] * np.eye(len(self.wiresmap.maps)),
    #                                np.r_[dmr[i]]) for i in range(len(dmr))]]
    #         else:
    #             r1 = np.r_[[np.dot(self.GMmodel.precisions_[membership[i]],
    #                                np.r_[dmr[i]]) for i in range(len(dmr))]]

    #         return 0.5 * r0.dot(W * mkvc(r1))

    #     elif self.evaltype == 'full':
    #         modellist = self.wiresmap * m
    #         model = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
    #         score = self.GMmodel.score_samples(
    #             model) + ComputeConstantTerm(self.GMmodel)
    #         score_vec = mkvc(
    #             np.r_[[score for maps in self.wiresmap.maps]])
    #         return -np.sum((W.T * W) * score_vec) / len(self.wiresmap.maps)

    # @timeIt
    # def deriv(self, m):

    #     if getattr(self, 'mref', None) is None:
    #         self.mref = mkvc(self.GMmodel.means_[self.membership(m)])

    #     membership = self.membership(self.mref)
    #     modellist = self.wiresmap * m
    #     mreflist = self.wiresmap * self.mref
    #     mD = [a.deriv(b) for a, b in zip(self.maplist, modellist)]
    #     mD = sp.block_diag(mD)

    #     if self.approx_gradient == True:
    #         dmmodel = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
    #         dmmref = np.c_[[a for a in mreflist]].T
    #         dm = dmmodel - dmmref

    #         if self.GMmodel.covariance_type == 'tied':
    #             r = self.W * \
    #                 mkvc(
    #                     np.r_[[np.dot(self.GMmodel.precisions_, dm[i]) for i in range(len(dm))]])
    #         elif self.GMmodel.covariance_type == 'diag' or self.GMmodel.covariance_type == 'spherical':
    #             r = self.W * \
    #                 mkvc(np.r_[[np.dot(self.GMmodel.precisions_[
    #                            membership[i]] * np.eye(len(self.wiresmap.maps)), dm[i]) for i in range(len(dm))]])
    #         else:
    #             r = self.W * \
    #                 mkvc(np.r_[[np.dot(self.GMmodel.precisions_[
    #                            membership[i]], dm[i]) for i in range(len(dm))]])
    #         return mkvc(mD.T * (self.W.T * r))

    #     else:
    #         modellist = self.wiresmap * m
    #         model = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
    #         score = self.GMmodel.score_samples(model)
    #         score_vec = np.hstack([score for maps in self.wiresmap.maps])
    #         logP = np.zeros((len(model), self.GMmodel.n_components))
    #         W = []
    #         for k in range(self.GMmodel.n_components):
    #             if self.GMmodel.covariance_type == 'tied':
    #                 logP[:, k] = mkvc(multivariate_normal(
    #                     self.GMmodel.means_[k], self.GMmodel.covariances_).logpdf(model))
    #                 W.append(self.GMmodel.weights_[k] * mkvc(np.r_[[np.dot(
    #                     self.GMmodel.precisions_, (model[i] - self.GMmodel.means_[k]).T)for i in range(len(model))]]))
    #             elif self.GMmodel.covariance_type == 'diag' or self.GMmodel.covariance_type == 'spherical':
    #                 logP[:, k] = mkvc(multivariate_normal(
    #                     self.GMmodel.means_[k], self.GMmodel.covariances_[k] * np.eye(len(self.wiresmap.maps))).logpdf(model))
    #                 W.append(self.GMmodel.weights_[k] * mkvc(np.r_[[np.dot(self.GMmodel.precisions_[k] * np.eye(
    #                     len(self.wiresmap.maps)), (model[i] - self.GMmodel.means_[k]).T)for i in range(len(model))]]))
    #             else:
    #                 logP[:, k] = mkvc(multivariate_normal(
    #                     self.GMmodel.means_[k], self.GMmodel.covariances_[k]).logpdf(model))
    #                 W.append(self.GMmodel.weights_[k] * mkvc(np.r_[[np.dot(self.GMmodel.precisions_[
    #                          k], (model[i] - self.GMmodel.means_[k]).T)for i in range(len(model))]]))
    #         W = (np.c_[W].T)
    #         logP = np.vstack([logP for maps in self.wiresmap.maps])
    #         numer, sign = logsumexp(logP, axis=1, b=W, return_sign=True)
    #         logderiv = numer - score_vec
    #         r = sign * np.exp(logderiv)
    #         return mkvc(mD.T * (self.W.T * (self.W * r)))

    # @timeIt
    # def deriv2(self, m, v=None):

    #     if getattr(self, 'mref', None) is None:
    #         self.mref = mkvc(self.GMmodel.means_[self.membership(m)])

    #     # For a positive definite Hessian,
    #     # we approximate it with the covariance of the cluster
    #     # whose each point belong
    #     membership = self.membership(self.mref)
    #     modellist = self.wiresmap * m
    #     mD = [a.deriv(b) for a, b in zip(self.maplist, modellist)]
    #     mD = sp.block_diag(mD)

    #     if self.GMmodel.covariance_type == 'tied':
    #         r = self.GMmodel.precisions_[np.newaxis, :, :][
    #             np.zeros_like(membership)]
    #     elif self.GMmodel.covariance_type == 'spherical' or self.GMmodel.covariance_type == 'diag':
    #         r = np.r_[[self.GMmodel.precisions_[memb] *
    #                    np.eye(len(self.wiresmap.maps)) for memb in membership]]
    #     else:
    #         r = self.GMmodel.precisions_[membership]

    #     if v is not None:
    #         mDv = self.wiresmap * (mD * v)
    #         mDv = np.c_[mDv]
    #         return mkvc(mD.T * ((self.W.T * self.W) *
    #                             mkvc(np.r_[[np.dot(r[i], mDv[i]) for i in range(len(mDv))]])))
    #     else:
    #         # Forming the Hessian by diagonal blocks
    #         hlist = [[r[:, i, j]
    #                   for i in range(len(self.wiresmap.maps))]
    #                  for j in range(len(self.wiresmap.maps))]

    #         Hr = sp.csc_matrix((0, 0), dtype=np.float64)
    #         for i in range(len(self.wiresmap.maps)):
    #             Hc = sp.csc_matrix((0, 0), dtype=np.float64)
    #             for j in range(len(self.wiresmap.maps)):
    #                 Hc = sp.hstack([Hc, sdiag(hlist[i][j])])
    #             Hr = sp.vstack([Hr, Hc])

    #         mDW = self.W * mD

    #         return (mDW.T * mDW) * Hr


class PetroRegularization(SimpleComboRegularization):

    def __init__(
        self, mesh, GMmref, GMmodel=None,
        wiresmap=None, maplist=None, approx_gradient=True,
        evaltype='approx',
        alpha_s=1.0, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
        alpha_xx=0., alpha_yy=0., alpha_zz=0.,
        **kwargs
    ):
        self.GMmref = copy.deepcopy(GMmref)
        order_clusters_GM_weight(self.GMmref)
        self._GMmodel = copy.deepcopy(GMmodel)
        self._wiresmap = wiresmap
        self._maplist = maplist
        self._mesh = mesh
        self.mesh = mesh
        self._approx_gradient = approx_gradient
        self._evaltype = evaltype
        self.mapping = IdentityMap(mesh, nP=self.wiresmap.nP)

        objfcts = [
            PetroSmallness(mesh=mesh, GMmodel=self.GMmodel,
                           wiresmap=self.wiresmap,
                           maplist=self.maplist,
                           approx_gradient=approx_gradient,
                           evaltype=evaltype,
                           mapping=self.mapping, **kwargs)
        ]
        objfcts += [
            SmoothDeriv(mesh=mesh, orientation='x',
                        mapping=maps * wire[1], **kwargs)
            for wire, maps in zip(self._wiresmap.maps, self._maplist)]
        objfcts += [
            SmoothDeriv2(mesh=mesh, orientation='x',
                         mapping=maps * wire[1], **kwargs)
            for wire, maps in zip(self._wiresmap.maps, self._maplist)]

        if mesh.dim > 1:
            objfcts += [
                SmoothDeriv(mesh=mesh, orientation='y',
                            mapping=maps * wire[1], **kwargs)
                for wire, maps in zip(self._wiresmap.maps, self._maplist)]
            objfcts += [
                SmoothDeriv2(mesh=mesh, orientation='y',
                             mapping=maps * wire[1], **kwargs)
                for wire, maps in zip(self._wiresmap.maps, self._maplist)]

        if mesh.dim > 2:
            objfcts += [
                SmoothDeriv(mesh=mesh, orientation='z',
                            mapping=maps * wire[1], **kwargs)
                for wire, maps in zip(self._wiresmap.maps, self._maplist)]
            objfcts += [
                SmoothDeriv2(mesh=mesh, orientation='z',
                             mapping=maps * wire[1], **kwargs)
                for wire, maps in zip(self._wiresmap.maps, self._maplist)]

        super(PetroRegularization, self).__init__(
            mesh=mesh,
            alpha_s=alpha_s, alpha_x=alpha_x, alpha_y=alpha_y, alpha_z=alpha_z,
            alpha_xx=alpha_xx, alpha_yy=alpha_yy, alpha_zz=alpha_zz,
            objfcts=objfcts, **kwargs)

        # setKwargs(self, **kwargs)

    # Properties
    alpha_s = props.Float("PetroPhysics weights")

    @property
    def GMmodel(self):
        if getattr(self, '_GMmodel', None) is None:
            self._GMmodel = copy.deepcopy(self.GMmref)
        return self._GMmodel

    @GMmodel.setter
    def GMmodel(self, gm):
        if gm is not None:
            self._GMmodel = copy.deepcopy(gm)
        self.objfcts[0].GMmodel = self.GMmodel

    def membership(self, m):
        return self.objfcts[0].membership(m)

    @property
    def wiresmap(self):
        if getattr(self, '_wiresmap', None) is None:
            self._wiresmap = Wires(('m', self._mesh.nC))
        return self._wiresmap

    @wiresmap.setter
    def wiresmap(self, wm):
        if wm is not None:
            self._wiresmap = wm
        self.objfcts[0].wiresmap = self.wiresmap

    @property
    def maplist(self):
        if getattr(self, '_maplist', None) is None:
            self._maplist = [IdentityMap(
                self._mesh) for maps in self.wiresmap.maps]
        return self._maplist

    @maplist.setter
    def maplist(self, mp):
        if mp is not None:
            self._maplist = mp
        self.objfcts[0].maplist = self.maplist

    @property
    def approx_gradient(self):
        if getattr(self, '_approx_gradient', None) is None:
            self._approx_gradient = True
        return self._approx_gradient

    @approx_gradient.setter
    def approx_gradient(self, ap):
        if ap is not None:
            self._approx_gradient = ap
        self.objfcts[0].approx_gradient = self.approx_gradient


class SimplePetroWithMappingSmallness(BaseRegularization):
    """
    Smallness term for the petrophysically constrained regularization
    """

    _multiplier_pair = 'alpha_s'

    def __init__(self, GMmodel, wiresmap=None,
                 maplist=None, mesh=None,
                 approx_gradient=True,
                 evaltype='approx',
                 **kwargs):

        self.approx_gradient = approx_gradient
        self.evaltype = evaltype

        super(SimplePetroWithMappingSmallness, self).__init__(
            mesh=mesh, **kwargs
        )
        self.GMmodel = GMmodel
        self.wiresmap = wiresmap
        self.maplist = maplist

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
                    speye(len(self.wiresmap.maps)),
                    sdiag(np.sqrt(self.cell_weights))
                )
        else:
            return Identity()

    @properties.validator('cell_weights')
    def _validate_cell_weights(self, change):
        if change['value'] is not None:
            if self._nC_residual != '*':
                if (len(change['value']) != self._nC_residual) and (len(change['value']) != len(self.wiresmap.maps) * self._nC_residual):
                    raise Exception(
                        'cell_weights must be length {} or {} not {}'.format(
                            self._nC_residual,
                            len(self.wiresmap.maps) * self._nC_residual,
                            len(change['value'])
                        )
                    )

    def membership(self, m):
        modellist = self.wiresmap * m
        model = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
        return self.GMmodel.predict(model)

    @timeIt
    def __call__(self, m, externalW=True):

        if externalW:
            W = self.W
        else:
            W = Identity()

        if getattr(self, 'mref', None) is None:
            self.mref = mkvc(self.GMmodel.means_[self.membership(m)])

        if self.evaltype == 'approx':
            membership = self.membership(self.mref)
            dm = self.wiresmap * (m)
            dmref = self.wiresmap * (self.mref)
            dmm = np.c_[[a * b for a, b in zip(self.maplist, dm)]].T
            dmm = np.r_[
                [
                    self.GMmodel.cluster_mapping[membership[i]] * dmm[i].reshape(-1, 2) for i in range(dmm.shape[0])
                ]
            ].reshape(-1, 2)
            dmmref = np.c_[[a for a in dmref]].T
            dmr = dmm - dmmref
            r0 = W * mkvc(dmr)

            if self.GMmodel.covariance_type == 'tied':
                r1 = np.r_[[np.dot(self.GMmodel.precisions_, np.r_[dmr[i]])
                            for i in range(len(dmr))]]
            elif self.GMmodel.covariance_type == 'diag' or self.GMmodel.covariance_type == 'spherical':
                r1 = np.r_[[np.dot(self.GMmodel.precisions_[membership[i]] * np.eye(len(self.wiresmap.maps)),
                                   np.r_[dmr[i]]) for i in range(len(dmr))]]
            else:
                r1 = np.r_[[np.dot(self.GMmodel.precisions_[membership[i]],
                                   np.r_[dmr[i]]) for i in range(len(dmr))]]

            return 0.5 * r0.dot(W * mkvc(r1))

        elif self.evaltype == 'full':
            modellist = self.wiresmap * m
            model = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
            score = self.GMmodel.score_samples(
                model) + ComputeConstantTerm(self.GMmodel)
            score_vec = mkvc(
                np.r_[[score for maps in self.wiresmap.maps]])
            return -np.sum((W.T * W) * score_vec) / len(self.wiresmap.maps)

    @timeIt
    def deriv(self, m):

        if getattr(self, 'mref', None) is None:
            self.mref = mkvc(self.GMmodel.means_[self.membership(m)])

        membership = self.membership(self.mref)
        modellist = self.wiresmap * m
        dmmodel = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
        mreflist = self.wiresmap * self.mref
        mD = [a.deriv(b) for a, b in zip(self.maplist, modellist)]
        mD = sp.block_diag(mD)

        if self.approx_gradient == True:
            dmm = np.r_[
                [
                    self.GMmodel.cluster_mapping[membership[i]] * dmmodel[i].reshape(-1, 2) for i in range(dmmodel.shape[0])
                ]
            ].reshape(-1, 2)
            dmmref = np.c_[[a for a in mreflist]].T
            dm = dmm - dmmref

            if self.GMmodel.covariance_type == 'tied':
                raise Exception('Not implemented')
            else:
                r = self.W * \
                    mkvc(np.r_[
                        [
                            mkvc(
                                self.GMmodel.cluster_mapping[membership[i]].deriv(
                                    dmmodel[i],
                                    v=np.dot(self.GMmodel.precisions_[membership[i]], dm[i])))
                            for i in range(dmmodel.shape[0])]])
            return mkvc(mD.T * (self.W.T * r))

        else:
            raise Exception('Not implemented')

    @timeIt
    def deriv2(self, m, v=None):

        if getattr(self, 'mref', None) is None:
            self.mref = mkvc(self.GMmodel.means_[self.membership(m)])

        # For a positive definite Hessian,
        # we approximate it with the covariance of the cluster
        # whose each point belong
        membership = self.membership(self.mref)
        modellist = self.wiresmap * m
        dmmodel = np.c_[[a * b for a, b in zip(self.maplist, modellist)]].T
        mD = [a.deriv(b) for a, b in zip(self.maplist, modellist)]
        mD = sp.block_diag(mD)

        if self._r_second_deriv is None:
            if self.GMmodel.covariance_type == 'tied':
                r = np.r_[
                    [
                        self.GMmodel.cluster_mapping[membership[i]].deriv(
                            dmmodel[i],
                            v=(self.GMmodel.cluster_mapping[membership[i]].deriv(
                                dmmodel[i],
                                v=self.GMmodel.precisions_
                            )).T
                        )
                        for i in range(len(dmmodel))
                    ]
                ]
            elif self.GMmodel.covariance_type == 'spherical' or self.GMmodel.covariance_type == 'diag':
                r = np.r_[
                    [
                        self.GMmodel.cluster_mapping[membership[i]].deriv(
                            dmmodel[i],
                            v=(self.GMmodel.cluster_mapping[membership[i]].deriv(
                                dmmodel[i],
                                v=self.GMmodel.precisions_[membership[i]] *
                                np.eye(len(self.wiresmap.maps))
                            )).T
                        )
                        for i in range(len(dmmodel))
                    ]
                ]
            else:
                r = np.r_[
                    [
                        self.GMmodel.cluster_mapping[membership[i]].deriv(
                            dmmodel[i],
                            v=(self.GMmodel.cluster_mapping[membership[i]].deriv(
                                dmmodel[i],
                                v=self.GMmodel.precisions_[membership[i]]
                            )).T
                        )
                        for i in range(len(dmmodel))
                    ]
                ]
            self._r_second_deriv = r

        if v is not None:
            mDv = self.wiresmap * (mD * v)
            mDv = np.c_[mDv]
            return mkvc(mD.T * ((self.W.T * self.W) *
                                mkvc(np.r_[[np.dot(self._r_second_deriv[i], mDv[i]) for i in range(len(mDv))]])))
        else:
            # Forming the Hessian by diagonal blocks
            hlist = [[self._r_second_deriv[:, i, j]
                      for i in range(len(self.wiresmap.maps))]
                     for j in range(len(self.wiresmap.maps))]

            Hr = sp.csc_matrix((0, 0), dtype=np.float64)
            for i in range(len(self.wiresmap.maps)):
                Hc = sp.csc_matrix((0, 0), dtype=np.float64)
                for j in range(len(self.wiresmap.maps)):
                    Hc = sp.hstack([Hc, sdiag(hlist[i][j])])
                Hr = sp.vstack([Hr, Hc])

            mDW = self.W * mD

            return (mDW.T * mDW) * Hr


class SimplePetroWithMappingRegularization(SimpleComboRegularization):

    def __init__(
        self, mesh, GMmref, GMmodel=None,
        wiresmap=None, maplist=None, approx_gradient=True,
        evaltype='approx',
        alpha_s=1.0, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
        alpha_xx=0., alpha_yy=0., alpha_zz=0.,
        **kwargs
    ):
        self.GMmref = copy.deepcopy(GMmref)
        order_clusters_GM_weight(self.GMmref)
        self._GMmodel = copy.deepcopy(GMmodel)
        self._wiresmap = wiresmap
        self._maplist = maplist
        self._mesh = mesh
        self.mesh = mesh
        self._approx_gradient = approx_gradient
        self._evaltype = evaltype
        self.mapping = IdentityMap(mesh, nP=self.wiresmap.nP)

        objfcts = [
            SimplePetroWithMappingSmallness(
                mesh=mesh,
                GMmodel=self.GMmodel,
                wiresmap=self.wiresmap,
                maplist=self.maplist,
                approx_gradient=approx_gradient,
                evaltype=evaltype,
                mapping=self.mapping, **kwargs)
        ]
        objfcts += [
            SimpleSmoothDeriv(mesh=mesh, orientation='x',
                              mapping=maps * wire[1], **kwargs)
            for wire, maps in zip(self._wiresmap.maps, self._maplist)]
        objfcts += [
            SmoothDeriv2(mesh=mesh, orientation='x',
                         mapping=maps * wire[1], **kwargs)
            for wire, maps in zip(self._wiresmap.maps, self._maplist)]

        if mesh.dim > 1:
            objfcts += [
                SimpleSmoothDeriv(mesh=mesh, orientation='y',
                                  mapping=maps * wire[1], **kwargs)
                for wire, maps in zip(self._wiresmap.maps, self._maplist)]
            objfcts += [
                SmoothDeriv2(mesh=mesh, orientation='y',
                             mapping=maps * wire[1], **kwargs)
                for wire, maps in zip(self._wiresmap.maps, self._maplist)]

        if mesh.dim > 2:
            objfcts += [
                SimpleSmoothDeriv(mesh=mesh, orientation='z',
                                  mapping=maps * wire[1], **kwargs)
                for wire, maps in zip(self._wiresmap.maps, self._maplist)]
            objfcts += [
                SmoothDeriv2(mesh=mesh, orientation='z',
                             mapping=maps * wire[1], **kwargs)
                for wire, maps in zip(self._wiresmap.maps, self._maplist)]

        super(SimplePetroWithMappingRegularization, self).__init__(
            mesh=mesh,
            alpha_s=alpha_s, alpha_x=alpha_x, alpha_y=alpha_y, alpha_z=alpha_z,
            alpha_xx=alpha_xx, alpha_yy=alpha_yy, alpha_zz=alpha_zz,
            objfcts=objfcts, **kwargs)

        #setKwargs(self, **kwargs)

    # Properties
    alpha_s = props.Float("PetroPhysics weights")

    @property
    def GMmodel(self):
        if getattr(self, '_GMmodel', None) is None:
            self._GMmodel = copy.deepcopy(self.GMmref)
        return self._GMmodel

    @GMmodel.setter
    def GMmodel(self, gm):
        if gm is not None:
            self._GMmodel = copy.deepcopy(gm)
        self.objfcts[0].GMmodel = self.GMmodel

    #@classmethod
    def membership(self, m):
        return self.objfcts[0].membership(m)

    @property
    def wiresmap(self):
        if getattr(self, '_wiresmap', None) is None:
            self._wiresmap = Wires(('m', self._mesh.nC))
        return self._wiresmap

    @wiresmap.setter
    def wiresmap(self, wm):
        if wm is not None:
            self._wiresmap = wm
        self.objfcts[0].wiresmap = self.wiresmap

    @property
    def maplist(self):
        if getattr(self, '_maplist', None) is None:
            self._maplist = [IdentityMap(
                self._mesh) for maps in self.wiresmap.maps]
        return self._maplist

    @maplist.setter
    def maplist(self, mp):
        if mp is not None:
            self._maplist = mp
        self.objfcts[0].maplist = self.maplist

    @property
    def approx_gradient(self):
        if getattr(self, '_approx_gradient', None) is None:
            self._approx_gradient = True
        return self._approx_gradient

    @approx_gradient.setter
    def approx_gradient(self, ap):
        if ap is not None:
            self._approx_gradient = ap
        self.objfcts[0].approx_gradient = self.approx_gradient


def MakeSimplePetroRegularization(
    mesh, GMmref, GMmodel=None,
    wiresmap=None, maplist=None,
    approx_gradient=True,
    evaltype='approx',
    gamma=1.,
    alpha_s=1.0, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
    alpha_xx=0., alpha_yy=0., alpha_zz=0.,
    cell_weights_list=None,
    **kwargs
):

    if wiresmap is None:
        wrmp = Wires(('m', mesh.nC))
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

    reg = SimplePetroRegularization(
        mesh=mesh, GMmref=GMmref, GMmodel=GMmodel,
        wiresmap=wiresmap, maplist=maplist,
        approx_gradient=approx_gradient,
        evaltype=evaltype,
        alpha_s=alpha_s,
        alpha_x=0., alpha_y=0., alpha_z=0.,
        **kwargs
    )
    reg.gamma = gamma
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
            alpha_s=0.,
            alpha_x=alph_x[i],
            alpha_y=alph_y[i],
            alpha_z=alph_z[i],
            cell_weights=clwhtlst[i],
            ** kwargs
        )

    return reg


def MakePetroRegularization(
    mesh, GMmref, GMmodel=None,
    wiresmap=None, maplist=None,
    approx_gradient=True,
    evaltype='approx',
    gamma=1.,
    alpha_s=1.0, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
    alpha_xx=0., alpha_yy=0., alpha_zz=0.,
    cell_weights_list=None,
    **kwargs
):

    if wiresmap is None:
        wrmp = Wires(('m', mesh.nC))
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

    reg = PetroRegularization(
        mesh=mesh, GMmref=GMmref, GMmodel=GMmodel,
        wiresmap=wiresmap, maplist=maplist,
        approx_gradient=approx_gradient,
        evaltype=evaltype,
        alpha_s=alpha_s,
        alpha_x=0., alpha_y=0., alpha_z=0.,
        **kwargs
    )
    reg.gamma = gamma
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
            alpha_s=0.,
            alpha_x=alph_x[i],
            alpha_y=alph_y[i],
            alpha_z=alph_z[i],
            cell_weights=clwhtlst[i],
            ** kwargs
        )

    return reg


def MakeSimplePetroWithMappingRegularization(
    mesh, GMmref, GMmodel=None,
    wiresmap=None, maplist=None,
    approx_gradient=True,
    evaltype='approx',
    gamma=1.,
    alpha_s=1.0, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
    alpha_xx=0., alpha_yy=0., alpha_zz=0.,
    cell_weights_list=None,
    **kwargs
):

    if wiresmap is None:
        wrmp = Wires(('m', mesh.nC))
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

    reg = SimplePetroWithMappingRegularization(
        mesh=mesh, GMmref=GMmref, GMmodel=GMmodel,
        wiresmap=wiresmap, maplist=maplist,
        approx_gradient=approx_gradient,
        evaltype=evaltype,
        alpha_s=alpha_s,
        alpha_x=0., alpha_y=0., alpha_z=0.,
        **kwargs
    )
    reg.gamma = gamma
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
            alpha_s=0.,
            alpha_x=alph_x[i],
            alpha_y=alph_y[i],
            alpha_z=alph_z[i],
            cell_weights=clwhtlst[i],
            ** kwargs
        )

    return reg
