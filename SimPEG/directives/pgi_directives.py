###############################################################################
#                                                                             #
#         Directives for PGI: Petrophysically guided Regularization           #
#                                                                             #
###############################################################################

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import copy
from ..regularization import (
    SimpleSmall,
    Small,
    SparseSmall,
    Simple,
    Tikhonov,
    Sparse,
    SimplePGIsmallness,
    PGIsmallness,
    SimplePGIwithRelationshipsSmallness,
    SimplePGI,
    PGI,
    SmoothDeriv,
    SimpleSmoothDeriv,
    SparseDeriv,
    SimplePGIwithRelationships,
)
from ..utils import (
    mkvc,
    order_cluster,
    WeightedGaussianMixture,
    GaussianMixtureWithPrior,
    GaussianMixtureWithNonlinearRelationships,
    GaussianMixtureWithNonlinearRelationshipsWithPrior,
    Zero,
)
from ..directives import InversionDirective, MultiTargetMisfits
from ..utils.code_utils import deprecate_property


class PGI_UpdateParameters(InversionDirective):

    verbose = False  # print info. at each iteration
    update_gmm = True  # update GMM
    zeta = 1e10  # default: keep GMM fixed
    nu = 1e10  # default: keep GMM fixed
    kappa = 1e10  # default: keep GMM fixed
    update_covariances = (
        True  # Average the covariances, If false: average the precisions
    )
    fixed_membership = None  # keep the membership of specific cells fixed
    keep_ref_fixed_in_Smooth = True  # keep mref fixed in the Smoothness

    def initialize(self):
        if getattr(self.invProb.reg.objfcts[0], "objfcts", None) is not None:
            pgi_reg = np.where(
                np.r_[
                    [
                        (
                            isinstance(regpart, SimplePGI)
                            or isinstance(regpart, PGI)
                            or isinstance(regpart, SimplePGIwithRelationships)
                        )
                        for regpart in self.invProb.reg.objfcts
                    ]
                ]
            )[0][0]

            self.pgi_reg = self.invProb.reg.objfcts[pgi_reg]

            if self.debug:
                print(type(self.self.pgi_reg))
            self._regmode = 1

        else:
            self._regmode = 2
            self.pgi_reg = self.invProb.reg

    def endIter(self):

        m = self.invProb.model
        modellist = self.pgi_reg.wiresmap * m
        model = np.c_[[a * b for a, b in zip(self.pgi_reg.maplist, modellist)]].T

        if self.pgi_reg.mrefInSmooth and self.keep_ref_fixed_in_Smooth:
            self.fixed_membership = np.c_[
                np.arange(len(self.pgi_reg.gmmref.cell_volumes)),
                self.pgi_reg.membership(self.pgi_reg.mref),
            ]

        if self.update_gmm and isinstance(
            self.pgi_reg.gmmref, GaussianMixtureWithNonlinearRelationships
        ):
            clfupdate = GaussianMixtureWithNonlinearRelationshipsWithPrior(
                gmmref=self.pgi_reg.gmmref,
                zeta=self.zeta,
                kappa=self.kappa,
                nu=self.nu,
                verbose=self.verbose,
                prior_type="semi",
                update_covariances=self.update_covariances,
                max_iter=self.pgi_reg.gmm.max_iter,
                n_init=self.pgi_reg.gmm.n_init,
                reg_covar=self.pgi_reg.gmm.reg_covar,
                weights_init=self.pgi_reg.gmm.weights_,
                means_init=self.pgi_reg.gmm.means_,
                precisions_init=self.pgi_reg.gmm.precisions_,
                random_state=self.pgi_reg.gmm.random_state,
                tol=self.pgi_reg.gmm.tol,
                verbose_interval=self.pgi_reg.gmm.verbose_interval,
                warm_start=self.pgi_reg.gmm.warm_start,
                fixed_membership=self.fixed_membership,
            )
            clfupdate = clfupdate.fit(model)
            # order_cluster(clfupdate, self.pgi_reg.gmmref)

        elif self.update_gmm and isinstance(
            self.pgi_reg.gmmref, WeightedGaussianMixture
        ):
            clfupdate = GaussianMixtureWithPrior(
                gmmref=self.pgi_reg.gmmref,
                zeta=self.zeta,
                kappa=self.kappa,
                nu=self.nu,
                verbose=self.verbose,
                prior_type="semi",
                update_covariances=self.update_covariances,
                max_iter=self.pgi_reg.gmm.max_iter,
                n_init=self.pgi_reg.gmm.n_init,
                reg_covar=self.pgi_reg.gmm.reg_covar,
                weights_init=self.pgi_reg.gmm.weights_,
                means_init=self.pgi_reg.gmm.means_,
                precisions_init=self.pgi_reg.gmm.precisions_,
                random_state=self.pgi_reg.gmm.random_state,
                tol=self.pgi_reg.gmm.tol,
                verbose_interval=self.pgi_reg.gmm.verbose_interval,
                warm_start=self.pgi_reg.gmm.warm_start,
                fixed_membership=self.fixed_membership,
            )
            clfupdate = clfupdate.fit(model)
            # order_cluster(clfupdate, self.pgi_reg.gmmref)

        else:
            clfupdate = copy.deepcopy(self.pgi_reg.gmmref)

        self.pgi_reg.gmm = clfupdate
        membership = self.pgi_reg.gmm.predict(model)

        if self.fixed_membership is not None:
            membership[self.fixed_membership[:, 0]] = self.fixed_membership[:, 1]

        mref = mkvc(self.pgi_reg.gmm.means_[membership])
        self.pgi_reg.mref = mref
        if getattr(self.fixed_membership, "shape", [0, 0])[0] < len(membership):
            self.pgi_reg.objfcts[0]._r_second_deriv = None


class PGI_BetaAlphaSchedule(InversionDirective):

    verbose = False
    tolerance = 0.0
    progress = 0.02
    coolingFactor = 2.0
    warmingFactor = 1.0
    mode = 1
    mode2_iter = 0
    alphasmax = 1e10
    betamin = 1e-10
    UpdateRate = 1
    ratio_in_cooling = False

    def initialize(self):
        targetclass = np.r_[
            [
                isinstance(dirpart, MultiTargetMisfits)
                for dirpart in self.inversion.directiveList.dList
            ]
        ]
        if ~np.any(targetclass):
            raise Exception(
                "You need to have a MultiTargetMisfits directives to use the PGI_BetaAlphaSchedule directive"
            )
        else:
            self.targetclass = np.where(targetclass)[0][-1]
            self.DMtarget = np.sum(
                np.r_[self.dmisfit.multipliers]
                * self.inversion.directiveList.dList[self.targetclass].DMtarget
            )
            self.previous_score = copy.deepcopy(
                self.inversion.directiveList.dList[self.targetclass].phims()
            )
            self.previous_dmlist = self.inversion.directiveList.dList[
                self.targetclass
            ].dmlist
            self.CLtarget = self.inversion.directiveList.dList[
                self.targetclass
            ].CLtarget

        updategaussianclass = np.r_[
            [
                isinstance(dirpart, PGI_UpdateParameters)
                for dirpart in self.inversion.directiveList.dList
            ]
        ]
        if ~np.any(updategaussianclass):
            self.DMtarget = None
        else:
            updategaussianclass = np.where(updategaussianclass)[0][-1]
            self.updategaussianclass = self.inversion.directiveList.dList[
                updategaussianclass
            ]

        if getattr(self.invProb.reg.objfcts[0], "objfcts", None) is not None:
            petrosmallness = np.where(
                np.r_[
                    [
                        (
                            isinstance(regpart, SimplePGI)
                            or isinstance(regpart, PGI)
                            or isinstance(regpart, SimplePGIwithRelationships)
                        )
                        for regpart in self.invProb.reg.objfcts
                    ]
                ]
            )[0][0]
            self.petrosmallness = petrosmallness
            if self.debug:
                print(type(self.invProb.reg.objfcts[self.petrosmallness]))
            self._regmode = 1
        else:
            self._regmode = 2

        if self._regmode == 1:
            self.pgi_reg = self.invProb.reg.objfcts[self.petrosmallness]
        else:
            self.pgi_reg = self.invProb.reg

    def endIter(self):

        self.DM = self.inversion.directiveList.dList[self.targetclass].DM
        self.dmlist = self.inversion.directiveList.dList[self.targetclass].dmlist
        self.DMtarget = self.inversion.directiveList.dList[self.targetclass].DMtarget
        self.TotalDMtarget = np.sum(
            np.r_[self.dmisfit.multipliers]
            * self.inversion.directiveList.dList[self.targetclass].DMtarget
        )
        self.score = self.inversion.directiveList.dList[self.targetclass].phims()
        self.targetlist = self.inversion.directiveList.dList[
            self.targetclass
        ].targetlist

        if self.DM:
            self.mode = 2
            self.mode2_iter += 1

        if self.opt.iter > 0 and self.opt.iter % self.UpdateRate == 0:
            if self.verbose:
                print(
                    "progress:",
                    self.dmlist,
                    "; progress targets:",
                    np.maximum(
                        (1.0 - self.progress) * self.previous_dmlist,
                        (1.0 + self.tolerance) * self.DMtarget,
                    ),
                )
            if np.any(
                [
                    np.all(
                        [
                            np.all(
                                self.dmlist[~self.targetlist]
                                > np.maximum(
                                    (1.0 - self.progress)
                                    * self.previous_dmlist[~self.targetlist],
                                    self.DMtarget[~self.targetlist],
                                )
                            ),
                            not self.DM,
                            self.mode == 1,
                        ]
                    ),
                    np.all(
                        [
                            np.all(
                                self.dmlist > (1.0 + self.tolerance) * self.DMtarget
                            ),
                            self.mode == 2,
                        ]
                    ),
                ]
            ):

                if np.all([self.invProb.beta > self.betamin]):

                    ratio = 1.0
                    indx = self.dmlist > (1.0 + self.tolerance) * self.DMtarget
                    if np.any(indx) and self.ratio_in_cooling:
                        ratio = np.median([self.dmlist[indx] / self.DMtarget[indx]])
                    self.invProb.beta /= self.coolingFactor * ratio

                    if self.verbose:
                        print("Decreasing beta to counter data misfit decrase plateau")

            elif np.all([self.DM, self.mode == 2]):

                if np.all([self.pgi_reg.alpha_s < self.alphasmax]):

                    ratio = np.median(self.DMtarget / self.dmlist)
                    self.pgi_reg.alpha_s *= self.warmingFactor * ratio

                    if self.verbose:
                        print(
                            "Warming alpha_s to favor clustering: ",
                            self.pgi_reg.alpha_s,
                        )

            elif np.all(
                [
                    np.any(self.dmlist > (1.0 + self.tolerance) * self.DMtarget),
                    self.mode == 2,
                ]
            ):

                if np.all([self.invProb.beta > self.betamin]):

                    ratio = 1.0
                    indx = self.dmlist > (1.0 + self.tolerance) * self.DMtarget
                    if np.any(indx) and self.ratio_in_cooling:
                        ratio = np.median([self.dmlist[indx] / self.DMtarget[indx]])
                    self.invProb.beta /= self.coolingFactor * ratio

                    if self.verbose:
                        print("Decrease beta for countering plateau")

        self.previous_score = copy.deepcopy(self.score)
        self.previous_dmlist = copy.deepcopy(
            self.inversion.directiveList.dList[self.targetclass].dmlist
        )


class PGI_AddMrefInSmooth(InversionDirective):

    # Chi factor for Data Misfit
    chifact = 1.0
    tolerance_phid = 0.0
    phi_d_target = None
    wait_till_stable = True
    tolerance = 0.0
    verbose = False

    def initialize(self):
        targetclass = np.r_[
            [
                isinstance(dirpart, MultiTargetMisfits)
                for dirpart in self.inversion.directiveList.dList
            ]
        ]
        if ~np.any(targetclass):
            self.DMtarget = None
        else:
            self.targetclass = np.where(targetclass)[0][-1]
            self._DMtarget = self.inversion.directiveList.dList[
                self.targetclass
            ].DMtarget

        self.pgi_updategmm_class = np.r_[
            [
                isinstance(dirpart, PGI_UpdateParameters)
                for dirpart in self.inversion.directiveList.dList
            ]
        ]

        if getattr(self.invProb.reg.objfcts[0], "objfcts", None) is not None:
            petrosmallness = np.where(
                np.r_[
                    [
                        (
                            isinstance(regpart, SimplePGI)
                            or isinstance(regpart, PGI)
                            or isinstance(regpart, SimplePGIwithRelationships)
                        )
                        for regpart in self.invProb.reg.objfcts
                    ]
                ]
            )[0][0]
            self.petrosmallness = petrosmallness
            if self.debug:
                print(type(self.invProb.reg.objfcts[self.petrosmallness]))
            self._regmode = 1
        else:
            self._regmode = 2

        if self._regmode == 1:
            self.pgi_reg = self.invProb.reg.objfcts[self.petrosmallness]
        else:
            self.pgi_reg = self.invProb.reg

        if getattr(self.invProb.reg.objfcts[0], "objfcts", None) is not None:
            self.nbr = np.sum(
                [
                    len(self.invProb.reg.objfcts[i].objfcts)
                    for i in range(len(self.invProb.reg.objfcts))
                ]
            )
            self.Smooth = np.r_[
                [
                    (
                        np.r_[
                            i,
                            j,
                            (
                                (
                                    isinstance(regpart, SmoothDeriv)
                                    or isinstance(regpart, SimpleSmoothDeriv)
                                    or isinstance(regpart, SparseDeriv)
                                )
                                and not (
                                    isinstance(regobjcts, SimplePGI)
                                    or isinstance(regobjcts, PGI)
                                    or isinstance(regobjcts, SimplePGIwithRelationships)
                                )
                            ),
                        ]
                    )
                    for i, regobjcts in enumerate(self.invProb.reg.objfcts)
                    for j, regpart in enumerate(regobjcts.objfcts)
                ]
            ]
            self._regmode = 1
        else:
            self.nbr = len(self.invProb.reg.objfcts)
            self.Smooth = np.r_[
                [
                    (
                        isinstance(regpart, SmoothDeriv)
                        or isinstance(regpart, SimpleSmoothDeriv)
                        or isinstance(regpart, SparseDeriv)
                    )
                    for regpart in self.invProb.reg.objfcts
                ]
            ]
            self._regmode = 2

        if ~np.any(self.pgi_updategmm_class):
            self.previous_membership = self.pgi_reg.membership(self.invProb.model)
        else:
            self.previous_membership = self.pgi_reg.membership(self.pgi_reg.mref)

    @property
    def DMtarget(self):
        if getattr(self, "_DMtarget", None) is None:
            self.phi_d_target = 0.5 * self.invProb.dmisfit.survey.nD
            self._DMtarget = self.chifact * self.phi_d_target
        return self._DMtarget

    @DMtarget.setter
    def DMtarget(self, val):
        self._DMtarget = val

    def endIter(self):
        self.DM = self.inversion.directiveList.dList[self.targetclass].DM
        self.dmlist = self.inversion.directiveList.dList[self.targetclass].dmlist

        if ~np.any(self.pgi_updategmm_class):
            self.membership = self.pgi_reg.membership(self.invProb.model)
        else:
            self.membership = self.pgi_reg.membership(self.pgi_reg.mref)

        same_mref = np.all(self.membership == self.previous_membership)
        percent_diff = (
            len(self.membership)
            - np.count_nonzero(self.previous_membership == self.membership)
        ) / len(self.membership)
        if self.verbose:
            print(
                "mref changed in ",
                len(self.membership)
                - np.count_nonzero(self.previous_membership == self.membership),
                " places",
            )
        if (
            self.DM or np.all(self.dmlist < (1 + self.tolerance_phid) * self.DMtarget)
        ) and (
            same_mref or not self.wait_till_stable or percent_diff <= self.tolerance
        ):
            self.invProb.reg.mrefInSmooth = True
            self.pgi_reg.mrefInSmooth = True

            if self._regmode == 2:
                for i in range(self.nbr):
                    if self.Smooth[i]:
                        self.invProb.reg.objfcts[i].mref = mkvc(
                            self.pgi_reg.gmm.means_[self.membership]
                        )
                if self.verbose:
                    print(
                        "Add mref to Smoothness. Changes in mref happened in {} % of the cells".format(
                            percent_diff
                        )
                    )

            elif self._regmode == 1:
                for i in range(self.nbr):
                    if self.Smooth[i, 2]:
                        idx = self.Smooth[i, :2]
                        if self.debug:
                            print(
                                type(self.invProb.reg.objfcts[idx[0]].objfcts[idx[1]])
                            )
                        self.invProb.reg.objfcts[idx[0]].objfcts[idx[1]].mref = mkvc(
                            self.pgi_reg.gmm.means_[self.membership]
                        )
                if self.verbose:
                    print(
                        "Add mref to Smoothness. Changes in mref happened in {} % of the cells".format(
                            percent_diff
                        )
                    )

        self.previous_membership = copy.deepcopy(self.membership)
