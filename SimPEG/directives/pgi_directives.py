###############################################################################
#                                                                             #
#         Directives for PGI: Petrophysically guided Regularization           #
#                                                                             #
###############################################################################

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import copy
from ..objective_function import ComboObjectiveFunction
from ..regularization import (
    Smallness,
    SparseSmallness,
    Sparse,
    PGIsmallness,
    PGIwithNonlinearRelationshipsSmallness,
    PGI,
    SmoothnessFirstOrder,
    SparseSmoothness,
    PGIwithRelationships,
)
from ..utils import (
    mkvc,
    WeightedGaussianMixture,
    GaussianMixtureWithPrior,
    GaussianMixtureWithNonlinearRelationships,
    GaussianMixtureWithNonlinearRelationshipsWithPrior,
    Zero,
)
from ..directives import InversionDirective, MultiTargetMisfits
from ..utils.code_utils import deprecate_property


class PGI_UpdateParameters(InversionDirective):
    """
    This directive is to be used with regularization from regularization.pgi.
    It updates:
        - the reference model and weights in the smallness (L2-approximation of PGI)
        - the GMM as a MAP estimate between the prior and the current model
    For more details, please consult:
     - https://doi.org/10.1093/gji/ggz389
    """

    verbose = False  # print info.  about the GMM at each iteration
    update_rate = 1  # updates at each `update_rate` iterations
    update_gmm = False  # update the GMM
    zeta = (
        1e10  # confidence in the prior proportions; default: high value, keep GMM fixed
    )
    nu = (
        1e10  # confidence in the prior covariances; default: high value, keep GMM fixed
    )
    kappa = 1e10  # confidence in the prior means;default: high value, keep GMM fixed
    update_covariances = (
        True  # Average the covariances, If false: average the precisions
    )
    fixed_membership = None  # keep the membership of specific cells fixed
    keep_ref_fixed_in_Smooth = True  # keep mref fixed in the Smoothness

    def initialize(self):

        pgi_reg = self.reg.get_functions_of_type(PGIsmallness)
        if len(pgi_reg) != 1:
            raise UserWarning(
                "'PGI_UpdateParameters' requires one 'PGIsmallness' regularization "
                "in the objective function."
            )
        self.pgi_reg = pgi_reg[0]

    def endIter(self):
        if self.opt.iter > 0 and self.opt.iter % self.update_rate == 0:
            m = self.invProb.model
            modellist = self.pgi_reg.wiresmap * m
            model = np.c_[[a * b for a, b in zip(self.pgi_reg.maplist, modellist)]].T

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

            else:
                clfupdate = copy.deepcopy(self.pgi_reg.gmmref)

            self.pgi_reg.gmm = clfupdate
            membership = self.pgi_reg.gmm.predict(model)

            if self.fixed_membership is not None:
                membership[self.fixed_membership[:, 0]] = self.fixed_membership[:, 1]

            mref = mkvc(self.pgi_reg.gmm.means_[membership])
            self.pgi_reg.reference_model = mref
            if getattr(self.fixed_membership, "shape", [0, 0])[0] < len(membership):
                self.pgi_reg._r_second_deriv = None


class PGI_BetaAlphaSchedule(InversionDirective):
    """
    This directive is to be used with regularizations from regularization.pgi.
    It implements the strategy described in https://doi.org/10.1093/gji/ggz389
    for iteratively updating beta and alpha_s for fitting the
    geophysical and smallness targets.
    """

    verbose = False  # print information (progress, updates made)
    tolerance = 0.0  # tolerance on the geophysical target misfit for cooling
    progress = 0.1  # minimum percentage progress (default 10%) before cooling beta
    coolingFactor = 2.0  # when cooled, beta is divided by it
    warmingFactor = 1.0  # when warmed, alpha_s is multiplied by the ratio of the
    # geophysical target with their current misfit, times this factor
    mode = 1  # mode 1: start with nothing fitted. Mode 2: warmstart with fitted geophysical data
    mode2_iter = 0  # counts how many iteration after the fit of the geophysical data
    alphasmax = 1e10  # max alpha_s
    betamin = 1e-10  # minimum beta
    update_rate = 1  # update every `update_rate` iterations
    pgi_reg = None
    ratio_in_cooling = (
        False  # add the ratio of geophysical misfit with their target in cooling
    )

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
        pgi_reg = self.reg.get_functions_of_type(PGI)
        if len(pgi_reg) != 1:
            raise UserWarning(
                "'PGI_UpdateParameters' requires one 'PGI' regularization "
                "in the objective function."
            )
        self.pgi_reg = pgi_reg[0]

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

        if self.opt.iter > 0 and self.opt.iter % self.update_rate == 0:
            if self.verbose:
                print(
                    "Beta cooling evaluation: progress:",
                    np.round(self.dmlist, decimals=1),
                    "; minimum progress targets:",
                    np.round(
                        np.maximum(
                            (1.0 - self.progress) * self.previous_dmlist,
                            (1.0 + self.tolerance) * self.DMtarget,
                        ),
                        decimals=1,
                    ),
                )
            if np.all(
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
            ):

                if np.all([self.invProb.beta > self.betamin]):

                    ratio = 1.0
                    indx = self.dmlist > (1.0 + self.tolerance) * self.DMtarget
                    if np.any(indx) and self.ratio_in_cooling:
                        ratio = np.median([self.dmlist[indx] / self.DMtarget[indx]])
                    self.invProb.beta /= self.coolingFactor * ratio

                    if self.verbose:
                        print("Decreasing beta to counter data misfit decrase plateau.")

            elif np.all([self.DM, self.mode == 2]):

                if np.all([self.pgi_reg.alpha_pgi < self.alphasmax]):

                    ratio = np.median(self.DMtarget / self.dmlist)
                    self.pgi_reg.alpha_pgi *= self.warmingFactor * ratio

                    if self.verbose:
                        print(
                            "Warming alpha_pgi to favor clustering: ",
                            self.pgi_reg.alpha_pgi,
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
                        print("Decreasing beta to counter data misfit increase.")

        self.previous_score = copy.deepcopy(self.score)
        self.previous_dmlist = copy.deepcopy(
            self.inversion.directiveList.dList[self.targetclass].dmlist
        )


class PGI_AddMrefInSmooth(InversionDirective):
    """
    This directive is to be used with regularizations from regularization.pgi.
    It implements the strategy described in https://doi.org/10.1093/gji/ggz389
    for including the learned reference model, once stable, in the smoothness terms.
    """

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

        if getattr(self.reg.objfcts[0], "objfcts", None) is not None:
            # Find the petrosmallness terms in a two-levels combo-regularization.
            petrosmallness = np.where(
                np.r_[
                    [
                        isinstance(regpart, (PGI, PGIwithRelationships))
                        for regpart in self.reg.objfcts
                    ]
                ]
            )[0][0]
            self.petrosmallness = petrosmallness

            # Find the smoothness terms in a two-levels combo-regularization.
            Smooth = []
            for i, regobjcts in enumerate(self.reg.objfcts):
                for j, regpart in enumerate(regobjcts.objfcts):
                    Smooth += [
                        [
                            i,
                            j,
                            isinstance(regpart, (SmoothnessFirstOrder, SparseSmoothness)),
                        ]
                    ]
            self.Smooth = np.r_[Smooth]

            self.nbr = np.sum(
                [len(self.reg.objfcts[i].objfcts) for i in range(len(self.reg.objfcts))]
            )
            self._regmode = 1
            self.pgi_reg = self.reg.objfcts[self.petrosmallness]

        else:
            self._regmode = 2
            self.pgi_reg = self.reg
            self.nbr = len(self.reg.objfcts)
            self.Smooth = np.r_[
                [
                    isinstance(regpart, (SmoothnessFirstOrder, SparseSmoothness))
                    for regpart in self.reg.objfcts
                ]
            ]
            self._regmode = 2

        if ~np.any(self.pgi_updategmm_class):
            self.previous_membership = self.pgi_reg.membership(self.invProb.model)
        else:
            self.previous_membership = self.pgi_reg.compute_quasi_geology_model()

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
            self.membership = self.pgi_reg.compute_quasi_geology_model()

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
            self.reg.reference_model_in_smooth = True
            self.pgi_reg.reference_model_in_smooth = True

            if self._regmode == 2:
                for i in range(self.nbr):
                    if self.Smooth[i]:
                        self.reg.objfcts[i].reference_model = mkvc(
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
                        self.reg.objfcts[idx[0]].objfcts[idx[1]].reference_model = mkvc(
                            self.pgi_reg.gmm.means_[self.membership]
                        )
                if self.verbose:
                    print(
                        "Add mref to Smoothness. Changes in mref happened in {} % of the cells".format(
                            percent_diff
                        )
                    )

        self.previous_membership = copy.deepcopy(self.membership)
