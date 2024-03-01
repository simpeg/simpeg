###############################################################################
#                                                                             #
#         Directives for PGI: Petrophysically guided Regularization           #
#                                                                             #
###############################################################################

import copy

import numpy as np

from ..directives import InversionDirective, MultiTargetMisfits
from ..regularization import (
    PGI,
    PGIwithRelationships,
    SmoothnessFirstOrder,
    SparseSmoothness,
)
from ..utils import (
    GaussianMixtureWithNonlinearRelationships,
    GaussianMixtureWithNonlinearRelationshipsWithPrior,
    GaussianMixtureWithPrior,
    WeightedGaussianMixture,
    mkvc,
)


class PGI_UpdateParameters(InversionDirective):
    """
    Directive to use with :class:`SimPEG.regularization.PGI`

    This directive updates the reference model and its weights in the smallness
    PGI regularization, along with the Gaussian Mixture as a map estimate
    between the prior and the current model.

    Parameters
    ----------
    update_gmm : bool, optional
        If True, updates the parameters of the Gaussian Mixture (mean,
        covariances, proportions). If False, the parameters of the are fixed
        throughout the inversion.
    zeta : float, optional
        Confidence in the prior proportions. Ignored if ``update_gmm`` is
        False.
    nu : float, optional
        Confidence in the prior covariances. Ignored if ``update_gmm`` is
        False.
    kappa : float, optional
        Confidence in the prior means. Ignored if ``update_gmm`` is False.
    update_rate : int, optional
        The Gaussian Mixture will be updated after this many iterations.
        Ignored if ``update_gmm`` is False.
    update_covariances : bool, optional
        Weather to update the covariances or the precisions. If True, update
        the covariances. If False, update the precisions. Ignored if
        ``update_gmm`` is False.
    fixed_membership : None or array_like, optional
        Array with the indices and membership of cells. Use when the membership
        of some cells is intended to be fixed.
    verbose : bool, optional
        If True, messages will be printed to stdout during each iteration.

    References
    ----------
     - https://doi.org/10.1093/gji/ggz389
    """

    def __init__(
        self,
        update_gmm=False,
        zeta=1e10,
        nu=1e10,
        kappa=1e10,
        update_rate=1,
        update_covariances=True,
        fixed_membership=None,
        verbose=False,
    ):
        super().__init__(self)
        self.update_gmm = update_gmm
        self.update_rate = update_rate
        self.zeta = zeta
        self.nu = nu
        self.kappa = kappa
        self.update_covariances = update_covariances
        self.fixed_membership = fixed_membership
        self.verbose = verbose

    def endIter(self):
        """Run after the end of each iteration in the inversion."""
        # Don't update the model if we are in the first iteration or if the
        # current iteration doesn't match the update rate
        if self.opt.iter == 0 or self.opt.iter % self.update_rate != 0:
            return None

        # Get current model
        m = self.invProb.model
        modellist = self.pgi_regularization.wiresmap * m
        model = np.c_[
            [a * b for a, b in zip(self.pgi_regularization.maplist, modellist)]
        ].T

        # Update the GMM parameters if required
        gmmref = self.pgi_regularization.gmmref
        if self.update_gmm:
            # Determine the class of the new gmm
            if isinstance(gmmref, GaussianMixtureWithNonlinearRelationships):
                new_gmm_class = GaussianMixtureWithNonlinearRelationshipsWithPrior
            elif isinstance(gmmref, WeightedGaussianMixture):
                new_gmm_class = GaussianMixtureWithPrior
            else:
                raise ValueError(
                    f"Invalid class '{gmmref.__class__}' for the Gaussian Mixture"
                    " in the PGI regularization."
                )
            new_gmm = new_gmm_class(
                gmmref=gmmref,
                zeta=self.zeta,
                kappa=self.kappa,
                nu=self.nu,
                verbose=self.verbose,
                prior_type="semi",
                update_covariances=self.update_covariances,
                max_iter=self.pgi_regularization.gmm.max_iter,
                n_init=self.pgi_regularization.gmm.n_init,
                reg_covar=self.pgi_regularization.gmm.reg_covar,
                weights_init=self.pgi_regularization.gmm.weights_,
                means_init=self.pgi_regularization.gmm.means_,
                precisions_init=self.pgi_regularization.gmm.precisions_,
                random_state=self.pgi_regularization.gmm.random_state,
                tol=self.pgi_regularization.gmm.tol,
                verbose_interval=self.pgi_regularization.gmm.verbose_interval,
                warm_start=self.pgi_regularization.gmm.warm_start,
                fixed_membership=self.fixed_membership,
            )
            new_gmm = new_gmm.fit(model)
        else:
            new_gmm = copy.deepcopy(gmmref)

        # Update the GMM in the PGI regularization
        self.pgi_regularization.gmm = new_gmm
        # Compute membership of the current model
        membership = self.pgi_regularization.gmm.predict(model)
        # Override the membership if some cells have fixed membership
        if self.fixed_membership is not None:
            membership[self.fixed_membership[:, 0]] = self.fixed_membership[:, 1]
        # Get the new reference model
        mref = mkvc(self.pgi_regularization.gmm.means_[membership])
        # Update the reference model in the PGI regularization
        self.pgi_regularization.reference_model = mref
        if getattr(self.fixed_membership, "shape", [0, 0])[0] < len(membership):
            self.pgi_regularization._r_second_deriv = None

    @property
    def pgi_regularization(self):
        """PGI regularization in the :class:`SimPEG.inverse_problem.BaseInvProblem``."""
        if not hasattr(self, "_pgi_regularization"):
            pgi_regularization = self.reg.get_functions_of_type(PGI)
            if len(pgi_regularization) != 1:
                raise UserWarning(
                    "'PGI_UpdateParameters' requires one 'PGI' regularization "
                    "in the objective function."
                )
            self._pgi_regularization = pgi_regularization[0]
        return self._pgi_regularization


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
    alphasmax = 1e10  # max alpha_s
    betamin = 1e-10  # minimum beta
    update_rate = 1  # update every `update_rate` iterations
    pgi_reg = None
    ratio_in_cooling = (
        False  # add the ratio of geophysical misfit with their target in cooling
    )

    def initialize(self):
        """Initialize the directive."""
        self.update_previous_score()
        self.update_previous_dmlist()

    def endIter(self):
        """Run after the end of each iteration in the inversion."""
        # Get some variables from the MultiTargetMisfits directive
        data_misfits_achieved = self.multi_target_misfits_directive.DM
        data_misfits_target = self.multi_target_misfits_directive.DMtarget
        dmlist = self.multi_target_misfits_directive.dmlist
        targetlist = self.multi_target_misfits_directive.targetlist

        # Change mode if data misfit targets have been achieved
        if data_misfits_achieved:
            self.mode = 2

        # Don't cool beta of warm alpha if we are in the first iteration or if
        # the current iteration doesn't match the update rate
        if self.opt.iter == 0 or self.opt.iter % self.update_rate != 0:
            self.update_previous_score()
            self.update_previous_dmlist()
            return None

        if self.verbose:
            targets = np.round(
                np.maximum(
                    (1.0 - self.progress) * self.previous_dmlist,
                    (1.0 + self.tolerance) * data_misfits_target,
                ),
                decimals=1,
            )
            dmlist_rounded = np.round(dmlist, decimals=1)
            print(
                f"Beta cooling evaluation: progress: {dmlist_rounded}; "
                f"minimum progress targets: {targets}"
            )

        # Decide if we should cool beta
        threshold = np.maximum(
            (1.0 - self.progress) * self.previous_dmlist[~targetlist],
            data_misfits_target[~targetlist],
        )
        if (
            (dmlist[~targetlist] > threshold).all()
            and not data_misfits_achieved
            and self.mode == 1
            and self.invProb.beta > self.betamin
        ):
            self.cool_beta()
            if self.verbose:
                print("Decreasing beta to counter data misfit decrase plateau.")

        # Decide if we should warm alpha instead
        elif (
            data_misfits_achieved
            and self.mode == 2
            and np.all(self.pgi_regularization.alpha_pgi < self.alphasmax)
        ):
            self.warm_alpha()
            if self.verbose:
                print(
                    "Warming alpha_pgi to favor clustering: ",
                    self.pgi_regularization.alpha_pgi,
                )

        # Decide if we should cool beta (to counter data misfit increase)
        elif (
            np.any(dmlist > (1.0 + self.tolerance) * data_misfits_target)
            and self.mode == 2
            and self.invProb.beta > self.betamin
        ):
            self.cool_beta()
            if self.verbose:
                print("Decreasing beta to counter data misfit increase.")

        # Update previous score and dmlist
        self.update_previous_score()
        self.update_previous_dmlist()

    def cool_beta(self):
        """Cool beta according to schedule."""
        data_misfits_target = self.multi_target_misfits_directive.DMtarget
        dmlist = self.multi_target_misfits_directive.dmlist
        ratio = 1.0
        indx = dmlist > (1.0 + self.tolerance) * data_misfits_target
        if np.any(indx) and self.ratio_in_cooling:
            ratio = np.median([dmlist[indx] / data_misfits_target[indx]])
        self.invProb.beta /= self.coolingFactor * ratio

    def warm_alpha(self):
        """Warm alpha according to schedule."""
        data_misfits_target = self.multi_target_misfits_directive.DMtarget
        dmlist = self.multi_target_misfits_directive.dmlist
        ratio = np.median(data_misfits_target / dmlist)
        self.pgi_regularization.alpha_pgi *= self.warmingFactor * ratio

    def update_previous_score(self):
        """
        Update the value of the ``previous_score`` attribute.

        Update it with the current value of the petrophysical misfit, obtained
        from the :meth:`MultiTargetMisfit.phims()` method.
        """
        self.previous_score = copy.deepcopy(self.multi_target_misfits_directive.phims())

    def update_previous_dmlist(self):
        """
        Update the value of the ``previous_dmlist`` attribute.

        Update it with the current value of the data misfits, obtained
        from the :meth:`MultiTargetMisfit.dmlist` attribute.
        """
        self.previous_dmlist = copy.deepcopy(self.multi_target_misfits_directive.dmlist)

    @property
    def directives(self):
        """List of all the directives in the :class:`SimPEG.inverison.BaseInversion``."""
        return self.inversion.directiveList.dList

    @property
    def multi_target_misfits_directive(self):
        """``MultiTargetMisfit`` directive in the :class:`SimPEG.inverison.BaseInversion``."""
        if not hasattr(self, "_mtm_directive"):
            # Obtain multi target misfits directive from the directive list
            multi_target_misfits_directive = [
                directive
                for directive in self.directives
                if isinstance(directive, MultiTargetMisfits)
            ]
            if not multi_target_misfits_directive:
                raise UserWarning(
                    "No MultiTargetMisfits directive found in the current inversion. "
                    "A MultiTargetMisfits directive is needed by the "
                    "PGI_BetaAlphaSchedule directive."
                )
            (self._mtm_directive,) = multi_target_misfits_directive
        return self._mtm_directive

    @property
    def pgi_update_params_directive(self):
        """``PGI_UpdateParam``s directive in the :class:`SimPEG.inverison.BaseInversion``."""
        if not hasattr(self, "_pgi_update_params"):
            # Obtain PGI_UpdateParams directive from the directive list
            pgi_update_params_directive = [
                directive
                for directive in self.directives
                if isinstance(directive, PGI_UpdateParameters)
            ]
            if pgi_update_params_directive:
                (self._pgi_update_params,) = pgi_update_params_directive
            else:
                self._pgi_update_params = None
        return self._pgi_update_params

    @property
    def pgi_regularization(self):
        """PGI regularization in the :class:`SimPEG.inverse_problem.BaseInvProblem``."""
        if not hasattr(self, "_pgi_regularization"):
            pgi_regularization = self.reg.get_functions_of_type(PGI)
            if len(pgi_regularization) != 1:
                raise UserWarning(
                    "'PGI_BetaAlphaSchedule' requires one 'PGI' regularization "
                    "in the objective function."
                )
            self._pgi_regularization = pgi_regularization[0]
        return self._pgi_regularization


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
                            isinstance(
                                regpart, (SmoothnessFirstOrder, SparseSmoothness)
                            ),
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
            self.phi_d_target = self.invProb.dmisfit.survey.nD
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
