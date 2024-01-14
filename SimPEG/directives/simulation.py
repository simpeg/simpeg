import warnings
import numpy as np
from ..maps import IdentityMap, Wires
from ..utils import (
    deprecate_property,
    diagEst,
    mkvc,
    validate_integer,
    validate_float,
    validate_string,
    validate_type,
)
from ..regularization import BaseSimilarityMeasure
from ..optimization import (
    IterationPrinters,
    StoppingCriteria,
    SimilarityMeasureInversionPrinters,
)
from .base import InversionDirective
from .optimization import UpdatePreconditioner
from .tradeoff_estimator import BaseBetaEstimator


###############################################################################
#                                                                             #
#              Directives of joint inversion                                  #
#                                                                             #
###############################################################################


class SimilarityMeasureInversionDirective(InversionDirective):
    """
    Directive for two model similiraty measure joint inversions. Sets Printers and
    StoppingCriteria.

    Notes
    -----
    Methods assume we are working with two models, and a single similarity measure.
    Also, the SimilarityMeasure objective function must be the last regularization.
    """

    printers = [
        IterationPrinters.iteration,
        SimilarityMeasureInversionPrinters.betas,
        SimilarityMeasureInversionPrinters.lambd,
        IterationPrinters.f,
        SimilarityMeasureInversionPrinters.phi_d_list,
        SimilarityMeasureInversionPrinters.phi_m_list,
        SimilarityMeasureInversionPrinters.phi_sim,
        SimilarityMeasureInversionPrinters.iterationCG,
    ]

    def initialize(self):
        if not isinstance(self.reg.objfcts[-1], BaseSimilarityMeasure):
            raise TypeError(
                f"The last regularization function must be an instance of "
                f"BaseSimilarityMeasure, got {type(self.reg.objfcts[-1])}."
            )

        # define relevant attributes
        self.betas = self.reg.multipliers[:-1]
        self.lambd = self.reg.multipliers[-1]
        self.phi_d_list = []
        self.phi_m_list = []
        self.phi_sim = 0.0

        # pass attributes to invProb
        self.invProb.betas = self.betas
        self.invProb.num_models = len(self.betas)
        self.invProb.lambd = self.lambd
        self.invProb.phi_d_list = self.phi_d_list
        self.invProb.phi_m_list = self.phi_m_list
        self.invProb.phi_sim = self.phi_sim

        self.opt.printers = self.printers
        self.opt.stoppers = [StoppingCriteria.iteration]

    def validate(self, directiveList):
        # check that this directive is first in the DirectiveList
        dList = directiveList.dList
        self_ind = dList.index(self)
        if self_ind != 0:
            raise IndexError(
                "The CrossGradientInversionDirective must be first in directive list."
            )
        return True

    def endIter(self):
        # compute attribute values
        phi_d = []
        for dmis in self.dmisfit.objfcts:
            phi_d.append(dmis(self.opt.xc))

        phi_m = []
        for reg in self.reg.objfcts:
            phi_m.append(reg(self.opt.xc))

        # pass attributes values to invProb
        self.invProb.phi_d_list = phi_d
        self.invProb.phi_m_list = phi_m[:-1]
        self.invProb.phi_sim = phi_m[-1]
        self.invProb.betas = self.reg.multipliers[:-1]
        # Assume last reg.objfct is the coupling
        self.invProb.lambd = self.reg.multipliers[-1]


class Update_Wj(InversionDirective):
    """
    Create approx-sensitivity base weighting using the probing method
    """

    def __init__(self, k=None, itr=None, **kwargs):
        self.k = k
        self.itr = itr
        super().__init__(**kwargs)

    @property
    def k(self):
        """Number of probing cycles for the estimator.

        Returns
        -------
        int
        """
        return self._k

    @k.setter
    def k(self, value):
        if value is not None:
            value = validate_integer("k", value, min_val=1)
        self._k = value

    @property
    def itr(self):
        """Which iteration to update the sensitivity.

        Will always update if `None`.

        Returns
        -------
        int or None
        """
        return self._itr

    @itr.setter
    def itr(self, value):
        if value is not None:
            value = validate_integer("itr", value, min_val=1)
        self._itr = value

    def endIter(self):
        if self.itr is None or self.itr == self.opt.iter:
            m = self.invProb.model
            if self.k is None:
                self.k = int(self.survey.nD / 10)

            def JtJv(v):
                Jv = self.simulation.Jvec(m, v)

                return self.simulation.Jtvec(m, Jv)

            JtJdiag = diagEst(JtJv, len(m), k=self.k)
            JtJdiag = JtJdiag / max(JtJdiag)

            self.reg.wght = JtJdiag


class UpdateSensitivityWeights(InversionDirective):
    r"""
    Sensitivity weighting for linear and non-linear least-squares inverse problems.

    This directive computes the root-mean squared sensitivities for the
    forward simulation(s) attached to the inverse problem, then truncates
    and scales the result to create cell weights which are applied in the regularization.
    The underlying theory is provided below in the `Notes` section.

    This directive **requires** that the map for the regularization function is either
    class:`SimPEG.maps.Wires` or class:`SimPEG.maps.Identity`. In other words, the
    sensitivity weighting cannot be applied for parametric inversion. In addition,
    the simulation(s) connected to the inverse problem **must** have a ``getJ`` or
    ``getJtJdiag`` method.

    This directive's place in the :class:`DirectivesList` **must** be
    before any directives which update the preconditioner for the inverse problem
    (i.e. :class:`UpdatePreconditioner`), and **must** be before any directives that
    estimate the starting trade-off parameter (i.e. :class:`EstimateBeta_ByEig`
    and :class:`EstimateBetaMaxDerivative`).

    Parameters
    ----------
    every_iteration : bool
        When ``True``, update sensitivity weighting at every model update; non-linear problems.
        When ``False``, create sensitivity weights for starting model only; linear problems.
    threshold : float
        Threshold value for smallest weighting value.
    threshold_method : {'amplitude', 'global', 'percentile'}
        Threshold method for how `threshold_value` is applied:

            - amplitude:
                the smallest root-mean squared sensitivity is a fractional percent of the largest value; must be between 0 and 1.
            - global:
                `threshold_value` is added to the cell weights prior to normalization; must be greater than 0.
            - percentile:
                the smallest root-mean squared sensitivity is set using percentile threshold; must be between 0 and 100.

    normalization_method : {'maximum', 'min_value', None}
        Normalization method applied to sensitivity weights.

        Options are:

            - maximum:
                sensitivity weights are normalized by the largest value such that the largest weight is equal to 1.
            - minimum:
                sensitivity weights are normalized by the smallest value, after thresholding, such that the smallest weights are equal to 1.
            - ``None``:
                normalization is not applied.

    Notes
    -----
    Let :math:`\mathbf{J}` represent the Jacobian. To create sensitivity weights, root-mean squared (RMS) sensitivities
    :math:`\mathbf{s}` are computed by summing the squares of the rows of the Jacobian:

    .. math::
        \mathbf{s} = \Bigg [ \sum_i \, \mathbf{J_{i, \centerdot }}^2 \, \Bigg ]^{1/2}

    The dynamic range of RMS sensitivities can span many orders of magnitude. When computing sensitivity
    weights, thresholding is generally applied to set a minimum value.

    Thresholding
    ^^^^^^^^^^^^

    If **global** thresholding is applied, we add a constant :math:`\tau` to the RMS sensitivities:

    .. math::
        \mathbf{\tilde{s}} = \mathbf{s} + \tau

    In the case of **percentile** thresholding, we let :math:`s_{\%}` represent a given percentile.
    Thresholding to set a minimum value is applied as follows:

    .. math::
        \tilde{s}_j = \begin{cases}
        s_j \;\; for \;\; s_j \geq s_{\%} \\
        s_{\%} \;\; for \;\; s_j < s_{\%}
        \end{cases}

    If **absolute** thresholding is applied, we define :math:`\eta` as a fractional percent.
    In this case, thresholding is applied as follows:

    .. math::
        \tilde{s}_j = \begin{cases}
        s_j \;\; for \;\; s_j \geq \eta s_{max} \\
        \eta s_{max} \;\; for \;\; s_j < \eta s_{max}
        \end{cases}
    """

    def __init__(
        self,
        every_iteration=False,
        threshold_value=1e-12,
        threshold_method="amplitude",
        normalization_method="maximum",
        **kwargs,
    ):
        if "everyIter" in kwargs.keys():
            warnings.warn(
                "'everyIter' property is deprecated and will be removed in SimPEG 0.20.0."
                "Please use 'every_iteration'.",
                stacklevel=2,
            )
            every_iteration = kwargs.pop("everyIter")

        if "threshold" in kwargs.keys():
            warnings.warn(
                "'threshold' property is deprecated and will be removed in SimPEG 0.20.0."
                "Please use 'threshold_value'.",
                stacklevel=2,
            )
            threshold_value = kwargs.pop("threshold")

        if "normalization" in kwargs.keys():
            warnings.warn(
                "'normalization' property is deprecated and will be removed in SimPEG 0.20.0."
                "Please define normalization using 'normalization_method'.",
                stacklevel=2,
            )
            normalization_method = kwargs.pop("normalization")
            if normalization_method is True:
                normalization_method = "maximum"
            else:
                normalization_method = None

        super().__init__(**kwargs)

        self.every_iteration = every_iteration
        self.threshold_value = threshold_value
        self.threshold_method = threshold_method
        self.normalization_method = normalization_method

    @property
    def every_iteration(self):
        """Update sensitivity weights when model is updated.

        When ``True``, update sensitivity weighting at every model update; non-linear problems.
        When ``False``, create sensitivity weights for starting model only; linear problems.

        Returns
        -------
        bool
        """
        return self._every_iteration

    @every_iteration.setter
    def every_iteration(self, value):
        self._every_iteration = validate_type("every_iteration", value, bool)

    everyIter = deprecate_property(
        every_iteration, "everyIter", "every_iteration", removal_version="0.20.0"
    )

    @property
    def threshold_value(self):
        """Threshold value used to set minimum weighting value.

        The way thresholding is applied to the weighting model depends on the
        `threshold_method` property. The choices for `threshold_method` are:

            - global:
                `threshold_value` is added to the cell weights prior to normalization; must be greater than 0.
            - percentile:
                `threshold_value` is a percentile cutoff; must be between 0 and 100
            - amplitude:
                `threshold_value` is the fractional percent of the largest value; must be between 0 and 1


        Returns
        -------
        float
        """
        return self._threshold_value

    @threshold_value.setter
    def threshold_value(self, value):
        self._threshold_value = validate_float("threshold_value", value, min_val=0.0)

    threshold = deprecate_property(
        threshold_value, "threshold", "threshold_value", removal_version="0.20.0"
    )

    @property
    def threshold_method(self):
        """Threshold method for how `threshold_value` is applied:

            - global:
                `threshold_value` is added to the cell weights prior to normalization; must be greater than 0.
            - percentile:
                the smallest root-mean squared sensitivity is set using percentile threshold; must be between 0 and 100
            - amplitude:
                the smallest root-mean squared sensitivity is a fractional percent of the largest value; must be between 0 and 1


        Returns
        -------
        str
        """
        return self._threshold_method

    @threshold_method.setter
    def threshold_method(self, value):
        self._threshold_method = validate_string(
            "threshold_method", value, string_list=["global", "percentile", "amplitude"]
        )

    @property
    def normalization_method(self):
        """Normalization method applied to sensitivity weights.

        Options are:

            - ``None``
                normalization is not applied
            - maximum:
                sensitivity weights are normalized by the largest value such that the largest weight is equal to 1.
            - minimum:
                sensitivity weights are normalized by the smallest value, after thresholding, such that the smallest weights are equal to 1.

        Returns
        -------
        None, str
        """
        return self._normalization_method

    @normalization_method.setter
    def normalization_method(self, value):
        if value is None:
            self._normalization_method = value

        elif isinstance(value, bool):
            warnings.warn(
                "Boolean type for 'normalization_method' is deprecated and will be removed in 0.20.0."
                "Please use None, 'maximum' or 'minimum'.",
                stacklevel=2,
            )
            if value:
                self._normalization_method = "maximum"
            else:
                self._normalization_method = None

        else:
            self._normalization_method = validate_string(
                "normalization_method", value, string_list=["minimum", "maximum"]
            )

    normalization = deprecate_property(
        normalization_method,
        "normalization",
        "normalization_method",
        removal_version="0.20.0",
    )

    def initialize(self):
        """Compute sensitivity weights upon starting the inversion."""
        for reg in self.reg.objfcts:
            if not isinstance(reg.mapping, (IdentityMap, Wires)):
                raise TypeError(
                    f"Mapping for the regularization must be of type {IdentityMap} or {Wires}. "
                    + f"Input mapping of type {type(reg.mapping)}."
                )

        self.update()

    def endIter(self):
        """Execute end of iteration."""

        if self.every_iteration:
            self.update()

    def update(self):
        """Update sensitivity weights"""

        jtj_diag = np.zeros_like(self.invProb.model)
        m = self.invProb.model

        for sim, dmisfit in zip(self.simulation, self.dmisfit.objfcts):
            if getattr(sim, "getJtJdiag", None) is None:
                if getattr(sim, "getJ", None) is None:
                    raise AttributeError(
                        "Simulation does not have a getJ attribute."
                        + "Cannot form the sensitivity explicitly"
                    )
                jtj_diag += mkvc(np.sum((dmisfit.W * sim.getJ(m)) ** 2.0, axis=0))
            else:
                jtj_diag += sim.getJtJdiag(m, W=dmisfit.W)

        # Compute and sum root-mean squared sensitivities for all objective functions
        wr = np.zeros_like(self.invProb.model)
        for reg in self.reg.objfcts:
            if isinstance(reg, BaseSimilarityMeasure):
                continue

            mesh = reg.regularization_mesh
            n_cells = mesh.nC
            mapped_jtj_diag = reg.mapping * jtj_diag
            # reshape the mapped, so you can divide by volume
            # (let's say it was a vector or anisotropic model)
            mapped_jtj_diag = mapped_jtj_diag.reshape((n_cells, -1), order="F")
            wr_temp = mapped_jtj_diag / reg.regularization_mesh.vol[:, None] ** 2.0
            wr_temp = wr_temp.reshape(-1, order="F")

            wr += reg.mapping.deriv(self.invProb.model).T * wr_temp

        wr **= 0.5

        # Apply thresholding
        if self.threshold_method == "global":
            wr += self.threshold_value
        elif self.threshold_method == "percentile":
            wr = np.clip(
                wr, a_min=np.percentile(wr, self.threshold_value), a_max=np.inf
            )
        else:
            wr = np.clip(wr, a_min=self.threshold_value * wr.max(), a_max=np.inf)

        # Apply normalization
        if self.normalization_method == "maximum":
            wr /= wr.max()
        elif self.normalization_method == "minimum":
            wr /= wr.min()

        # Add sensitivity weighting to all model objective functions
        for reg in self.reg.objfcts:
            if not isinstance(reg, BaseSimilarityMeasure):
                sub_regs = getattr(reg, "objfcts", [reg])
                for sub_reg in sub_regs:
                    sub_reg.set_weights(sensitivity=sub_reg.mapping * wr)

    def validate(self, directiveList):
        """Validate directive against directives list.

        The ``UpdateSensitivityWeights`` directive impacts the regularization by applying
        cell weights. As a result, its place in the :class:`DirectivesList` must be
        before any directives which update the preconditioner for the inverse problem
        (i.e. :class:`UpdatePreconditioner`), and must be before any directives that
        estimate the starting trade-off parameter (i.e. :class:`EstimateBeta_ByEig`
        and :class:`EstimateBetaMaxDerivative`).


        Returns
        -------
        bool
            Returns ``True`` if validation passes. Otherwise, an error is thrown.
        """
        # check if a beta estimator is in the list after setting the weights
        dList = directiveList.dList
        self_ind = dList.index(self)

        beta_estimator_ind = [isinstance(d, BaseBetaEstimator) for d in dList]
        lin_precond_ind = [isinstance(d, UpdatePreconditioner) for d in dList]

        if any(beta_estimator_ind):
            assert beta_estimator_ind.index(True) > self_ind, (
                "The directive for setting intial beta must be after UpdateSensitivityWeights "
                "in the directiveList"
            )

        if any(lin_precond_ind):
            assert lin_precond_ind.index(True) > self_ind, (
                "The directive 'UpdatePreconditioner' must be after UpdateSensitivityWeights "
                "in the directiveList"
            )

        return True
