from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

from datetime import datetime
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy.sparse as sp
from ..typing import RandomSeed
from ..data_misfit import BaseDataMisfit
from ..objective_function import BaseObjectiveFunction, ComboObjectiveFunction
from ..maps import IdentityMap, Wires
from ..regularization import (
    WeightedLeastSquares,
    BaseRegularization,
    Smallness,
    Sparse,
    SparseSmallness,
    PGIsmallness,
    SmoothnessFirstOrder,
    SparseSmoothness,
    BaseSimilarityMeasure,
)
from ..utils import (
    mkvc,
    set_kwargs,
    sdiag,
    estimate_diagonal,
    spherical2cartesian,
    cartesian2spherical,
    Zero,
    eigenvalue_by_power_iteration,
    validate_string,
    get_logger,
)
from ..utils.code_utils import (
    deprecate_property,
    validate_type,
    validate_integer,
    validate_float,
    validate_ndarray_with_shape,
)

if TYPE_CHECKING:
    from ..simulation import BaseSimulation
    from ..survey import BaseSurvey


class InversionDirective:
    """Base inversion directive class.

    SimPEG directives initialize and update parameters used by the inversion algorithm;
    e.g. setting the initial beta or updating the regularization. ``InversionDirective``
    is a parent class responsible for connecting directives to the data misfit, regularization
    and optimization defining the inverse problem.

    Parameters
    ----------
    inversion : simpeg.inversion.BaseInversion, None
        An SimPEG inversion object; i.e. an instance of :class:`simpeg.inversion.BaseInversion`.
    dmisfit : simpeg.data_misfit.BaseDataMisfit, None
        A data data misfit; i.e. an instance of :class:`simpeg.data_misfit.BaseDataMisfit`.
    reg : simpeg.regularization.BaseRegularization, None
        The regularization, or model objective function; i.e. an instance of :class:`simpeg.regularization.BaseRegularization`.
    verbose : bool
        Whether or not to print debugging information.
    """

    _REGISTRY = {}

    _regPair = [WeightedLeastSquares, BaseRegularization, ComboObjectiveFunction]
    _dmisfitPair = [BaseDataMisfit, ComboObjectiveFunction]

    def __init__(self, inversion=None, dmisfit=None, reg=None, verbose=False, **kwargs):
        self.inversion = inversion
        self.dmisfit = dmisfit
        self.reg = reg
        self.verbose = verbose
        set_kwargs(self, **kwargs)

    @property
    def verbose(self):
        """Whether or not to print debugging information.

        Returns
        -------
        bool
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = validate_type("verbose", value, bool)

    @property
    def inversion(self):
        """Inversion object associated with the directive.

        Returns
        -------
        simpeg.inversion.BaseInversion
            The inversion associated with the directive.
        """
        if not hasattr(self, "_inversion"):
            return None
        return self._inversion

    @inversion.setter
    def inversion(self, i):
        if getattr(self, "_inversion", None) is not None:
            warnings.warn(
                "InversionDirective {0!s} has switched to a new inversion.".format(
                    self.__class__.__name__
                ),
                stacklevel=2,
            )
        self._inversion = i

    @property
    def invProb(self):
        """Inverse problem associated with the directive.

        Returns
        -------
        simpeg.inverse_problem.BaseInvProblem
            The inverse problem associated with the directive.
        """
        return self.inversion.invProb

    @property
    def opt(self):
        """Optimization algorithm associated with the directive.

        Returns
        -------
        simpeg.optimization.Minimize
            Optimization algorithm associated with the directive.
        """
        return self.invProb.opt

    @property
    def reg(self) -> BaseObjectiveFunction:
        """Regularization associated with the directive.

        Returns
        -------
        simpeg.regularization.BaseRegularization
            The regularization associated with the directive.
        """
        if getattr(self, "_reg", None) is None:
            self.reg = self.invProb.reg  # go through the setter
        return self._reg

    @reg.setter
    def reg(self, value):
        if value is not None:
            assert any(
                [isinstance(value, regtype) for regtype in self._regPair]
            ), "Regularization must be in {}, not {}".format(self._regPair, type(value))

            if isinstance(value, WeightedLeastSquares):
                value = 1 * value  # turn it into a combo objective function
        self._reg = value

    @property
    def dmisfit(self) -> BaseObjectiveFunction:
        """Data misfit associated with the directive.

        Returns
        -------
        simpeg.data_misfit.BaseDataMisfit
            The data misfit associated with the directive.
        """
        if getattr(self, "_dmisfit", None) is None:
            self.dmisfit = self.invProb.dmisfit  # go through the setter
        return self._dmisfit

    @dmisfit.setter
    def dmisfit(self, value):
        if value is not None:
            assert any(
                [isinstance(value, dmisfittype) for dmisfittype in self._dmisfitPair]
            ), "Misfit must be in {}, not {}".format(self._dmisfitPair, type(value))

            if not isinstance(value, ComboObjectiveFunction):
                value = 1 * value  # turn it into a combo objective function
        self._dmisfit = value

    @property
    def survey(self) -> list["BaseSurvey"]:
        """Return survey for all data misfits

        Assuming that ``dmisfit`` is always a ``ComboObjectiveFunction``,
        return a list containing the survey for each data misfit; i.e.
        [survey1, survey2, ...]

        Returns
        -------
        list of simpeg.survey.Survey
            Survey for all data misfits.
        """
        return [objfcts.simulation.survey for objfcts in self.dmisfit.objfcts]

    @property
    def simulation(self) -> list["BaseSimulation"]:
        """Return simulation for all data misfits.

        Assuming that ``dmisfit`` is always a ``ComboObjectiveFunction``,
        return a list containing the simulation for each data misfit; i.e.
        [sim1, sim2, ...].

        Returns
        -------
        list of simpeg.simulation.BaseSimulation
            Simulation for all data misfits.
        """
        return [objfcts.simulation for objfcts in self.dmisfit.objfcts]

    def initialize(self):
        """Initialize inversion parameter(s) according to directive."""
        pass

    def endIter(self):
        """Update inversion parameter(s) according to directive at end of iteration."""
        pass

    def finish(self):
        """Update inversion parameter(s) according to directive at end of inversion."""
        pass

    def validate(self, directiveList=None):
        """Validate directive.

        The `validate` method returns ``True`` if the directive and its location within
        the directives list does not encounter conflicts. Otherwise, an appropriate error
        message is returned describing the conflict.

        Parameters
        ----------
        directive_list : simpeg.directives.DirectiveList
            List of directives used in the inversion.

        Returns
        -------
        bool
            Returns ``True`` if validated, otherwise an approriate error is returned.
        """
        return True


class DirectiveList(object):
    """Directives list

    SimPEG directives initialize and update parameters used by the inversion algorithm;
    e.g. setting the initial beta or updating the regularization. ``DirectiveList`` stores
    the set of directives used in the inversion algorithm.

    Parameters
    ----------
    *directives : simpeg.directives.InversionDirective
        Directives for the inversion.
    inversion : simpeg.inversion.BaseInversion
        The inversion associated with the directives list.
    debug : bool
        Whether to print debugging information.

    """

    def __init__(self, *directives, inversion=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.dList = []
        for d in directives:
            assert isinstance(
                d, InversionDirective
            ), "All directives must be InversionDirectives not {}".format(type(d))
            self.dList.append(d)
        self.inversion = inversion
        self.verbose = debug

    @property
    def debug(self):
        """Whether or not to print debugging information

        Returns
        -------
        bool
        """
        return getattr(self, "_debug", False)

    @debug.setter
    def debug(self, value):
        for d in self.dList:
            d.debug = value
        self._debug = value

    @property
    def inversion(self):
        """Inversion object associated with the directives list.

        Returns
        -------
        simpeg.inversion.BaseInversion
            The inversion associated with the directives list.
        """
        return getattr(self, "_inversion", None)

    @inversion.setter
    def inversion(self, i):
        if self.inversion is i:
            return
        if getattr(self, "_inversion", None) is not None:
            warnings.warn(
                "{0!s} has switched to a new inversion.".format(
                    self.__class__.__name__
                ),
                stacklevel=2,
            )
        for d in self.dList:
            d.inversion = i
        self._inversion = i

    def call(self, ruleType):
        if self.dList is None:
            if self.verbose:
                print("DirectiveList is None, no directives to call!")
            return

        directives = ["initialize", "endIter", "finish"]
        assert ruleType in directives, 'Directive type must be in ["{0!s}"]'.format(
            '", "'.join(directives)
        )
        for r in self.dList:
            getattr(r, ruleType)()

    def validate(self):
        [directive.validate(self) for directive in self]
        return True

    def __iter__(self):
        return iter(self.dList)


class BaseBetaEstimator(InversionDirective):
    """Base class for estimating initial trade-off parameter (beta).

    This class has properties and methods inherited by directive classes which estimate
    the initial trade-off parameter (beta). This class is not used directly to create
    directives for the inversion.

    Parameters
    ----------
    beta0_ratio : float
        Desired ratio between data misfit and model objective function at initial beta iteration.
    random_seed : None or :class:`~simpeg.typing.RandomSeed`, optional
        Random seed used for random sampling. It can either be an int,
        a predefined Numpy random number generator, or any valid input to
        ``numpy.random.default_rng``.

    """

    def __init__(
        self,
        beta0_ratio=1.0,
        random_seed: RandomSeed | None = None,
        **kwargs,
    ):
        # Deprecate seed argument
        if kwargs.pop("seed", None) is not None:
            raise TypeError(
                "'seed' has been removed in "
                " SimPEG v0.24.0, please use 'random_seed' instead.",
            )
        super().__init__(**kwargs)
        self.beta0_ratio = beta0_ratio
        self.random_seed = random_seed

    @property
    def beta0_ratio(self):
        """The estimated ratio is multiplied by this to obtain beta.

        Returns
        -------
        float
        """
        return self._beta0_ratio

    @beta0_ratio.setter
    def beta0_ratio(self, value):
        self._beta0_ratio = validate_float(
            "beta0_ratio", value, min_val=0.0, inclusive_min=False
        )

    @property
    def random_seed(self):
        """Random seed to initialize with.

        Returns
        -------
        int, numpy.random.Generator or None
        """
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value):
        try:
            np.random.default_rng(value)
        except TypeError as err:
            msg = (
                "Unable to initialize the random number generator with "
                f"a {type(value).__name__}"
            )
            raise TypeError(msg) from err
        self._random_seed = value

    def validate(self, directive_list):
        ind = [isinstance(d, BaseBetaEstimator) for d in directive_list.dList]
        assert np.sum(ind) == 1, (
            "Multiple directives for computing initial beta detected in directives list. "
            "Only one directive can be used to set the initial beta."
        )

        return True

    seed = deprecate_property(
        random_seed,
        "seed",
        "random_seed",
        removal_version="0.24.0",
        error=True,
    )


class BetaEstimateMaxDerivative(BaseBetaEstimator):
    r"""Estimate initial trade-off parameter (beta) using largest derivatives.

    The initial trade-off parameter (beta) is estimated by scaling the ratio
    between the largest derivatives in the gradient of the data misfit and
    model objective function. The estimated trade-off parameter is used to
    update the **beta** property in the associated :class:`simpeg.inverse_problem.BaseInvProblem`
    object prior to running the inversion. A separate directive is used for updating the
    trade-off parameter at successive beta iterations; see :class:`BetaSchedule`.

    Parameters
    ----------
    beta0_ratio: float
        Desired ratio between data misfit and model objective function at initial beta iteration.
    random_seed : None or :class:`~simpeg.typing.RandomSeed`, optional
        Random seed used for random sampling. It can either be an int,
        a predefined Numpy random number generator, or any valid input to
        ``numpy.random.default_rng``.

    Notes
    -----
    Let :math:`\phi_d` represent the data misfit, :math:`\phi_m` represent the model
    objective function and :math:`\mathbf{m_0}` represent the starting model. The first
    model update is obtained by minimizing the a global objective function of the form:

    .. math::
        \phi (\mathbf{m_0}) = \phi_d (\mathbf{m_0}) + \beta_0 \phi_m (\mathbf{m_0})

    where :math:`\beta_0` represents the initial trade-off parameter (beta).

    We define :math:`\gamma` as the desired ratio between the data misfit and model objective
    functions at the initial beta iteration (defined by the 'beta0_ratio' input argument).
    Here, the initial trade-off parameter is computed according to:

    .. math::
        \beta_0 = \gamma \frac{| \nabla_m \phi_d (\mathbf{m_0}) |_{max}}{| \nabla_m \phi_m (\mathbf{m_0 + \delta m}) |_{max}}

    where

    .. math::
        \delta \mathbf{m} = \frac{m_{max}}{\mu_{max}} \boldsymbol{\mu}

    and :math:`\boldsymbol{\mu}` is a set of independent samples from the
    continuous uniform distribution between 0 and 1.

    """

    def __init__(
        self, beta0_ratio=1.0, random_seed: RandomSeed | None = None, **kwargs
    ):
        super().__init__(beta0_ratio=beta0_ratio, random_seed=random_seed, **kwargs)

    def initialize(self):
        rng = np.random.default_rng(seed=self.random_seed)

        if self.verbose:
            print("Calculating the beta0 parameter.")

        m = self.invProb.model

        x0 = rng.random(size=m.shape)
        phi_d_deriv = np.abs(self.dmisfit.deriv(m)).max()
        dm = x0 / x0.max() * m.max()
        phi_m_deriv = np.abs(self.reg.deriv(m + dm)).max()

        self.ratio = np.asarray(phi_d_deriv / phi_m_deriv)
        self.beta0 = self.beta0_ratio * self.ratio
        self.invProb.beta = self.beta0


class BetaEstimate_ByEig(BaseBetaEstimator):
    r"""Estimate initial trade-off parameter (beta) by power iteration.

    The initial trade-off parameter (beta) is estimated by scaling the ratio
    between the largest eigenvalue in the second derivative of the data
    misfit and the model objective function. The largest eigenvalues are estimated
    using the power iteration method; see :func:`simpeg.utils.eigenvalue_by_power_iteration`.
    The estimated trade-off parameter is used to update the **beta** property in the
    associated :class:`simpeg.inverse_problem.BaseInvProblem` object prior to running the inversion.
    Note that a separate directive is used for updating the trade-off parameter at successive
    beta iterations; see :class:`BetaSchedule`.

    Parameters
    ----------
    beta0_ratio: float
        Desired ratio between data misfit and model objective function at initial beta iteration.
    n_pw_iter : int
        Number of power iterations used to estimate largest eigenvalues.
    random_seed : None or :class:`~simpeg.typing.RandomSeed`, optional
        Random seed used for random sampling. It can either be an int,
        a predefined Numpy random number generator, or any valid input to
        ``numpy.random.default_rng``.

    Notes
    -----
    Let :math:`\phi_d` represent the data misfit, :math:`\phi_m` represent the model
    objective function and :math:`\mathbf{m_0}` represent the starting model. The first
    model update is obtained by minimizing the a global objective function of the form:

    .. math::
        \phi (\mathbf{m_0}) = \phi_d (\mathbf{m_0}) + \beta_0 \phi_m (\mathbf{m_0})

    where :math:`\beta_0` represents the initial trade-off parameter (beta).
    Let :math:`\gamma` define the desired ratio between the data misfit and model
    objective functions at the initial beta iteration (defined by the 'beta0_ratio' input argument).
    Using the power iteration approach, our initial trade-off parameter is given by:

    .. math::
        \beta_0 = \gamma \frac{\lambda_d}{\lambda_m}

    where :math:`\lambda_d` as the largest eigenvalue of the Hessian of the data misfit, and
    :math:`\lambda_m` as the largest eigenvalue of the Hessian of the model objective function.
    For each Hessian, the largest eigenvalue is computed using power iteration. The input
    parameter 'n_pw_iter' sets the number of power iterations used in the estimate.

    For a description of the power iteration approach for estimating the larges eigenvalue,
    see :func:`simpeg.utils.eigenvalue_by_power_iteration`.

    """

    def __init__(
        self,
        beta0_ratio=1.0,
        n_pw_iter=4,
        random_seed: RandomSeed | None = None,
        **kwargs,
    ):
        super().__init__(beta0_ratio=beta0_ratio, random_seed=random_seed, **kwargs)
        self.n_pw_iter = n_pw_iter

    @property
    def n_pw_iter(self):
        """Number of power iterations for estimating largest eigenvalues.

        Returns
        -------
        int
            Number of power iterations for estimating largest eigenvalues.
        """
        return self._n_pw_iter

    @n_pw_iter.setter
    def n_pw_iter(self, value):
        self._n_pw_iter = validate_integer("n_pw_iter", value, min_val=1)

    def initialize(self):
        rng = np.random.default_rng(seed=self.random_seed)

        if self.verbose:
            print("Calculating the beta0 parameter.")

        m = self.invProb.model

        dm_eigenvalue = eigenvalue_by_power_iteration(
            self.dmisfit,
            m,
            n_pw_iter=self.n_pw_iter,
            random_seed=rng,
        )
        reg_eigenvalue = eigenvalue_by_power_iteration(
            self.reg,
            m,
            n_pw_iter=self.n_pw_iter,
            random_seed=rng,
        )

        self.ratio = np.asarray(dm_eigenvalue / reg_eigenvalue)
        self.beta0 = self.beta0_ratio * self.ratio
        self.invProb.beta = self.beta0


class BetaSchedule(InversionDirective):
    """Reduce trade-off parameter (beta) at successive iterations using a cooling schedule.

    Updates the **beta** property in the associated :class:`simpeg.inverse_problem.BaseInvProblem`
    while the inversion is running.
    For linear least-squares problems, the optimization problem can be solved in a
    single step and the cooling rate can be set to *1*. For non-linear optimization
    problems, multiple steps are required obtain the minimizer for a fixed trade-off
    parameter. In this case, the cooling rate should be larger than 1.

    Parameters
    ----------
    coolingFactor : float
        The factor by which the trade-off parameter is decreased when updated.
        The preexisting value of the trade-off parameter is divided by the cooling factor.
    coolingRate : int
        Sets the number of successive iterations before the trade-off parameter is reduced.
        Use *1* for linear least-squares optimization problems. Use *2* for weakly non-linear
        optimization problems. Use *3* for general non-linear optimization problems.

    """

    def __init__(self, coolingFactor=8.0, coolingRate=3, **kwargs):
        super().__init__(**kwargs)
        self.coolingFactor = coolingFactor
        self.coolingRate = coolingRate

    @property
    def coolingFactor(self):
        """Beta is divided by this value every `coolingRate` iterations.

        Returns
        -------
        float
        """
        return self._coolingFactor

    @coolingFactor.setter
    def coolingFactor(self, value):
        self._coolingFactor = validate_float(
            "coolingFactor", value, min_val=0.0, inclusive_min=False
        )

    @property
    def coolingRate(self):
        """Cool after this number of iterations.

        Returns
        -------
        int
        """
        return self._coolingRate

    @coolingRate.setter
    def coolingRate(self, value):
        self._coolingRate = validate_integer("coolingRate", value, min_val=1)

    def endIter(self):
        it = self.opt.iter
        if 0 < it < self.opt.maxIter and it % self.coolingRate == 0:
            if self.verbose:
                print(
                    "BetaSchedule is cooling Beta. Iteration: {0:d}".format(
                        self.opt.iter
                    )
                )
            self.invProb.beta /= self.coolingFactor


class AlphasSmoothEstimate_ByEig(InversionDirective):
    """
    Estimate the alphas multipliers for the smoothness terms of the regularization
    as a multiple of the ratio between the highest eigenvalue of the
    smallness term and the highest eigenvalue of each smoothness term of the regularization.
    The highest eigenvalue are estimated through power iterations and Rayleigh quotient.
    """

    def __init__(
        self,
        alpha0_ratio=1.0,
        n_pw_iter=4,
        random_seed: RandomSeed | None = None,
        **kwargs,
    ):
        # Deprecate seed argument
        if kwargs.pop("seed", None) is not None:
            raise TypeError(
                "'seed' has been removed in "
                " SimPEG v0.24.0, please use 'random_seed' instead.",
            )
        super().__init__(**kwargs)
        self.alpha0_ratio = alpha0_ratio
        self.n_pw_iter = n_pw_iter
        self.random_seed = random_seed

    @property
    def alpha0_ratio(self):
        """the estimated Alpha_smooth is multiplied by this ratio (int or array).

        Returns
        -------
        numpy.ndarray
        """
        return self._alpha0_ratio

    @alpha0_ratio.setter
    def alpha0_ratio(self, value):
        self._alpha0_ratio = validate_ndarray_with_shape(
            "alpha0_ratio", value, shape=("*",)
        )

    @property
    def n_pw_iter(self):
        """Number of power iterations for estimation.

        Returns
        -------
        int
        """
        return self._n_pw_iter

    @n_pw_iter.setter
    def n_pw_iter(self, value):
        self._n_pw_iter = validate_integer("n_pw_iter", value, min_val=1)

    @property
    def random_seed(self):
        """Random seed to initialize with.

        Returns
        -------
        int, numpy.random.Generator or None
        """
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value):
        try:
            np.random.default_rng(value)
        except TypeError as err:
            msg = (
                "Unable to initialize the random number generator with "
                f"a {type(value).__name__}"
            )
            raise TypeError(msg) from err
        self._random_seed = value

    seed = deprecate_property(
        random_seed,
        "seed",
        "random_seed",
        removal_version="0.24.0",
        error=True,
    )

    def initialize(self):
        """"""
        rng = np.random.default_rng(seed=self.random_seed)

        smoothness = []
        smallness = []
        parents = {}
        for regobjcts in self.reg.objfcts:
            if isinstance(regobjcts, ComboObjectiveFunction):
                objfcts = regobjcts.objfcts
            else:
                objfcts = [regobjcts]

            for obj in objfcts:
                if isinstance(
                    obj,
                    (
                        Smallness,
                        SparseSmallness,
                        PGIsmallness,
                    ),
                ):
                    smallness += [obj]

                elif isinstance(obj, (SmoothnessFirstOrder, SparseSmoothness)):
                    parents[obj] = regobjcts
                    smoothness += [obj]

        if len(smallness) == 0:
            raise UserWarning(
                "Directive 'AlphasSmoothEstimate_ByEig' requires a regularization with at least one Small instance."
            )

        smallness_eigenvalue = eigenvalue_by_power_iteration(
            smallness[0],
            self.invProb.model,
            n_pw_iter=self.n_pw_iter,
            random_seed=rng,
        )

        self.alpha0_ratio = self.alpha0_ratio * np.ones(len(smoothness))

        if len(self.alpha0_ratio) != len(smoothness):
            raise ValueError(
                f"Input values for 'alpha0_ratio' should be of len({len(smoothness)}). Provided {self.alpha0_ratio}"
            )

        alphas = []
        for user_alpha, obj in zip(self.alpha0_ratio, smoothness):
            smooth_i_eigenvalue = eigenvalue_by_power_iteration(
                obj,
                self.invProb.model,
                n_pw_iter=self.n_pw_iter,
                random_seed=rng,
            )
            ratio = smallness_eigenvalue / smooth_i_eigenvalue

            mtype = obj._multiplier_pair

            new_alpha = getattr(parents[obj], mtype) * user_alpha * ratio
            setattr(parents[obj], mtype, new_alpha)
            alphas += [new_alpha]

        if self.verbose:
            print(f"Alpha scales: {alphas}")


class ScalingMultipleDataMisfits_ByEig(InversionDirective):
    """
    For multiple data misfits only: multiply each data misfit term
    by the inverse of its highest eigenvalue and then
    normalize the sum of the data misfit multipliers to one.
    The highest eigenvalue are estimated through power iterations and Rayleigh quotient.
    """

    def __init__(
        self,
        chi0_ratio=None,
        n_pw_iter=4,
        random_seed: RandomSeed | None = None,
        **kwargs,
    ):
        # Deprecate seed argument
        if kwargs.pop("seed", None) is not None:
            raise TypeError(
                "'seed' has been removed in "
                " SimPEG v0.24.0, please use 'random_seed' instead.",
            )
        super().__init__(**kwargs)
        self.chi0_ratio = chi0_ratio
        self.n_pw_iter = n_pw_iter
        self.random_seed = random_seed

    @property
    def chi0_ratio(self):
        """the estimated Alpha_smooth is multiplied by this ratio (int or array)

        Returns
        -------
        numpy.ndarray
        """
        return self._chi0_ratio

    @chi0_ratio.setter
    def chi0_ratio(self, value):
        if value is not None:
            value = validate_ndarray_with_shape("chi0_ratio", value, shape=("*",))
        self._chi0_ratio = value

    @property
    def n_pw_iter(self):
        """Number of power iterations for estimation.

        Returns
        -------
        int
        """
        return self._n_pw_iter

    @n_pw_iter.setter
    def n_pw_iter(self, value):
        self._n_pw_iter = validate_integer("n_pw_iter", value, min_val=1)

    @property
    def random_seed(self):
        """Random seed to initialize with

        Returns
        -------
        int, numpy.random.Generator or None
        """
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value):
        try:
            np.random.default_rng(value)
        except TypeError as err:
            msg = (
                "Unable to initialize the random number generator with "
                f"a {type(value).__name__}"
            )
            raise TypeError(msg) from err
        self._random_seed = value

    seed = deprecate_property(
        random_seed,
        "seed",
        "random_seed",
        removal_version="0.24.0",
        error=True,
    )

    def initialize(self):
        """"""
        rng = np.random.default_rng(seed=self.random_seed)

        if self.verbose:
            print("Calculating the scaling parameter.")

        if (
            getattr(self.dmisfit, "objfcts", None) is None
            or len(self.dmisfit.objfcts) == 1
        ):
            raise TypeError(
                "ScalingMultipleDataMisfits_ByEig only applies to joint inversion"
            )

        ndm = len(self.dmisfit.objfcts)
        if self.chi0_ratio is not None:
            self.chi0_ratio = self.chi0_ratio * np.ones(ndm)
        else:
            self.chi0_ratio = self.dmisfit.multipliers

        m = self.invProb.model

        dm_eigenvalue_list = []
        for dm in self.dmisfit.objfcts:
            dm_eigenvalue_list += [
                eigenvalue_by_power_iteration(dm, m, random_seed=rng)
            ]

        self.chi0 = self.chi0_ratio / np.r_[dm_eigenvalue_list]
        self.chi0 = self.chi0 / np.sum(self.chi0)
        self.dmisfit.multipliers = self.chi0

        if self.verbose:
            print("Scale Multipliers: ", self.dmisfit.multipliers)


class JointScalingSchedule(InversionDirective):
    """
    For multiple data misfits only: rebalance each data misfit term
    during the inversion when some datasets are fit, and others not
    using the ratios of current misfits and their respective target.
    It implements the strategy described in https://doi.org/10.1093/gji/ggaa378.
    """

    def __init__(
        self, warmingFactor=1.0, chimax=1e10, chimin=1e-10, update_rate=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.mode = 1
        self.warmingFactor = warmingFactor
        self.chimax = chimax
        self.chimin = chimin
        self.update_rate = update_rate

    @property
    def mode(self):
        """The type of update to perform.

        Returns
        -------
        {1, 2}
        """
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = validate_integer("mode", value, min_val=1, max_val=2)

    @property
    def warmingFactor(self):
        """Factor to adjust scaling of the data misfits by.

        Returns
        -------
        float
        """
        return self._warmingFactor

    @warmingFactor.setter
    def warmingFactor(self, value):
        self._warmingFactor = validate_float(
            "warmingFactor", value, min_val=0.0, inclusive_min=False
        )

    @property
    def chimax(self):
        """Maximum chi factor.

        Returns
        -------
        float
        """
        return self._chimax

    @chimax.setter
    def chimax(self, value):
        self._chimax = validate_float("chimax", value, min_val=0.0, inclusive_min=False)

    @property
    def chimin(self):
        """Minimum chi factor.

        Returns
        -------
        float
        """
        return self._chimin

    @chimin.setter
    def chimin(self, value):
        self._chimin = validate_float("chimin", value, min_val=0.0, inclusive_min=False)

    @property
    def update_rate(self):
        """Will update the data misfit scalings after this many iterations.

        Returns
        -------
        int
        """
        return self._update_rate

    @update_rate.setter
    def update_rate(self, value):
        self._update_rate = validate_integer("update_rate", value, min_val=1)

    def initialize(self):
        if (
            getattr(self.dmisfit, "objfcts", None) is None
            or len(self.dmisfit.objfcts) == 1
        ):
            raise TypeError("JointScalingSchedule only applies to joint inversion")

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
            self.DMtarget = self.inversion.directiveList.dList[
                self.targetclass
            ].DMtarget

        if self.verbose:
            print("Initial data misfit scales: ", self.dmisfit.multipliers)

    def endIter(self):
        self.dmlist = self.inversion.directiveList.dList[self.targetclass].dmlist

        if np.any(self.dmlist < self.DMtarget):
            self.mode = 2
        else:
            self.mode = 1

        if self.opt.iter > 0 and self.opt.iter % self.update_rate == 0:
            if self.mode == 2:
                if np.all(np.r_[self.dmisfit.multipliers] > self.chimin) and np.all(
                    np.r_[self.dmisfit.multipliers] < self.chimax
                ):
                    indx = self.dmlist > self.DMtarget
                    if np.any(indx):
                        multipliers = self.warmingFactor * np.median(
                            self.DMtarget[~indx] / self.dmlist[~indx]
                        )
                        if np.sum(indx) == 1:
                            indx = np.where(indx)[0][0]
                        self.dmisfit.multipliers[indx] *= multipliers
                        self.dmisfit.multipliers /= np.sum(self.dmisfit.multipliers)

                        if self.verbose:
                            print("Updating scaling for data misfits by ", multipliers)
                            print("New scales:", self.dmisfit.multipliers)


class TargetMisfit(InversionDirective):
    """
    ... note:: Currently this target misfit is not set up for joint inversion.
    Check out MultiTargetMisfits
    """

    def __init__(self, target=None, phi_d_star=None, chifact=1.0, **kwargs):
        super().__init__(**kwargs)
        self.chifact = chifact
        self.phi_d_star = phi_d_star
        if phi_d_star is not None and target is not None:
            raise AttributeError("Attempted to set both target and phi_d_star.")
        if target is not None:
            self.target = target

    @property
    def target(self):
        """The target value for the data misfit

        Returns
        -------
        float
        """
        if getattr(self, "_target", None) is None:
            self._target = self.chifact * self.phi_d_star
        return self._target

    @target.setter
    def target(self, val):
        self._target = validate_float("target", val, min_val=0.0, inclusive_min=False)

    @property
    def chifact(self):
        """The a multiplier for the target data misfit value.

        The target value is `chifact` times `phi_d_star`

        Returns
        -------
        float
        """
        return self._chifact

    @chifact.setter
    def chifact(self, value):
        self._chifact = validate_float(
            "chifact", value, min_val=0.0, inclusive_min=False
        )
        self._target = None

    @property
    def phi_d_star(self):
        """The target phi_d value for the data misfit.

        The target value is `chifact` times `phi_d_star`

        Returns
        -------
        float
        """
        # phid = ||dpred - dobs||^2
        if self._phi_d_star is None:
            nD = 0
            for survey in self.survey:
                nD += survey.nD
            self._phi_d_star = nD
        return self._phi_d_star

    @phi_d_star.setter
    def phi_d_star(self, value):
        # phid = ||dpred - dobs||^2
        if value is not None:
            value = validate_float(
                "phi_d_star", value, min_val=0.0, inclusive_min=False
            )
        self._phi_d_star = value
        self._target = None

    def initialize(self):
        logger = get_logger()
        logger.info(
            f"Directive {self.__class__.__name__}: Target data misfit is {self.target}"
        )

    def endIter(self):
        if self.invProb.phi_d < self.target:
            self.opt.stopNextIteration = True
            self.print_final_misfit()

    def print_final_misfit(self):
        if self.opt.print_type == "ubc":
            self.opt.print_target = (
                ">> Target misfit: %.1f (# of data) is achieved"
            ) % (self.target)


class MultiTargetMisfits(InversionDirective):
    def __init__(
        self,
        WeightsInTarget=False,
        chifact=1.0,
        phi_d_star=None,
        TriggerSmall=True,
        chiSmall=1.0,
        phi_ms_star=None,
        TriggerTheta=False,
        ToleranceTheta=1.0,
        distance_norm=np.inf,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.WeightsInTarget = WeightsInTarget
        # Chi factor for Geophsyical Data Misfit
        self.chifact = chifact
        self.phi_d_star = phi_d_star

        # Chifact for Clustering/Smallness
        self.TriggerSmall = TriggerSmall
        self.chiSmall = chiSmall
        self.phi_ms_star = phi_ms_star

        # Tolerance for parameters difference with their priors
        self.TriggerTheta = TriggerTheta  # deactivated by default
        self.ToleranceTheta = ToleranceTheta
        self.distance_norm = distance_norm

        self._DM = False
        self._CL = False
        self._DP = False

    @property
    def WeightsInTarget(self):
        """Whether to account for weights in the petrophysical misfit.

        Returns
        -------
        bool
        """
        return self._WeightsInTarget

    @WeightsInTarget.setter
    def WeightsInTarget(self, value):
        self._WeightsInTarget = validate_type("WeightsInTarget", value, bool)

    @property
    def chifact(self):
        """The a multiplier for the target Geophysical data misfit value.

        The target value is `chifact` times `phi_d_star`

        Returns
        -------
        numpy.ndarray
        """
        return self._chifact

    @chifact.setter
    def chifact(self, value):
        self._chifact = validate_ndarray_with_shape("chifact", value, shape=("*",))
        self._DMtarget = None

    @property
    def phi_d_star(self):
        """The target phi_d value for the Geophysical data misfit.

        The target value is `chifact` times `phi_d_star`

        Returns
        -------
        float
        """
        # phid = || dpred - dobs||^2
        if getattr(self, "_phi_d_star", None) is None:
            # Check if it is a ComboObjective
            if isinstance(self.dmisfit, ComboObjectiveFunction):
                value = np.r_[[survey.nD for survey in self.survey]]
            else:
                value = np.r_[[self.survey.nD]]
            self._phi_d_star = value
            self._DMtarget = None

        return self._phi_d_star

    @phi_d_star.setter
    def phi_d_star(self, value):
        # phid =|| dpred - dobs||^2
        if value is not None:
            value = validate_ndarray_with_shape("phi_d_star", value, shape=("*",))
        self._phi_d_star = value
        self._DMtarget = None

    @property
    def chiSmall(self):
        """The a multiplier for the target petrophysical misfit value.

        The target value is `chiSmall` times `phi_ms_star`

        Returns
        -------
        float
        """
        return self._chiSmall

    @chiSmall.setter
    def chiSmall(self, value):
        self._chiSmall = validate_float("chiSmall", value)
        self._CLtarget = None

    @property
    def phi_ms_star(self):
        """The target value for the petrophysical data misfit.

        The target value is `chiSmall` times `phi_ms_star`

        Returns
        -------
        float
        """
        return self._phi_ms_star

    @phi_ms_star.setter
    def phi_ms_star(self, value):
        if value is not None:
            value = validate_float("phi_ms_star", value)
        self._phi_ms_star = value
        self._CLtarget = None

    @property
    def TriggerSmall(self):
        """Whether to trigger the smallness misfit test.

        Returns
        -------
        bool
        """
        return self._TriggerSmall

    @TriggerSmall.setter
    def TriggerSmall(self, value):
        self._TriggerSmall = validate_type("TriggerSmall", value, bool)

    @property
    def TriggerTheta(self):
        """Whether to trigger the GMM misfit test.

        Returns
        -------
        bool
        """
        return self._TriggerTheta

    @TriggerTheta.setter
    def TriggerTheta(self, value):
        self._TriggerTheta = validate_type("TriggerTheta", value, bool)

    @property
    def ToleranceTheta(self):
        """Target value for the GMM misfit.

        Returns
        -------
        float
        """
        return self._ToleranceTheta

    @ToleranceTheta.setter
    def ToleranceTheta(self, value):
        self._ToleranceTheta = validate_float("ToleranceTheta", value, min_val=0.0)

    @property
    def distance_norm(self):
        """Distance norm to use for GMM misfit measure.

        Returns
        -------
        float
        """
        return self._distance_norm

    @distance_norm.setter
    def distance_norm(self, value):
        self._distance_norm = validate_float("distance_norm", value, min_val=0.0)

    def initialize(self):
        self.dmlist = np.r_[[dmis(self.invProb.model) for dmis in self.dmisfit.objfcts]]

        if getattr(self.invProb.reg.objfcts[0], "objfcts", None) is not None:
            smallness = np.r_[
                [
                    (
                        np.r_[
                            i,
                            j,
                            isinstance(regpart, PGIsmallness),
                        ]
                    )
                    for i, regobjcts in enumerate(self.invProb.reg.objfcts)
                    for j, regpart in enumerate(regobjcts.objfcts)
                ]
            ]
            if smallness[smallness[:, 2] == 1][:, :2].size == 0:
                warnings.warn(
                    "There is no PGI regularization. Smallness target is turned off (TriggerSmall flag)",
                    stacklevel=2,
                )
                self.smallness = -1
                self.pgi_smallness = None

            else:
                self.smallness = smallness[smallness[:, 2] == 1][:, :2][0]
                self.pgi_smallness = self.invProb.reg.objfcts[
                    self.smallness[0]
                ].objfcts[self.smallness[1]]

                if self.verbose:
                    print(
                        type(
                            self.invProb.reg.objfcts[self.smallness[0]].objfcts[
                                self.smallness[1]
                            ]
                        )
                    )

            self._regmode = 1

        else:
            smallness = np.r_[
                [
                    (
                        np.r_[
                            j,
                            isinstance(regpart, PGIsmallness),
                        ]
                    )
                    for j, regpart in enumerate(self.invProb.reg.objfcts)
                ]
            ]
            if smallness[smallness[:, 1] == 1][:, :1].size == 0:
                if self.TriggerSmall:
                    warnings.warn(
                        "There is no PGI regularization. Smallness target is turned off (TriggerSmall flag).",
                        stacklevel=2,
                    )
                    self.TriggerSmall = False
                self.smallness = -1
            else:
                self.smallness = smallness[smallness[:, 1] == 1][:, :1][0]
                self.pgi_smallness = self.invProb.reg.objfcts[self.smallness[0]]

                if self.verbose:
                    print(type(self.invProb.reg.objfcts[self.smallness[0]]))

            self._regmode = 2

    @property
    def DM(self):
        """Whether the geophysical data misfit target was satisfied.

        Returns
        -------
        bool
        """
        return self._DM

    @property
    def CL(self):
        """Whether the petrophysical misfit target was satisified.

        Returns
        -------
        bool
        """
        return self._CL

    @property
    def DP(self):
        """Whether the GMM misfit was below the threshold.

        Returns
        -------
        bool
        """
        return self._DP

    @property
    def AllStop(self):
        """Whether all target misfit values have been met.

        Returns
        -------
        bool
        """

        return self.DM and self.CL and self.DP

    @property
    def DMtarget(self):
        if getattr(self, "_DMtarget", None) is None:
            self._DMtarget = self.chifact * self.phi_d_star
        return self._DMtarget

    @DMtarget.setter
    def DMtarget(self, val):
        self._DMtarget = val

    @property
    def CLtarget(self):
        if not getattr(self.pgi_smallness, "approx_eval", True):
            # if nonlinear prior, compute targer numerically at each GMM update
            samples, _ = self.pgi_smallness.gmm.sample(
                len(self.pgi_smallness.gmm.cell_volumes)
            )
            self.phi_ms_star = self.pgi_smallness(
                mkvc(samples), externalW=self.WeightsInTarget
            )

            self._CLtarget = self.chiSmall * self.phi_ms_star

        elif getattr(self, "_CLtarget", None) is None:
            # phid = ||dpred - dobs||^2
            if self.phi_ms_star is None:
                # Expected value is number of active cells * number of physical
                # properties
                self.phi_ms_star = len(self.invProb.model)

            self._CLtarget = self.chiSmall * self.phi_ms_star

        return self._CLtarget

    @property
    def CLnormalizedConstant(self):
        if ~self.WeightsInTarget:
            return 1.0
        elif np.any(self.smallness == -1):
            return np.sum(
                sp.csr_matrix.diagonal(self.invProb.reg.objfcts[0].W) ** 2.0
            ) / len(self.invProb.model)
        else:
            return np.sum(sp.csr_matrix.diagonal(self.pgi_smallness.W) ** 2.0) / len(
                self.invProb.model
            )

    @CLtarget.setter
    def CLtarget(self, val):
        self._CLtarget = val

    def phims(self):
        if np.any(self.smallness == -1):
            return self.invProb.reg.objfcts[0](self.invProb.model)
        else:
            return (
                self.pgi_smallness(
                    self.invProb.model, external_weights=self.WeightsInTarget
                )
                / self.CLnormalizedConstant
            )

    def ThetaTarget(self):
        maxdiff = 0.0

        for i in range(self.invProb.reg.gmm.n_components):
            meandiff = np.linalg.norm(
                (self.invProb.reg.gmm.means_[i] - self.invProb.reg.gmmref.means_[i])
                / self.invProb.reg.gmmref.means_[i],
                ord=self.distance_norm,
            )
            maxdiff = np.maximum(maxdiff, meandiff)

            if (
                self.invProb.reg.gmm.covariance_type == "full"
                or self.invProb.reg.gmm.covariance_type == "spherical"
            ):
                covdiff = np.linalg.norm(
                    (
                        self.invProb.reg.gmm.covariances_[i]
                        - self.invProb.reg.gmmref.covariances_[i]
                    )
                    / self.invProb.reg.gmmref.covariances_[i],
                    ord=self.distance_norm,
                )
            else:
                covdiff = np.linalg.norm(
                    (
                        self.invProb.reg.gmm.covariances_
                        - self.invProb.reg.gmmref.covariances_
                    )
                    / self.invProb.reg.gmmref.covariances_,
                    ord=self.distance_norm,
                )
            maxdiff = np.maximum(maxdiff, covdiff)

            pidiff = np.linalg.norm(
                [
                    (
                        self.invProb.reg.gmm.weights_[i]
                        - self.invProb.reg.gmmref.weights_[i]
                    )
                    / self.invProb.reg.gmmref.weights_[i]
                ],
                ord=self.distance_norm,
            )
            maxdiff = np.maximum(maxdiff, pidiff)

        return maxdiff

    def endIter(self):
        self._DM = False
        self._CL = True
        self._DP = True
        self.dmlist = np.r_[[dmis(self.invProb.model) for dmis in self.dmisfit.objfcts]]
        self.targetlist = np.r_[
            [dm < tgt for dm, tgt in zip(self.dmlist, self.DMtarget)]
        ]

        if np.all(self.targetlist):
            self._DM = True

        if self.TriggerSmall and np.any(self.smallness != -1):
            if self.phims() > self.CLtarget:
                self._CL = False

        if self.TriggerTheta:
            if self.ThetaTarget() > self.ToleranceTheta:
                self._DP = False

        if self.verbose:
            message = "geophys. misfits: " + "; ".join(
                map(
                    str,
                    [
                        "{0} (target {1} [{2}])".format(val, tgt, cond)
                        for val, tgt, cond in zip(
                            np.round(self.dmlist, 1),
                            np.round(self.DMtarget, 1),
                            self.targetlist,
                        )
                    ],
                )
            )
            if self.TriggerSmall:
                message += (
                    " | smallness misfit: {0:.1f} (target: {1:.1f} [{2}])".format(
                        self.phims(), self.CLtarget, self.CL
                    )
                )
            if self.TriggerTheta:
                message += " | GMM parameters within tolerance: {}".format(self.DP)
            print(message)

        if self.AllStop:
            self.opt.stopNextIteration = True
            if self.verbose:
                print("All targets have been reached")


class SaveEveryIteration(InversionDirective, metaclass=ABCMeta):
    """SaveEveryIteration

    This directive saves information at each iteration.

    Parameters
    ----------
    directory : pathlib.Path or str, optional
        The directory to store output information to, defaults to current directory.
    name : str, optional
        Root of the filename to be saved, commonly this will get iteration specific
        details appended to it.
    on_disk : bool, optional
        Whether this directive will save a log file to disk.
    """

    def __init__(self, directory=".", name="InversionModel", on_disk=True, **kwargs):
        self._on_disk = validate_type("on_disk", on_disk, bool)

        super().__init__(**kwargs)
        if self.on_disk:
            self.directory = directory
        else:
            self.directory = None
        self.name = name
        self._time_string_format = "%Y-%m-%d-%H-%M"
        self._iter_format = "03d"
        self._iter_string = "###"
        self._start_time = self._time_string_format

    def initialize(self):
        self._start_time = datetime.now().strftime(self._time_string_format)
        if opt := getattr(self, "opt", None):
            max_digit = len(str(opt.maxIter))
            self._iter_format = f"0{max_digit}d"

    @property
    def on_disk(self) -> bool:
        """Whether this object stores information to `file_abs_path`."""
        return self._on_disk

    @on_disk.setter
    def on_disk(self, value):
        self._on_disk = validate_type("on_disk", value, bool)

    @property
    def directory(self) -> pathlib.Path:
        """Directory to save results in.

        Returns
        -------
        pathlib.Path
        """
        if not self.on_disk:
            raise AttributeError(
                f"'{type(self).__qualname__}.directory' is only available if saving to disk."
            )
        return self._directory

    @directory.setter
    def directory(self, value):
        if value is None and self.on_disk:
            raise ValueError("Directory is not optional if 'on_disk==True'.")
        if value is not None:
            value = validate_type("directory", value, pathlib.Path).resolve()
        self._directory = value

    @property
    def name(self) -> str:
        """Root of the filename to be saved.

        Returns
        -------
        str
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = validate_string("name", value)

    @property
    def _time_iter_file_name(self) -> pathlib.Path:
        time_string = self._start_time
        if not getattr(self, "opt", None):
            iter_string = "###"
        else:
            itr = getattr(self.opt, "iter", 0)
            iter_string = f"{itr:{self._iter_format}}"

        return pathlib.Path(f"{self.name}_{time_string}_{iter_string}")

    @property
    def _time_file_name(self) -> pathlib.Path:
        return pathlib.Path(f"{self.name}_{self._start_time}")

    def _mkdir_and_check_output_file(self, should_exist=False):
        """
        Use this to ensure a directory exists, and to check if file_abs_path exists.
        Issues a warning if the output file exists but should not,
        or if it doesn't exist but does.

        Parameters
        ----------
        should_exist : bool, optional
            Whether file_abs_path should exist.
        """
        self.directory.mkdir(exist_ok=True)
        fp = self.file_abs_path
        exists = fp.exists()
        if exists and not should_exist:
            warnings.warn(f"Overwriting file {fp}", UserWarning, stacklevel=2)
        if not exists and should_exist:
            warnings.warn(
                f"File {fp} was not found, creating a new one.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def fileName(self):
        warnings.warn(
            "'fileName' has been deprecated and will be removed in SimPEG 0.26.0 use 'file_abs_path'",
            FutureWarning,
            stacklevel=2,
        )
        return self.file_abs_path.stem

    @property
    @abstractmethod
    def file_abs_path(self) -> pathlib.Path:
        """The absolute path to the saved output file.

        Returns
        -------
        pathlib.Path
        """


class SaveModelEveryIteration(SaveEveryIteration):
    """Saves the inversion model at the end of every iteration to a directory

    Parameters
    ----------
    directory : pathlib.Path or str, optional
        The directory to store output information to, defaults to current directory.
    name : str, optional
        Root of the filename to be saved, defaults to ``'InversionModel'``

    Notes
    -----

    This directive saves the model as a numpy array at each iteration. The
    default directory is the current directory and the models are saved as
    `name` + ``'_YYYY-MM-DD-HH-MM_iter.npy'``
    """

    def __init__(self, **kwargs):
        if "on_disk" in kwargs:
            msg = (
                f"The 'on_disk' argument is ignored by the '{type(self).__name__}' "
                "directive, it's always True."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            kwargs.pop("on_disk")
        super().__init__(on_disk=True, **kwargs)

    def initialize(self):
        super().initialize()
        print(
            f"{type(self).__qualname__} will save your models as: "
            f"'{self.file_abs_path}'"
        )

    @property
    def on_disk(self) -> bool:
        """This class always saves to disk.

        Returns
        -------
        bool
        """
        return True

    @on_disk.setter
    def on_disk(self, value):  # noqa: F811
        """This class always saves to disk."""
        msg = (
            f"Cannot modify value of 'on_disk' for {type(self).__name__}' directive. "
            "It's always True."
        )
        raise AttributeError(msg)

    @property
    def file_abs_path(self) -> pathlib.Path:
        return self.directory / self._time_iter_file_name.with_suffix(".npy")

    def endIter(self):
        self._mkdir_and_check_output_file(should_exist=False)
        np.save(self.file_abs_path, self.opt.xc)


class SaveOutputEveryIteration(SaveEveryIteration):
    """Keeps track of the objective function values.

    Parameters
    ----------
    on_disk : bool, optional
        Whether this directive additionally stores the log to a text file.
    directory : pathlib.Path, optional
        The directory to store output information to if `on_disk`, defaults to current directory.
    name : str, optional
        The root name of the file to save to, will append the inversion start time to this value.
    """

    def __init__(self, on_disk=True, **kwargs):
        if (save_txt := kwargs.pop("save_txt", None)) is not None:
            self.save_txt = save_txt
            on_disk = self.save_txt
        super().__init__(on_disk=on_disk, **kwargs)

    def initialize(self):
        super().initialize()
        if self.on_disk:
            fp = self.file_abs_path
            print(
                f"'{type(self).__qualname__}' will save your inversion "
                f"progress to: '{fp}'"
            )
            self._mkdir_and_check_output_file(should_exist=False)
            with open(fp, "w") as f:
                f.write(f"{self._header}\n")
        self._initialize_lists()

    @property
    def _header(self):
        return "  #     beta     phi_d     phi_m   phi_m_small     phi_m_smoomth_x     phi_m_smoomth_y     phi_m_smoomth_z      phi"

    def _initialize_lists(self):
        # Create a list of each
        self.beta = []
        self.phi_d = []
        self.phi_m = []
        self.phi_m_small = []
        self.phi_m_smooth_x = []
        self.phi_m_smooth_y = []
        self.phi_m_smooth_z = []
        self.phi = []

    @property
    def file_abs_path(self) -> pathlib.Path | None:
        """The absolute path to the saved log file."""
        if self.on_disk:
            return self.directory / self._time_file_name.with_suffix(".txt")

    save_txt = deprecate_property(
        SaveEveryIteration.on_disk,
        "save_txt",
        removal_version="0.26.0",
        future_warn=True,
    )

    def endIter(self):
        phi_s, phi_x, phi_y, phi_z = 0, 0, 0, 0

        for reg in self.reg.objfcts:
            if isinstance(reg, Sparse):
                i_s, i_x, i_y, i_z = 0, 1, 2, 3
            else:
                i_s, i_x, i_y, i_z = 0, 1, 3, 5
            if getattr(reg, "alpha_s", None):
                phi_s += reg.objfcts[i_s](self.invProb.model) * reg.alpha_s
            if getattr(reg, "alpha_x", None):
                phi_x += reg.objfcts[i_x](self.invProb.model) * reg.alpha_x

            if reg.regularization_mesh.dim > 1 and getattr(reg, "alpha_y", None):
                phi_y += reg.objfcts[i_y](self.invProb.model) * reg.alpha_y
            if reg.regularization_mesh.dim > 2 and getattr(reg, "alpha_z", None):
                phi_z += reg.objfcts[i_z](self.invProb.model) * reg.alpha_z

        self.beta.append(self.invProb.beta)
        self.phi_d.append(self.invProb.phi_d)
        self.phi_m.append(self.invProb.phi_m)
        self.phi_m_small.append(phi_s)
        self.phi_m_smooth_x.append(phi_x)
        self.phi_m_smooth_y.append(phi_y)
        self.phi_m_smooth_z.append(phi_z)
        self.phi.append(self.opt.f)

        if self.on_disk:
            self._mkdir_and_check_output_file(should_exist=True)
            with open(self.file_abs_path, "a") as f:
                f.write(
                    " {0:3d} {1:1.4e} {2:1.4e} {3:1.4e} {4:1.4e} {5:1.4e} "
                    "{6:1.4e}  {7:1.4e}  {8:1.4e}\n".format(
                        self.opt.iter,
                        self.beta[-1],
                        self.phi_d[-1],
                        self.phi_m[-1],
                        self.phi_m_small[-1],
                        self.phi_m_smooth_x[-1],
                        self.phi_m_smooth_y[-1],
                        self.phi_m_smooth_z[-1],
                        self.phi[-1],
                    )
                )

    def load_results(self, file_name=None):
        if file_name is None:
            if not self.on_disk:
                raise TypeError(
                    f"'file_name' is a required argument if '{type(self).__qualname__}.on_disk' is `False`"
                )
            file_name = self.file_abs_path
        results = np.loadtxt(file_name, comments="#")
        if results.shape[1] != 9:
            raise ValueError(f"{file_name} does not have valid results")

        self.beta = results[:, 1]
        self.phi_d = results[:, 2]
        self.phi_m = results[:, 3]
        self.phi_m_small = results[:, 4]
        self.phi_m_smooth_x = results[:, 5]
        self.phi_m_smooth_y = results[:, 6]
        self.phi_m_smooth_z = results[:, 7]
        self.f = results[:, 8]

        self.phi_m_smooth = (
            self.phi_m_smooth_x + self.phi_m_smooth_y + self.phi_m_smooth_z
        )

        self.target_misfit = self.invProb.dmisfit.simulation.survey.nD
        self.i_target = None

        if self.invProb.phi_d < self.target_misfit:
            i_target = 0
            while self.phi_d[i_target] > self.target_misfit:
                i_target += 1
            self.i_target = i_target

    def plot_misfit_curves(
        self,
        fname=None,
        dpi=300,
        plot_small_smooth=False,
        plot_phi_m=True,
        plot_small=False,
        plot_smooth=False,
    ):
        self.target_misfit = np.sum([dmis.nD for dmis in self.invProb.dmisfit.objfcts])
        self.i_target = None

        if self.invProb.phi_d < self.target_misfit:
            i_target = 0
            while self.phi_d[i_target] > self.target_misfit:
                i_target += 1
            self.i_target = i_target

        fig = plt.figure(figsize=(5, 2))
        ax = plt.subplot(111)
        ax_1 = ax.twinx()
        ax.semilogy(
            np.arange(len(self.phi_d)), self.phi_d, "k-", lw=2, label=r"$\phi_d$"
        )

        if plot_phi_m:
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m, "r", lw=2, label=r"$\phi_m$"
            )

        if plot_small_smooth or plot_small:
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m_small, "ro", label="small"
            )
        if plot_small_smooth or plot_smooth:
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m_smooth_x, "rx", label="smooth_x"
            )
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m_smooth_y, "rx", label="smooth_y"
            )
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m_smooth_z, "rx", label="smooth_z"
            )

        ax.legend(loc=1)
        ax_1.legend(loc=2)

        ax.plot(
            np.r_[ax.get_xlim()[0], ax.get_xlim()[1]],
            np.ones(2) * self.target_misfit,
            "k:",
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\phi_d$")
        ax_1.set_ylabel(r"$\phi_m$", color="r")
        ax_1.tick_params(axis="y", which="both", colors="red")

        plt.show()
        if fname is not None:
            fig.savefig(fname, dpi=dpi)

    def plot_tikhonov_curves(self, fname=None, dpi=200):
        self.target_misfit = self.invProb.dmisfit.simulation.survey.nD
        self.i_target = None

        if self.invProb.phi_d < self.target_misfit:
            i_target = 0
            while self.phi_d[i_target] > self.target_misfit:
                i_target += 1
            self.i_target = i_target

        fig = plt.figure(figsize=(5, 8))
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)

        ax1.plot(self.beta, self.phi_d, "k-", lw=2, ms=4)
        ax1.set_xlim(np.hstack(self.beta).min(), np.hstack(self.beta).max())
        ax1.set_xlabel(r"$\beta$", fontsize=14)
        ax1.set_ylabel(r"$\phi_d$", fontsize=14)

        ax2.plot(self.beta, self.phi_m, "k-", lw=2)
        ax2.set_xlim(np.hstack(self.beta).min(), np.hstack(self.beta).max())
        ax2.set_xlabel(r"$\beta$", fontsize=14)
        ax2.set_ylabel(r"$\phi_m$", fontsize=14)

        ax3.plot(self.phi_m, self.phi_d, "k-", lw=2)
        ax3.set_xlim(np.hstack(self.phi_m).min(), np.hstack(self.phi_m).max())
        ax3.set_xlabel(r"$\phi_m$", fontsize=14)
        ax3.set_ylabel(r"$\phi_d$", fontsize=14)

        if self.i_target is not None:
            ax1.plot(self.beta[self.i_target], self.phi_d[self.i_target], "k*", ms=10)
            ax2.plot(self.beta[self.i_target], self.phi_m[self.i_target], "k*", ms=10)
            ax3.plot(self.phi_m[self.i_target], self.phi_d[self.i_target], "k*", ms=10)

        for ax in [ax1, ax2, ax3]:
            ax.set_xscale("linear")
            ax.set_yscale("linear")
        plt.tight_layout()
        plt.show()
        if fname is not None:
            fig.savefig(fname, dpi=dpi)


class SaveOutputDictEveryIteration(SaveEveryIteration):
    """Saves inversion parameters to a dictionary at every iteration.

    At the end of every iteration, information about the current iteration is
    saved to the `outDict` property of this object.

    Parameters
    ----------
    on_disk : bool, optional
        Whether to also save the parameters to an `npz` file at the end of each iteration.
    directory : pathlib.Path or str, optional
        Directory to save inversion parameters to if `on_disk`, defaults to current directory.
    name : str, optional
        Root name of the output file. The inversion start time and the iteration are appended to this.
    """

    # Initialize the output dict
    def __init__(self, on_disk=False, **kwargs):
        if (save_on_disk := kwargs.pop("saveOnDisk", None)) is not None:
            self.saveOnDisk = save_on_disk
            on_disk = self.saveOnDisk
        super().__init__(on_disk=on_disk, **kwargs)

    saveOnDisk = deprecate_property(
        SaveEveryIteration.on_disk,
        "saveOnDisk",
        removal_version="0.26.0",
        future_warn=True,
    )

    @property
    def file_abs_path(self) -> pathlib.Path | None:
        if self.on_disk:
            return self.directory / self._time_iter_file_name.with_suffix(".npz")

    def initialize(self):
        super().initialize()
        self.outDict = {}
        if self.on_disk:
            print(
                f"'{type(self).__qualname__}' will save your inversion progress as a dictionary to: "
                f"'{self.file_abs_path}'"
            )

    def endIter(self):
        # regCombo = ["phi_ms", "phi_msx"]

        # if self.simulation[0].mesh.dim >= 2:
        #     regCombo += ["phi_msy"]

        # if self.simulation[0].mesh.dim == 3:
        #     regCombo += ["phi_msz"]

        # Initialize the output dict
        iterDict = {}

        # Save the data.
        iterDict["iter"] = self.opt.iter
        iterDict["beta"] = self.invProb.beta
        iterDict["phi_d"] = self.invProb.phi_d
        iterDict["phi_m"] = self.invProb.phi_m

        # for label, fcts in zip(regCombo, self.reg.objfcts[0].objfcts):
        #     iterDict[label] = fcts(self.invProb.model)

        iterDict["f"] = self.opt.f
        iterDict["m"] = self.invProb.model
        iterDict["dpred"] = self.invProb.dpred

        for reg in self.reg.objfcts:
            if isinstance(reg, Sparse):
                for reg_part, norm in zip(reg.objfcts, reg.norms):
                    reg_name = f"{type(reg_part).__name__}"
                    if hasattr(reg_part, "orientation"):
                        reg_name = reg_part.orientation + " " + reg_name
                    iterDict[reg_name + ".irls_threshold"] = reg_part.irls_threshold
                    iterDict[reg_name + ".norm"] = norm

        # Save the file as a npz
        if self.on_disk:
            self._mkdir_and_check_output_file(should_exist=False)
            np.savez(self.file_abs_path, iterDict)

        self.outDict[self.opt.iter] = iterDict


class UpdatePreconditioner(InversionDirective):
    """
    Create a Jacobi preconditioner for the linear problem
    """

    def __init__(self, update_every_iteration=True, **kwargs):
        super().__init__(**kwargs)
        self.update_every_iteration = update_every_iteration

    @property
    def update_every_iteration(self):
        """Whether to update the preconditioner at every iteration.

        Returns
        -------
        bool
        """
        return self._update_every_iteration

    @update_every_iteration.setter
    def update_every_iteration(self, value):
        self._update_every_iteration = validate_type(
            "update_every_iteration", value, bool
        )

    def initialize(self):
        # Create the pre-conditioner
        regDiag = np.zeros_like(self.invProb.model)
        m = self.invProb.model

        for reg in self.reg.objfcts:
            # Check if regularization has a projection
            rdg = reg.deriv2(m)
            if not isinstance(rdg, Zero):
                regDiag += rdg.diagonal()

        JtJdiag = np.zeros_like(self.invProb.model)
        for sim, dmisfit in zip(self.simulation, self.dmisfit.objfcts):
            if getattr(sim, "getJtJdiag", None) is None:
                assert getattr(sim, "getJ", None) is not None, (
                    "Simulation does not have a getJ attribute."
                    + "Cannot form the sensitivity explicitly"
                )
                JtJdiag += np.sum(np.power((dmisfit.W * sim.getJ(m)), 2), axis=0)
            else:
                JtJdiag += sim.getJtJdiag(m, W=dmisfit.W)

        diagA = JtJdiag + self.invProb.beta * regDiag
        diagA[diagA != 0] = diagA[diagA != 0] ** -1.0
        PC = sdiag((diagA))

        self.opt.approxHinv = PC

    def endIter(self):
        # Cool the threshold parameter
        if self.update_every_iteration is False:
            return

        # Create the pre-conditioner
        regDiag = np.zeros_like(self.invProb.model)
        m = self.invProb.model

        for reg in self.reg.objfcts:
            # Check if he has wire
            regDiag += reg.deriv2(m).diagonal()

        JtJdiag = np.zeros_like(self.invProb.model)
        for sim, dmisfit in zip(self.simulation, self.dmisfit.objfcts):
            if getattr(sim, "getJtJdiag", None) is None:
                assert getattr(sim, "getJ", None) is not None, (
                    "Simulation does not have a getJ attribute."
                    + "Cannot form the sensitivity explicitly"
                )
                JtJdiag += np.sum(np.power((dmisfit.W * sim.getJ(m)), 2), axis=0)
            else:
                JtJdiag += sim.getJtJdiag(m, W=dmisfit.W)

        diagA = JtJdiag + self.invProb.beta * regDiag
        diagA[diagA != 0] = diagA[diagA != 0] ** -1.0
        PC = sdiag((diagA))
        self.opt.approxHinv = PC


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

            JtJdiag = estimate_diagonal(JtJv, len(m), k=self.k)
            JtJdiag = JtJdiag / max(JtJdiag)

            self.reg.wght = JtJdiag


class UpdateSensitivityWeights(InversionDirective):
    r"""
    Sensitivity weighting for linear and non-linear least-squares inverse problems.

    This directive computes the root-mean squared sensitivities for the forward
    simulation(s) attached to the inverse problem, then truncates and scales the result
    to create cell weights which are applied in the regularization.

    .. important::

        This directive **requires** that the map for the regularization function is
        either :class:`simpeg.maps.Wires` or :class:`simpeg.maps.IdentityMap`. In other
        words, the sensitivity weighting cannot be applied for parametric inversion. In
        addition, the simulation(s) connected to the inverse problem **must** have
        a ``getJ`` or ``getJtJdiag`` method.

    .. important::

        This directive **must** be placed before any directives which update the
        preconditioner for the inverse problem (i.e. :class:`UpdatePreconditioner`), and
        **must** be before any directives that estimate the starting trade-off parameter
        (i.e. :class:`BetaEstimate_ByEig` and :class:`BetaEstimateMaxDerivative`).

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
            The smallest root-mean squared sensitivity is a fractional percent of the
            largest value; must be between 0 and 1.
        - global:
            The ``threshold_value`` is added to the cell weights prior to normalization;
            must be greater than 0.
        - percentile:
            The smallest root-mean squared sensitivity is set using percentile
            threshold; must be between 0 and 100.

    normalization_method : {'maximum', 'min_value', None}
        Normalization method applied to sensitivity weights.

        Options are:

        - maximum:
            Sensitivity weights are normalized by the largest value such that the
            largest weight is equal to 1.
        - minimum:
            Sensitivity weights are normalized by the smallest value, after
            thresholding, such that the smallest weights are equal to 1.
        - ``None``:
            Normalization is not applied.

    Notes
    -----
    Let :math:`\mathbf{J}` represent the Jacobian. To create sensitivity weights, root-mean squared (RMS) sensitivities
    :math:`\mathbf{s}` are computed by summing the squares of the rows of the Jacobian:

    .. math::
        \mathbf{s} = \Bigg [ \sum_i \, \mathbf{J_{i, \centerdot }}^2 \, \Bigg ]^{1/2}

    The dynamic range of RMS sensitivities can span many orders of magnitude. When computing sensitivity
    weights, thresholding is generally applied to set a minimum value.

    **Thresholding:**

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
        else:
            self._normalization_method = validate_string(
                "normalization_method", value, string_list=["minimum", "maximum"]
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


class ProjectSphericalBounds(InversionDirective):
    r"""
    Trick for spherical coordinate system.
    Project :math:`\theta` and :math:`\phi` angles back to :math:`[-\pi,\pi]`
    using back and forth conversion.
    spherical->cartesian->spherical
    """

    def initialize(self):
        x = self.invProb.model
        # Convert to cartesian than back to avoid over rotation
        nC = int(len(x) / 3)

        xyz = spherical2cartesian(x.reshape((nC, 3), order="F"))
        m = cartesian2spherical(xyz.reshape((nC, 3), order="F"))

        self.invProb.model = m

        for sim in self.simulation:
            sim.model = m

        self.opt.xc = m

    def endIter(self):
        x = self.invProb.model
        nC = int(len(x) / 3)

        # Convert to cartesian than back to avoid over rotation
        xyz = spherical2cartesian(x.reshape((nC, 3), order="F"))
        m = cartesian2spherical(xyz.reshape((nC, 3), order="F"))

        self.invProb.model = m

        phi_m_last = []
        for reg in self.reg.objfcts:
            reg.model = self.invProb.model
            phi_m_last += [reg(self.invProb.model)]

        self.invProb.phi_m_last = phi_m_last

        for sim in self.simulation:
            sim.model = m

        self.opt.xc = m
