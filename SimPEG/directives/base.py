import warnings

from ..data_misfit import BaseDataMisfit
from ..objective_function import ComboObjectiveFunction

from ..regularization import (
    WeightedLeastSquares,
    BaseRegularization,
)
from ..utils import (
    set_kwargs,
)
from ..utils.code_utils import (
    deprecate_property,
    validate_type,
)


class InversionDirective:
    """Base inversion directive class.

    SimPEG directives initialize and update parameters used by the inversion algorithm;
    e.g. setting the initial beta or updating the regularization. ``InversionDirective``
    is a parent class responsible for connecting directives to the data misfit, regularization
    and optimization defining the inverse problem.

    Parameters
    ----------
    inversion : SimPEG.inversion.BaseInversion, None
        An SimPEG inversion object; i.e. an instance of :class:`SimPEG.inversion.BaseInversion`.
    dmisfit : SimPEG.data_misfit.BaseDataMisfit, None
        A data data misfit; i.e. an instance of :class:`SimPEG.data_misfit.BaseDataMisfit`.
    reg : SimPEG.regularization.BaseRegularization, None
        The regularization, or model objective function; i.e. an instance of :class:`SimPEG.regularization.BaseRegularization`.
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
        debug = kwargs.pop("debug", None)
        if debug is not None:
            self.debug = debug
        else:
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

    debug = deprecate_property(
        verbose, "debug", "verbose", removal_version="0.19.0", future_warn=True
    )

    @property
    def inversion(self):
        """Inversion object associated with the directive.

        Returns
        -------
        SimPEG.inversion.BaseInversion
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
        SimPEG.inverse_problem.BaseInvProblem
            The inverse problem associated with the directive.
        """
        return self.inversion.invProb

    @property
    def opt(self):
        """Optimization algorithm associated with the directive.

        Returns
        -------
        SimPEG.optimization.Minimize
            Optimization algorithm associated with the directive.
        """
        return self.invProb.opt

    @property
    def reg(self):
        """Regularization associated with the directive.

        Returns
        -------
        SimPEG.regularization.BaseRegularization
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
    def dmisfit(self):
        """Data misfit associated with the directive.

        Returns
        -------
        SimPEG.data_misfit.BaseDataMisfit
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
    def survey(self):
        """Return survey for all data misfits

        Assuming that ``dmisfit`` is always a ``ComboObjectiveFunction``,
        return a list containing the survey for each data misfit; i.e.
        [survey1, survey2, ...]

        Returns
        -------
        list of SimPEG.survey.Survey
            Survey for all data misfits.
        """
        return [objfcts.simulation.survey for objfcts in self.dmisfit.objfcts]

    @property
    def simulation(self):
        """Return simulation for all data misfits.

        Assuming that ``dmisfit`` is always a ``ComboObjectiveFunction``,
        return a list containing the simulation for each data misfit; i.e.
        [sim1, sim2, ...].

        Returns
        -------
        list of SimPEG.simulation.BaseSimulation
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
        directive_list : SimPEG.directives.DirectiveList
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
    directives : list of SimPEG.directives.InversionDirective
        List of directives.
    inversion : SimPEG.inversion.BaseInversion
        The inversion associated with the directives list.
    debug : bool
        Whether or not to print debugging information.

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
        SimPEG.inversion.BaseInversion
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
        [directive.validate(self) for directive in self.dList]
        return True
