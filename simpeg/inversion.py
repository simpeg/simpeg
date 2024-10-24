import numpy as np

from .optimization import IterationPrinters, StoppingCriteria
from .directives import DirectiveList
from .utils import timeIt, Counter, validate_type, validate_string


class BaseInversion(object):
    """Inversion Class"""

    def __init__(
        self,
        invProb,
        directiveList=None,
        counter=None,
        debug=False,
        name="BaseInversion",
        **kwargs,
    ):
        if directiveList is None:
            directiveList = []
        self.directiveList = directiveList
        self.counter = counter
        self.debug = debug
        self.name = name
        super().__init__(**kwargs)

        self.invProb = invProb

        self.opt = invProb.opt
        self.opt.callback = self._optCallback

        self.stoppers = [StoppingCriteria.iteration]

        # Check if we have inserted printers into the optimization
        if IterationPrinters.phi_d not in self.opt.printers:
            self.opt.printers.insert(1, IterationPrinters.beta)
            self.opt.printers.insert(2, IterationPrinters.phi_d)
            self.opt.printers.insert(3, IterationPrinters.phi_m)

    @property
    def name(self):
        """The name of the inversion.

        Returns
        -------
        str
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = validate_string("name", value)

    #: Print debugging information
    @property
    def debug(self):
        """Debugging flag.

        Returns
        -------
        bool
        """
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = validate_type("debug", value, bool)

    #: Set this to a simpeg.utils.Counter() if you want to count things
    @property
    def counter(self):
        """The counter.

        Returns
        -------
        None or simpeg.utils.Counter
        """
        return self._counter

    @counter.setter
    def counter(self, value):
        if value is not None:
            value = validate_type("counter", value, Counter, cast=False)
        self._counter = value

    @property
    def directiveList(self):
        if getattr(self, "_directiveList", None) is None:
            self._directiveList = DirectiveList(inversion=self)
            self._directiveList.validate()  # validate if we skip setter
        return self._directiveList

    @directiveList.setter
    def directiveList(self, value):
        if isinstance(value, list):
            value = DirectiveList(*value)
        assert isinstance(value, DirectiveList), "Must be a DirectiveList"
        value.validate()  # validate before setting
        self._directiveList = value
        self._directiveList.inversion = self

    @timeIt
    def run(self, m0):
        """run(m0)

        Runs the inversion!

        """
        self.invProb.startup(m0)
        self.directiveList.call("initialize")
        print("model has any nan: {:b}".format(np.any(np.isnan(self.invProb.model))))
        self.m = self.opt.minimize(self.invProb.evalFunction, self.invProb.model)
        self.directiveList.call("finish")

        return self.m

    def _optCallback(self, xt):
        self.directiveList.call("endIter")
