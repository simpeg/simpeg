from __future__ import print_function
import scipy.sparse as sp
import numpy as np
from SimPEG import Utils
from .Optimization import Remember, IterationPrinters, StoppingCriteria
from . import Directives


class BaseInversion(object):
    """Inversion Class"""

    name = 'BaseInversion'

    #: Print debugging information
    debug = False

    #: Set this to a SimPEG.Utils.Counter() if you want to count things
    counter = None

    @property
    def directiveList(self):
        if getattr(self, '_directiveList', None) is None:
            self._directiveList = Directives.DirectiveList(inversion=self)
            self._directiveList.validate()  # validate if we skip setter
        return self._directiveList

    @directiveList.setter
    def directiveList(self, value):
        if type(value) is list:
            value = Directives.DirectiveList(*value)
        assert isinstance(value, Directives.DirectiveList), (
            'Must be a DirectiveList'
        )
        value.validate()  # validate before setting
        self._directiveList = value
        self._directiveList.inversion = self

    def __init__(self, invProb, directiveList=None, **kwargs):
        if directiveList is None:
            directiveList = []
        self.directiveList = directiveList
        Utils.setKwargs(self, **kwargs)

        self.invProb = invProb

        self.opt = invProb.opt
        self.opt.callback = self._optCallback

        self.stoppers = [StoppingCriteria.iteration]

        # Check if we have inserted printers into the optimization
        if IterationPrinters.phi_d not in self.opt.printers:
            self.opt.printers.insert(1, IterationPrinters.beta)
            self.opt.printers.insert(2, IterationPrinters.phi_d)
            self.opt.printers.insert(3, IterationPrinters.phi_m)

    @Utils.timeIt
    def run(self, m0):
        """run(m0)

            Runs the inversion!

        """
        self.invProb.startup(m0)
        self.directiveList.call('initialize')
        print('model has any nan: {:b}'.format(np.any(np.isnan(self.invProb.model))))
        self.m = self.opt.minimize(self.invProb.evalFunction, self.invProb.model)
        self.directiveList.call('finish')

        return self.m

    def _optCallback(self, xt):
        self.directiveList.call('endIter')
