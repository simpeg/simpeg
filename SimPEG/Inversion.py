import SimPEG
from SimPEG import Utils, sp, np
from Optimization import Remember, IterationPrinters, StoppingCriteria


class BaseInversion(object):
    """BaseInversion(objFunc, opt, **kwargs)
    """

    __metaclass__ = Utils.SimPEGMetaClass

    name = 'BaseInversion'

    debug   = False    #: Print debugging information

    counter = None     #: Set this to a SimPEG.Utils.Counter() if you want to count things

    def __init__(self, objFunc, opt, **kwargs):
        Utils.setKwargs(self, **kwargs)

        self.objFunc = objFunc
        self.objFunc.parent = self

        self.opt = opt
        self.opt.parent = self

        self.stoppers = [StoppingCriteria.iteration]

        # Check if we have inserted printers into the optimization
        if IterationPrinters.phi_d not in self.opt.printers:
            self.opt.printers.insert(1,IterationPrinters.beta)
            self.opt.printers.insert(2,IterationPrinters.phi_d)
            self.opt.printers.insert(3,IterationPrinters.phi_m)

    @Utils.timeIt
    def run(self, m0):
        """run(m0)

            Runs the inversion!

        """
        self.objFunc.startup(m0)
        self.m = self.opt.minimize(self.objFunc.evalFunction, m0)
        self.finish()

        return self.m

    @Utils.callHooks('finish')
    def finish(self):
        """finish()

            **finish** is called at the end of the optimization.
        """
        pass
