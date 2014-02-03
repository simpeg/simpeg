import SimPEG
from SimPEG import Utils, sp, np
from Optimization import Remember, IterationPrinters


class BaseInversion(object):
    """BaseInversion(objFunc, opt, **kwargs)
    """

    __metaclass__ = Utils.Save.Savable

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

        if not hasattr(opt, '_bfgsH0') and hasattr(opt, 'bfgsH0'): # Check if it has been set by the user and the default is not being used.
            #TODO: I don't think that this if statement is working...
            print 'Setting bfgsH0 to the inverse of the modelObj2Deriv. Done using direct methods.'
            opt.bfgsH0 = SimPEG.Solver(objFunc.reg.modelObj2Deriv())


    #TODO: Move this to the data class?
    @property
    def phi_d_target(self):
        """
        target for phi_d

        By default this is the number of data.

        Note that we do not set the target if it is None, but we return the default value.
        """
        if getattr(self, '_phi_d_target', None) is None:
            return self.data.dobs.size #
        return self._phi_d_target

    @phi_d_target.setter
    def phi_d_target(self, value):
        self._phi_d_target = value

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


    def save(self, group):
        group.attrs['phi_d'] = self.phi_d
        group.attrs['phi_m'] = self.phi_m
        group.setArray('m', self.m)
        group.setArray('dpred', self.dpred)



# class Inversion(Cooling, Remember, BaseInversion):

#     maxIter = 10
#     name = "SimPEG Inversion"

#     def __init__(self, prob, reg, opt, data, **kwargs):
#         BaseInversion.__init__(self, prob, reg, opt, data, **kwargs)

#         self.stoppers.append(StoppingCriteria.phi_d_target_Inversion)

#         if StoppingCriteria.phi_d_target_Minimize not in self.opt.stoppers:
#             self.opt.stoppers.append(StoppingCriteria.phi_d_target_Minimize)

# class TimeSteppingInversion(Remember, BaseInversion):
#     """
#         A slightly different view on regularization parameters,
#         let Beta be viewed as 1/dt, and timestep by updating the
#         reference model every optimization iteration.
#     """
#     maxIter = 1
#     name = "Time-Stepping SimPEG Inversion"

#     def __init__(self, prob, reg, opt, data, **kwargs):
#         BaseInversion.__init__(self, prob, reg, opt, data, **kwargs)

#         self.stoppers.append(StoppingCriteria.phi_d_target_Inversion)

#         if StoppingCriteria.phi_d_target_Minimize not in self.opt.stoppers:
#             self.opt.stoppers.append(StoppingCriteria.phi_d_target_Minimize)

#     def _startup_TimeSteppingInversion(self, m0):

#         def _doEndIteration_updateMref(self, xt):
#             if self.debug: 'Updating the reference model.'
#             self.parent.reg.mref = self.xc

#         self.opt.hook(_doEndIteration_updateMref, overwrite=True)
