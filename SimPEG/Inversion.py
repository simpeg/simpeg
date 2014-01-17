import SimPEG
from SimPEG import Utils, sp, np
from Optimization import Remember, IterationPrinters, StoppingCriteria

class BaseInversion(object):
    """BaseInversion(prob, reg, opt, data, **kwargs)
    """

    __metaclass__ = Utils.Save.Savable

    maxIter = 1        #: Maximum number of iterations
    name = 'BaseInversion'

    debug   = False    #: Print debugging information

    comment = ''       #: Used by some functions to indicate what is going on in the algorithm
    counter = None     #: Set this to a SimPEG.Utils.Counter() if you want to count things

    def __init__(self, dataObj, opt, **kwargs):
        Utils.setKwargs(self, **kwargs)

        self.dataObj = dataObj
        self.dataObj.parent = self

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
            opt.bfgsH0 = SimPEG.Solver(reg.modelObj2Deriv())


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
        self.startup(m0)
        while True:
            self.doStartIteration()
            self.m = self.opt.minimize(self.evalFunction, self.m)
            self.doEndIteration()
            if self.stoppingCriteria(): break

        self.printDone()
        self.finish()

        return self.m

    @Utils.callHooks('startup')
    def startup(self, m0):
        """
            **startup** is called at the start of any new run call.

            :param numpy.ndarray x0: initial x
            :rtype: None
            :return: None
        """

        if not hasattr(self.reg, '_mref'):
            print 'Regularization has not set mref. SimPEG will set it to m0.'
            self.reg.mref = m0

        self.m = m0
        self._iter = 0
        self._beta = None
        self.phi_d_last = np.nan
        self.phi_m_last = np.nan

    @Utils.callHooks('doStartIteration')
    def doStartIteration(self):
        """
            **doStartIteration** is called at the end of each run iteration.

            :rtype: None
            :return: None
        """
        self._beta = self.getBeta()


    @Utils.callHooks('doEndIteration')
    def doEndIteration(self):
        """
            **doEndIteration** is called at the end of each run iteration.

            :rtype: None
            :return: None
        """
        # store old values
        self.phi_d_last = self.phi_d
        self.phi_m_last = self.phi_m
        self._iter += 1


    def stoppingCriteria(self):
        if self.debug: print 'checking stoppingCriteria'
        return Utils.checkStoppers(self, self.stoppers)


    def printDone(self):
        """
            **printDone** is called at the end of the inversion routine.

        """
        Utils.printStoppers(self, self.stoppers)

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
