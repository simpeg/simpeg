import numpy as np
import matplotlib.pyplot as plt
from SimPEG.utils import mkvc, sdiag, setKwargs, printTitles, printLine, printStoppers, checkStoppers
norm = np.linalg.norm
import scipy.sparse as sp
from SimPEG import Solver

try:
    from pubsub import pub
    doPub = True
except Exception, e:
    print 'Warning: you may not have the required pubsub installed, use pypubsub. You will not be able to listen to events.'
    doPub = False

class StoppingCriteria(object):
    """docstring for StoppingCriteria"""

    iteration   = { "str": "%d : maxIter   =     %3d    <= iter          =    %3d",
                    "left": lambda M: M.maxIter, "right": lambda M: M._iter,
                    "stopType": "critical"}

    iterationLS = { "str": "%d : maxIterLS =     %3d    <= iterLS          =    %3d",
                    "left": lambda M: M.maxIterLS, "right": lambda M: M._iterLS,
                    "stopType": "critical"}

    armijoGoldstein = { "str": "%d :    ft     = %1.4e <= alp*descent     = %1.4e",
                        "left": lambda M: M._LS_ft, "right": lambda M: M.f + M.LSreduction * M._LS_descent,
                        "stopType": "optimal"}

    tolerance_f = { "str": "%d : |fc-fOld| = %1.4e <= tolF*(1+|f0|) = %1.4e",
                    "left":     lambda M: 1 if M._iter==0 else abs(M.f-M.f_last), "right":    lambda M: 0 if M._iter==0 else M.tolF*(1+abs(M.f0)),
                    "stopType": "optimal"}

    moving_x = { "str": "%d : |xc-x_last| = %1.4e <= tolX*(1+|x0|) = %1.4e",
                 "left": lambda M: 1 if M._iter==0 else norm(M.xc-M.x_last), "right": lambda M: 0 if M._iter==0 else M.tolX*(1+norm(M.x0)),
                 "stopType": "optimal"}

    tolerance_g = { "str": "%d : |g|       = %1.4e <= tolG          = %1.4e",
                   "left": lambda M: norm(M.projection(M.g)), "right": lambda M: M.tolG,
                   "stopType": "optimal"}

    norm_g = { "str": "%d : |g|       = %1.4e <= 1e3*eps       = %1.4e",
               "left": lambda M: norm(M.g), "right": lambda M: 1e3*M.eps,
               "stopType": "critical"}

    bindingSet = { "str": "%d : probSize  =    %3d   <= bindingSet      =    %3d",
                   "left": lambda M: M.xc.size, "right": lambda M: np.sum(M.bindingSet(M.xc)),
                   "stopType": "critical"}

    bindingSet_LS = { "str": "%d : probSize  =    %3d   <= bindingSet      =    %3d",
                      "left": lambda M: M._LS_xt.size, "right": lambda M: np.sum(M.bindingSet(M._LS_xt)),
                      "stopType": "critical"}

    phi_d_target_Minimize = { "str": "%d : phi_d  = %1.4e <= phi_d_target  = %1.4e ",
                              "left": lambda M: M.parent.phi_d, "right": lambda M: M.parent.phi_d_target,
                              "stopType": "critical"}

    phi_d_target_Inversion = { "str": "%d : phi_d  = %1.4e <= phi_d_target  = %1.4e ",
                               "left": lambda I: I.phi_d, "right": lambda I: I.phi_d_target,
                               "stopType": "critical"}


class IterationPrinters(object):
    """docstring for IterationPrinters"""

    iteration = {"title": "#", "value": lambda M: M._iter, "width": 5, "format": "%3d"}
    f = {"title": "f", "value": lambda M: M.f, "width": 10, "format": "%1.2e"}
    norm_g = {"title": "|g|", "value": lambda M: norm(M.g), "width": 10, "format": "%1.2e"}
    totalLS = {"title": "LS", "value": lambda M: M._iterLS, "width": 5, "format": "%d"}

    iterationLS = {"title": "#", "value": lambda M: (M._iter, M._iterLS), "width": 5, "format": "%3d.%d"}
    LS_ft = {"title": "ft", "value": lambda M: M._LS_ft, "width": 10, "format": "%1.2e"}
    LS_t = {"title": "t", "value": lambda M: M._LS_t, "width": 10, "format": "%0.5f"}
    LS_armijoGoldstein = {"title": "f + alp*g.T*p", "value": lambda M: M.f + M.LSreduction*M._LS_descent, "width": 16, "format": "%1.2e"}

    itType = {"title": "itType", "value": lambda M: M._itType, "width": 8, "format": "%s"}
    aSet = {"title": "aSet", "value": lambda M: np.sum(M.activeSet(M.xc)), "width": 8, "format": "%d"}
    bSet = {"title": "bSet", "value": lambda M: np.sum(M.bindingSet(M.xc)), "width": 8, "format": "%d"}
    comment = {"title": "Comment", "value": lambda M: M.projComment, "width": 7, "format": "%s"}

    beta = {"title": "beta", "value": lambda M: M.parent._beta, "width": 10, "format":   "%1.2e"}
    phi_d = {"title": "phi_d", "value": lambda M: M.parent.phi_d, "width": 10, "format":   "%1.2e"}
    phi_m = {"title": "phi_m", "value": lambda M: M.parent.phi_m, "width": 10, "format":   "%1.2e"}


class Minimize(object):
    """

        Minimize is a general class for derivative based optimization.


    """

    name = "General Optimization Algorithm"

    maxIter = 20
    maxIterLS = 10
    maxStep = np.inf
    LSreduction = 1e-4
    LSshorten = 0.5
    tolF = 1e-1
    tolX = 1e-1
    tolG = 1e-1
    eps = 1e-5

    debug   = False
    debugLS = False

    def __init__(self, **kwargs):
        self._id = int(np.random.rand()*1e6)  # create a unique identifier to this program to be used in pubsub
        self.stoppers = [StoppingCriteria.tolerance_f, StoppingCriteria.moving_x, StoppingCriteria.tolerance_g, StoppingCriteria.norm_g, StoppingCriteria.iteration]
        self.stoppersLS = [StoppingCriteria.armijoGoldstein, StoppingCriteria.iterationLS]

        self.printers = [IterationPrinters.iteration, IterationPrinters.f, IterationPrinters.norm_g, IterationPrinters.totalLS]
        self.printersLS = [IterationPrinters.iterationLS, IterationPrinters.LS_ft, IterationPrinters.LS_t, IterationPrinters.LS_armijoGoldstein]

        setKwargs(self, **kwargs)

    def minimize(self, evalFunction, x0):
        """
        Minimizes the function (evalFunction) starting at the location x0.

        :param def evalFunction: function handle that evaluates: f, g, H = F(x)
        :param numpy.ndarray x0: starting location
        :rtype: numpy.ndarray
        :return: x, the last iterate of the optimization algorithm

        evalFunction is a function handle::

            (f[, g][, H]) = evalFunction(x, return_g=False, return_H=False )

            def evalFunction(x, return_g=False, return_H=False):
                out = (f,)
                if return_g:
                    out += (g,)
                if return_H:
                    out += (H,)
                return out if len(out) > 1 else out[0]


        Events are fired with the following inputs via pypubsub::

            Minimize.printInit              (minimize)
            Minimize.evalFunction           (minimize, f, g, H)
            Minimize.printIter              (minimize)
            Minimize.searchDirection        (minimize, p)
            Minimize.scaleSearchDirection   (minimize, p)
            Minimize.modifySearchDirection  (minimize, xt, passLS)
            Minimize.endIteration           (minimize, xt)
            Minimize.printDone              (minimize)

        To hook into one of these events (must have pypubsub installed)::

            from pubsub import pub
            def listener(minimize,p):
                print 'The search direction is:  ', p
            pub.subscribe(listener, 'Minimize.searchDirection')

        You can use pubsub communication to debug your code, it is not used internally.


        The algorithm for general minimization is as follows::

            startup(x0)
            printInit()

            while True:
                f, g, H = evalFunction(xc)
                printIter()
                if stoppingCriteria(): break
                p = findSearchDirection()
                p = scaleSearchDirection(p)
                xt, passLS = modifySearchDirection(p)
                if not passLS:
                    xt, caught = modifySearchDirectionBreak(p)
                    if not caught: return xc
                doEndIteration(xt)

            printDone()
            return xc
        """
        self.evalFunction = evalFunction
        self.startup(x0)
        self.printInit()

        while True:
            self.f, self.g, self.H = evalFunction(self.xc, return_g=True, return_H=True)
            if doPub: pub.sendMessage('Minimize.evalFunction', minimize=self, f=self.f, g=self.g, H=self.H)
            self.printIter()
            if self.stoppingCriteria(): break
            p = self.findSearchDirection()
            if doPub: pub.sendMessage('Minimize.searchDirection', minimize=self, p=p)
            p = self.scaleSearchDirection(p)
            if doPub: pub.sendMessage('Minimize.scaleSearchDirection', minimize=self, p=p)
            xt, passLS = self.modifySearchDirection(p)
            if doPub: pub.sendMessage('Minimize.modifySearchDirection', minimize=self, xt=xt, passLS=passLS)
            if not passLS:
                xt, caught = self.modifySearchDirectionBreak(p)
                if not caught: return self.xc
            self.doEndIteration(xt)
            if doPub: pub.sendMessage('Minimize.endIteration', minimize=self, xt=xt)

        self.printDone()

        return self.xc

    @property
    def parent(self):
        """
            This is the parent of the optimization routine.
        """
        return getattr(self, '_parent', None)
    @parent.setter
    def parent(self, value):
        self._parent = value

    def startup(self, x0):
        """
            **startup** is called at the start of any new minimize call.

            This will set::

                x0 = x0
                xc = x0
                _iter = _iterLS = 0

            If you have things that also need to run on startup, you can create a method::

                def _startup*(self, x0):
                    pass

            Where the * can be any string. If present, _startup* will be called at the start of the default startup call.
            You may also completely overwrite this function.

            :param numpy.ndarray x0: initial x
            :rtype: None
            :return: None
        """
        for method in [posible for posible in dir(self) if '_startup' in posible]:
            if self.debug: print 'startup is calling self.'+method
            getattr(self,method)(x0)

        self._iter = 0
        self._iterLS = 0

        x0 = self.projection(x0)  # ensure that we start of feasible.
        self.x0 = x0
        self.xc = x0
        self.f_last = np.nan
        self.x_last = x0


    def printInit(self, inLS=False):
        """
            **printInit** is called at the beginning of the optimization routine.

            If there is a parent object, printInit will check for a
            parent.printInit function and call that.

        """
        if doPub and not inLS: pub.sendMessage('Minimize.printInit', minimize=self)
        pad = ' '*10 if inLS else ''
        name = self.name if not inLS else self.nameLS
        printTitles(self, self.printers if not inLS else self.printersLS, name, pad)

    def printIter(self, inLS=False):
        """
            **printIter** is called directly after function evaluations.

            If there is a parent object, printIter will check for a
            parent.printIter function and call that.

        """
        if doPub and not inLS: pub.sendMessage('Minimize.printIter', minimize=self)
        pad = ' '*10 if inLS else ''
        printLine(self, self.printers if not inLS else self.printersLS, pad=pad)

    def printDone(self, inLS=False):
        """
            **printDone** is called at the end of the optimization routine.

            If there is a parent object, printDone will check for a
            parent.printDone function and call that.

        """
        if doPub and not inLS: pub.sendMessage('Minimize.printDone', minimize=self)
        pad = ' '*10 if inLS else ''
        stop, done = (' STOP! ', ' DONE! ') if not inLS else ('----------------', ' End Linesearch ')
        stoppers = self.stoppers if not inLS else self.stoppersLS
        printStoppers(self, stoppers, pad='', stop=stop, done=done)


    def stoppingCriteria(self, inLS=False):
        if self._iter == 0:
            self.f0 = self.f
            self.g0 = self.g
        return checkStoppers(self, self.stoppers if not inLS else self.stoppersLS)


    def projection(self, p):
        """
            projects the search direction.

            by default, no projection is applied.

            :param numpy.ndarray p: searchDirection
            :rtype: numpy.ndarray
            :return: p, projected search direction
        """
        return p

    def findSearchDirection(self):
        """
            **findSearchDirection** should return an approximation of:

            .. math::

                H p = - g

            Where you are solving for the search direction, p

            The default is:

            .. math::

                H = I

                p = - g

            And corresponds to SteepestDescent.

            The latest function evaluations are present in::

                self.f, self.g, self.H

            :rtype: numpy.ndarray
            :return: p, Search Direction
        """
        return -self.g

    def scaleSearchDirection(self, p):
        """
            **scaleSearchDirection** should scale the search direction if appropriate.

            Set the parameter **maxStep** in the minimize object, to scale back the gradient to a maximum size.

            :param numpy.ndarray p: searchDirection
            :rtype: numpy.ndarray
            :return: p, Scaled Search Direction
        """

        if self.maxStep < np.abs(p.max()):
            p = self.maxStep*p/np.abs(p.max())
        return p

    nameLS = "Armijo linesearch"

    def modifySearchDirection(self, p):
        """
            **modifySearchDirection** changes the search direction based on some sort of linesearch or trust-region criteria.

            By default, an Armijo backtracking linesearch is preformed with the following parameters:

                * maxIterLS, the maximum number of linesearch iterations
                * LSreduction, the expected reduction expected, default: 1e-4
                * LSshorten, how much the step is reduced, default: 0.5

            If the linesearch is completed, and a descent direction is found, passLS is returned as True.

            Else, a modifySearchDirectionBreak call is preformed.

            :param numpy.ndarray p: searchDirection
            :rtype: numpy.ndarray,bool
            :return: (xt, passLS)
        """
        # Projected Armijo linesearch
        self._LS_t = 1
        self._iterLS = 0
        while self._iterLS < self.maxIterLS:
            self._LS_xt      = self.projection(self.xc + self._LS_t*p)
            self._LS_ft      = self.evalFunction(self._LS_xt, return_g=False, return_H=False)
            self._LS_descent = np.inner(self.g, self._LS_xt - self.xc)  # this takes into account multiplying by t, but is important for projection.
            if self.stoppingCriteria(inLS=True): break
            self._iterLS += 1
            self._LS_t = self.LSshorten*self._LS_t
            if self.debugLS:
                if self._iterLS == 1: self.printInit(inLS=True)
                self.printIter(inLS=True)

        if self.debugLS and self._iterLS > 0: self.printDone(inLS=True)

        return self._LS_xt, self._iterLS < self.maxIterLS

    def modifySearchDirectionBreak(self, p):
        """
            Code is called if modifySearchDirection fails
            to find a descent direction.

            The search direction is passed as input and
            this function must pass back both a new searchDirection,
            and if the searchDirection break has been caught.

            By default, no additional work is done, and the
            evalFunction returns a False indicating the break was not caught.

            :param numpy.ndarray p: searchDirection
            :rtype: numpy.ndarray,bool
            :return: (xt, breakCaught)
        """
        self.printDone(inLS=True)
        print 'The linesearch got broken. Boo.'
        return p, False

    def doEndIteration(self, xt):
        """
            **doEndIteration** is called at the end of each minimize iteration.

            By default, function values and x locations are shuffled to store 1 past iteration in memory.

            self.xc must be updated in this code.


            If you have things that also need to run at the end of every iteration, you can create a method::

                def _doEndIteration*(self, xt):
                    pass

            Where the * can be any string. If present, _doEndIteration* will be called at the start of the default doEndIteration call.
            You may also completely overwrite this function.

            :param numpy.ndarray xt: tested new iterate that ensures a descent direction.
            :rtype: None
            :return: None
        """
        for method in [posible for posible in dir(self) if '_doEndIteration' in posible]:
            if self.debug: print 'doEndIteration is calling self.'+method
            getattr(self,method)(xt)


        # store old values
        self.f_last = self.f
        self.x_last, self.xc = self.xc, xt
        self._iter += 1


class Remember(object):
    """
        This mixin remembers all the things you tend to forget.

        You can remember parameters directly, naming the str in Minimize,
        or pass a tuple with the name and the function that takes Minimize.

        For Example::

            opt.remember('f',('norm_g', lambda M: np.linalg.norm(M.g)))

            opt.minimize(evalFunction, x0)

            opt.recall('f')

        The param name (str) can also be located in the parent (if no conflicts),
        and it will be looked up by default.
    """

    _rememberThese = []

    def remember(self, *args):
        self._rememberThese = args

    def recall(self, param):
        assert param in self._rememberList, "You didn't tell me to remember "+param+", you gotta tell me what to remember!"
        return self._rememberList[param]

    def _startupRemember(self, x0):
        self._rememberList = {}
        for param in self._rememberThese:
            if type(param) is str:
                self._rememberList[param] = []
            elif type(param) is tuple:
                self._rememberList[param[0]] = []

    def _doEndIterationRemember(self, *args):
        for param in self._rememberThese:
            if type(param) is str:
                if self.debug: print 'Remember is remembering: ' + param
                val = getattr(self, param, None)
                if val is None and getattr(self, 'parent', None) is not None:
                    # Look to the parent for the param if not found here.
                    val = getattr(self.parent, param, None)
                self._rememberList[param].append( val )
            elif type(param) is tuple:
                if self.debug: print 'Remember is remembering: ' + param[0]
                self._rememberList[param[0]].append( param[1](self) )



class ProjectedGradient(Minimize, Remember):
    name = 'Projected Gradient'

    maxIterCG = 10
    tolCG = 1e-3

    lower = -0.4
    upper = 0.9

    def __init__(self,**kwargs):
        super(ProjectedGradient, self).__init__(**kwargs)

        self.stoppers.append(StoppingCriteria.bindingSet)
        self.stoppersLS.append(StoppingCriteria.bindingSet_LS)

        self.printers.extend([ IterationPrinters.itType, IterationPrinters.aSet, IterationPrinters.bSet, IterationPrinters.comment ])


    def _startup(self, x0):
        # ensure bound vectors are the same size as the model
        if type(self.lower) is not np.ndarray:
            self.lower = np.ones_like(x0)*self.lower
        if type(self.upper) is not np.ndarray:
            self.upper = np.ones_like(x0)*self.upper

        self.explorePG = True
        self.exploreCG = False
        self.stopDoingPG = False

        self._itType = 'SD'
        self.projComment = ''

        self.aSet_prev = self.activeSet(x0)

    def projection(self, x):
        """Make sure we are feasible."""
        return np.median(np.c_[self.lower,x,self.upper],axis=1)

    def activeSet(self, x):
        """If we are on a bound"""
        return np.logical_or(x == self.lower, x == self.upper)

    def inactiveSet(self, x):
        """The free variables."""
        return np.logical_not(self.activeSet(x))

    def bindingSet(self, x):
        """
            If we are on a bound and the negative gradient points away from the feasible set.

            Optimality condition. (Satisfies Kuhn-Tucker) MoreToraldo91

        """
        bind_up  = np.logical_and(x == self.lower, self.g >= 0)
        bind_low = np.logical_and(x == self.upper, self.g <= 0)
        return np.logical_or(bind_up, bind_low)

    def findSearchDirection(self):
        self.aSet_prev = self.activeSet(self.xc)
        allBoundsAreActive = sum(self.aSet_prev) == self.xc.size

        if self.debug: print 'findSearchDirection: stopDoingPG: ', self.stopDoingPG
        if self.debug: print 'findSearchDirection: explorePG: ', self.explorePG
        if self.debug: print 'findSearchDirection: exploreCG: ', self.exploreCG
        if self.debug: print 'findSearchDirection: aSet', np.sum(self.activeSet(self.xc))
        if self.debug: print 'findSearchDirection: bSet', np.sum(self.bindingSet(self.xc))
        if self.debug: print 'findSearchDirection: allBoundsAreActive: ', allBoundsAreActive

        if self.explorePG or not self.exploreCG or allBoundsAreActive:
            if self.debug: print 'findSearchDirection.PG: doingPG'
            self._itType = 'SD'
            p = -self.g
        else:
            if self.debug: print 'findSearchDirection.CG: doingCG'
            self.f_decrease_max = -np.inf # Reset the max decrease each time you do a CG iteration
            self._itType = '.CG.'

            iSet  = self.inactiveSet(self.xc)  # The inactive set (free variables)
            bSet = self.bindingSet(self.xc)
            shape = (self.xc.size, np.sum(iSet))
            v = np.ones(shape[1])
            i = np.where(iSet)[0]
            j = np.arange(shape[1])
            if self.debug: print 'findSearchDirection.CG: Z.shape', shape
            Z = sp.csr_matrix((v, (i, j)), shape=shape)

            def reduceHess(v):
                # Z is tall and skinny
                return Z.T*(self.H*(Z*v))
            operator = sp.linalg.LinearOperator( (shape[1], shape[1]), reduceHess, dtype=float )
            p, info  = sp.linalg.cg(operator, -Z.T*self.g, tol=self.tolCG, maxiter=self.maxIterCG)
            p = Z*p  #  bring up to full size
            # aSet_after = self.activeSet(self.xc+p)
        return p

    def _doEndIteration_ProjectedGradient(self, xt):
        aSet = self.activeSet(xt)
        bSet = self.bindingSet(xt)

        self.explorePG = not np.all(aSet == self.aSet_prev) # explore proximal gradient
        self.exploreCG = np.all(aSet == bSet) # explore conjugate gradient

        f_current_decrease = self.f_last - self.f
        self.projComment = ''
#         print f_current_decrease
        if self._iter < 1:
            self.f_decrease_max = -np.inf
        else:
            # Note that I reset this if we do a CG iteration.
            self.f_decrease_max = max(self.f_decrease_max, f_current_decrease)
            self.stopDoingPG = f_current_decrease < 0.25 * self.f_decrease_max
#             print 'f_decrease_max: ', self.f_decrease_max
#             print 'stopDoingSD: ', self.stopDoingSD
            if self.stopDoingPG:
                self.projComment = 'Stop SD'
                self.explorePG = False
                self.exploreCG = True
        # implement 3.8, MoreToraldo91
        #self.eta_2 * max_decrease where max decrease
        # if true go to CG
        # don't do too many steps of PG in a row.
        if self.debug: self.printDone()

class GaussNewton(Minimize, Remember):
    name = 'Gauss Newton'
    def findSearchDirection(self):
        return Solver(self.H).solve(-self.g)


class InexactGaussNewton(Minimize, Remember):
    name = 'Inexact Gauss Newton'

    maxIterCG = 10
    tolCG = 1e-5

    def findSearchDirection(self):
        # TODO: use BFGS as a preconditioner or gauss sidel of the WtW or solve WtW directly
        p, info = sp.linalg.cg(self.H, -self.g, tol=self.tolCG, maxiter=self.maxIterCG)
        return p


class SteepestDescent(Minimize, Remember):
    name = 'Steepest Descent'
    def findSearchDirection(self):
        return -self.g

if __name__ == '__main__':
    from SimPEG.tests import Rosenbrock, checkDerivative
    import matplotlib.pyplot as plt
    x0 = np.array([2.6, 3.7])
    checkDerivative(Rosenbrock, x0, plotIt=False)

    # def listener1(minimize,p):
    #     print 'hi:  ', p
    # if doPub: pub.subscribe(listener1, 'Minimize.searchDirection')

    xOpt = GaussNewton(maxIter=20,tolF=1e-10,tolX=1e-10,tolG=1e-10).minimize(Rosenbrock,x0)
    print "xOpt=[%f, %f]" % (xOpt[0], xOpt[1])
    xOpt = SteepestDescent(maxIter=30, maxIterLS=15,tolF=1e-10,tolX=1e-10,tolG=1e-10).minimize(Rosenbrock, x0)
    print "xOpt=[%f, %f]" % (xOpt[0], xOpt[1])
