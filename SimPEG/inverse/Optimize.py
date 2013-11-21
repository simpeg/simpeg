import numpy as np
import matplotlib.pyplot as plt
from SimPEG.utils import mkvc, sdiag, setKwargs, printTitles, printLine, printStoppers, checkStoppers, count, timeIt
norm = np.linalg.norm
import scipy.sparse as sp
from SimPEG import Solver


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

    tolerance_g = { "str": "%d : |proj(x-g)-x|    = %1.4e <= tolG          = %1.4e",
                   "left": lambda M: norm(M.projection(M.xc - M.g) - M.xc), "right": lambda M: M.tolG,
                   "stopType": "optimal"}

    norm_g = { "str": "%d : |proj(x-g)-x|    = %1.4e <= 1e3*eps       = %1.4e",
               "left": lambda M: norm(M.projection(M.xc - M.g) - M.xc), "right": lambda M: 1e3*M.eps,
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
    norm_g = {"title": "|proj(x-g)-x|", "value": lambda M: norm(M.projection(M.xc - M.g) - M.xc), "width": 15, "format": "%1.2e"}
    totalLS = {"title": "LS", "value": lambda M: M._iterLS, "width": 5, "format": "%d"}

    iterationLS = {"title": "#", "value": lambda M: (M._iter, M._iterLS), "width": 5, "format": "%3d.%d"}
    LS_ft = {"title": "ft", "value": lambda M: M._LS_ft, "width": 10, "format": "%1.2e"}
    LS_t = {"title": "t", "value": lambda M: M._LS_t, "width": 10, "format": "%0.5f"}
    LS_armijoGoldstein = {"title": "f + alp*g.T*p", "value": lambda M: M.f + M.LSreduction*M._LS_descent, "width": 16, "format": "%1.2e"}

    itType = {"title": "itType", "value": lambda M: M._itType, "width": 8, "format": "%s"}
    aSet = {"title": "aSet", "value": lambda M: np.sum(M.activeSet(M.xc)), "width": 8, "format": "%d"}
    bSet = {"title": "bSet", "value": lambda M: np.sum(M.bindingSet(M.xc)), "width": 8, "format": "%d"}
    comment = {"title": "Comment", "value": lambda M: M.comment, "width": 12, "format": "%s"}

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

    comment = ''
    counter = None

    def __init__(self, **kwargs):
        self._id = int(np.random.rand()*1e6)  # create a unique identifier to this program to be used in pubsub
        self.stoppers = [StoppingCriteria.tolerance_f, StoppingCriteria.moving_x, StoppingCriteria.tolerance_g, StoppingCriteria.norm_g, StoppingCriteria.iteration]
        self.stoppersLS = [StoppingCriteria.armijoGoldstein, StoppingCriteria.iterationLS]

        self.printers = [IterationPrinters.iteration, IterationPrinters.f, IterationPrinters.norm_g, IterationPrinters.totalLS]
        self.printersLS = [IterationPrinters.iterationLS, IterationPrinters.LS_ft, IterationPrinters.LS_t, IterationPrinters.LS_armijoGoldstein]

        setKwargs(self, **kwargs)

    @timeIt
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
            self.printIter()
            if self.stoppingCriteria(): break
            p = self.findSearchDirection()
            p = self.scaleSearchDirection(p)
            xt, passLS = self.modifySearchDirection(p)
            if not passLS:
                xt, caught = self.modifySearchDirectionBreak(p)
                if not caught: return self.xc
            self.doEndIteration(xt)

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

    @count
    def printIter(self, inLS=False):
        """
            **printIter** is called directly after function evaluations.

            If there is a parent object, printIter will check for a
            parent.printIter function and call that.

        """

        for method in [posible for posible in dir(self) if '_printIter' in posible]:
            if self.debug: print 'printIter is calling self.'+method
            getattr(self,method)(inLS)

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

    @timeIt
    def stoppingCriteria(self, inLS=False):
        if self._iter == 0:
            self.f0 = self.f
            self.g0 = self.g
        return checkStoppers(self, self.stoppers if not inLS else self.stoppersLS)

    @timeIt
    def projection(self, p):
        """
            projects the search direction.

            by default, no projection is applied.

            :param numpy.ndarray p: searchDirection
            :rtype: numpy.ndarray
            :return: p, projected search direction
        """
        return p

    @timeIt
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

    @count
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

    @timeIt
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

    @count
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

    @count
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
        if self.debug: self.printDone()



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

    lower = -np.inf
    upper = np.inf

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
        self.comment = ''

        self.aSet_prev = self.activeSet(x0)

    @count
    def projection(self, x):
        """Make sure we are feasible."""
        return np.median(np.c_[self.lower,x,self.upper],axis=1)

    @count
    def activeSet(self, x):
        """If we are on a bound"""
        return np.logical_or(x == self.lower, x == self.upper)

    @count
    def inactiveSet(self, x):
        """The free variables."""
        return np.logical_not(self.activeSet(x))

    @count
    def bindingSet(self, x):
        """
            If we are on a bound and the negative gradient points away from the feasible set.

            Optimality condition. (Satisfies Kuhn-Tucker) MoreToraldo91

        """
        bind_up  = np.logical_and(x == self.lower, self.g >= 0)
        bind_low = np.logical_and(x == self.upper, self.g <= 0)
        return np.logical_or(bind_up, bind_low)

    @timeIt
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
            # Reset the max decrease each time you do a CG iteration
            self.f_decrease_max = -np.inf

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
            operator = sp.linalg.LinearOperator( (shape[1], shape[1]), reduceHess, dtype=self.xc.dtype )
            p, info  = sp.linalg.cg(operator, -Z.T*self.g, tol=self.tolCG, maxiter=self.maxIterCG)
            p = Z*p  #  bring up to full size
            # aSet_after = self.activeSet(self.xc+p)
        return p

    @timeIt
    def _doEndIteration_ProjectedGradient(self, xt):
        aSet = self.activeSet(xt)
        bSet = self.bindingSet(xt)

        self.explorePG = not np.all(aSet == self.aSet_prev) # explore proximal gradient
        self.exploreCG = np.all(aSet == bSet) # explore conjugate gradient

        f_current_decrease = self.f_last - self.f
        self.comment = ''
        if self._iter < 1:
            # Note that this is reset on every CG iteration.
            self.f_decrease_max = -np.inf
        else:
            self.f_decrease_max = max(self.f_decrease_max, f_current_decrease)
            self.stopDoingPG = f_current_decrease < 0.25 * self.f_decrease_max
            if self.stopDoingPG:
                self.comment = 'Stop SD'
                self.explorePG = False
                self.exploreCG = True
        # implement 3.8, MoreToraldo91
        #self.eta_2 * max_decrease where max decrease
        # if true go to CG
        # don't do too many steps of PG in a row.

        if self.debug: print 'doEndIteration.ProjGrad, f_current_decrease: ', f_current_decrease
        if self.debug: print 'doEndIteration.ProjGrad, f_decrease_max: ', self.f_decrease_max
        if self.debug: print 'doEndIteration.ProjGrad, stopDoingSD: ', self.stopDoingSD



class BFGS(Minimize, Remember):
    name  = 'BFGS'
    nbfgs = 10

    @property
    def bfgsH0(self):
        """
            Approximate Hessian used in preconditioning the problem.

            Must be a SimPEG.Solver
        """
        _bfgsH0 = getattr(self,'_bfgsH0',None)
        if _bfgsH0 is None:
            return Solver(sp.identity(self.xc.size).tocsc(), flag='D')
        return _bfgsH0
    @bfgsH0.setter
    def bfgsH0(self, value):
        assert type(value) is Solver, 'bfgsH0 must be a SimPEG.Solver'
        self._bfgsH0 = value

    def _startup_BFGS(self,x0):
        self._bfgscnt = -1
        self._bfgsY   = np.zeros((x0.size, self.nbfgs))
        self._bfgsS   = np.zeros((x0.size, self.nbfgs))
        if not np.any([p is IterationPrinters.comment for p in self.printers]):
            self.printers.append(IterationPrinters.comment)

    def bfgs(self, d):
        n  = self._bfgscnt
        nn = ktop = min(self._bfgsS.shape[1],n)
        return self.bfgsrec(ktop,n,nn,self._bfgsS,self._bfgsY,d)

    def bfgsrec(self,k,n,nn,S,Y,d):
        """BFGS recursion"""
        if k < 0:
            d = self.bfgsH0.solve(d)
        else:
            khat    = 0 if nn is 0 else np.mod(n-nn+k,nn)
            gamma   = np.vdot(S[:,khat],d)/np.vdot(Y[:,khat],S[:,khat])
            d       = d - gamma*Y[:,khat]
            d       = self.bfgsrec(k-1,n,nn,S,Y,d)
            d       = d + (gamma - np.vdot(Y[:,khat],d)/np.vdot(Y[:,khat],S[:,khat]))*S[:,khat]
        return d

    def findSearchDirection(self):
        return self.bfgs(-self.g)

    def _doEndIteration_BFGS(self, xt):
        if self._iter is 0:
            self.g_last = self.g
            return

        yy = self.g - self.g_last;
        ss = self.xc - xt;
        self.g_last = self.g

        if yy.dot(ss) > 0:
            self._bfgscnt += 1
            ktop = np.mod(self._bfgscnt,self.nbfgs)
            self._bfgsY[:,ktop] = yy
            self._bfgsS[:,ktop] = ss
            self.comment = ''
        else:
            self.comment = 'Skip BFGS'


class GaussNewton(Minimize, Remember):
    name = 'Gauss Newton'

    @timeIt
    def findSearchDirection(self):
        return Solver(self.H).solve(-self.g)


class InexactGaussNewton(BFGS, Minimize, Remember):
    """
        Minimizes using CG as the inexact solver of

        .. math::

            \mathbf{H p = -g}

        By default BFGS is used as the preconditioner.

        Use *nbfgs* to set the memory limitation of BFGS.

        To set the initial H0 to be used in BFGS, set *bfgsH0* to be a SimPEG.Solver

    """

    def __init__(self, **kwargs):
        Minimize.__init__(self, **kwargs)

    name = 'Inexact Gauss Newton'

    maxIterCG = 10
    tolCG = 1e-3

    @property
    def approxHinv(self):
        """
            The approximate Hessian inverse is used to precondition CG.

            Default uses BFGS, with an initial H0 of *bfgsH0*.

            Must be a scipy.sparse.linalg.LinearOperator
        """
        _approxHinv = getattr(self,'_approxHinv',None)
        if _approxHinv is None:
            M = sp.linalg.LinearOperator( (self.xc.size, self.xc.size), self.bfgs, dtype=self.xc.dtype )
            return M
        return _approxHinv
    @approxHinv.setter
    def approxHinv(self, value):
        self._approxHinv = value

    @timeIt
    def findSearchDirection(self):
        Hinv = Solver(self.H, doDirect=False, options={'iterSolver': 'CG', 'M': self.approxHinv, 'tol': self.tolCG, 'maxIter': self.maxIterCG})
        p = Hinv.solve(-self.g)
        return p


class SteepestDescent(Minimize, Remember):
    name = 'Steepest Descent'

    @timeIt
    def findSearchDirection(self):
        return -self.g


class NewtonRoot(object):
    """
        Newton Method - Root Finding

        root = newtonRoot(fun,x);

        Where fun is the function that returns the function value as well as the
        gradient.

        For iterative solving of dh = -J\\r, use O.solveTol = TOL. For direct
        solves, use SOLVETOL = 0 (default)

        Rowan Cockett
        16-May-2013 16:29:51
        University of British Columbia
        rcockett@eos.ubc.ca

    """

    tol      = 1.000e-06
    solveTol = 0 # Default direct solve.
    maxIter  = 20
    stepDcr  = 0.5
    maxLS    = 30
    comments = False
    doLS     = True

    def __init__(self, **kwargs):
        setKwargs(self, **kwargs)

    def root(self, fun, x):
        if self.comments: print 'Newton Method:\n'

        self._iter = 0
        while True:

            [r,J] = fun(x);
            if self.solveTol == 0:
                Jinv = Solver(J)
                dh   = - Jinv.solve(r)
            else:
                raise NotImplementedError('Iterative solve on NewtonRoot is not yet implemented.')
                # M = @(x) tril(J)\(diag(J).*(triu(J)\x));
                # [dh, ~] = bicgstab(J,-r,O.solveTol,500,M);

            muLS = 1.
            LScnt  = 1
            xt = x + dh
            rt, Jt = fun(xt) # TODO: get rid of Jt

            if self.comments: print '\tLinesearch:\n'
            # Enter Linesearch
            while True and self.doLS:
                if self.comments:
                    print '\t\tResid: %e\n'%norm(rt)
                if norm(rt) <= norm(r) or norm(rt) < self.tol:
                    break

                muLS = muLS*self.stepDcr
                LScnt = LScnt + 1
                print '.'
                if LScnt > self.maxLS:
                    print 'Newton Method: Line search break.'
                    root = NaN
                    return
                xt = x + muLS*dh
                rt, Jt = fun(xt) # TODO: get rid of Jt

            x = xt
            self._iter += 1
            if norm(rt) < self.tol or self._iter > self.maxIter:
                break

        return x



if __name__ == '__main__':
    from SimPEG.tests import Rosenbrock, checkDerivative
    import matplotlib.pyplot as plt
    x0 = np.array([2.6, 3.7])
    checkDerivative(Rosenbrock, x0, plotIt=False)

    xOpt = GaussNewton(maxIter=20,tolF=1e-10,tolX=1e-10,tolG=1e-10).minimize(Rosenbrock,x0)
    print "xOpt=[%f, %f]" % (xOpt[0], xOpt[1])
    xOpt = SteepestDescent(maxIter=30, maxIterLS=15,tolF=1e-10,tolX=1e-10,tolG=1e-10).minimize(Rosenbrock, x0)
    print "xOpt=[%f, %f]" % (xOpt[0], xOpt[1])


    print 'test the newtonRoot finding.'
    fun = lambda x: (np.sin(x), sdiag(np.cos(x)))
    x = np.array([np.pi-0.3, np.pi+0.1, 0])
    pnt = NewtonRoot(comments=False).root(fun,x)
    print pnt
