from __future__ import print_function
from . import Utils
import numpy as np
import scipy.sparse as sp
from six import string_types
from .Utils.SolverUtils import *
norm = np.linalg.norm


__all__ = [
    'Minimize', 'Remember', 'SteepestDescent', 'BFGS', 'GaussNewton',
    'InexactGaussNewton', 'ProjectedGradient', 'NewtonRoot',
    'StoppingCriteria', 'IterationPrinters'
]

SolverICG = SolverWrapI(sp.linalg.cg, checkAccuracy=False)


class StoppingCriteria(object):
    """docstring for StoppingCriteria"""

    iteration = {
        "str": "%d : maxIter   =     %3d    <= iter          =    %3d",
        "left": lambda M: M.maxIter, "right": lambda M: M.iter,
        "stopType": "critical"
    }

    iterationLS = {
        "str": "%d : maxIterLS =     %3d    <= iterLS          =    %3d",
        "left": lambda M: M.maxIterLS, "right": lambda M: M.iterLS,
        "stopType": "critical"
    }

    armijoGoldstein = {
        "str": "%d :    ft     = %1.4e <= alp*descent     = %1.4e",
        "left": lambda M: M._LS_ft,
        "right": lambda M: M.f + M.LSreduction * M._LS_descent,
        "stopType": "optimal"
    }

    tolerance_f = {
        "str": "%d : |fc-fOld| = %1.4e <= tolF*(1+|f0|) = %1.4e",
        "left": lambda M: 1 if M.iter==0 else abs(M.f-M.f_last),
        "right": lambda M: 0 if M.iter==0 else M.tolF*(1+abs(M.f0)),
        "stopType": "optimal"
    }

    moving_x = {
        "str": "%d : |xc-x_last| = %1.4e <= tolX*(1+|x0|) = %1.4e",
        "left": lambda M: 1 if M.iter==0 else norm(M.xc-M.x_last),
        "right": lambda M: 0 if M.iter==0 else M.tolX*(1+norm(M.x0)),
        "stopType": "optimal"
    }

    tolerance_g = {
        "str": "%d : |proj(x-g)-x|    = %1.4e <= tolG          = %1.4e",
        "left": lambda M: norm(M.projection(M.xc - M.g) - M.xc),
        "right": lambda M: M.tolG,
        "stopType": "optimal"
    }

    norm_g = {
        "str": "%d : |proj(x-g)-x|    = %1.4e <= 1e3*eps       = %1.4e",
        "left": lambda M: norm(M.projection(M.xc - M.g) - M.xc),
        "right": lambda M: 1e3*M.eps,
        "stopType": "critical"
    }

    bindingSet = {
        "str": "%d : probSize  =    %3d   <= bindingSet      =    %3d",
        "left": lambda M: M.xc.size,
        "right": lambda M: np.sum(M.bindingSet(M.xc)),
        "stopType": "critical"
    }

    bindingSet_LS = {
        "str": "%d : probSize  =    %3d   <= bindingSet      =    %3d",
        "left": lambda M: M._LS_xt.size,
        "right": lambda M: np.sum(M.bindingSet(M._LS_xt)),
        "stopType": "critical"
    }

    phi_d_target_Minimize = {
        "str": "%d : phi_d  = %1.4e <= phi_d_target  = %1.4e ",
        "left": lambda M: M.parent.phi_d,
        "right": lambda M: M.parent.phi_d_target,
        "stopType": "critical"
    }

    phi_d_target_Inversion = {
        "str": "%d : phi_d  = %1.4e <= phi_d_target  = %1.4e ",
        "left": lambda I: I.phi_d, "right": lambda I: I.phi_d_target,
        "stopType": "critical"
    }


class IterationPrinters(object):
    """docstring for IterationPrinters"""

    iteration = {
        "title": "#", "value": lambda M: M.iter, "width": 5, "format": "%3d"
    }
    f = {
        "title": "f", "value": lambda M: M.f, "width": 10, "format": "%1.2e"
    }
    norm_g = {
        "title": "|proj(x-g)-x|",
        "value": lambda M: norm(M.projection(M.xc - M.g) - M.xc),
        "width": 15, "format": "%1.2e"
    }
    totalLS = {
        "title": "LS", "value": lambda M: M.iterLS, "width": 5, "format": "%d"
    }

    iterationLS = {
        "title": "#", "value": lambda M: (M.iter, M.iterLS), "width": 5,
        "format": "%3d.%d"
    }
    LS_ft = {
        "title": "ft", "value": lambda M: M._LS_ft, "width": 10,
        "format": "%1.2e"
    }
    LS_t = {
        "title": "t", "value": lambda M: M._LS_t, "width": 10,
        "format": "%0.5f"
    }
    LS_armijoGoldstein = {
        "title": "f + alp*g.T*p",
        "value": lambda M: M.f + M.LSreduction*M._LS_descent, "width": 16,
        "format": "%1.2e"
    }

    itType = {
        "title": "itType", "value": lambda M: M._itType, "width": 8,
        "format": "%s"
    }
    aSet = {
        "title": "aSet", "value": lambda M: np.sum(M.activeSet(M.xc)),
        "width": 8, "format": "%d"
    }
    bSet = {
        "title": "bSet", "value": lambda M: np.sum(M.bindingSet(M.xc)),
        "width": 8, "format": "%d"
    }
    comment = {
        "title": "Comment", "value": lambda M: M.comment, "width": 12,
        "format": "%s"
    }

    beta = {
        "title": "beta", "value": lambda M: M.parent.beta, "width": 10,
        "format":   "%1.2e"
    }
    phi_d = {
        "title": "phi_d", "value": lambda M: M.parent.phi_d, "width": 10,
        "format":   "%1.2e"
    }
    phi_m = {
        "title": "phi_m", "value": lambda M: M.parent.phi_m, "width": 10,
        "format":   "%1.2e"
    }


class Minimize(object):
    """
        Minimize is a general class for derivative based optimization.
    """

    name = "General Optimization Algorithm"  #: The name of the optimization algorithm

    maxIter = 20  #: Maximum number of iterations
    maxIterLS = 10  #: Maximum number of iterations for the line-search
    maxStep = np.inf  #: Maximum step possible, used in scaling before the line-search.
    LSreduction = 1e-4  #: Expected decrease in the line-search
    LSshorten = 0.5  #: Line-search step is shortened by this amount each time.
    tolF = 1e-1  #: Tolerance on function value decrease
    tolX = 1e-1  #: Tolerance on norm(x) movement
    tolG = 1e-1  #: Tolerance on gradient norm
    eps = 1e-5  #: Small value

    stopNextIteration = False #: Stops the optimization program nicely.

    debug   = False  #: Print debugging information
    debugLS = False  #: Print debugging information for the line-search

    comment = ''  #: Used by some functions to indicate what is going on in the algorithm
    counter = None  #: Set this to a SimPEG.Utils.Counter() if you want to count things
    parent = None  #: This is the parent of the optimization routine.

    def __init__(self, **kwargs):
        self.stoppers = [
            StoppingCriteria.tolerance_f, StoppingCriteria.moving_x,
            StoppingCriteria.tolerance_g, StoppingCriteria.norm_g,
            StoppingCriteria.iteration
        ]
        self.stoppersLS = [
            StoppingCriteria.armijoGoldstein, StoppingCriteria.iterationLS
        ]

        self.printers = [
            IterationPrinters.iteration, IterationPrinters.f,
            IterationPrinters.norm_g, IterationPrinters.totalLS
        ]
        self.printersLS = [
            IterationPrinters.iterationLS, IterationPrinters.LS_ft,
            IterationPrinters.LS_t, IterationPrinters.LS_armijoGoldstein
        ]

        Utils.setKwargs(self, **kwargs)

    @property
    def callback(self):
        return getattr(self, '_callback', None)

    @callback.setter
    def callback(self, value):
        if self.callback is not None:
            print(
                'The callback on the {0!s} Optimization was '
                'replaced.'.format(self.__name__)
            )
        self._callback = value


    @Utils.timeIt
    def minimize(self, evalFunction, x0):
        """minimize(evalFunction, x0)

        Minimizes the function (evalFunction) starting at the location x0.

        :param callable evalFunction: function handle that evaluates: f, g, H = F(x)
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
                doStartIteration()
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
            finish()
            return xc
        """
        self.evalFunction = evalFunction
        self.startup(x0)
        self.printInit()
        print('x0 has any nan: {:b}'.format(np.any(np.isnan(x0))))
        while True:
            self.doStartIteration()
            self.f, self.g, self.H = evalFunction(
                self.xc, return_g=True, return_H=True
            )
            self.printIter()
            if self.stoppingCriteria():
                break
            self.searchDirection = self.findSearchDirection()
            del self.H #: Doing this saves memory, as it is not needed in the rest of the computations.
            p = self.scaleSearchDirection(self.searchDirection)
            xt, passLS = self.modifySearchDirection(p)
            if not passLS:
                xt, caught = self.modifySearchDirectionBreak(p)
                if not caught:
                    return self.xc
            self.doEndIteration(xt)
            if self.stopNextIteration:
                break

        self.printDone()
        self.finish()

        return self.xc

    @Utils.callHooks('startup')
    def startup(self, x0):
        """
            **startup** is called at the start of any new minimize call.

            This will set::

                x0 = x0
                xc = x0
                iter = iterLS = 0

            :param numpy.ndarray x0: initial x
            :rtype: None
            :return: None
        """

        self.iter = 0
        self.iterLS = 0
        self.stopNextIteration = False

        x0 = self.projection(x0)  # ensure that we start of feasible.
        self.x0 = x0
        self.xc = x0
        self.f_last = np.nan
        self.x_last = x0

    @Utils.count
    @Utils.callHooks('doStartIteration')
    def doStartIteration(self):
        """doStartIteration()

            **doStartIteration** is called at the start of each minimize
            iteration.

            :rtype: None
            :return: None
        """
        pass


    def printInit(self, inLS=False):
        """
            **printInit** is called at the beginning of the optimization
            routine.

            If there is a parent object, printInit will check for a
            parent.printInit function and call that.

        """
        pad = ' '*10 if inLS else ''
        name = self.name if not inLS else self.nameLS
        Utils.printTitles(
            self, self.printers if not inLS else self.printersLS, name, pad
        )

    @Utils.callHooks('printIter')
    def printIter(self, inLS=False):
        """
            **printIter** is called directly after function evaluations.

            If there is a parent object, printIter will check for a
            parent.printIter function and call that.

        """
        pad = ' '*10 if inLS else ''
        Utils.printLine(
            self, self.printers if not inLS else self.printersLS, pad=pad
        )

    def printDone(self, inLS=False):
        """
            **printDone** is called at the end of the optimization routine.

            If there is a parent object, printDone will check for a
            parent.printDone function and call that.

        """
        pad = ' '*10 if inLS else ''
        stop, done = (
            (' STOP! ', ' DONE! ') if not inLS else
            ('----------------', ' End Linesearch ')
        )
        stoppers = self.stoppers if not inLS else self.stoppersLS
        Utils.printStoppers(self, stoppers, pad='', stop=stop, done=done)


    @Utils.callHooks('finish')
    def finish(self):
        """finish()

            **finish** is called at the end of the optimization.

            :rtype: None
            :return: None

        """
        pass

    def stoppingCriteria(self, inLS=False):
        if self.iter == 0:
            self.f0 = self.f
            self.g0 = self.g
        return Utils.checkStoppers(
            self, self.stoppers if not inLS else self.stoppersLS
        )

    @Utils.timeIt
    @Utils.callHooks('projection')
    def projection(self, p):
        """projection(p)

            projects the search direction.

            by default, no projection is applied.

            :param numpy.ndarray p: searchDirection
            :rtype: numpy.ndarray
            :return: p, projected search direction
        """
        return p

    @Utils.timeIt
    def findSearchDirection(self):
        """findSearchDirection()

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

    @Utils.count
    def scaleSearchDirection(self, p):
        """scaleSearchDirection(p)

            **scaleSearchDirection** should scale the search direction if
            appropriate.

            Set the parameter **maxStep** in the minimize object, to scale back
            the gradient to a maximum size.

            :param numpy.ndarray p: searchDirection
            :rtype: numpy.ndarray
            :return: p, Scaled Search Direction
        """

        if self.maxStep < np.abs(p.max()):
            p = self.maxStep*p/np.abs(p.max())
        return p

    nameLS = "Armijo linesearch" #: The line-search name

    @Utils.timeIt
    def modifySearchDirection(self, p):
        """modifySearchDirection(p)

            **modifySearchDirection** changes the search direction based on
            some sort of linesearch or trust-region criteria.

            By default, an Armijo backtracking linesearch is preformed with the
            following parameters:

                * maxIterLS, the maximum number of linesearch iterations
                * LSreduction, the expected reduction expected, default: 1e-4
                * LSshorten, how much the step is reduced, default: 0.5

            If the linesearch is completed, and a descent direction is found,
            passLS is returned as True.

            Else, a modifySearchDirectionBreak call is preformed.

            :param numpy.ndarray p: searchDirection
            :rtype: tuple
            :return: (xt, passLS) numpy.ndarray, bool
        """
        # Projected Armijo linesearch
        self._LS_t = 1
        self.iterLS = 0
        while self.iterLS < self.maxIterLS:
            self._LS_xt = self.projection(self.xc + self._LS_t*p)
            self._LS_ft = self.evalFunction(
                self._LS_xt, return_g=False, return_H=False
            )
            self._LS_descent = np.inner(self.g, self._LS_xt - self.xc)  # this takes into account multiplying by t, but is important for projection.
            if self.stoppingCriteria(inLS=True):
                break
            self.iterLS += 1
            self._LS_t = self.LSshorten*self._LS_t
            if self.debugLS:
                if self.iterLS == 1: self.printInit(inLS=True)
                self.printIter(inLS=True)

        if self.debugLS and self.iterLS > 0:
            self.printDone(inLS=True)

        return self._LS_xt, self.iterLS < self.maxIterLS

    @Utils.count
    def modifySearchDirectionBreak(self, p):
        """modifySearchDirectionBreak(p)

            Code is called if modifySearchDirection fails
            to find a descent direction.

            The search direction is passed as input and
            this function must pass back both a new searchDirection,
            and if the searchDirection break has been caught.

            By default, no additional work is done, and the
            evalFunction returns a False indicating the break was not caught.

            :param numpy.ndarray p: searchDirection
            :rtype: tuple
            :return: (xt, breakCaught) numpy.ndarray, bool
        """
        self.printDone(inLS=True)
        print('The linesearch got broken. Boo.')
        return p, False

    @Utils.count
    @Utils.callHooks('doEndIteration')
    def doEndIteration(self, xt):
        """doEndIteration(xt)

            **doEndIteration** is called at the end of each minimize iteration.

            By default, function values and x locations are shuffled to store 1
            past iteration in memory.

            self.xc must be updated in this code.

            :param numpy.ndarray xt: tested new iterate that ensures a descent direction.
            :rtype: None
            :return: None
        """
        # store old values
        self.f_last = self.f
        self.x_last, self.xc = self.xc, xt
        self.iter += 1
        if self.debug:
            self.printDone()

        if self.callback is not None:
            self.callback(xt)

    def save(self, group):
        group.setArray('searchDirection', self.searchDirection)

        if getattr(self, 'parent', None) is None:
            group.setArray('x', self.xc)
        else: # Assume inversion is the parent
            group.attrs['phi_d'] = self.parent.phi_d
            group.attrs['phi_m'] = self.parent.phi_m
            group.attrs['beta'] = self.parent.beta
            group.setArray('m', self.xc)
            group.setArray('dpred', self.parent.dpred)


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
        assert param in self._rememberList, (
            "You didn't tell me to remember " + param +
            ", you gotta tell me what to remember!"
        )
        return self._rememberList[param]

    def _startupRemember(self, x0):
        self._rememberList = {}
        for param in self._rememberThese:
            if isinstance(param, string_types):
                self._rememberList[param] = []
            elif isinstance(param, tuple):
                self._rememberList[param[0]] = []

    def _doEndIterationRemember(self, *args):
        for param in self._rememberThese:
            if isinstance(param, string_types):
                if self.debug: print('Remember is remembering: ' + param)
                val = getattr(self, param, None)
                if val is None and getattr(self, 'parent', None) is not None:
                    # Look to the parent for the param if not found here.
                    val = getattr(self.parent, param, None)
                self._rememberList[param].append( val )
            elif isinstance(param, tuple):
                if self.debug: print('Remember is remembering: ' + param[0])
                self._rememberList[param[0]].append( param[1](self) )


class ProjectedGradient(Minimize, Remember):
    name = 'Projected Gradient'

    maxIterCG = 5
    tolCG = 1e-1

    lower = -np.inf
    upper = np.inf

    def __init__(self,**kwargs):
        super(ProjectedGradient, self).__init__(**kwargs)

        self.stoppers.append(StoppingCriteria.bindingSet)
        self.stoppersLS.append(StoppingCriteria.bindingSet_LS)

        self.printers.extend([
            IterationPrinters.itType, IterationPrinters.aSet,
            IterationPrinters.bSet, IterationPrinters.comment
        ])

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

    @Utils.count
    def projection(self, x):
        """projection(x)

            Make sure we are feasible.

        """
        return np.median(np.c_[self.lower, x, self.upper], axis=1)

    @Utils.count
    def activeSet(self, x):
        """activeSet(x)

            If we are on a bound

        """
        return np.logical_or(x == self.lower, x == self.upper)

    @Utils.count
    def inactiveSet(self, x):
        """inactiveSet(x)

            The free variables.

        """
        return np.logical_not(self.activeSet(x))

    @Utils.count
    def bindingSet(self, x):
        """bindingSet(x)

            If we are on a bound and the negative gradient points away from the
            feasible set.

            Optimality condition. (Satisfies Kuhn-Tucker) MoreToraldo91

        """
        bind_up  = np.logical_and(x == self.lower, self.g >= 0)
        bind_low = np.logical_and(x == self.upper, self.g <= 0)
        return np.logical_or(bind_up, bind_low)

    @Utils.timeIt
    def findSearchDirection(self):
        """findSearchDirection()

            Finds the search direction based on either CG or steepest descent.
        """
        self.aSet_prev = self.activeSet(self.xc)
        allBoundsAreActive = sum(self.aSet_prev) == self.xc.size

        if self.debug:
            print('findSearchDirection: stopDoingPG: ', self.stopDoingPG)
        if self.debug:
            print('findSearchDirection: explorePG: ', self.explorePG)
        if self.debug:
            print('findSearchDirection: exploreCG: ', self.exploreCG)
        if self.debug:
            print('findSearchDirection: aSet', np.sum(self.activeSet(self.xc)))
        if self.debug:
            print(
                'findSearchDirection: bSet', np.sum(self.bindingSet(self.xc))
            )
        if self.debug:
            print(
                'findSearchDirection: allBoundsAreActive: ', allBoundsAreActive
            )

        if self.explorePG or not self.exploreCG or allBoundsAreActive:
            if self.debug:
                print('findSearchDirection.PG: doingPG')
            self._itType = 'SD'
            p = -self.g
        else:
            if self.debug:
                print('findSearchDirection.CG: doingCG')
            # Reset the max decrease each time you do a CG iteration
            self.f_decrease_max = -np.inf

            self._itType = '.CG.'

            iSet  = self.inactiveSet(self.xc)  # The inactive set (free variables)
            bSet = self.bindingSet(self.xc)
            shape = (self.xc.size, np.sum(iSet))
            v = np.ones(shape[1])
            i = np.where(iSet)[0]
            j = np.arange(shape[1])
            if self.debug:
                print('findSearchDirection.CG: Z.shape', shape)
            Z = sp.csr_matrix((v, (i, j)), shape=shape)

            def reduceHess(v):
                # Z is tall and skinny
                return Z.T*(self.H*(Z*v))
            operator = sp.linalg.LinearOperator(
                (shape[1], shape[1]), reduceHess, dtype=self.xc.dtype
            )
            p, info = sp.linalg.cg(
                operator, -Z.T*self.g, tol=self.tolCG, maxiter=self.maxIterCG
            )
            p = Z*p  # bring up to full size
            # aSet_after = self.activeSet(self.xc+p)
        return p

    @Utils.timeIt
    def _doEndIteration_ProjectedGradient(self, xt):
        """_doEndIteration_ProjectedGradient(xt)"""
        aSet = self.activeSet(xt)
        bSet = self.bindingSet(xt)

        self.explorePG = not np.all(aSet == self.aSet_prev) # explore proximal gradient
        self.exploreCG = np.all(aSet == bSet) # explore conjugate gradient

        f_current_decrease = self.f_last - self.f
        self.comment = ''
        if self.iter < 1:
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
        # self.eta_2 * max_decrease where max decrease
        # if true go to CG
        # don't do too many steps of PG in a row.

        if self.debug:
            print(
                'doEndIteration.ProjGrad, f_current_decrease: ',
                f_current_decrease
            )
        if self.debug:
            print(
                'doEndIteration.ProjGrad, f_decrease_max: ',
                self.f_decrease_max
            )
        if self.debug:
            print('doEndIteration.ProjGrad, stopDoingSD: ', self.stopDoingPG)


class BFGS(Minimize, Remember):
    name = 'BFGS'
    nbfgs = 10

    def __init__(self, **kwargs):
        Minimize.__init__(self, **kwargs)

    @property
    def bfgsH0(self):
        """
            Approximate Hessian used in preconditioning the problem.

            Must be a SimPEG.Solver
        """
        if getattr(self, '_bfgsH0', None) is None:
            print("""
                Default solver: SolverDiag is being used in bfgsH0
                """
            )
            self._bfgsH0 = SolverDiag(sp.identity(self.xc.size))
        return self._bfgsH0

    @bfgsH0.setter
    def bfgsH0(self, value):
        self._bfgsH0 = value

    def _startup_BFGS(self, x0):
        self._bfgscnt = -1
        self._bfgsY = np.zeros((x0.size, self.nbfgs))
        self._bfgsS = np.zeros((x0.size, self.nbfgs))
        if not np.any([p is IterationPrinters.comment for p in self.printers]):
            self.printers.append(IterationPrinters.comment)

    def bfgs(self, d):
        n = self._bfgscnt
        nn = ktop = min(self._bfgsS.shape[1], n)
        return self.bfgsrec(ktop, n, nn, self._bfgsS, self._bfgsY, d)

    def bfgsrec(self, k, n, nn, S, Y, d):
        """BFGS recursion"""
        if k < 0:
            d = self.bfgsH0 * d  # Assume that bfgsH0 is a SimPEG.Solver
        else:
            khat = 0 if nn is 0 else np.mod(n-nn+k,nn)
            gamma = np.vdot(S[:, khat], d)/np.vdot(Y[:, khat], S[:, khat])
            d = d - gamma*Y[:, khat]
            d = self.bfgsrec(k-1, n, nn, S, Y, d)
            d = d + (
                gamma - np.vdot(Y[:, khat], d)/np.vdot(Y[:, khat], S[:, khat])
            ) * S[:, khat]
        return d

    def findSearchDirection(self):
        return self.bfgs(-self.g)

    def _doEndIteration_BFGS(self, xt):
        if self.iter is 0:
            self.g_last = self.g
            return

        yy = self.g - self.g_last
        ss = self.xc - xt
        self.g_last = self.g

        if yy.dot(ss) > 0:
            self._bfgscnt += 1
            ktop = np.mod(self._bfgscnt, self.nbfgs)
            self._bfgsY[:, ktop] = yy
            self._bfgsS[:, ktop] = ss
            self.comment = ''
        else:
            self.comment = 'Skip BFGS'


class GaussNewton(Minimize, Remember):
    name = 'Gauss Newton'

    def __init__(self, **kwargs):
        Minimize.__init__(self, **kwargs)

    @Utils.timeIt
    def findSearchDirection(self):
        return Solver(self.H) * (-self.g)


class InexactGaussNewton(BFGS, Minimize, Remember):
    """
        Minimizes using CG as the inexact solver of

        .. math::

            \mathbf{H p = -g}

        By default BFGS is used as the preconditioner.

        Use *nbfgs* to set the memory limitation of BFGS.

        To set the initial H0 to be used in BFGS, set *bfgsH0* to be a
        SimPEG.Solver

    """

    def __init__(self, **kwargs):
        Minimize.__init__(self, **kwargs)

    name = 'Inexact Gauss Newton'

    maxIterCG = 5
    tolCG = 1e-1

    @property
    def approxHinv(self):
        """
            The approximate Hessian inverse is used to precondition CG.

            Default uses BFGS, with an initial H0 of *bfgsH0*.

            Must be a scipy.sparse.linalg.LinearOperator
        """
        _approxHinv = getattr(self, '_approxHinv', None)
        if _approxHinv is None:
            M = sp.linalg.LinearOperator(
                (self.xc.size, self.xc.size), self.bfgs, dtype=self.xc.dtype
            )
            return M
        return _approxHinv

    @approxHinv.setter
    def approxHinv(self, value):
        self._approxHinv = value

    @Utils.timeIt
    def findSearchDirection(self):
        Hinv = SolverICG(
            self.H, M=self.approxHinv, tol=self.tolCG, maxiter=self.maxIterCG
        )
        p = Hinv * (-self.g)
        return p


class SteepestDescent(Minimize, Remember):
    name = 'Steepest Descent'

    def __init__(self, **kwargs):
        Minimize.__init__(self, **kwargs)

    @Utils.timeIt
    def findSearchDirection(self):
        return -self.g


class NewtonRoot(object):
    """
        Newton Method - Root Finding

        root = newtonRoot(fun,x);

        Where fun is the function that returns the function value as well as
        the gradient.

        For iterative solving of dh = -J\\r, use O.solveTol = TOL. For direct
        solves, use SOLVETOL = 0 (default)

        Rowan Cockett
        16-May-2013 16:29:51
        University of British Columbia
        rcockett@eos.ubc.ca

    """

    tol = 1.000e-06
    maxIter = 20
    stepDcr = 0.5
    maxLS = 30
    comments = False
    doLS = True

    Solver = Solver
    solverOpts = {}

    def __init__(self, **kwargs):
        Utils.setKwargs(self, **kwargs)

    def root(self, fun, x):
        """root(fun, x)

        Function Should have the form::

            def evalFunction(x, return_g=False):
                    out = (f,)
                    if return_g:
                        out += (g,)
                    return out if len(out) > 1 else out[0]

        """
        if self.comments:
            print('Newton Method:\n')

        self.iter = 0
        while True:

            r, J = fun(x, return_g=True)

            Jinv = self.Solver(J, **self.solverOpts)
            dh = - (Jinv * r)

            muLS = 1.
            LScnt  = 1
            xt = x + dh
            rt = fun(xt, return_g=False)

            if self.comments and self.doLS:
                print('\tLinesearch:\n')
            # Enter Linesearch
            while True and self.doLS:
                if self.comments:
                    print('\t\tResid: {0:e}\n'.format(norm(rt)))
                if norm(rt) <= norm(r) or norm(rt) < self.tol:
                    break

                muLS = muLS*self.stepDcr
                LScnt = LScnt + 1
                print('.')
                if LScnt > self.maxLS:
                    print('Newton Method: Line search break.')
                    return None
                xt = x + muLS*dh
                rt = fun(xt, return_g=False)

            x = xt
            self.iter += 1
            if norm(rt) < self.tol:
                break
            if self.iter > self.maxIter:
                print(
                    'NewtonRoot stopped by maxIters ({0:d}). '
                    'norm: {1:4.4e}'.format(self.maxIter, norm(rt))
                )
                break

        return x


class ProjectedGNCG(BFGS, Minimize, Remember):

    def __init__(self, **kwargs):
        Minimize.__init__(self, **kwargs)

    name = 'Projected GNCG'

    maxIterCG = 5
    tolCG = 1e-1

    stepOffBoundsFact = 0.1 # perturbation of the inactive set off the bounds
    stepActiveset = True
    lower = -np.inf
    upper = np.inf

    def _startup(self, x0):
        # ensure bound vectors are the same size as the model
        if type(self.lower) is not np.ndarray:
            self.lower = np.ones_like(x0)*self.lower
        if type(self.upper) is not np.ndarray:
            self.upper = np.ones_like(x0)*self.upper

    @Utils.count
    def projection(self, x):
        """projection(x)

            Make sure we are feasible.

        """
        return np.median(np.c_[self.lower, x, self.upper], axis=1)

    @Utils.count
    def activeSet(self, x):
        """activeSet(x)

            If we are on a bound

        """
        return np.logical_or(x <= self.lower, x >= self.upper)

    @property
    def approxHinv(self):
        """
            The approximate Hessian inverse is used to precondition CG.

            Default uses BFGS, with an initial H0 of *bfgsH0*.

            Must be a scipy.sparse.linalg.LinearOperator
        """
        _approxHinv = getattr(self, '_approxHinv', None)
        if _approxHinv is None:
            M = sp.linalg.LinearOperator(
                (self.xc.size, self.xc.size), self.bfgs, dtype=self.xc.dtype
            )
            return M
        return _approxHinv

    @approxHinv.setter
    def approxHinv(self, value):
        self._approxHinv = value

    @Utils.timeIt
    def findSearchDirection(self):

        """
            findSearchDirection()
            Finds the search direction based on either CG or steepest descent.
        """
        Active = self.activeSet(self.xc)
        temp = sum((np.ones_like(self.xc.size)-Active))
        allBoundsAreActive = temp == self.xc.size

        if allBoundsAreActive:
            Hinv = SolverICG(
                self.H, M=self.approxHinv, tol=self.tolCG,
                maxiter=self.maxIterCG
            )
            p = Hinv * (-self.g)
            return p
        else:
            delx = np.zeros(self.g.size)
            resid = -(1-Active) * self.g

            # Begin CG iterations.
            cgiter = 0
            cgFlag = 0
            normResid0 = norm(resid)

            while cgFlag == 0:

                cgiter = cgiter + 1
                dc = (1-Active)*(self.approxHinv*resid)
                rd = np.dot(resid, dc)

                #  Compute conjugate direction pc.
                if cgiter == 1:
                    pc = dc
                else:
                    betak = rd / rdlast
                    pc = dc + betak * pc

                #  Form product Hessian*pc.
                Hp = self.H*pc
                Hp = (1-Active)*Hp

                #  Update delx and residual.
                alphak = rd / np.dot(pc, Hp)
                delx = delx + alphak*pc
                resid = resid - alphak*Hp
                rdlast = rd

                if np.logical_or(
                    norm(resid)/normResid0 <= self.tolCG,
                    cgiter == self.maxIterCG
                ):
                    cgFlag = 1
                # End CG Iterations

            # Take a gradient step on the active cells if exist
            if self.stepActiveset:
                if temp != self.xc.size:

                    rhs_a = (Active) * -self.g

                    dm_i = max( abs( delx ) )
                    dm_a = max( abs(rhs_a) )

                # perturb inactive set off of bounds so that they are included
                # in the step
                delx = delx + self.stepOffBoundsFact * (rhs_a * dm_i / dm_a)


            # Only keep gradients going in the right direction on the active
            # set
            indx = (
                ((self.xc<=self.lower) & (delx < 0)) |
                ((self.xc>=self.upper) & (delx > 0))
            )
            delx[indx] = 0.

            return delx
