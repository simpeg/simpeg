"""
========================================================
SimPEG Optimizers (:mod:`simpeg.optimization`)
========================================================
.. currentmodule:: simpeg.optimization

Optimizers
==========

These optimizers are available within SimPEG for use during inversion.

Unbound Optimizers
------------------

These optimizers all work on unbound minimization functions.

.. autosummary::
  :toctree: generated/

  SteepestDescent
  BFGS
  GaussNewton
  InexactGaussNewton

Box Bounded Optimizers
----------------------
These optimizers support box bound constraints on the model parameters

.. autosummary::
  :toctree: generated/

  ProjectedGradient
  ProjectedGNCG

Root Finding
------------
.. autosummary::
  :toctree: generated/

  NewtonRoot

Minimization Base Classes
===========================

These classes are usually inherited or used by the optimization algorithms
above to control their execution.

Base Minimizer
--------------
.. autosummary::
  :toctree: generated/

  Minimize


Minimizer Mixins
----------------
.. autosummary::
  :toctree: generated/

  Remember
  Bounded
  InexactCG

Iteration Printers and Stoppers
-------------------------------
.. autosummary::
  :toctree: generated/

  IterationPrinters
  StoppingCriteria

"""

import warnings
from collections.abc import Callable
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from discretize.utils import Identity

from pymatsolver import Solver, SolverCG

from .typing import MinimizeCallable
from .utils import (
    call_hooks,
    check_stoppers,
    count,
    set_kwargs,
    timeIt,
    print_titles,
    print_line,
    print_stoppers,
    print_done,
    validate_float,
    validate_integer,
    validate_type,
    validate_ndarray_with_shape,
    deprecate_property,
)

norm = np.linalg.norm


__all__ = [
    "Minimize",
    "Remember",
    "SteepestDescent",
    "BFGS",
    "GaussNewton",
    "InexactGaussNewton",
    "ProjectedGradient",
    "ProjectedGNCG",
    "NewtonRoot",
    "StoppingCriteria",
    "IterationPrinters",
]


class StoppingCriteria(object):
    """docstring for StoppingCriteria"""

    iteration = {
        "str": "%d : maxIter   =     %3d    <= iter          =    %3d",
        "left": lambda M: M.maxIter,
        "right": lambda M: M.iter,
        "stopType": "critical",
    }

    iterationLS = {
        "str": "%d : maxIterLS =     %3d    <= iterLS          =    %3d",
        "left": lambda M: M.maxIterLS,
        "right": lambda M: M.iterLS,
        "stopType": "critical",
    }

    armijoGoldstein = {
        "str": "%d :    ft     = %1.4e <= alp*descent     = %1.4e",
        "left": lambda M: M._LS_ft,
        "right": lambda M: M.f + M.LSreduction * M._LS_descent,
        "stopType": "optimal",
    }

    WolfeCurvature = {
        "str": "%d :    -newgradient*descent  = %1.4e <= -alp*oldgradient*descent     = %1.4e",
        "left": lambda M: -M._LS_ft_descent,
        "right": lambda M: -M.LScurvature * M._LS_descent,
        "stopType": "optimal",
    }

    tolerance_f = {
        "str": "%d : |fc-fOld| = %1.4e <= tolF*(1+|f0|) = %1.4e",
        "left": lambda M: 1 if M.iter == 0 else abs(M.f - M.f_last),
        "right": lambda M: 0 if M.iter == 0 else M.tolF * (1 + abs(M.f0)),
        "stopType": "optimal",
    }

    moving_x = {
        "str": "%d : |xc-x_last| = %1.4e <= tolX*(1+|x0|) = %1.4e",
        "left": lambda M: 1 if M.iter == 0 else norm(M.xc - M.x_last),
        "right": lambda M: 0 if M.iter == 0 else M.tolX * (1 + norm(M.x0)),
        "stopType": "optimal",
    }

    tolerance_g = {
        "str": "%d : |proj(x-g)-x|    = %1.4e <= tolG          = %1.4e",
        "left": lambda M: norm(M.projection(M.xc - M.g) - M.xc),
        "right": lambda M: M.tolG,
        "stopType": "optimal",
    }

    norm_g = {
        "str": "%d : |proj(x-g)-x|    = %1.4e <= 1e3*eps       = %1.4e",
        "left": lambda M: norm(M.projection(M.xc - M.g) - M.xc),
        "right": lambda M: 1e3 * M.eps,
        "stopType": "critical",
    }

    bindingSet = {
        "str": "%d : probSize  =    %3d   <= bindingSet      =    %3d",
        "left": lambda M: M.xc.size,
        "right": lambda M: np.sum(M.bindingSet(M.xc)),
        "stopType": "critical",
    }

    bindingSet_LS = {
        "str": "%d : probSize  =    %3d   <= bindingSet      =    %3d",
        "left": lambda M: M._LS_xt.size,
        "right": lambda M: np.sum(M.bindingSet(M._LS_xt)),
        "stopType": "critical",
    }

    phi_d_target_Minimize = {
        "str": "%d : phi_d  = %1.4e <= phi_d_target  = %1.4e ",
        "left": lambda M: M.parent.phi_d,
        "right": lambda M: M.parent.phi_d_target,
        "stopType": "critical",
    }

    phi_d_target_Inversion = {
        "str": "%d : phi_d  = %1.4e <= phi_d_target  = %1.4e ",
        "left": lambda I: I.phi_d,
        "right": lambda I: I.phi_d_target,
        "stopType": "critical",
    }


class IterationPrinters(object):
    """docstring for IterationPrinters"""

    iteration = {
        "title": "#",
        "value": lambda M: M.iter,
        "width": 5,
        "format": lambda v: f"{v:3d}",
    }
    f = {
        "title": "f",
        "value": lambda M: M.f,
        "width": 10,
        "format": lambda v: f"{v:1.2e}",
    }
    norm_g = {
        "title": "|proj(x-g)-x|",
        "value": lambda M: (
            None if M.iter == 0 else norm(M.projection(M.xc - M.g) - M.xc)
        ),
        "width": 15,
        "format": lambda v: f"{v:1.2e}",
    }
    totalLS = {
        "title": "LS",
        "value": lambda M: None if M.iter == 0 else M.iterLS,
        "width": 5,
        "format": lambda v: f"{v:d}",
    }

    iterationLS = {
        "title": "#",
        "value": lambda M: (M.iter, M.iterLS),
        "width": 5,
        "format": lambda v: f"{v[0]:3d}.{v[1]:d}",
    }
    LS_ft = {
        "title": "ft",
        "value": lambda M: M._LS_ft,
        "width": 10,
        "format": lambda v: f"{v:1.2e}",
    }
    LS_t = {
        "title": "t",
        "value": lambda M: M._LS_t,
        "width": 10,
        "format": lambda v: f"{v:0.5f}",
    }
    LS_armijoGoldstein = {
        "title": "f + alp*g.T*p",
        "value": lambda M: M.f + M.LSreduction * M._LS_descent,
        "width": 16,
        "format": lambda v: f"{v:1.2e}",
    }
    LS_WolfeCurvature = {
        "title": "alp*g.T*p",
        "str": "%d :    ft     = %1.4e >= alp*descent     = %1.4e",
        "value": lambda M: M.LScurvature * M._LS_descent,
        "width": 16,
        "format": lambda v: f"{v:1.2e}",
    }

    itType = {
        "title": "itType",
        "value": lambda M: M._itType,
        "width": 8,
        "format": lambda v: f"{v:s}",
    }
    aSet = {
        "title": "aSet",
        "value": lambda M: None if M.iter == 0 else np.sum(M.activeSet(M.xc)),
        "width": 8,
        "format": lambda v: f"{v:d}",
    }
    bSet = {
        "title": "bSet",
        "value": lambda M: None if M.iter == 0 else np.sum(M.bindingSet(M.xc)),
        "width": 8,
        "format": lambda v: f"{v:d}",
    }
    comment = {
        "title": "Comment",
        "value": lambda M: M.comment,
        "width": 12,
        "format": lambda v: f"{v:s}",
    }

    beta = {
        "title": "beta",
        "value": lambda M: M.parent.beta,
        "width": 10,
        "format": lambda v: f"{v:1.2e}",
    }
    phi_d = {
        "title": "phi_d",
        "value": lambda M: M.parent.phi_d,
        "width": 10,
        "format": lambda v: f"{v:1.2e}",
    }
    phi_m = {
        "title": "phi_m",
        "value": lambda M: M.parent.phi_m,
        "width": 10,
        "format": lambda v: f"{v:1.2e}",
    }

    phi_s = {
        "title": "phi_s",
        "value": lambda M: M.parent.phi_s,
        "width": 10,
        "format": lambda v: f"{v:1.2e}",
    }
    phi_x = {
        "title": "phi_x",
        "value": lambda M: M.parent.phi_x,
        "width": 10,
        "format": lambda v: f"{v:1.2e}",
    }
    phi_y = {
        "title": "phi_y",
        "value": lambda M: M.parent.phi_y,
        "width": 10,
        "format": lambda v: f"{v:1.2e}",
    }
    phi_z = {
        "title": "phi_z",
        "value": lambda M: M.parent.phi_z,
        "width": 10,
        "format": lambda v: f"{v:1.2e}",
    }

    iterationCG = {
        "title": "iter_CG",
        "value": lambda M: getattr(M, "cg_count", None),
        "width": 10,
        "format": lambda v: f"{v:d}",
    }

    iteration_CG_rel_residual = {
        "title": "CG |Ax-b|/|b|",
        "value": lambda M: getattr(M, "cg_rel_resid", None),
        "width": 15,
        "format": lambda v: f"{v:1.2e}",
    }

    iteration_CG_abs_residual = {
        "title": "CG |Ax-b|",
        "value": lambda M: getattr(M, "cg_abs_resid", None),
        "width": 11,
        "format": lambda v: f"{v:1.2e}",
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
    LScurvature = (
        0.9  #: Expected decrease of the slope for line search Wolfe Curvature criteria
    )
    LSshorten = 0.5  #: Line-search step is shortened by this amount each time.
    tolF = 1e-1  #: Tolerance on function value decrease
    tolX = 1e-1  #: Tolerance on norm(x) movement
    tolG = 1e-1  #: Tolerance on gradient norm
    eps = 1e-5  #: Small value
    require_decrease = True  #: Require decrease in the objective function. If False, we will still take a step when the linesearch fails

    stopNextIteration = False  #: Stops the optimization program nicely.
    use_WolfeCurvature = False  #: add the Wolfe Curvature criteria for line search

    debug = False  #: Print debugging information
    debugLS = False  #: Print debugging information for the line-search

    comment = (
        ""  #: Used by some functions to indicate what is going on in the algorithm
    )
    counter = None  #: Set this to a simpeg.utils.Counter() if you want to count things
    parent = None  #: This is the parent of the optimization routine.

    print_type = None

    def __init__(self, **kwargs):
        set_kwargs(self, **kwargs)

        self.stoppersLS = [
            StoppingCriteria.armijoGoldstein,
            StoppingCriteria.iterationLS,
        ]

        if self.use_WolfeCurvature:
            self.stoppersLS.append(StoppingCriteria.WolfeCurvature)

        self.printersLS = [
            IterationPrinters.iterationLS,
            IterationPrinters.LS_ft,
            IterationPrinters.LS_t,
            IterationPrinters.LS_armijoGoldstein,
        ]

        if self.print_type == "ubc":
            self.stoppers = [StoppingCriteria.iteration]
            self.printers = [
                IterationPrinters.iteration,
                IterationPrinters.phi_s,
                IterationPrinters.phi_x,
                IterationPrinters.phi_y,
                IterationPrinters.phi_z,
                IterationPrinters.totalLS,
            ]
        else:
            self.stoppers = [
                StoppingCriteria.tolerance_f,
                StoppingCriteria.moving_x,
                StoppingCriteria.tolerance_g,
                StoppingCriteria.norm_g,
                StoppingCriteria.iteration,
            ]
            self.printers = [
                IterationPrinters.iteration,
                IterationPrinters.f,
                IterationPrinters.norm_g,
                IterationPrinters.totalLS,
            ]

    @property
    def callback(self) -> Optional[Callable[[np.ndarray], Any]]:
        """A used defined callback function.

        Returns
        -------
        None or Callable[[np.ndarray], Any]
            The optional user supplied callback function accepting the current iteration
            value as an input.
        """
        return getattr(self, "_callback", None)

    @callback.setter
    def callback(self, value: Callable[[np.ndarray], Any]):
        if self.callback is not None:
            print(
                f"The callback on the {self.__class__.__name__} minimizer was replaced."
            )
        self._callback = value

    @timeIt
    def minimize(self, evalFunction: MinimizeCallable, x0: np.ndarray) -> np.ndarray:
        """minimize(evalFunction, x0)

        Minimizes the function (evalFunction) starting at the location x0.

        Parameters
        ----------
        evalFunction : callable
            The objective function to be minimized::

                evalFunction(
                    x: numpy.ndarray,
                    return_g: bool,
                    return_H: bool
                ) -> (
                    float
                    | tuple[float, numpy.ndarray]
                    | tuple[float, LinearOperator]
                    | tuple[float, numpy.ndarray, LinearOperator]
                )

            That will optionally return the gradient as a ``numpy.ndarray`` and the Hessian as any class
            that supports matrix vector multiplication using the `*` operator.

        x0 : numpy.ndarray
            Initial guess.

        Returns
        -------
        x_min : numpy.ndarray
            The last iterate of the optimization algorithm.
        """
        self.evalFunction = evalFunction
        self.startup(x0)
        self.printInit()

        if np.any(np.isnan(x0)):
            raise ValueError("x0 has a nan.")
        self.f = evalFunction(
            self.xc, return_g=False, return_H=False
        )  # will stash the fields objects
        self.printIter()
        while True:
            self.doStartIteration()
            self.f, self.g, self.H = evalFunction(self.xc, return_g=True, return_H=True)
            if self.stoppingCriteria():
                break
            self.searchDirection = self.findSearchDirection()
            del (
                self.H
            )  #: Doing this saves memory, as it is not needed in the rest of the computations.
            p = self.scaleSearchDirection(self.searchDirection)
            xt, passLS = self.modifySearchDirection(p)
            if not passLS:
                if self.require_decrease is True:
                    xt, caught = self.modifySearchDirectionBreak(p)
                    if not caught:
                        return self.xc
                else:
                    print("Linesearch failed. Stepping anyways...")
            self.doEndIteration(xt)
            if self.stopNextIteration:
                break

        self.printDone()
        self.finish()

        return self.xc

    @call_hooks("startup")
    def startup(self, x0: np.ndarray) -> None:
        """Called at the start of any new minimize call.

        This will set::

            x0 = x0
            xc = x0
            iter = iterLS = 0

        Parameters
        ----------
        x0 : numpy.ndarray
            initial x
        """

        self.iter = 0
        self.iterLS = 0
        self.stopNextIteration = False

        try:
            x0 = self.projection(x0)  # ensure that we start of feasible.
        except Exception as err:
            raise RuntimeError("Initial model is not projectable") from err

        self.x0 = x0
        self.xc = x0
        self.f_last = np.nan
        self.x_last = x0

    @count
    @call_hooks("doStartIteration")
    def doStartIteration(self) -> None:
        """Called at the start of each minimize iteration."""
        pass

    def printInit(self, inLS: bool = False) -> None:
        """Called at the beginning of the optimization routine.

        If there is a parent object, printInit will check for a
        parent.printInit function and call that.

        Parameters
        ----------
        inLS : bool
            Whether this is being called from a line search.

        """
        pad = " " * 10 if inLS else ""
        name = self.name if not inLS else self.nameLS
        print_titles(self, self.printers if not inLS else self.printersLS, name, pad)

    @call_hooks("printIter")
    def printIter(self, inLS: bool = False) -> None:
        """Called directly after function evaluations.

        Parameters
        ----------
        inLS : bool
            Whether this is being called from a line search.

        If there is a parent object, printIter will check for a
        parent.printIter function and call that.

        """
        pad = " " * 10 if inLS else ""
        print_line(self, self.printers if not inLS else self.printersLS, pad=pad)

    def printDone(self, inLS: bool = False) -> None:
        """Called at the end of the optimization routine.

        If there is a parent object, printDone will check for a
        parent.printDone function and call that.

        Parameters
        ----------
        inLS : bool
            Whether this is being called from a line search.

        """
        pad = " " * 10 if inLS else ""
        stop, done = (
            (" STOP! ", " DONE! ")
            if not inLS
            else ("----------------", " End Linesearch ")
        )
        stoppers = self.stoppers if not inLS else self.stoppersLS

        if self.print_type == "ubc":
            try:
                print_line(
                    self, self.printers if not inLS else self.printersLS, pad=pad
                )
                print_done(
                    self,
                    self.printers,
                    pad=pad,
                )
            except AttributeError:
                print_done(
                    self,
                    self.printers,
                    pad=pad,
                )
        else:
            print_stoppers(self, stoppers, pad="", stop=stop, done=done)

    @call_hooks("finish")
    def finish(self) -> None:
        """Called at the end of the optimization."""
        pass

    def stoppingCriteria(self, inLS: bool = False) -> bool:
        if self.iter == 0:
            self.f0 = self.f
            self.g0 = self.g
        return check_stoppers(self, self.stoppers if not inLS else self.stoppersLS)

    @timeIt
    @call_hooks("projection")
    def projection(self, p: np.ndarray) -> np.ndarray:
        """Projects a model onto bounds (if given)

        By default, no projection is applied.

        Parameters
        ----------
        p : numpy.ndarray
            The model to project

        Returns
        -------
        numpy.ndarray
            The projected model.
        """
        return p

    @timeIt
    def findSearchDirection(self) -> np.ndarray:
        """Return the direction to search along for a minimum value.

        Returns
        -------
        numpy.ndarray
            The search direction.

        Notes
        -----
        This should usually return an approximation of:

        .. math::

            p = - H^{-1} g

        The default is:

        .. math::

            p = - g

        Corresponding to the steepest descent direction

        The latest function evaluations are present in::

            self.f, self.g, self.H
        """
        return -self.g

    @count
    def scaleSearchDirection(self, p: np.ndarray) -> np.ndarray:
        """Scales the search direction if appropriate.

        Set the parameter ``maxStep`` in the minimize object, to scale back
        the search direction to a maximum size.

        Parameters
        ----------
        p : numpy.ndarray
            The current search direction.

        Returns
        -------
        numpy.ndarray
            The scaled search direction.
        """

        if self.maxStep < np.abs(p.max()):
            p = self.maxStep * p / np.abs(p.max())
        return p

    nameLS = "Armijo linesearch"  #: The line-search name

    @timeIt
    def modifySearchDirection(self, p: np.ndarray) -> np.ndarray:
        """Changes the search direction based on some sort of linesearch or trust-region criteria.

        Parameters
        ----------
        p : numpy.ndarray
            The current search direction.

        Returns
        -------
        numpy.ndarray
            The modified search direction.

        Notes
        -----
        By default, an Armijo backtracking linesearch is preformed with the
        following parameters:

            * maxIterLS, the maximum number of linesearch iterations
            * LSreduction, the expected reduction expected, default: 1e-4
            * LSshorten, how much the step is reduced, default: 0.5

        If the linesearch is completed, and a descent direction is found,
        passLS is returned as True.

        Else, a `modifySearchDirectionBreak` call is preformed.
        """
        # Projected Armijo linesearch
        self._LS_t = 1.0
        self.iterLS = 0
        while self.iterLS < self.maxIterLS:
            self._LS_xt = self.projection(self.xc + self._LS_t * p)
            if self.use_WolfeCurvature:
                self._LS_ft, self._LS_ft_descent = self.evalFunction(
                    self._LS_xt, return_g=self.use_WolfeCurvature, return_H=False
                )
                self._LS_ft_descent = np.inner(
                    self._LS_ft_descent, self._LS_xt - self.xc
                )  # This is the curvature WolfeCurvature condition
            else:
                self._LS_ft = self.evalFunction(
                    self._LS_xt, return_g=self.use_WolfeCurvature, return_H=False
                )
            self._LS_descent = np.inner(
                self.g, self._LS_xt - self.xc
            )  # this takes into account multiplying by t, but is important for projection.
            if self.stoppingCriteria(inLS=True):
                break
            self.iterLS += 1
            self._LS_t = self.LSshorten * self._LS_t
            if self.debugLS:
                if self.iterLS == 1:
                    self.printInit(inLS=True)
                self.printIter(inLS=True)

        if self.debugLS and self.iterLS > 0:
            self.printDone(inLS=True)

        return self._LS_xt, self.iterLS < self.maxIterLS

    @count
    def modifySearchDirectionBreak(self, p: np.ndarray) -> np.ndarray:
        """Called if modifySearchDirection fails to find a descent direction.

        The search direction is passed as input and
        this function must pass back both a new searchDirection,
        and if the searchDirection break has been caught.

        By default, no additional work is done, and the
        evalFunction returns a False indicating the break was not caught.

        Parameters
        ----------
        p : numpy.ndarray
            The failed search direction.

        Returns
        -------
        xt : numpy.ndarray
            An alternative search direction to use.
        was_caught : bool
            Whether the break was caught. The minimization algorithm will
            break early if ``not was_caught``.
        """
        self.printDone(inLS=True)
        print("The linesearch got broken. Boo.")
        return p, False

    @count
    @call_hooks("doEndIteration")
    def doEndIteration(self, xt: np.ndarray) -> None:
        """Operation called at the end of each minimize iteration.

        By default, function values and x locations are shuffled to store 1
        past iteration in memory.

        Parameters
        ----------
        xt : numpy.ndarray
            An accepted model at the end of each iteration.
        """
        # store old values
        self.f_last = self.f
        if hasattr(self, "_LS_ft"):
            self.f = self._LS_ft

        # the current iterate, `self.xc`, must be set in this function if overridden in a base class
        self.x_last, self.xc = self.xc, xt
        self.iter += 1
        self.printIter()  # before callbacks (from directives...)
        if self.debug:
            self.printDone()

        if self.callback is not None:
            self.callback(xt)

    def save(self, group):
        group.setArray("searchDirection", self.searchDirection)

        if getattr(self, "parent", None) is None:
            group.setArray("x", self.xc)
        else:  # Assume inversion is the parent
            group.attrs["phi_d"] = self.parent.phi_d
            group.attrs["phi_m"] = self.parent.phi_m
            group.attrs["beta"] = self.parent.beta
            group.setArray("m", self.xc)
            group.setArray("dpred", self.parent.dpred)


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
            "You didn't tell me to remember "
            + param
            + ", you gotta tell me what to remember!"
        )
        return self._rememberList[param]

    def _startupRemember(self, x0):
        self._rememberList = {}
        for param in self._rememberThese:
            if isinstance(param, str):
                self._rememberList[param] = []
            elif isinstance(param, tuple):
                self._rememberList[param[0]] = []

    def _doEndIterationRemember(self, *args):
        for param in self._rememberThese:
            if isinstance(param, str):
                if self.debug:
                    print("Remember is remembering: " + param)
                val = getattr(self, param, None)
                if val is None and getattr(self, "parent", None) is not None:
                    # Look to the parent for the param if not found here.
                    val = getattr(self.parent, param, None)
                self._rememberList[param].append(val)
            elif isinstance(param, tuple):
                if self.debug:
                    print("Remember is remembering: " + param[0])
                self._rememberList[param[0]].append(param[1](self))


class Bounded(object):
    """Mixin class for bounded minimizers

    Parameters
    ----------
    lower, upper : float or numpy.ndarray, optional
        The lower and upper bounds.
    """

    def __init__(
        self,
        *,
        lower: None | float | npt.NDArray[np.float64],
        upper: None | float | npt.NDArray[np.float64] = None,
        **kwargs,
    ):
        self.lower = lower
        self.upper = upper
        super().__init__(**kwargs)

    @property
    def lower(self) -> None | float | npt.NDArray[np.float64]:
        """The lower bound value.

        Returns
        -------
        lower : None, float, numpy.ndarray
        """
        return self._lower

    @lower.setter
    def lower(self, value):
        if value is not None:
            try:
                value = validate_float("lower", value)
            except TypeError:
                value = validate_ndarray_with_shape("lower", value, shape=("*",))
        self._lower = value

    @property
    def upper(self) -> None | float | npt.NDArray[np.float64]:
        """The upper bound value.

        Returns
        -------
        upper : None, float, numpy.ndarray
        """
        return self._upper

    @upper.setter
    def upper(self, value):
        if value is not None:
            try:
                value = validate_float("upper", value)
            except TypeError:
                value = validate_ndarray_with_shape("upper", value, shape=("*",))
        self._upper = value

    @count
    def projection(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """projection(x)

        Make sure we are feasible.

        """
        if self.lower is not None:
            x = np.maximum(x, self.lower)
        if self.upper is not None:
            x = np.minimum(x, self.upper)
        return x

    @count
    def activeSet(self, x: npt.NDArray[np.float64]) -> npt.NDArray[bool]:
        """activeSet(x)

        If we are on a bound

        """
        out = np.zeros(x.shape, dtype=bool)
        if self.lower is not None:
            out |= x <= self.lower
        if self.upper is not None:
            out |= x >= self.upper
        return out

    @count
    def inactiveSet(self, x: npt.NDArray[np.float64]) -> npt.NDArray[bool]:
        """inactiveSet(x)

        The free variables.

        """
        return np.logical_not(self.activeSet(x))

    @count
    def bindingSet(self, x: npt.NDArray[np.float64]) -> npt.NDArray[bool]:
        """bindingSet(x)

        If we are on a bound and the negative gradient points away from the
        feasible set.

        Optimality condition. (Satisfies Kuhn-Tucker) MoreToraldo91

        """
        out = np.zeros(x.shape, dtype=bool)
        if self.lower is not None:
            out |= (x <= self.lower) & (self.g >= 0)
        if self.upper is not None:
            out |= (x >= self.upper) & (self.g <= 0)
        return out


class InexactCG(object):
    """Mixin to hold common parameters for a CG solver.

    Parameters
    ----------
    cg_rtol : float, optional
        Relative tolerance stopping condition on the CG residual
    cg_atol : float, optional
        Absolute tolerance stopping condition on the CG residual
    cg_maxiter : int, optional
        Maximum number of CG iterations to perform

    Notes
    -----

    The convergence check for CG is:
    >>> norm(A @ x_k - b) <= max(cg_rtol * norm(A @ x_0 - b), cg_atol)

    See Also
    --------
    scipy.sparse.linalg.cg

    """

    def __init__(
        self,
        *,
        cg_rtol: float = 1e-1,
        cg_atol: float = 0,
        cg_maxiter: int = 5,
        **kwargs,
    ):

        if (val := kwargs.pop("tolCG", None)) is not None:
            self.tolCG = val  # Deprecated cg_rtol
        else:
            self.cg_rtol = cg_rtol
        self.cg_atol = cg_atol

        if (val := kwargs.pop("maxIterCG", None)) is not None:
            self.maxIterCG = val
        else:
            self.cg_maxiter = cg_maxiter

        super().__init__(**kwargs)

    @property
    def cg_atol(self) -> float:
        """Absolute tolerance for inner CG iterations.

        CG iterations are terminated if:
        >>> norm(A @ x_k - b) <= max(cg_rtol * norm(A @ x_0 - b), cg_atol)

        or if the maximum number of CG iterations is reached.

        Returns
        -------
        float

        See Also
        --------
        cg_rtol, scipy.sparse.linalg.cg
        """
        return self._cg_atol

    @cg_atol.setter
    def cg_atol(self, value):
        self._cg_atol = validate_float("cg_atol", value, min_val=0, inclusive_min=True)

    @property
    def cg_rtol(self) -> float:
        """Relative tolerance for inner CG iterations.

        CG iterations are terminated if:
        >>> norm(A @ x_k - b) <= max(cg_rtol * norm(A @ x_0 - b), cg_atol)

        or if the maximum number of CG iterations is reached.

        Returns
        -------
        float

        See Also
        --------
        cg_rtol, scipy.sparse.linalg.cg
        """
        return self._cg_rtol

    @cg_rtol.setter
    def cg_rtol(self, value):
        self._cg_rtol = validate_float("cg_rtol", value, min_val=0, inclusive_min=True)

    @property
    def cg_maxiter(self) -> int:
        """Maximum number of CG iterations.
        Returns
        -------
        int
        """
        return self._cg_maxiter

    @cg_maxiter.setter
    def cg_maxiter(self, value):
        self._cg_maxiter = validate_integer("cg_maxiter", value, min_val=1)

    maxIterCG = deprecate_property(
        cg_maxiter, old_name="maxIterCG", removal_version="0.26.0", future_warn=True
    )
    tolCG = deprecate_property(
        cg_rtol, old_name="tolCG", removal_version="0.26.0", future_warn=True
    )


class ProjectedGradient(Bounded, InexactCG, Minimize, Remember):
    name = "Projected Gradient"

    def __init__(
        self, *, lower=-np.inf, upper=np.inf, cg_rtol=1e-1, cg_maxiter=5, **kwargs
    ):
        super().__init__(
            lower=lower, upper=upper, cg_rtol=cg_rtol, cg_maxiter=cg_maxiter, **kwargs
        )

        self.stoppers.append(StoppingCriteria.bindingSet)
        self.stoppersLS.append(StoppingCriteria.bindingSet_LS)

        self.printers.extend(
            [
                IterationPrinters.itType,
                IterationPrinters.aSet,
                IterationPrinters.bSet,
                IterationPrinters.comment,
            ]
        )

    def startup(self, x0):
        super().startup(x0)

        self.explorePG = True
        self.exploreCG = False
        self.stopDoingPG = False

        self._itType = "SD"
        self.comment = ""

        self.aSet_prev = self.activeSet(x0)

    @timeIt
    def findSearchDirection(self):
        """findSearchDirection()

        Finds the search direction based on either CG or steepest descent.
        """
        self.aSet_prev = self.activeSet(self.xc)
        allBoundsAreActive = sum(self.aSet_prev) == self.xc.size

        if self.debug:
            print("findSearchDirection: stopDoingPG: ", self.stopDoingPG)
        if self.debug:
            print("findSearchDirection: explorePG: ", self.explorePG)
        if self.debug:
            print("findSearchDirection: exploreCG: ", self.exploreCG)
        if self.debug:
            print("findSearchDirection: aSet", np.sum(self.activeSet(self.xc)))
        if self.debug:
            print("findSearchDirection: bSet", np.sum(self.bindingSet(self.xc)))
        if self.debug:
            print("findSearchDirection: allBoundsAreActive: ", allBoundsAreActive)

        if self.explorePG or not self.exploreCG or allBoundsAreActive:
            if self.debug:
                print("findSearchDirection.PG: doingPG")
            self._itType = "SD"
            p = -self.g
        else:
            if self.debug:
                print("findSearchDirection.CG: doingCG")
            # Reset the max decrease each time you do a CG iteration
            self.f_decrease_max = -np.inf

            self._itType = ".CG."

            iSet = self.inactiveSet(self.xc)  # The inactive set (free variables)
            shape = (self.xc.size, np.sum(iSet))
            v = np.ones(shape[1])
            i = np.where(iSet)[0]
            j = np.arange(shape[1])
            if self.debug:
                print("findSearchDirection.CG: Z.shape", shape)
            Z = sp.csr_matrix((v, (i, j)), shape=shape)

            def reduceHess(v):
                # Z is tall and skinny
                return Z.T * (self.H * (Z * v))

            operator = sp.linalg.LinearOperator(
                (shape[1], shape[1]), reduceHess, dtype=self.xc.dtype
            )

            p, info = sp.linalg.cg(
                operator,
                -Z.T * self.g,
                rtol=self.cg_rtol,
                atol=self.cg_atol,
                maxiter=self.cg_maxiter,
            )
            p = Z * p  # bring up to full size
            # aSet_after = self.activeSet(self.xc+p)
        return p

    @timeIt
    def _doEndIteration_ProjectedGradient(self, xt):
        """_doEndIteration_ProjectedGradient(xt)"""
        aSet = self.activeSet(xt)
        bSet = self.bindingSet(xt)

        self.explorePG = not np.all(aSet == self.aSet_prev)  # explore proximal gradient
        self.exploreCG = np.all(aSet == bSet)  # explore conjugate gradient

        f_current_decrease = self.f_last - self.f
        self.comment = ""
        if self.iter < 1:
            # Note that this is reset on every CG iteration.
            self.f_decrease_max = -np.inf
        else:
            self.f_decrease_max = max(self.f_decrease_max, f_current_decrease)
            self.stopDoingPG = f_current_decrease < 0.25 * self.f_decrease_max
            if self.stopDoingPG:
                self.comment = "Stop SD"
                self.explorePG = False
                self.exploreCG = True
        # implement 3.8, MoreToraldo91
        # self.eta_2 * max_decrease where max decrease
        # if true go to CG
        # don't do too many steps of PG in a row.

        if self.debug:
            print("doEndIteration.ProjGrad, f_current_decrease: ", f_current_decrease)
        if self.debug:
            print("doEndIteration.ProjGrad, f_decrease_max: ", self.f_decrease_max)
        if self.debug:
            print("doEndIteration.ProjGrad, stopDoingSD: ", self.stopDoingPG)


class BFGS(Minimize, Remember):
    name = "BFGS"
    nbfgs = 10

    @property
    def bfgsH0(self):
        """
        Approximate Hessian used in preconditioning the problem.

        Must be a simpeg.Solver
        """
        if getattr(self, "_bfgsH0", None) is None:
            self._bfgsH0 = Identity()
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
            d = self.bfgsH0 * d  # Assume that bfgsH0 is a simpeg.Solver
        else:
            khat = 0 if nn == 0 else np.mod(n - nn + k, nn)
            gamma = np.vdot(S[:, khat], d) / np.vdot(Y[:, khat], S[:, khat])
            d = d - gamma * Y[:, khat]
            d = self.bfgsrec(k - 1, n, nn, S, Y, d)
            d = (
                d
                + (gamma - np.vdot(Y[:, khat], d) / np.vdot(Y[:, khat], S[:, khat]))
                * S[:, khat]
            )
        return d

    def findSearchDirection(self):
        return self.bfgs(-self.g)

    def _doEndIteration_BFGS(self, xt):
        if self.iter == 0:
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
            self.comment = ""
        else:
            self.comment = "Skip BFGS"


class GaussNewton(Minimize, Remember):
    name = "Gauss Newton"

    def __init__(self, **kwargs):
        Minimize.__init__(self, **kwargs)

    @timeIt
    def findSearchDirection(self):
        return Solver(self.H) * (-self.g)


class InexactGaussNewton(InexactCG, BFGS):
    r"""
    Minimizes using CG as the inexact solver of

    .. math::

        \mathbf{H p = -g}

    By default BFGS is used as the preconditioner.

    Use *nbfgs* to set the memory limitation of BFGS.

    To set the initial H0 to be used in BFGS, set *bfgsH0* to be a
    simpeg.Solver

    """

    def __init__(
        self,
        *,
        cg_rtol: float = 1e-1,
        cg_atol: float = 0.0,
        cg_maxiter: int = 5,
        **kwargs,
    ):
        super().__init__(
            cg_rtol=cg_rtol, cg_atol=cg_atol, cg_maxiter=cg_maxiter, **kwargs
        )

        self._was_default_hinv = False

    name = "Inexact Gauss Newton"

    @property
    def approxHinv(self):
        """
        The approximate Hessian inverse is used to precondition CG.

        Default uses BFGS, with an initial H0 of *bfgsH0*.

        Must be a scipy.sparse.linalg.LinearOperator
        """
        _approxHinv = getattr(self, "_approxHinv", None)
        if _approxHinv is None:
            M = sp.linalg.LinearOperator(
                (self.xc.size, self.xc.size), self.bfgs, dtype=self.xc.dtype
            )
            self._was_default_hinv = True
            return M
        self._was_default_hinv = False
        return _approxHinv

    @approxHinv.setter
    def approxHinv(self, value):
        self._approxHinv = value

    @timeIt
    def findSearchDirection(self):
        Hinv = SolverCG(
            self.H,
            M=self.approxHinv,
            rtol=self.cg_rtol,
            atol=self.cg_atol,
            maxiter=self.cg_maxiter,
        )
        p = Hinv * (-self.g)
        return p

    def _doEndIteration_BFGS(self, xt):
        if self._was_default_hinv:
            super()._doEndIteration_BFGS(xt)


class SteepestDescent(Minimize, Remember):
    name = "Steepest Descent"

    def __init__(self, **kwargs):
        Minimize.__init__(self, **kwargs)

    @timeIt
    def findSearchDirection(self):
        return -self.g


class NewtonRoot(object):
    r"""
    Newton Method - Root Finding

    root = newtonRoot(fun,x);

    Where fun is the function that returns the function value as well as
    the gradient.

    For iterative solving of dh = -J\r, use O.solveTol = TOL. For direct
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
        set_kwargs(self, **kwargs)

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
            print("Newton Method:\n")

        self.iter = 0
        while True:
            r, J = fun(x, return_g=True)

            Jinv = self.Solver(J, **self.solverOpts)
            dh = -(Jinv * r)

            muLS = 1.0
            LScnt = 1
            xt = x + dh
            rt = fun(xt, return_g=False)

            if self.comments and self.doLS:
                print("\tLinesearch:\n")
            # Enter Linesearch
            while True and self.doLS:
                if self.comments:
                    print("\t\tResid: {0:e}\n".format(norm(rt)))
                if norm(rt) <= norm(r) or norm(rt) < self.tol:
                    break

                muLS = muLS * self.stepDcr
                LScnt = LScnt + 1
                print(".")
                if LScnt > self.maxLS:
                    print("Newton Method: Line search break.")
                    return None
                xt = x + muLS * dh
                rt = fun(xt, return_g=False)

            x = xt
            self.iter += 1
            if norm(rt) < self.tol:
                break
            if self.iter > self.maxIter:
                print(
                    "NewtonRoot stopped by maxIters ({0:d}). "
                    "norm: {1:4.4e}".format(self.maxIter, norm(rt))
                )
                break

        return x


class ProjectedGNCG(Bounded, InexactGaussNewton):
    def __init__(
        self,
        *,
        lower: None | float | npt.NDArray[np.float64] = -np.inf,
        upper: None | float | npt.NDArray[np.float64] = np.inf,
        cg_maxiter: int = 5,
        cg_rtol: float = None,
        cg_atol: float = None,
        step_active_set: bool = True,
        active_set_grad_scale: float = 1e-2,
        **kwargs,
    ):
        if (val := kwargs.pop("tolCG", None)) is not None:
            # Deprecated path when tolCG is passed.
            self.tolCG = val
            cg_atol = val
            cg_rtol = 0.0
        elif cg_rtol is None and cg_atol is None:
            # Note these defaults match previous settings...
            # but they're not good in general...
            # Ideally they will change to cg_rtol=1E-3 and cg_atol=0.0
            warnings.warn(
                "The defaults for ProjectedGNCG will change in SimPEG 0.26.0. If you want to maintain the "
                "previous behavior, explicitly set 'cg_atol=1E-3' and 'cg_rtol=0.0'.",
                FutureWarning,
                stacklevel=2,
            )
            cg_atol = 1e-3
            cg_rtol = 0.0
        # defaults for if someone passes just cg_rtol or just cg_atol (to be removed on deprecation removal)
        # These will likely be the future defaults
        elif cg_atol is None:
            cg_atol = 0.0
        elif cg_rtol is None:
            cg_rtol = 1e-3

        if (val := kwargs.pop("stepActiveSet", None)) is not None:
            self.stepActiveSet = val
        else:
            self.step_active_set = step_active_set

        if (val := kwargs.pop("stepOffBoundsFact", None)) is not None:
            self.stepOffBoundsFact = val
        else:
            self.active_set_grad_scale = active_set_grad_scale

        super().__init__(
            lower=lower,
            upper=upper,
            cg_maxiter=cg_maxiter,
            cg_rtol=cg_rtol,
            cg_atol=cg_atol,
            **kwargs,
        )

        # initialize some tracking parameters
        self.cg_count = 0
        self.cg_abs_resid = np.inf
        self.cg_rel_resid = np.inf

        self.printers.extend(
            [
                IterationPrinters.iterationCG,
                IterationPrinters.iteration_CG_rel_residual,
                IterationPrinters.iteration_CG_abs_residual,
            ]
        )

    name = "Projected GNCG"

    @property
    def step_active_set(self) -> bool:
        """Whether to include the active set's gradient in the step direction.

        Returns
        -------
        bool
        """
        return self._step_active_set

    @step_active_set.setter
    def step_active_set(self, value: bool):
        self._step_active_set = validate_type("step_active_set", value, bool)

    @property
    def active_set_grad_scale(self) -> float:
        """Scalar to apply to the active set's gradient

        if `step_active_set` is `True`, then the active set's gradient is multiplied by this value
        when including it in the search direction.

        Returns
        -------
        float
        """
        return self._active_set_grad_scale

    @active_set_grad_scale.setter
    def active_set_grad_scale(self, value: float):
        self._active_set_grad_scale = validate_float(
            "active_set_grad_scale", value, min_val=0, inclusive_min=True
        )

    @timeIt
    def findSearchDirection(self):
        """
        findSearchDirection()
        Finds the search direction based on projected CG
        """
        # remember, "active" means cell with values equal to the limit
        # "inactive" are cells with values inside the limits.

        # The basic logic of this method is to do CG iterations only
        # on the inactive set, then also add a scaled gradient for the
        # active set, (if that gradient points away from the limits.)

        self.cg_count = 0
        active = self.activeSet(self.xc)
        inactive = ~active

        step = np.zeros(self.g.size)
        resid = inactive * (-self.g)

        r = resid  # - Inactive * (self.H * step)#  step is zero

        p = self.approxHinv * r

        sold = np.dot(r, p)

        count = 0
        r_norm0 = norm(r)

        atol = max(self.cg_rtol * norm(r_norm0), self.cg_atol)
        if self.debug:
            print(f"CG Target tolerance: {atol}")
        r_norm = r_norm0
        while r_norm > atol and count < self.cg_maxiter:
            if self.debug:
                print(f"CG Iteration: {count}, residual norm: {r_norm}")
            count += 1

            q = inactive * (self.H * p)

            alpha = sold / (np.dot(p, q))

            step += alpha * p

            r -= alpha * q
            r_norm = norm(r)

            h = self.approxHinv * r

            snew = np.dot(r, h)

            p = h + (snew / sold) * p

            sold = snew
            # End CG Iterations
        self.cg_count = count
        self.cg_abs_resid = r_norm
        self.cg_rel_resid = r_norm / r_norm0

        # Also include the gradient for cells on the boundary
        # if that gradient would move them away from the boundary.
        # aka, active and not bound.
        bound = self.bindingSet(self.xc)
        active_not_bound = active & (~bound)
        if self.step_active_set and np.any(active_not_bound):
            rhs_a = active_not_bound * -self.g

            # active means x == boundary
            # bound means x == boundary and g == 0  or -g points beyond boundary
            # active and not bound means
            # x == boundary and g neq 0 and g points inside
            # so can safely discard a non-zero check on
            # if np.any(rhs_a)

            # reasonable guess at the step length for the gradient on the
            # active cell boundaries. Basically scale it to have the same
            # maximum as the cg step on the cells that are not on the
            # boundary.
            dm_i = max(abs(step))
            dm_a = max(abs(rhs_a))

            # add the active set's gradients.
            step += self.active_set_grad_scale * (rhs_a * dm_i / dm_a)

        # Only keep search directions going in the right direction
        step[bound] = 0

        return step

    stepActiveSet = deprecate_property(
        step_active_set,
        old_name="stepActiveSet",
        removal_version="0.26.0",
        future_warn=True,
    )

    stepOffBoundsFact = deprecate_property(
        active_set_grad_scale,
        old_name="stepOffBoundsFact",
        removal_version="0.26.0",
        future_warn=True,
    )

    # This was the weird part from before... the default tolerance was used as an absolute tolerance...
    tolCG = deprecate_property(
        InexactGaussNewton.cg_atol,
        old_name="tolCG",
        removal_version="0.26.0",
        future_warn=True,
    )
