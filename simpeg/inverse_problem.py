import numpy as np
import scipy.sparse as sp
import gc
from .data_misfit import BaseDataMisfit
from .regularization import BaseRegularization, WeightedLeastSquares, Sparse
from .objective_function import BaseObjectiveFunction, ComboObjectiveFunction
from .optimization import Minimize
from .utils import (
    call_hooks,
    timeIt,
    Counter,
    validate_float,
    validate_type,
    validate_ndarray_with_shape,
)
from .simulation import DefaultSolver
from .version import __version__ as simpeg_version


class BaseInvProblem:
    """BaseInvProblem(dmisfit, reg, opt)"""

    def __init__(
        self,
        dmisfit,
        reg,
        opt,
        beta=1.0,
        debug=False,
        counter=None,
        print_version=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert isinstance(reg, BaseRegularization) or isinstance(
            reg, BaseObjectiveFunction
        ), "reg must be a Regularization or Objective Function class."

        self.dmisfit = dmisfit

        self.reg = reg
        self.opt = opt
        self.beta = beta
        self.debug = debug
        self.counter = counter
        self.model = None
        self.print_version = print_version
        # TODO: Remove: (and make iteration printers better!)
        self.opt.parent = self
        self.reg.parent = self
        self.dmisfit.parent = self

    #: Trade-off parameter
    @property
    def beta(self):
        """Trade-off parameter

        Returns
        -------
        float
        """
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = validate_float("beta", value, min_val=0.0)

    @property
    def debug(self):
        """Whether to print debugging information.

        Returns
        -------
        bool
        """
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = validate_type("debug", value, bool)

    @property
    def counter(self):
        """Set this to a `simpeg.utils.Counter` if you want to count things.

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
    def dmisfit(self):
        """The data misfit.

        Returns
        -------
        simpeg.objective_function.ComboObjectiveFunction
        """
        return self._dmisfit

    @dmisfit.setter
    def dmisfit(self, value):
        value = validate_type("dmisfit", value, BaseObjectiveFunction, cast=False)
        if not isinstance(value, ComboObjectiveFunction):
            value = ComboObjectiveFunction(objfcts=[value])
        self._dmisfit = value

    @property
    def reg(self):
        """The regularization object for the inversion

        Returns
        -------
        simpeg.objective_function.ComboObjectiveFunction
        """
        return self._reg

    @reg.setter
    def reg(self, value):
        value = validate_type("reg", value, BaseObjectiveFunction, cast=False)
        if not isinstance(value, ComboObjectiveFunction):
            value = ComboObjectiveFunction(objfcts=[value])
        self._reg = value

    @property
    def opt(self):
        """The optimization routine.

        Returns
        -------
        simpeg.optimization.Minimize
        """
        return self._opt

    @opt.setter
    def opt(self, value):
        self._opt = validate_type("opt", value, Minimize, cast=False)

    @property
    def deleteTheseOnModelUpdate(self):
        """A list of properties stored on this object to delete when the model is updated

        Returns
        -------
        list of str
            For example `['_MeSigma', '_MeSigmaI']`.
        """
        return []

    @property
    def model(self):
        """The inversion model.

        Returns
        -------
        numpy.ndarray
        """
        return self._model

    @model.setter
    def model(self, value):
        if value is not None:
            value = validate_ndarray_with_shape(
                "model", value, shape=[("*",), ("*", "*")], dtype=None
            )
        for prop in self.deleteTheseOnModelUpdate:
            if hasattr(self, prop):
                delattr(self, prop)
        self._model = value

    @call_hooks("startup")
    def startup(self, m0):
        """startup(m0)

        Called when inversion is first starting.
        """
        if self.debug:
            print("Calling InvProblem.startup")

        if self.print_version:
            print(f"\nRunning inversion with SimPEG v{simpeg_version}")

        for fct in self.reg.objfcts:
            if (
                hasattr(fct, "reference_model")
                and getattr(fct, "reference_model", None) is None
            ):
                print(
                    "simpeg.InvProblem will set Regularization.reference_model to m0."
                )
                fct.reference_model = m0

        self.phi_d = np.nan
        self.phi_m = np.nan

        self.model = m0

        solver = DefaultSolver
        set_default = True
        for objfct in self.dmisfit.objfcts:
            if (
                isinstance(objfct, BaseDataMisfit)
                and getattr(objfct.simulation, "solver", None) is not None
            ):
                solver = objfct.simulation.solver
                solver_opts = objfct.simulation.solver_opts
                print(
                    """
                        simpeg.InvProblem is setting bfgsH0 to the inverse of the eval2Deriv.
                        ***Done using same Solver, and solver_opts as the {} problem***
                        """.format(
                        objfct.simulation.__class__.__name__
                    )
                )
                set_default = False
                break
        if set_default:
            print(
                """
                    simpeg.InvProblem is setting bfgsH0 to the inverse of the eval2Deriv.
                    ***Done using the default solver {} and no solver_opts.***
                    """.format(
                    DefaultSolver.__name__
                )
            )
            solver = DefaultSolver
            solver_opts = {}

        self.opt.bfgsH0 = solver(
            sp.csr_matrix(self.reg.deriv2(self.model)), **solver_opts
        )

    @property
    def warmstart(self):
        return getattr(self, "_warmstart", [])

    @warmstart.setter
    def warmstart(self, value):
        assert type(value) is list, "warmstart must be a list."
        for v in value:
            assert type(v) is tuple, "warmstart must be a list of tuples (m, u)."
            assert (
                len(v) == 2
            ), "warmstart must be a list of tuples (m, u). YOURS IS NOT LENGTH 2!"
            assert isinstance(
                v[0], np.ndarray
            ), "first warmstart value must be a model."
        self._warmstart = value

    def getFields(self, m, store=False, deleteWarmstart=True):
        f = None

        for mtest, u_ofmtest in self.warmstart:
            if m is mtest:
                f = u_ofmtest
                if self.debug:
                    print("InvProb is Warm Starting!")
                break

        if f is None:
            if isinstance(self.dmisfit, BaseDataMisfit):
                f = self.dmisfit.simulation.fields(m)

            elif isinstance(self.dmisfit, BaseObjectiveFunction):
                f = []
                for objfct in self.dmisfit.objfcts:
                    if hasattr(objfct, "simulation"):
                        f += [objfct.simulation.fields(m)]
                    else:
                        f += []

        if deleteWarmstart:
            self.warmstart = []
        if store:
            self.warmstart += [(m, f)]

        return f

    def get_dpred(self, m, f):
        dpred = []
        for i, objfct in enumerate(self.dmisfit.objfcts):
            if hasattr(objfct, "simulation"):
                dpred += [objfct.simulation.dpred(m, f=f[i])]
            else:
                dpred += []
        return np.hstack(dpred)

    @timeIt
    def evalFunction(self, m, return_g=True, return_H=True):
        """evalFunction(m, return_g=True, return_H=True)"""

        self.model = m
        gc.collect()

        # Store fields if doing a line-search
        f = self.getFields(m, store=(return_g is False and return_H is False))

        # if isinstance(self.dmisfit, BaseDataMisfit):
        phi_d = self.dmisfit(m, f=f)
        self.dpred = self.get_dpred(m, f=f)

        phi_m = self.reg(m)

        self.phi_d, self.phi_d_last = phi_d, self.phi_d
        self.phi_m, self.phi_m_last = phi_m, self.phi_m

        # Only works for WeightedLeastSquares regularization
        if self.opt.print_type == "ubc":
            self.phi_s = 0.0
            self.phi_x = 0.0
            self.phi_y = 0.0
            self.phi_z = 0.0

            if not isinstance(self.reg, WeightedLeastSquares):
                regs = self.reg.objfcts
                mults = self.reg.multipliers
            else:
                regs = [self.reg]
                mults = [1.0]
            for reg, mult in zip(regs, mults):
                if isinstance(reg, Sparse):
                    i_s, i_x, i_y, i_z = 0, 1, 2, 3
                else:
                    i_s, i_x, i_y, i_z = 0, 1, 3, 5
                dim = reg.regularization_mesh.dim
                self.phi_s += mult * reg.objfcts[i_s](m) * reg.alpha_s
                self.phi_x += mult * reg.objfcts[i_x](m) * reg.alpha_x
                if dim > 1:
                    self.phi_z += mult * reg.objfcts[i_y](m) * reg.alpha_y
                if dim > 2:
                    self.phi_y = self.phi_z
                    self.phi_z += mult * reg.objfcts[i_z](m) * reg.alpha_z

        phi = phi_d + self.beta * phi_m

        out = (phi,)
        if return_g:
            phi_dDeriv = self.dmisfit.deriv(m, f=f)
            phi_mDeriv = self.reg.deriv(m)

            g = phi_dDeriv + self.beta * phi_mDeriv
            out += (g,)

        if return_H:

            def H_fun(v):
                phi_d2Deriv = self.dmisfit.deriv2(m, v, f=f)
                phi_m2Deriv = self.reg.deriv2(m, v=v)

                return phi_d2Deriv + self.beta * phi_m2Deriv

            H = sp.linalg.LinearOperator((m.size, m.size), H_fun, dtype=m.dtype)
            out += (H,)
        return out if len(out) > 1 else out[0]
