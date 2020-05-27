from __future__ import print_function

import properties
import numpy as np
import scipy.sparse as sp
import gc
from .data_misfit import BaseDataMisfit
from .props import BaseSimPEG, Model
from .regularization import BaseRegularization, BaseComboRegularization, Sparse
from .objective_function import BaseObjectiveFunction, ComboObjectiveFunction
from .utils import callHooks, timeIt


class BaseInvProblem(BaseSimPEG):
    """BaseInvProblem(dmisfit, reg, opt)"""

    #: Trade-off parameter
    beta = 1.0

    #: Print debugging information
    debug = False

    #: Set this to a SimPEG.utils.Counter() if you want to count things
    counter = None

    #: DataMisfit
    dmisfit = None

    #: Regularization
    reg = None

    #: Optimization program
    opt = None

    #: List of strings, e.g. ['_MeSigma', '_MeSigmaI']
    deleteTheseOnModelUpdate = []

    model = Model("Inversion model.")

    @properties.observer("model")
    def _on_model_update(self, value):
        """
            Sets the current model, and removes dependent properties
        """
        for prop in self.deleteTheseOnModelUpdate:
            if hasattr(self, prop):
                delattr(self, prop)

    def __init__(self, dmisfit, reg, opt, **kwargs):
        super(BaseInvProblem, self).__init__(**kwargs)
        assert isinstance(dmisfit, BaseDataMisfit) or isinstance(
            dmisfit, BaseObjectiveFunction
        ), "dmisfit must be a DataMisfit or ObjectiveFunction class."
        assert isinstance(reg, BaseRegularization) or isinstance(
            reg, BaseObjectiveFunction
        ), "reg must be a Regularization or Objective Function class."
        self.dmisfit = dmisfit
        self.reg = reg
        self.opt = opt
        # TODO: Remove: (and make iteration printers better!)
        self.opt.parent = self
        self.reg.parent = self
        self.dmisfit.parent = self

    @callHooks("startup")
    def startup(self, m0):
        """startup(m0)

            Called when inversion is first starting.
        """
        if self.debug:
            print("Calling InvProblem.startup")

        if hasattr(self.reg, "mref") and getattr(self.reg, "mref", None) is None:
            print("SimPEG.InvProblem will set Regularization.mref to m0.")
            self.reg.mref = m0

        if isinstance(self.reg, ComboObjectiveFunction) and not isinstance(
            self.reg, BaseComboRegularization
        ):
            for fct in self.reg.objfcts:
                if hasattr(fct, "mref") and getattr(fct, "mref", None) is None:
                    print("SimPEG.InvProblem will set Regularization.mref to m0.")
                    fct.mref = m0

        self.phi_d = np.nan
        self.phi_m = np.nan

        self.model = m0

        if isinstance(self.dmisfit, BaseDataMisfit):
            if getattr(self.dmisfit.simulation, "solver", None) is not None:
                print(
                    """
        SimPEG.InvProblem is setting bfgsH0 to the inverse of the eval2Deriv.
        ***Done using same Solver and solverOpts as the problem***"""
                )
                self.opt.bfgsH0 = self.dmisfit.simulation.solver(
                    self.reg.deriv2(self.model), **self.dmisfit.simulation.solver_opts
                )
        elif isinstance(self.dmisfit, BaseObjectiveFunction):
            for objfct in self.dmisfit.objfcts:
                if isinstance(objfct, BaseDataMisfit):
                    if getattr(objfct.simulation, "solver", None) is not None:
                        print(
                            """
        SimPEG.InvProblem is setting bfgsH0 to the inverse of the eval2Deriv.
        ***Done using same Solver and solver_opts as the {} problem***""".format(
                                objfct.simulation.__class__.__name__
                            )
                        )
                        self.opt.bfgsH0 = objfct.simulation.solver(
                            self.reg.deriv2(self.model), **objfct.simulation.solver_opts
                        )
                        break

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
        if isinstance(self.dmisfit, BaseDataMisfit):
            return self.dmisfit.simulation.dpred(m, f=f)
        elif isinstance(self.dmisfit, BaseObjectiveFunction):
            dpred = []
            for i, objfct in enumerate(self.dmisfit.objfcts):
                if hasattr(objfct, "survey"):
                    dpred += [objfct.survey.dpred(m, f=f[i])]
                else:
                    dpred += []
            return dpred

    @timeIt
    def evalFunction(self, m, return_g=True, return_H=True):
        """evalFunction(m, return_g=True, return_H=True)
        """

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

        # Only works for Tikhonov
        if self.opt.print_type == "ubc":

            self.phi_s = 0.0
            self.phi_x = 0.0
            self.phi_y = 0.0
            self.phi_z = 0.0

            if not isinstance(self.reg, BaseComboRegularization):
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
                dim = reg.regmesh.dim
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
