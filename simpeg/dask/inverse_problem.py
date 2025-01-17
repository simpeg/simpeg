from ..inverse_problem import BaseInvProblem
import numpy as np

from .objective_function import DaskComboMisfits
from scipy.sparse.linalg import LinearOperator
from ..regularization import WeightedLeastSquares, Sparse
from ..objective_function import ComboObjectiveFunction
from simpeg.utils import call_hooks
from simpeg.version import __version__ as simpeg_version


def get_dpred(self, m, f=None):
    dpreds = []

    if isinstance(self.dmisfit, DaskComboMisfits):
        return self.dmisfit.get_dpred(m, f=f)

    for objfct in self.dmisfit.objfcts:
        dpred = objfct.simulation.dpred(m, f=f)
        dpreds += [np.asarray(dpred)]

    return dpreds


BaseInvProblem.get_dpred = get_dpred


def dask_evalFunction(self, m, return_g=True, return_H=True):
    """evalFunction(m, return_g=True, return_H=True)"""
    self.model = m
    self.dpred = self.get_dpred(m)
    residuals = []

    if isinstance(self.dmisfit, DaskComboMisfits):
        residuals = self.dmisfit.residuals(m)
    else:
        for (_, objfct), pred in zip(self.dmisfit, self.dpred):
            residuals.append(objfct.W * (objfct.data.dobs - pred))

    phi_d = 0.0
    for residual in residuals:
        phi_d += np.vdot(residual, residual)

    reg2Deriv = []
    if isinstance(self.reg, ComboObjectiveFunction):
        for constant, objfct in self.reg:
            if isinstance(objfct, ComboObjectiveFunction):
                reg2Deriv += [constant * multi * obj.deriv2(m) for multi, obj in objfct]
            else:
                reg2Deriv += [constant * objfct.deriv2(m)]
    else:
        reg2Deriv = [self.reg.deriv2(m)]

    self.reg2Deriv = np.sum(reg2Deriv)

    # reg = np.linalg.norm(self.reg2Deriv * self.reg._delta_m(m))
    phi_m = self.reg(m)

    self.phi_d, self.phi_d_last = phi_d, self.phi_d
    self.phi_m, self.phi_m_last = phi_m, self.phi_m

    phi = phi_d + self.beta * phi_m

    # Only works for Tikhonov
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
            dim = reg.regmesh.dim
            self.phi_s += mult * reg.objfcts[i_s](m) * reg.alpha_s
            self.phi_x += mult * reg.objfcts[i_x](m) * reg.alpha_x
            if dim > 1:
                self.phi_z += mult * reg.objfcts[i_y](m) * reg.alpha_y
            if dim > 2:
                self.phi_y = self.phi_z
                self.phi_z += mult * reg.objfcts[i_z](m) * reg.alpha_z

    # phi = phi_d + self.beta * phi_m

    out = (phi,)
    if return_g:
        phi_dDeriv = self.dmisfit.deriv(m)
        phi_mDeriv = self.reg.deriv(m)

        g = np.asarray(phi_dDeriv) + self.beta * phi_mDeriv
        out += (g,)

    if return_H:

        def H_fun(v):
            phi_d2Deriv = self.dmisfit.deriv2(m, v)
            phi_m2Deriv = self.reg2Deriv * v
            H = phi_d2Deriv + self.beta * phi_m2Deriv

            return H

        H = LinearOperator((m.size, m.size), H_fun, dtype=m.dtype)
        out += (H,)
    return out if len(out) > 1 else out[0]


BaseInvProblem.evalFunction = dask_evalFunction


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
            print("simpeg.InvProblem will set Regularization.reference_model to m0.")
            fct.reference_model = m0

    self.phi_d = np.nan
    self.phi_m = np.nan

    self.model = m0


BaseInvProblem.startup = startup
