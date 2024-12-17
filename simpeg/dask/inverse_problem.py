from ..inverse_problem import BaseInvProblem
import numpy as np

from dask.distributed import get_client, Future
from scipy.sparse.linalg import LinearOperator
from ..regularization import WeightedLeastSquares, Sparse

from ..objective_function import ComboObjectiveFunction


def get_dpred(self, m, f=None):
    dpreds = []

    for objfct in self.dmisfit.objfcts:
        dpred = objfct.simulation.dpred(m)
        dpreds += [dpred]

    if isinstance(dpreds[0], Future):
        client = get_client()
        dpreds = client.gather(dpreds)
    else:
        for i, dpred in enumerate(dpreds):
            dpreds[i] = np.asarray(dpred)

    return dpreds


BaseInvProblem.get_dpred = get_dpred


def dask_evalFunction(self, m, return_g=True, return_H=True):
    """evalFunction(m, return_g=True, return_H=True)"""
    self.model = m

    self.dpred = self.get_dpred(m)

    phi_d = 0
    for (_, objfct), pred in zip(self.dmisfit, self.dpred):
        residual = objfct.W * (objfct.data.dobs - pred)
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
