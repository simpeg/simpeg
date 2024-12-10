from ..inverse_problem import BaseInvProblem
import numpy as np


from dask.distributed import Future, get_client
from scipy.sparse.linalg import LinearOperator
from ..regularization import WeightedLeastSquares, Sparse

from ..objective_function import ComboObjectiveFunction


def get_dpred(self, m, f=None, compute_J=False):
    dpreds = []

    for i, objfct in enumerate(self.dmisfit.objfcts):

        if compute_J and i == 0:
            print("Computing forward & sensitivities")

        if f is not None:
            fields = f[i]
        else:
            fields = objfct.simulation.fields(m)

        future = objfct.simulation.dpred(m, f=fields)

        if compute_J:
            objfct.simulation.compute_J(m, f=fields)

        dpreds += [future]

    if isinstance(dpreds[0], Future):
        client = get_client()
        dpreds = client.gather(dpreds)

    return dpreds


BaseInvProblem.get_dpred = get_dpred


def dask_evalFunction(self, m, return_g=True, return_H=True):
    """evalFunction(m, return_g=True, return_H=True)"""
    self.model = m

    # Store fields if doing a line-search
    fields = self.getFields(m, store=(return_g is False and return_H is False))

    # if isinstance(self.dmisfit, BaseDataMisfit):
    phi_d = self.dmisfit(m, f=fields)
    self.dpred = self.get_dpred(m, f=fields, compute_J=return_H)

    phi_d = 0
    for (_, objfct), pred in zip(self.dmisfit, self.dpred):
        residual = objfct.W * (objfct.data.dobs - pred)
        phi_d += np.vdot(residual, residual)

    phi_d = np.asarray(phi_d)
    # print(self.dpred[0])

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
        phi_dDeriv = self.dmisfit.deriv(m, f=fields)
        # if hasattr(self.reg.objfcts[0], "space") and self.reg.objfcts[0].space == "spherical":
        phi_mDeriv = self.reg.deriv(m)
        # else:
        #     phi_mDeriv = np.sum([reg2Deriv * obj.f_m for reg2Deriv, obj in zip(self.reg2Deriv, self.reg.objfcts)], axis=0)

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
