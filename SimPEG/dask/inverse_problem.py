from ..inverse_problem import BaseInvProblem
import numpy as np

from dask import delayed
from dask.distributed import Future, get_client
import dask.array as da
import gc
from ..regularization import BaseComboRegularization, Sparse
from ..data_misfit import BaseDataMisfit
from ..objective_function import BaseObjectiveFunction


def dask_getFields(self, m, store=False, deleteWarmstart=True):
    f = None

    try:
        client = get_client()
        fields = lambda f, x, workers: client.compute(f(x), workers=workers)
    except:
        fields = lambda f, x: f(x)

    for mtest, u_ofmtest in self.warmstart:
        if m is mtest:
            f = u_ofmtest
            if self.debug:
                print("InvProb is Warm Starting!")
            break

    if f is None:
        if isinstance(self.dmisfit, BaseDataMisfit):

            if self.dmisfit.model_map is not None:
                vec = self.dmisfit.model_map @ m
            else:
                vec = m

            f = fields(self.dmisfit.simulation.fields, vec)

        elif isinstance(self.dmisfit, BaseObjectiveFunction):
            f = []
            for objfct in self.dmisfit.objfcts:
                if hasattr(objfct, "simulation"):
                    if objfct.model_map is not None:
                        vec = objfct.model_map @ m
                    else:
                        vec = m

                    f += [fields(objfct.simulation.fields, vec, objfct.workers)]
                else:
                    f += []

    if isinstance(f, Future) or isinstance(f[0], Future):
        f = client.gather(f)

    if deleteWarmstart:
        self.warmstart = []
    if store:
        self.warmstart += [(m, f)]

    return f


BaseInvProblem.getFields = dask_getFields


# def dask_formJ(self, m):
#     j = None
#
#     try:
#         client = get_client()
#         jsub = lambda f, x, fields: client.compute(f(x), fields=None)
#     except:
#         jsub = lambda f, x: f(x)
#
#     if j is None:
#         if isinstance(self.dmisfit, BaseDataMisfit):
#             j = jsub(self.dmisfit.simulation.getJ, m)
#
#         elif isinstance(self.dmisfit, BaseObjectiveFunction):
#             j = []
#             for objfct in self.dmisfit.objfcts:
#                 if hasattr(objfct, "simulation"):
#                     j += [jsub(objfct.simulation.getJ, m, None)]
#                 else:
#                     j += []
#
#     if isinstance(j, Future) or isinstance(j[0], Future):
#         j = client.gather(j)
#
#     return da.vstack(j).compute()


# BaseInvProblem.formJ = dask_formJ


def get_dpred(self, m, f=None, compute_J=False):
    dpreds = []
    client = get_client()

    if isinstance(self.dmisfit, BaseDataMisfit):
        return self.dmisfit.simulation.dpred(m)
    elif isinstance(self.dmisfit, BaseObjectiveFunction):

        for i, objfct in enumerate(self.dmisfit.objfcts):
            if hasattr(objfct, "simulation"):
                if objfct.model_map is not None:
                    vec = objfct.model_map @ m
                else:
                    vec = m

                future = client.compute(
                    objfct.simulation.dpred(
                        vec, compute_J=compute_J and (objfct.simulation._Jmatrix is None)
                    ), workers=objfct.workers
                )
                dpreds += [future]

            else:
                dpreds += []

    if isinstance(dpreds[0], Future):
        dpreds = client.gather(dpreds)
        preds = []
        if isinstance(dpreds[0], tuple):  # Jmatrix was computed
            for future, objfct in zip(dpreds, self.dmisfit.objfcts):
                preds += [future[0]]
                objfct.simulation._Jmatrix = future[1]

            return preds
    return dpreds


BaseInvProblem.get_dpred = get_dpred


def dask_evalFunction(self, m, return_g=True, return_H=True):
    """evalFunction(m, return_g=True, return_H=True)
    """

    self.model = m
    # gc.collect()

    # Store fields if doing a line-search
    # f = self.getFields(m, store=(return_g is False and return_H is False))

    # if isinstance(self.dmisfit, BaseDataMisfit):
    # phi_d = np.asarray(self.dmisfit(m, f=f))
    self.dpred = self.get_dpred(m, compute_J=return_H)

    phi_d = 0
    for objfct, pred in zip(self.dmisfit.objfcts, self.dpred):
        residual = objfct.W * (objfct.data.dobs - pred)
        phi_d += 0.5 * np.vdot(residual, residual)

    phi_d = np.asarray(phi_d)
    # print(self.dpred[0])
    self.reg2Deriv = self.reg.deriv2(m)
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

    # phi = phi_d + self.beta * phi_m

    out = (phi,)
    if return_g:
        phi_dDeriv = self.dmisfit.deriv(m, f=self.dpred)
        if hasattr(self.reg.objfcts[0], "space") and self.reg.objfcts[0].space == "spherical":
            phi_mDeriv = self.reg2Deriv * self.reg._delta_m(m)
        else:
            phi_mDeriv = self.reg.deriv(m)

        g = np.asarray(phi_dDeriv) + self.beta * phi_mDeriv
        out += (g,)

    if return_H:

        def H_fun(v):
            phi_d2Deriv = self.dmisfit.deriv2(m, v)
            if hasattr(self.reg.objfcts[0], "space") and self.reg.objfcts[0].space == "spherical":
                phi_m2Deriv = self.reg2Deriv * v
            else:
                phi_m2Deriv = self.reg.deriv2(m, v=v)

            H = phi_d2Deriv + self.beta * phi_m2Deriv

            return H

        H = H_fun
        out += (H,)
    return out if len(out) > 1 else out[0]


BaseInvProblem.evalFunction = dask_evalFunction
