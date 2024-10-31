from ..inverse_problem import BaseInvProblem
import numpy as np
from time import time
from datetime import timedelta
from dask.distributed import Future, get_client
import dask.array as da
from scipy.sparse.linalg import LinearOperator
from ..regularization import WeightedLeastSquares, Sparse
from ..data_misfit import BaseDataMisfit
from ..objective_function import BaseObjectiveFunction, ComboObjectiveFunction


def dask_getFields(self, m, store=False, deleteWarmstart=True):
    f = None

    # try:
    #     client = get_client()
    #     fields = lambda f, x, workers: client.compute(f(x), workers=workers)
    # except:
    #     fields = lambda f, x: f(x)

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


def get_dpred(self, m, f=None, compute_J=False):
    dpreds = []

    if isinstance(self.dmisfit, BaseDataMisfit):
        return self.dmisfit.simulation.dpred(m)
    elif isinstance(self.dmisfit, BaseObjectiveFunction):
        for i, objfct in enumerate(self.dmisfit.objfcts):
            if hasattr(objfct, "simulation"):
                if getattr(objfct, "model_map", None) is not None:
                    vec = objfct.model_map @ m
                else:
                    vec = m

                compute_sensitivities = compute_J and (
                    objfct.simulation._Jmatrix is None
                )

                if compute_sensitivities and i == 0:
                    print("Computing forward & sensitivities")

                if objfct.workers is not None:
                    client = get_client()
                    future = client.compute(
                        objfct.simulation.dpred(vec, compute_J=compute_sensitivities),
                        workers=objfct.workers,
                    )
                else:
                    # For locals, the future is now
                    ct = time()

                    future = objfct.simulation.dpred(
                        vec, compute_J=compute_sensitivities
                    )

                    if compute_sensitivities:
                        runtime = time() - ct
                        total = len(self.dmisfit.objfcts)

                        message = f"{i+1} of {total} in {timedelta(seconds=runtime)}. "
                        if (total - i - 1) > 0:
                            message += (
                                f"ETA -> {timedelta(seconds=(total - i - 1) * runtime)}"
                            )
                        print(message)

                dpreds += [future]

            else:
                dpreds += []

    if isinstance(dpreds[0], Future):
        client = get_client()
        dpreds = client.gather(dpreds)

    preds = []
    if isinstance(dpreds[0], tuple):  # Jmatrix was computed
        for future, objfct in zip(dpreds, self.dmisfit.objfcts):
            preds += [future[0]]
            objfct.simulation._Jmatrix = future[1]
        return preds

    else:
        dpreds = da.compute(dpreds)[0]
    return dpreds


BaseInvProblem.get_dpred = get_dpred


def dask_evalFunction(self, m, return_g=True, return_H=True):
    """evalFunction(m, return_g=True, return_H=True)"""
    self.model = m
    self.dpred = self.get_dpred(m, compute_J=return_H)

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
        phi_dDeriv = self.dmisfit.deriv(m, f=self.dpred)
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
