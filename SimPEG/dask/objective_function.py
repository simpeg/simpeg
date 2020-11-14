from ..objective_function import ComboObjectiveFunction
import dask
import dask.array as da
import os
import shutil
import numpy as np
from dask.distributed import Future, get_client


def dask_call(self, m, f=None):
    fcts = []
    multipliers = []
    for i, phi in enumerate(self):
        multiplier, objfct = phi
        if multiplier == 0.0:  # don't evaluate the fct
            continue
        else:

            if f is not None and objfct._hasFields:
                fct = objfct(m, f=f[i])
            else:
                fct = objfct(m)

            if isinstance(fct, Future):
                fcts += [fct]
            else:
                future = self.client.compute(
                    self.client.submit(da.multiply, multiplier, fct).result()
                )
                fcts += [future]

            multipliers += [multiplier]

    if isinstance(fcts[0], Future):
        phi = self.client.submit(
            da.sum, self.client.submit(da.vstack, fcts), axis=0
        ).result()
        return phi

    else:
        return np.sum(
            np.r_[multipliers][:, None] * np.vstack(fcts), axis=0
        ).squeeze()


ComboObjectiveFunction.__call__ = dask_call


def dask_deriv(self, m, f=None):
    """
    First derivative of the composite objective function is the sum of the
    derivatives of each objective function in the list, weighted by their
    respective multplier.

    :param numpy.ndarray m: model
    :param SimPEG.Fields f: Fields object (if applicable)
    """

    g = []
    multipliers = []
    for i, phi in enumerate(self):
        multiplier, objfct = phi
        if multiplier == 0.0:  # don't evaluate the fct
            continue
        else:

            if f is not None and objfct._hasFields:
                fct = objfct.deriv(m, f=f[i])
            else:
                fct = objfct.deriv(m)

            if isinstance(fct, Future):
                g += [fct]
            else:
                future = self.client.compute(
                    self.client.submit(da.multiply, multiplier, fct).result()
                )
                g += [future]

            multipliers += [multiplier]

    if isinstance(g[0], Future):
        phi_deriv = self.client.submit(
            da.sum, self.client.submit(da.vstack, g), axis=0
        ).result()
        return phi_deriv

    else:
        return np.sum(
            np.r_[multipliers][:, None] * np.vstack(g), axis=0
        ).squeeze()


ComboObjectiveFunction.deriv = dask_deriv


def dask_deriv2(self, m, v=None, f=None):
    """
    Second derivative of the composite objective function is the sum of the
    second derivatives of each objective function in the list, weighted by
    their respective multplier.

    :param numpy.ndarray m: model
    :param numpy.ndarray v: vector we are multiplying by
    :param SimPEG.Fields f: Fields object (if applicable)
    """

    H = []
    multipliers = []
    for i, phi in enumerate(self):
        multiplier, objfct = phi
        if multiplier == 0.0:  # don't evaluate the fct
            continue
        else:

            if f is not None and objfct._hasFields:
                fct = objfct.deriv2(m, v, f=f[i])
            else:
                fct = objfct.deriv2(m, v)

            if isinstance(fct, Future):
                H += [fct]
            else:
                future = self.client.compute(
                    self.client.submit(da.multiply, multiplier, fct).result()
                )
                H += [future]

            multipliers += [multiplier]

    if isinstance(H[0], Future):
        phi_deriv2 = self.client.submit(
            da.sum, self.client.submit(da.vstack, H), axis=0
        ).result()
        return phi_deriv2

    else:
        return np.sum(
            np.r_[multipliers][:, None] * np.vstack(H), axis=0
        ).squeeze()


ComboObjectiveFunction.deriv2 = dask_deriv2
