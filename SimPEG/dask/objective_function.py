from ..objective_function import ComboObjectiveFunction as Cobjfct
import dask
import dask.array as da
import os
import shutil
import numpy as np


def dask_call(self, m, f=None):
    fct = []
    for i, phi in enumerate(self):
        multiplier, objfct = phi
        if multiplier == 0.0:  # don't evaluate the fct
            continue
        else:
            if f is not None and objfct._hasFields:
                fct += [multiplier * objfct(m, f=f[i])]
            else:
                fct += [multiplier * objfct(m)]

    stack = da.vstack(fct)

    return da.sum(stack, axis=0).compute()


Cobjfct.__call__ = dask_call


def dask_deriv(self, m, f=None):
    """
    First derivative of the composite objective function is the sum of the
    derivatives of each objective function in the list, weighted by their
    respective multplier.

    :param numpy.ndarray m: model
    :param SimPEG.Fields f: Fields object (if applicable)
    """

    # @dask.delayed
    # def rowSum(arr):
    #     sumIt = 0
    #     for i in range(len(arr)):
    #         sumIt += arr[i]
    #     return sumIt

    g = []
    for i, phi in enumerate(self):
        multiplier, objfct = phi
        if multiplier == 0.0:  # don't evaluate the fct
            continue
        else:
            if f is not None and objfct._hasFields:
                g += [multiplier * objfct.deriv(m, f=f[i])]
            else:
                g += [multiplier * objfct.deriv(m)]

    stack = da.vstack(g)

    return da.sum(stack, axis=0).compute()


Cobjfct.deriv = dask_deriv


def dask_deriv2(self, m, v=None, f=None):
    """
    Second derivative of the composite objective function is the sum of the
    second derivatives of each objective function in the list, weighted by
    their respective multplier.

    :param numpy.ndarray m: model
    :param numpy.ndarray v: vector we are multiplying by
    :param SimPEG.Fields f: Fields object (if applicable)
    """
    # @dask.delayed
    # def rowSum(arr):
    #     sumIt = 0
    #     for i in range(len(arr)):
    #         sumIt += arr[i]
    #     return sumIt

    H = []
    for i, phi in enumerate(self):
        multiplier, objfct = phi
        if multiplier == 0.0:  # don't evaluate the fct
            continue
        else:
            if f is not None and objfct._hasFields:

                H += [multiplier * objfct.deriv2(m, v, f=f[i])]
            else:
                H += [multiplier * objfct.deriv2(m, v)]

    if isinstance(H[0], dask.array.Array):

        stack = da.vstack(H)

        return da.sum(stack, axis=0).compute()

    else:
        sumIt = 0
        for i in range(len(H)):
            sumIt += H[i]
        return sumIt


Cobjfct.deriv2 = dask_deriv2
