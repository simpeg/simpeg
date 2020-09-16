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
