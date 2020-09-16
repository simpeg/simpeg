from ..data_misfit import L2DataMisfit as Dmis

import dask
import dask.array as da
import os
import shutil
import numpy as np


def dask_deriv(self, m, f=None):
    """
    deriv(m, f=None)
    Derivative of the data misfit

    .. math::

        \mathbf{J}^{\top} \mathbf{W}^{\top} \mathbf{W}
        (\mathbf{d} - \mathbf{d}^{obs})

    :param numpy.ndarray m: model
    :param SimPEG.fields.Fields f: fields object
    """

    if f is None:
        f = self.simulation.fields(m)

    return self.simulation.Jtvec(
        m, self.W.T * (self.W * self.residual(m, f=f)), f=f
    )

Dmis.deriv = dask_deriv


def dask_deriv2(self, m, v, f=None):
    """
    deriv2(m, v, f=None)

    .. math::

        \mathbf{J}^{\top} \mathbf{W}^{\top} \mathbf{W} \mathbf{J}

    :param numpy.ndarray m: model
    :param numpy.ndarray v: vector
    :param SimPEG.fields.Fields f: fields object
    """

    if f is None:
        f = self.simulation.fields(m)

    return self.simulation.Jtvec_approx(
        m, self.W * (self.W * self.simulation.Jvec_approx(m, v, f=f)), f=f
    )

Dmis.deriv = dask_deriv2
