from .....electromagnetics.static.induced_polarization.simulation import (
    BaseIPSimulation as Sim,
)
from .....utils import Zero, mkvc
from .....data import Data
from ....utils import compute_chunk_sizes
import dask
import dask.array as da
from dask.distributed import Future
import numpy as np
import zarr
import os
import shutil
import numcodecs

numcodecs.blosc.use_threads = False

Sim.sensitivity_path = './sensitivity/'

from ..resistivity.simulation import (
    dask_fields, dask_getJtJdiag, dask_Jvec, dask_Jtvec,
    compute_J, dask_getSourceTerm,
)

Sim.fields = dask_fields
Sim.getJtJdiag = dask_getJtJdiag
Sim.Jvec = dask_Jvec
Sim.Jtvec = dask_Jtvec
Sim.compute_J = compute_J
Sim.getSourceTerm = dask_getSourceTerm


@dask.delayed
def dask_dpred(self, m=None, f=None, compute_J=False):
    """
    dpred(m, f=None)
    Create the projected data from a model.
    The fields, f, (if provided) will be used for the predicted data
    instead of recalculating the fields (which may be expensive!).

    .. math::

        d_\\text{pred} = P(f(m))

    Where P is a projection of the fields onto the data space.
    """
    if self.survey is None:
        raise AttributeError(
            "The survey has not yet been set and is required to compute "
            "data. Please set the survey for the simulation: "
            "simulation.survey = survey"
        )

    if f is None:
        if m is None:
            m = self.model
        f, Ainv = self.fields(m, return_Ainv=compute_J)

    if self._scale is None:
        scale = Data(self.survey, np.full(self.survey.nD, self._sign))
        # loop through receievers to check if they need to set the _dc_voltage
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                if (
                        rx.data_type == "apparent_chargeability"
                        or self._data_type == "apparent_chargeability"
                ):
                    scale[src, rx] = self._sign / rx.eval(src, self.mesh, f)
        self._scale = scale.dobs

    if compute_J:
        Jmatrix = self.compute_J(f=f, Ainv=Ainv)

    data = self._scale * self.Jvec(m, m)

    if compute_J:
        return (np.asarray(data), Jmatrix)

    return np.asarray(data)


Sim.dpred = dask_dpred

#
# if m is not None:
#     self.model = m
#     # sensitivity matrix is fixed
#     # self._Jmatrix = None
#
# if self._f is None:
#     if self.verbose is True:
#         print(">> Solve DC problem")
#     self._f = super().fields(m=None)
#
# if self._scale is None:
#     scale = Data(self.survey, np.full(self.survey.nD, self._sign))
#     # loop through receievers to check if they need to set the _dc_voltage
#     for src in self.survey.source_list:
#         for rx in src.receiver_list:
#             if (
#                     rx.data_type == "apparent_chargeability"
#                     or self._data_type == "apparent_chargeability"
#             ):
#                 scale[src, rx] = self._sign / rx.eval(src, self.mesh, self._f)
#     self._scale = scale.dobs
#
# if self.verbose is True:
#     print(">> Compute predicted data")
#
# self._pred = self.forward(m, f=self._f)
#
# # if not self.storeJ:
# #     self.Ainv.clean()
#
# return self._f