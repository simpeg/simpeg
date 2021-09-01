from ..simulation import BaseSimulation as Sim
from dask.distributed import get_client, Future
from dask import array, delayed
from dask.delayed import Delayed
import warnings
from ..data import SyntheticData
import numpy as np
from ..utils import mkvc
from ..data import Data

Sim._max_ram = 16

@property
def max_ram(self):
    "Maximum ram in (Gb)"
    return self._max_ram


@max_ram.setter
def max_ram(self, other):
    if other <= 0:
        raise ValueError("max_ram must be greater than 0")
    self._max_ram = other


Sim.max_ram = max_ram

Sim._max_chunk_size = 128


@property
def max_chunk_size(self):
    "Largest chunk size (Mb) used by Dask"
    return self._max_chunk_size


@max_chunk_size.setter
def max_chunk_size(self, other):
    if other <= 0:
        raise ValueError("max_chunk_size must be greater than 0")
    self._max_chunk_size = other


Sim.max_chunk_size = max_chunk_size


def make_synthetic_data(
        self, m, relative_error=0.05, noise_floor=0.0, f=None, add_noise=False, **kwargs
):
    """
    Make synthetic data given a model, and a standard deviation.
    :param numpy.ndarray m: geophysical model
    :param numpy.ndarray relative_error: standard deviation
    :param numpy.ndarray noise_floor: noise floor
    :param numpy.ndarray f: fields for the given model (if pre-calculated)
    """

    std = kwargs.pop("std", None)
    if std is not None:
        warnings.warn(
            "The std parameter will be deprecated in SimPEG 0.15.0. "
            "Please use relative_error.",
            DeprecationWarning,
        )
        relative_error = std

    # if f is None:
    #     f = self.fields(m)
    #
    #     if isinstance(f, Delayed):
    #         f = f.compute()

    # client = get_client()
    dpred = self.dpred(m, f=f)
    if isinstance(dpred, Delayed):
        client = get_client()
        dclean = client.compute(dpred, workers=self.workers).result()
    else:
        dclean = np.asarray(dpred)

    if add_noise is True:
        std = relative_error * abs(dclean) + noise_floor
        noise = std * np.random.randn(*dclean.shape)
        dobs = dclean + noise
    else:
        dobs = dclean

    return SyntheticData(
        survey=self.survey,
        dobs=dobs,
        dclean=dclean,
        relative_error=relative_error,
        noise_floor=noise_floor,
    )

Sim.make_synthetic_data = make_synthetic_data
# @property
# def client(self):
#     if getattr(self, '_client', None) is None:
#         self._client = get_client()
#
#     return self._client
#
#
# @client.setter
# def client(self, client):
#     assert isinstance(client, Client)
#     self._client = client
#
#
# Sim.client = client
#

@property
def workers(self):
    if getattr(self, '_workers', None) is None:
        self._workers = None

    return self._workers


@workers.setter
def workers(self, workers):
    self._workers = workers


Sim.workers = workers


def dask_Jvec(self, m, v):
    """
        Compute sensitivity matrix (J) and vector (v) product.
    """
    self.model = m
    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    return array.dot(self.Jmatrix, v)


Sim.Jvec = dask_Jvec


def dask_Jtvec(self, m, v):
    """
        Compute adjoint sensitivity matrix (J^T) and vector (v) product.
    """
    self.model = m
    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    return array.dot(v, self.Jmatrix)


Sim.Jtvec = dask_Jtvec


@property
def Jmatrix(self):
    """
    Sensitivity matrix stored on disk
    """
    if getattr(self, "_Jmatrix", None) is None:
        client = get_client()
        self._Jmatrix = client.compute(
                delayed(self.compute_J)(),
            workers=self.workers
        )
    elif isinstance(self._Jmatrix, Future):
        # client = get_client()
        self._Jmatrix.result()
        self._Jmatrix = array.from_zarr(self.sensitivity_path + f"J.zarr")

    return self._Jmatrix


Sim.Jmatrix = Jmatrix


@delayed
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

    data = Data(self.survey)
    for src in self.survey.source_list:
        for rx in src.receiver_list:
            data[src, rx] = rx.eval(src, self.mesh, f)

    if compute_J:
        Jmatrix = self.compute_J(f=f, Ainv=Ainv)
        return (mkvc(data), Jmatrix)

    return mkvc(data)


Sim.dpred = dask_dpred

