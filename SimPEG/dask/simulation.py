from ..simulation import BaseSimulation as Sim
from dask.distributed import get_client, Client
from dask.delayed import Delayed
import warnings
from ..data import SyntheticData
import numpy as np

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

    client = get_client()
    dclean = client.compute(self.dpred(m, f=f), workers=self.workers).result()

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
