from ..simulation import BaseSimulation as Sim
from dask.distributed import get_client, Client

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
