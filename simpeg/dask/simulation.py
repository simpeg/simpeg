from ..simulation import BaseSimulation as Sim

from dask import array
from dask.distributed import get_client
import numpy as np
from multiprocessing import cpu_count

Sim._delete_on_model_update = ["_Jmatrix", "_jtjdiag", "_stashed_fields"]
Sim.sensitivity_path = "./sensitivity/"
Sim._max_ram = 16
Sim._max_chunk_size = 128


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


def getJtJdiag(self, m, W=None, f=None):
    """
    Return the diagonal of JtJ
    """
    if getattr(self, "_jtjdiag", None) is None:
        self.model = m
        if W is None:
            W = np.ones(self.Jmatrix.shape[0])
        else:
            W = W.diagonal()

        self._jtj_diag = np.asarray(
            np.einsum("i,ij,ij->j", W**2, self.Jmatrix, self.Jmatrix)
        )

    return self._jtj_diag


Sim.getJtJdiag = getJtJdiag


# def __init__(
#     self,
#     survey=None,
#     sensitivity_path="./sensitivity/",
#     counter=None,
#     verbose=False,
#     chunk_format="row",
#     max_ram=16,
#     max_chunk_size=128,
#     **kwargs,
# ):
#     _old_init(
#         self,
#         survey=survey,
#         sensitivity_path=sensitivity_path,
#         counter=counter,
#         verbose=verbose,
#         **kwargs,
#     )
#     self.chunk_format = chunk_format
#     self.max_ram = max_ram
#     self.max_chunk_size = max_chunk_size


def Jvec(self, m, v, **_):
    """
    Compute sensitivity matrix (J) and vector (v) product.
    """
    self.model = m

    if isinstance(self.Jmatrix, np.ndarray):
        return self.Jmatrix @ v.astype(np.float32)

    return array.dot(self.Jmatrix, v.astype(np.float32))


Sim.Jvec = Jvec


def Jtvec(self, m, v, **_):
    """
    Compute adjoint sensitivity matrix (J^T) and vector (v) product.
    """
    self.model = m

    if isinstance(self.Jmatrix, np.ndarray):
        return self.Jmatrix.T @ v.astype(np.float32)

    return array.dot(v.astype(np.float32), self.Jmatrix)


Sim.Jtvec = Jtvec


@property
def Jmatrix(self):
    """
    Sensitivity matrix stored on disk
    Return the diagonal of JtJ
    """
    if getattr(self, "_Jmatrix", None) is None:
        self._Jmatrix = self.compute_J(self.model)

    return self._Jmatrix


Sim.Jmatrix = Jmatrix


def n_threads(self, client=None, worker=None):
    """
    Number of threads used by Dask
    """
    if getattr(self, "_n_threads", None) is None:
        if client:
            n_threads = client.nthreads()
            if not worker:
                self._n_threads = list(n_threads.values())[0]
            else:
                self._n_threads = n_threads[self.worker[0]]
        else:
            self._n_threads = cpu_count()

    return self._n_threads


Sim.n_threads = n_threads


def _get_client_worker(self) -> tuple:
    """
    Get the Dask client and worker if they exist.
    """
    try:
        client = get_client()
    except ValueError:
        client = None

    try:
        worker = self.worker
    except AttributeError:
        worker = None

    return client, worker


Sim._get_client_worker = _get_client_worker
