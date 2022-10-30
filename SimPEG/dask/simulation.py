from ..simulation import BaseSimulation as Sim
from dask.distributed import get_client, Future, Client
from dask import array, delayed
import warnings
from ..data import SyntheticData
import numpy as np
from .utils import compute

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

    dpred = self.dpred(m, f=f)

    if not isinstance(dpred, np.ndarray):
        dpred = compute(self, dpred)
        if isinstance(dpred, Future):
            client = get_client()
            dpred = client.gather(dpred)

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

    return array.dot(self.Jmatrix, v.astype(np.float32))


Sim.Jvec = dask_Jvec


def dask_Jtvec(self, m, v):
    """
        Compute adjoint sensitivity matrix (J^T) and vector (v) product.
    """
    self.model = m
    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    return array.dot(v.astype(np.float32), self.Jmatrix)


Sim.Jtvec = dask_Jtvec


def dask_getJtJdiag(self, m, W=None):
    """
        Return the diagonal of JtJ
    """
    self.model = m
    if self.gtgdiag is None:
        if isinstance(self.Jmatrix, Future):
            self.Jmatrix  # Wait to finish

        if W is None:
            W = np.ones(self.nD)
        else:
            W = W.diagonal()

        diag = array.einsum('i,ij,ij->j', W, self.Jmatrix, self.Jmatrix)

        if isinstance(diag, array.Array):
            diag = np.asarray(diag.compute())

        self.gtgdiag = diag
    return self.gtgdiag


Sim.getJtJdiag = dask_getJtJdiag


@property
def Jmatrix(self):
    """
    Sensitivity matrix stored on disk
    """
    if getattr(self, "_Jmatrix", None) is None:
        if self.workers is None:
            self._Jmatrix = self.compute_J()
        else:
            try:
                client = get_client()
            except ValueError:
                client = Client()

            self._Jmatrix = client.compute(
                    delayed(self.compute_J)()
            )
    elif isinstance(self._Jmatrix, Future):
        # client = get_client()
        self._Jmatrix.result()
        self._Jmatrix = array.from_zarr(self.sensitivity_path + f"J.zarr")

    return self._Jmatrix


Sim.Jmatrix = Jmatrix

