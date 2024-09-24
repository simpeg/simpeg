from ..simulation import BaseSimulation as Sim
from dask.distributed import get_client, Future
from dask import array, delayed
import multiprocessing
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


@property
def n_cpu(self):
    """Number of cpu's available."""
    if getattr(self, "_n_cpu", None) is None:
        self._n_cpu = int(multiprocessing.cpu_count())
    return self._n_cpu


@n_cpu.setter
def n_cpu(self, other):
    if other <= 0:
        raise ValueError("n_cpu must be greater than 0")
    self._n_cpu = other


Sim.n_cpu = n_cpu


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
            stacklevel=2,
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
    if getattr(self, "_workers", None) is None:
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

    if isinstance(self.Jmatrix, np.ndarray):
        return self.Jmatrix @ v.astype(np.float32)

    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    return array.dot(self.Jmatrix, v).astype(np.float32)


Sim.Jvec = dask_Jvec


def dask_Jtvec(self, m, v):
    """
    Compute adjoint sensitivity matrix (J^T) and vector (v) product.
    """
    self.model = m

    if isinstance(self.Jmatrix, np.ndarray):
        return self.Jmatrix.T @ v.astype(np.float32)

    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    return array.dot(v, self.Jmatrix).astype(np.float32)


Sim.Jtvec = dask_Jtvec


@property
def Jmatrix(self):
    """
    Sensitivity matrix stored on disk
    """
    if getattr(self, "_Jmatrix", None) is None:
        if self.workers is None:
            self._Jmatrix = self.compute_J()
            self._G = self._Jmatrix
        else:
            client = get_client()  # Assumes a Client already exists

            if self.store_sensitivities == "ram":
                self._Jmatrix = client.persist(
                    delayed(self.compute_J)(), workers=self.workers
                )
            else:
                self._Jmatrix = client.compute(
                    delayed(self.compute_J)(), workers=self.workers
                )

    elif isinstance(self._Jmatrix, Future):
        self._Jmatrix.result()
        if self.store_sensitivities == "disk":
            self._Jmatrix = array.from_zarr(self.sensitivity_path + "J.zarr")

    return self._Jmatrix


Sim.Jmatrix = Jmatrix


def dask_dpred(self, m=None, f=None, compute_J=False):
    r"""
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
        f = self.fields(m, return_Ainv=compute_J)

    def evaluate_receiver(source, receiver, mesh, fields):
        return receiver.eval(source, mesh, fields).flatten()

    row = delayed(evaluate_receiver, pure=True)
    rows = []
    for src in self.survey.source_list:
        for rx in src.receiver_list:
            rows.append(
                array.from_delayed(
                    row(src, rx, self.mesh, f),
                    dtype=np.float32,
                    shape=(rx.nD,),
                )
            )

    data = array.hstack(rows).compute()

    if compute_J and self._Jmatrix is None:
        Jmatrix = self.compute_J(f=f)
        return data, Jmatrix

    return data


Sim.dpred = dask_dpred


def dask_getJtJdiag(self, m, W=None):
    """
    Return the diagonal of JtJ
    """
    self.model = m
    if getattr(self, "_jtjdiag", None) is None:
        if isinstance(self.Jmatrix, Future):
            self.Jmatrix  # Wait to finish

        if W is None:
            W = np.ones(self.nD)
        else:
            W = W.diagonal() ** 2.0

        diag = array.einsum("i,ij,ij->j", W, self.Jmatrix, self.Jmatrix)

        if isinstance(diag, array.Array):
            diag = np.asarray(diag.compute())

        self._jtjdiag = diag

    return self._jtjdiag
