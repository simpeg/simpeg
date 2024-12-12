from ..simulation import BaseSimulation as Sim

from dask import array

import numpy as np


class BaseSimulation(Sim):
    """
    Base class for SimPEG simulations
    """

    clean_on_model_update = ["_Jmatrix", "_jtjdiag", "_stashed_fields"]
    sensitivity_path = "./sensitivity/"
    _max_ram = 16
    _max_chunk_size = 128

    @property
    def max_ram(self):
        "Maximum ram in (Gb)"
        return self._max_ram

    @max_ram.setter
    def max_ram(self, other):
        if other <= 0:
            raise ValueError("max_ram must be greater than 0")
        self._max_ram = other

    @property
    def max_chunk_size(self):
        "Largest chunk size (Mb) used by Dask"
        return self._max_chunk_size

    @max_chunk_size.setter
    def max_chunk_size(self, other):
        if other <= 0:
            raise ValueError("max_chunk_size must be greater than 0")
        self._max_chunk_size = other

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

            self._jtj_diag = array.einsum(
                "i,ij,ij->j", W**2, self.Jmatrix, self.Jmatrix
            )

        return self._jtj_diag

    def Jvec(self, m, v, **_):
        """
        Compute sensitivity matrix (J) and vector (v) product.
        """
        self.model = m

        if isinstance(self.Jmatrix, np.ndarray):
            return self.Jmatrix @ v.astype(np.float32)

        return array.dot(self.Jmatrix, v.astype(np.float32))

    def Jtvec(self, m, v, **_):
        """
        Compute adjoint sensitivity matrix (J^T) and vector (v) product.
        """
        self.model = m

        if isinstance(self.Jmatrix, np.ndarray):
            return self.Jmatrix.T @ v.astype(np.float32)

        return array.dot(v.astype(np.float32), self.Jmatrix)

    @property
    def Jmatrix(self):
        """
        Sensitivity matrix stored on disk
        Return the diagonal of JtJ
        """
        if getattr(self, "_Jmatrix", None) is None:
            self._Jmatrix = self.compute_J(self.model)

        return self._Jmatrix
