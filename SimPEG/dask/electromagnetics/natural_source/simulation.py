from ....electromagnetics.natural_source.simulation import Simulation2DMagneticField as Sim
from ....utils import Zero, mkvc
import numpy as np
import scipy.sparse as sp
import dask.array as da
from dask.distributed import Future
import zarr
from time import time

def dask_boundary_fields(self, model=None):
    "Returns the 1D field objects at the boundaries"
    
    if getattr(self, "_boundary_fields", None) is None:
        if model is None:
            model = self.model
        sim = self._sim_left
        if self.muiMap is None:
            try:
                sim.mui = self._P_l @ self.mui
            except Exception:
                sim.mui = self.mui
        if self.sigmaMap is None:
            try:
                sim.sigma = self._P_l @ self.sigma
            except Exception:
                sim.sigma = self.sigma
        f_left, Ainv = sim.fields(model, return_Ainv=True)
        self._sim_left.Ainv = Ainv

        sim = self._sim_right
        if self.muiMap is None:
            try:
                sim.mui = self._P_r @ self.mui
            except Exception:
                sim.mui = self.mui
        if self.sigmaMap is None:
            try:
                sim.sigma = self._P_r @ self.sigma
            except Exception:
                sim.sigma = self.sigma
        f_right, Ainv = sim.fields(model, return_Ainv=True)
        self._sim_right.Ainv = Ainv

        self._boundary_fields = (f_left, f_right)
    return self._boundary_fields


Sim.boundary_fields = dask_boundary_fields
