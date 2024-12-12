from .....electromagnetics.static.induced_polarization.simulation import (
    Simulation2DNodal as Sim,
)
from ....simulation import BaseSimulation
from .....data import Data
import numpy as np
import numcodecs

numcodecs.blosc.use_threads = False

from ..resistivity.simulation_2d import Simulation2DNodal as SimulationDC2D


class Simulation2DNodal(BaseSimulation, Sim):
    """
    Overloaded Simulation2DNodal to include the dask methods
    """

    def fields(self, m=None):
        if m is not None:
            self.model = m

        kys = self._quad_points
        f = self.fieldsPair(self)
        f._quad_weights = self._quad_weights

        Ainv = {}
        for iky, ky in enumerate(kys):
            A = self.getA(ky)
            Ainv[iky] = self.solver(A, **self.solver_opts)

            RHS = self.getRHS(ky)
            f[:, self._solutionType, iky] = Ainv[iky] * RHS

        if self._scale is None:
            scale = Data(self.survey, np.ones(self.survey.nD))
            f_fwd = self.fields_to_space(f)
            # loop through receievers to check if they need to set the _dc_voltage
            for src in self.survey.source_list:
                for rx in src.receiver_list:
                    if (
                        rx.data_type == "apparent_chargeability"
                        or self._data_type == "apparent_chargeability"
                    ):
                        scale[src, rx] = 1.0 / rx.eval(src, self.mesh, f_fwd)
            self._scale = scale.dobs

        self.Ainv = Ainv

        return f


Simulation2DNodal.compute_J = SimulationDC2D.compute_J
Simulation2DNodal.getSourceTerm = SimulationDC2D.getSourceTerm
