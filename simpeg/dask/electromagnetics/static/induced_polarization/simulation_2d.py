from .....electromagnetics.static.induced_polarization.simulation import (
    Simulation2DNodal as Sim,
)
from .....data import Data
import numpy as np
import numcodecs

numcodecs.blosc.use_threads = False
from .simulation import getJtJdiag, Jvec, Jtvec, dpred
from ..resistivity.simulation_2d import compute_J, getSourceTerm


def fields(self, m=None, return_Ainv=False):
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

    self._stashed_fields = f
    if return_Ainv:
        return f, Ainv
    return f


Sim.getJtJdiag = getJtJdiag
Sim.Jvec = Jvec
Sim.Jtvec = Jtvec
Sim.dpred = dpred
Sim.fields = fields
Sim.compute_J = compute_J
Sim.getSourceTerm = getSourceTerm
