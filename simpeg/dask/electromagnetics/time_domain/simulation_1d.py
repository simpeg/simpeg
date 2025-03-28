from ....electromagnetics.time_domain.simulation_1d import Simulation1DLayered as Sim

from ...simulation import getJtJdiag, Jvec, Jtvec

Sim._delete_on_model_update = ["_Jmatrix", "_jtjdiag", "_J"]


@property
def Jmatrix(self):
    """
    Sensitivity matrix stored on disk
    Return the diagonal of JtJ
    """
    if getattr(self, "_Jmatrix", None) is None:
        Jmat = self.getJ(self.model)
        self._Jmatrix = Jmat["ds"]

    return self._Jmatrix


Sim.getJtJdiag = getJtJdiag
Sim.Jvec = Jvec
Sim.Jtvec = Jtvec
Sim.Jmatrix = Jmatrix
