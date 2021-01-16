from .....electromagnetics.static.resistivity.receivers import Dipole, Pole
from .....utils import sdiag


def dask_Dipole_getP(self, mesh, Gloc, transpose=False):
    if self._Ps:
        return self._Ps[0]

    P0 = mesh.getInterpolationMat(self.locations[0], Gloc)
    P1 = mesh.getInterpolationMat(self.locations[1], Gloc)
    P = P0 - P1

    if self.data_type == "apparent_resistivity":
        P = sdiag(1.0 / self.geometric_factor) * P
    elif self.data_type == "apparent_chargeability":
        P = sdiag(1.0 / self.dc_voltage) * P

    if self.storeProjections:
        self._Ps[0] = P

    if transpose:
        P = P.toarray().T

    return P


Dipole.getP = dask_Dipole_getP


def dask_Pole_getP(self, mesh, Gloc):
    if self._Ps:
        return self._Ps[0]

    P = mesh.getInterpolationMat(self.locations, Gloc)

    if self.data_type == "apparent_resistivity":
        P = sdiag(1.0 / self.geometric_factor) * P
    elif self.data_type == "apparent_chargeability":
        P = sdiag(1.0 / self.dc_voltage) * P
    if self.storeProjections:
        self._Ps[0] = P

    return P


Pole.getP = dask_Dipole_getP