import numpy as np
import properties
import scipy.sparse as sp
from ....utils.code_utils import deprecate_property

from .... import props, maps
from ....data import Data
from ....base import BasePDESimulation

from ..resistivity import Simulation3DCellCentered as DC_3D_CC
from ..resistivity import Simulation3DNodal as DC_3D_N
from ..resistivity import Simulation2DCellCentered as DC_2D_CC
from ..resistivity import Simulation2DNodal as DC_2D_N


class BaseIPSimulation(BasePDESimulation):
    sigma = props.PhysicalProperty("Electrical Conductivity (S/m)")
    rho = props.PhysicalProperty("Electrical Resistivity (Ohm m)")
    props.Reciprocal(sigma, rho)

    @property
    def sigmaMap(self):
        return maps.IdentityMap()

    @sigmaMap.setter
    def sigmaMap(self, arg):
        pass

    @property
    def rhoMap(self):
        return maps.IdentityMap()

    @rhoMap.setter
    def rhoMap(self, arg):
        pass

    @property
    def sigmaDeriv(self):
        return -sp.diags(self.sigma) @ self.etaDeriv

    @property
    def rhoDeriv(self):
        return sp.diags(self.rho) @ self.etaDeriv

    eta, etaMap, etaDeriv = props.Invertible("Electrical Chargeability (V/V)")

    _data_type = properties.StringChoice(
        "IP data type",
        default="volt",
        choices=["volt", "apparent_chargeability"],
    )

    data_type = deprecate_property(
        _data_type,
        "data_type",
        new_name="receiver.data_type",
        removal_version="0.17.0",
        future_warn=True,
    )

    _Jmatrix = None
    _f = None  # the DC fields
    _pred = None
    _scale = None
    gtgdiag = None

    def fields(self, m):
        if self.verbose:
            print(">> Compute DC fields")
        if self._f is None:
            # re-uses the DC simulation's fields method
            self._f = super().fields(None)

        if self._scale is None:
            scale = Data(self.survey, np.ones(self.survey.nD))
            try:
                f = self.fields_to_space(self._f)
            except AttributeError:
                f = self._f
            # loop through receievers to check if they need to set the _dc_voltage
            for src in self.survey.source_list:
                for rx in src.receiver_list:
                    if (
                        rx.data_type == "apparent_chargeability"
                        or self._data_type == "apparent_chargeability"
                    ):
                        scale[src, rx] = 1.0 / rx.eval(src, self.mesh, f)
            self._scale = scale.dobs

        self._pred = self.forward(m, f=self._f)

        return self._f

    def dpred(self, m=None, f=None):
        """
        Predicted data.

        .. math::

            d_\\text{pred} = Pf(m)

        """
        # return self.Jvec(m, m, f=f)
        if f is None:
            f = self.fields(m)

        return self._pred

    def getJtJdiag(self, m, W=None):
        if self.gtgdiag is None:
            J = self.getJ(m)
            if W is None:
                W = self._scale ** 2
            else:
                W = (self._scale * W.diagonal()) ** 2

            self.gtgdiag = np.einsum("i,ij,ij->j", W, J, J)

        return self.gtgdiag

    def Jvec(self, m, v, f=None):
        return self._scale * super().Jvec(m, v, f)

    def forward(self, m, f=None):
        return np.asarray(self.Jvec(m, m, f=f))

    def Jtvec(self, m, v, f=None):
        return super().Jtvec(m, v * self._scale, f)

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        return toDelete


class Simulation2DCellCentered(BaseIPSimulation, DC_2D_CC):
    """
    2.5D cell centered IP problem
    """


class Simulation2DNodal(BaseIPSimulation, DC_2D_N):
    """
    2.5D nodal IP problem
    """


class Simulation3DCellCentered(BaseIPSimulation, DC_3D_CC):
    """
    3D cell centered IP problem
    """


class Simulation3DNodal(BaseIPSimulation, DC_3D_N):
    """
    3D nodal IP problem
    """


Simulation2DCellCentred = Simulation2DCellCentered
Simulation3DCellCentred = Simulation3DCellCentered
