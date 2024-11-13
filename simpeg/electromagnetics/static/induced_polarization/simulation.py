from functools import cached_property

import numpy as np
import scipy.sparse as sp

from .... import maps, props
from ....base import BasePDESimulation
from ....data import Data
from ..resistivity import Simulation2DCellCentered as DC_2D_CC
from ..resistivity import Simulation2DNodal as DC_2D_N
from ..resistivity import Simulation3DCellCentered as DC_3D_CC
from ..resistivity import Simulation3DNodal as DC_3D_N


class BaseIPSimulation(BasePDESimulation):
    sigma = props.PhysicalProperty("Electrical Conductivity (S/m)")
    resistivity = props.PhysicalProperty("Electrical Resistivity (Ohm m)")
    props.Reciprocal(sigma, resistivity)

    @property
    def conductivity_map(self):
        return maps.IdentityMap()

    @conductivity_map.setter
    def conductivity_map(self, arg):
        pass

    @property
    def resistivity_map(self):
        return maps.IdentityMap()

    @resistivity_map.setter
    def resistivity_map(self, arg):
        pass

    @property
    def _con_deriv(self):
        return -sp.diags(self.conductivity) @ self.etaDeriv

    @property
    def _res_deriv(self):
        return sp.diags(self.resistivity) @ self.etaDeriv

    @cached_property
    def _scale(self):
        scale = Data(self.survey, np.ones(self.survey.nD))
        if self._f is None:
            # re-uses the DC simulation's fields method
            self._f = super().fields(None)
        try:
            f = self.fields_to_space(self._f)
        except AttributeError:
            f = self._f
        # loop through receivers to check if they need to set the _dc_voltage
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                if rx.data_type == "apparent_chargeability":
                    scale[src, rx] = 1.0 / rx.eval(src, self.mesh, f)
        return scale.dobs

    eta, etaMap, etaDeriv = props.Invertible("Electrical Chargeability (V/V)")

    def __init__(
        self,
        mesh=None,
        survey=None,
        sigma=None,
        resistivity=None,
        eta=None,
        etaMap=None,
        Ainv=None,  # A DC's Ainv
        _f=None,  # A pre-computed DC field
        **kwargs,
    ):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.conductivity = sigma
        self.resistivity = resistivity
        self.eta = eta
        self.etaMap = etaMap
        if Ainv is not None:
            self.Ainv = Ainv
        self._f = _f

    _Jmatrix = None
    _pred = None

    def fields(self, m):
        if self.verbose:
            print(">> Compute DC fields")
        if self._f is None:
            # re-uses the DC simulation's fields method
            self._f = super().fields(None)

        self._pred = self.forward(m, f=self._f)

        return self._f

    def dpred(self, m=None, f=None):
        r"""
        Predicted data.

        .. math::

            d_\text{pred} = Pf(m)

        """
        # return self.Jvec(m, m, f=f)
        if f is None:
            f = self.fields(m)

        return self._pred

    def getJtJdiag(self, m, W=None, f=None):
        if getattr(self, "_gtgdiag", None) is None:
            J = self.getJ(m, f=f)
            if W is None:
                W = self._scale**2
            else:
                W = (self._scale * W.diagonal()) ** 2

            self._gtgdiag = np.einsum("i,ij,ij->j", W, J, J)

        return self._gtgdiag

    def Jvec(self, m, v, f=None):
        return self._scale * super().Jvec(m, v, f)

    def forward(self, m, f=None):
        return np.asarray(self.Jvec(m, m, f=f))

    def Jtvec(self, m, v, f=None):
        return super().Jtvec(m, v * self._scale, f)

    @property
    def _delete_on_model_change(self):
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
