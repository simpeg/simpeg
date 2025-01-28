from functools import cached_property

import numpy as np
import scipy.sparse as sp

from .... import props
from ....base import BaseElectricalPDESimulation
from ....data import Data
from ..resistivity import Simulation2DCellCentered as DC_2D_CC
from ..resistivity import Simulation2DNodal as DC_2D_N
from ..resistivity import Simulation3DCellCentered as DC_3D_CC
from ..resistivity import Simulation3DNodal as DC_3D_N
from ....props import _add_deprecated_physical_property_functions


@_add_deprecated_physical_property_functions("eta")
class BaseIPSimulation(BaseElectricalPDESimulation):
    sigma = BaseElectricalPDESimulation.sigma.set_feature(invertible=False)
    rho = BaseElectricalPDESimulation.rho.set_feature(invertible=False)

    eta = props.PhysicalProperty("Electrical Chargeability (V/V)")

    def _prop_deriv(self, attr):
        if attr == "sigma":
            return -sp.diags(self.sigma)
        elif attr == "rho":
            return sp.diags(self.rho)
        return super()._prop_deriv(attr)

    def is_parametrized(self, attr):
        if attr == "sigma" or attr == "rho":
            return True
        else:
            return super().is_parametrized(attr)

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

    def __init__(
        self,
        mesh,
        survey=None,
        sigma=None,
        rho=None,
        eta=None,
        Ainv=None,  # A DC's Ainv
        _f=None,  # A pre-computed DC field
        **kwargs,
    ):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self._init_recip_properties(sigma=sigma, rho=rho)
        self._init_property(eta=eta)
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
        self.model = m
        if getattr(self, "_gtgdiag", None) is None:
            J = self.getJ(m, f=f)
            if W is None:
                W = self._scale**2
            else:
                W = (self._scale * W.diagonal()) ** 2

            self._gtgdiag = np.einsum("i,ij,ij->j", W, J, J)

        return self._gtgdiag

    def Jvec(self, m, v, f=None):
        self.model = m
        return self._scale * super().Jvec(m, self._prop_deriv("eta") @ v, f)

    def forward(self, m, f=None):
        self.model = m
        jvec = super().Jvec(m, self.eta, f=f)
        return np.asarray(self._scale * jvec)

    def Jtvec(self, m, v, f=None):
        self.model = m
        return self._prop_deriv("eta").T @ super().Jtvec(m, v * self._scale, f)

    @property
    def _delete_on_model_update(self):
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
