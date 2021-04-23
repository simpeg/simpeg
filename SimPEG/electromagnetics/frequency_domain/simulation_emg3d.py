import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0
import properties
from ...utils.code_utils import deprecate_class

from ... import props
from ...data import Data
from ...utils import mkvc
from .simulation import BaseFDEMSimulation
from ..utils import omega
from .survey import Survey

from emg3d import models, surveys, simulations


class Simulation3DEMG3D(BaseFDEMSimulation):
    """
    By eliminating the magnetic flux density using

        .. math ::

            \mathbf{b} = \\frac{1}{i \omega}\\left(-\mathbf{C} \mathbf{e} +
            \mathbf{s_m}\\right)


    we can write Maxwell's equations as a second order system in
    \\\(\\\mathbf{e}\\\) only:

    .. math ::

        \\left(\mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{C} +
        i \omega \mathbf{M^e_{\sigma}} \\right)\mathbf{e} =
        \mathbf{C}^{\\top} \mathbf{M_{\mu^{-1}}^f}\mathbf{s_m}
        - i\omega\mathbf{M^e}\mathbf{s_e}

    which we solve for :math:`\mathbf{e}`.

    :param discretize.base.BaseMesh mesh: mesh
    """

    _solutionType = "eSolution"
    _formulation = "EB"
    fieldsPair = Fields3DElectricField
    _emg3d_simulation = None
    _simulation_inputs={}

    def __init__(self, mesh, **kwargs):
        super(Simulation3DElectricField, self).__init__(mesh, **kwargs)
  
    @property
    def emg3d_survey(self):
        if getattr(self, "_emg3d_survey", None) is None:
            self._emg3d_survey = self.generate_emg3d_survey_from_simpeg_survey()
        return self._emg3d_survey
    
    @property
    def emg3d_sigma(self):
        if getattr(self, "_emg3d_sigma", None) is None:
            self._emg3d_sigma = models.Model(
                self.mesh, 
                self.sigma.reshape(self.mesh.vnC, order='F'), 
                mapping='Conductivity'
            )        
        return self._emg3d_sigma
    
    def generate_emg3d_survey_from_simpeg_survey(self):
        pass

    def Jvec(self, m, v, f=None):
        
        self.model = m        
        dsig_dm_v = self.sigmaDeriv * v
        return self._emg3d_simulation.jvec(vec=dsig_dm_v)

    def Jtvec(self, m, v, f=None):
        
        self.model = m
        vec_xr = self._emg3d_simulation.data.synthetic.copy()
        vec_xr.values = v
        jt_sigma_vec = self._emg3d_simulation.gradient(vec=vec_xr)
        dsimg_dmT = self.sigmaDeriv.T
        jt_vec = dsimg_dmT * j_sigma_vec
        return jt_vec


    def dpred(self, m=None, f=None):
        data = self._emg3d_simulation.data.synthetic.values.flatten()
        return data 
  
    def fields(self, m=None):
        self.model = m
 
        if self._emg3d_simulation is not None:
           del self._emg3d_simulation 

        self.emg3d_simulation = simulations.Simulation(
            model=self.emg3d_sigma, 
            **self._simulation_inputs
        )
        self._emg3d_simulation.compute()
        return None
