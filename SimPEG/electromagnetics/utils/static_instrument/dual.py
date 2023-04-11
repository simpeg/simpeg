import numpy as np
import os
from matplotlib import pyplot as plt
from discretize import TensorMesh

from SimPEG import maps
from SimPEG.electromagnetics import time_domain as tdem
from SimPEG.electromagnetics.utils.em1d_utils import plot_layer
import libaarhusxyz
import pandas as pd

import numpy as np
from scipy.spatial import cKDTree, Delaunay
import os, tarfile
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from discretize import TensorMesh, SimplexMesh
#from pymatsolver import PardisoSolver

from SimPEG.utils import mkvc
from SimPEG import (
    maps, data, data_misfit, inverse_problem, regularization, optimization,
    directives, inversion, utils
    )

from SimPEG.utils import mkvc
import SimPEG.electromagnetics.time_domain as tdem
import SimPEG.electromagnetics.utils.em1d_utils
from SimPEG.electromagnetics.utils.em1d_utils import get_2d_mesh,plot_layer, get_vertical_discretization_time
from SimPEG.regularization import LaterallyConstrained, RegularizationMesh

import scipy.stats
from . import base

class DualMomentTEMXYZSystem(base.XYZSystem):
    """Dual moment system, suitable for describing e.g. the SkyTEM
    instruments. This class can not be directly instantiated, but
    instead, instantiable subclasses can created using the class
    method

    ```
    MySurveyInstrument = DualMomentTEMXYZSystem.load_gex(
        libaarhusxyz.GEX("instrument.gex"))
    ```

    which reads a gex file containing among other things the waveform
    description of the instrument.

    See the help for `XYZSystem` for more information on basic usage.
    """
    gate_start_lm=5
    gate_end_lm=11
    gate_start_hm=12
    gate_end_hm=26

    rx_orientation = 'z'
    tx_orientation = 'z'
    
    @classmethod
    def load_gex(cls, gex):
        """Accepts a GEX file loaded using libaarhusxyz.GEX() and
        returns a new subclass of XYZSystem that can be used to do
        inversion and forward modelling."""
        
        class GexSystem(cls):
            pass
        GexSystem.gex = gex
        return GexSystem   
    
    @property
    def area(self):
        return self.gex.General['TxLoopArea']
    
    @property
    def waveform_hm(self):
        return self.gex.General['WaveformHMPoint']
    
    @property
    def waveform_lm(self):
        return self.gex.General['WaveformLMPoint']

    @property
    def lm_data(self):
        dbdt = self.xyz.dbdt_ch1gt.values
        dbdt = dbdt * self.xyz.model_info.get("scalefactor", 1)
        if "dbdt_inuse_ch1gt" in self.xyz.layer_data:
            dbdt = np.where(self.xyz.dbdt_inuse_ch1gt == 0, np.nan, dbdt)
        return -(dbdt*self.gex.Channel1['GateFactor'])[:,self.gate_start_lm:self.gate_end_lm]
    
    @property
    def hm_data(self):
        dbdt = self.xyz.dbdt_ch2gt.values
        dbdt = dbdt * self.xyz.model_info.get("scalefactor", 1)
        if "dbdt_inuse_ch1gt" in self.xyz.layer_data:
            dbdt = np.where(self.xyz.dbdt_inuse_ch2gt == 0, np.nan, dbdt)
        return -(dbdt*self.gex.Channel2['GateFactor'])[:,self.gate_start_hm:self.gate_end_hm]
    
    @property
    def lm_std(self):
        return (self.xyz.dbdt_std_ch1gt.values*self.gex.Channel1['GateFactor'])[:,self.gate_start_lm:self.gate_end_lm]
    
    @property
    def hm_std(self):
        return (self.xyz.dbdt_std_ch2gt.values*self.gex.Channel2['GateFactor'])[:,self.gate_start_hm:self.gate_end_hm]

    @property
    def data_array_nan(self):
        return np.hstack((self.lm_data, self.hm_data)).flatten()

    @property
    def data_array(self):
        dobs = self.data_array_nan
        return np.where(np.isnan(dobs), 9999., dobs)
    
    uncertainties_floor = 1e-13
    uncertainties_data = 0.05 # If None, use data std:s
    @property
    def uncert_array(self):
        if self.uncertainties_data is None:
            uncertainties = np.hstack((self.lm_std, self.hm_std)).flatten()
        else:
            uncertainties = self.uncertainties_data
        uncertainties = uncertainties * np.abs(self.data_array) + self.uncertainties_floor
        return np.where(np.isnan(self.data_array_nan), np.Inf, uncertainties)

    @property
    def times_full(self):
        return (np.array(self.gex.gate_times('Channel1')[:,0]),
                np.array(self.gex.gate_times('Channel2')[:,0]))    
    
    @property
    def times(self):
        lmtimes, hmtimes = self.times_full        
        return (lmtimes[self.gate_start_lm:self.gate_end_lm],
                hmtimes[self.gate_start_hm:self.gate_end_hm])    
    
    def make_waveforms(self):
        time_input_currents_hm = self.waveform_hm[:,0]
        input_currents_hm = self.waveform_hm[:,1]
        time_input_currents_lm = self.waveform_lm[:,0]
        input_currents_lm = self.waveform_lm[:,1]

        waveform_hm = tdem.sources.PiecewiseLinearWaveform(time_input_currents_hm, input_currents_hm)
        waveform_lm = tdem.sources.PiecewiseLinearWaveform(time_input_currents_lm, input_currents_lm)
        return waveform_lm, waveform_hm
    
    def make_system(self, idx, location, times):
        # FIXME: Martin says set z to altitude, not z (subtract topo), original code from seogi doesn't work!
        # Note: location[2] is already == altitude
        receiver_location = (location[0] + self.gex.General['RxCoilPosition'][0],
                             location[1],
                             location[2] + np.abs(self.gex.General['RxCoilPosition'][2]))
        waveform_lm, waveform_hm = self.make_waveforms()        

        return [
            tdem.sources.MagDipole(
                [tdem.receivers.PointMagneticFluxTimeDerivative(
                    receiver_location, times[0], self.rx_orientation)],
                location=location,
                waveform=waveform_lm,
                orientation=self.tx_orientation,
                i_sounding=idx),
            tdem.sources.MagDipole(
                [tdem.receivers.PointMagneticFluxTimeDerivative(
                    receiver_location, times[1], self.rx_orientation)],
                location=location,
                waveform=waveform_hm,
                orientation=self.tx_orientation,
                i_sounding=idx)]
    
    thicknesses_type = "geometric"
    thicknesses_minimum_dz = 3
    thicknesses_geomtric_factor = 1.07
    thicknesses_sigma_background = 0.1
    def make_thicknesses(self):
        if self.thicknesses_type == "geometric":
            return SimPEG.electromagnetics.utils.em1d_utils.get_vertical_discretization(
                self.n_layer-1, self.thicknesses_minimum_dz, self.thicknesses_geomtric_factor)
        else:
            if "dep_top" in self.xyz.layer_params:
                return np.diff(self.xyz.layer_params["dep_top"].values)
            return SimPEG.electromagnetics.utils.em1d_utils.get_vertical_discretization_time(
                np.sort(np.concatenate(self.times)),
                sigma_background=self.thicknesses_sigma_background,
                n_layer=self.n_layer-1
            )

    def make_misfit_weights(self, thicknesses):
        return 1./self.uncert_array

    def forward_data_to_xyz(self, resp):
        times_lm, times_hm = self.times_full
        
        xyzresp = libaarhusxyz.XYZ()
        xyzresp.model_info.update(self.xyz.model_info)
        xyzresp.flightlines = self.xyz.flightlines

        lm = np.full((len(xyzresp.flightlines), len(times_lm)), np.nan)
        hm = np.full((len(xyzresp.flightlines), len(times_hm)), np.nan)

        gate_count_lm = self.gate_end_lm-self.gate_start_lm
        gate_count_hm = self.gate_end_hm-self.gate_start_hm
        lm[:,self.gate_start_lm:self.gate_end_lm] = resp[:,:gate_count_lm]
        hm[:,self.gate_start_hm:self.gate_end_hm] = resp[:,gate_count_lm:]
        
        lm /= self.xyz.model_info.get("scalefactor", 1)
        hm /= self.xyz.model_info.get("scalefactor", 1)

        xyzresp.layer_data = {
            "dbdt_ch1gt": pd.DataFrame(lm),
            "dbdt_ch2gt": pd.DataFrame(hm),
        }

        # XYZ assumes all receivers have the same times
        xyzresp.model_info["gate times for channel 1"] = list(times_lm)
        xyzresp.model_info["gate times for channel 2"] = list(times_hm)

        return xyzresp
    
