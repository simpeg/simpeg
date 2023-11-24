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
import typing

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
    gate_filter__start_lm=5
    gate_filter__end_lm=11
    gate_filter__start_hm=12
    gate_filter__end_hm=26

    rx_orientation : typing.Literal['x', 'y', 'z'] = 'z'
    tx_orientation : typing.Literal['x', 'y', 'z'] = 'z'
    
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
    def correct_tilt_pitch_for1Dinv(self):
        cos_roll = np.cos(self.xyz.flightlines.tilt_x.values/180*np.pi)
        cos_pitch = np.cos(self.xyz.flightlines.tilt_y.values/180*np.pi)
        return 1 / (cos_roll * cos_pitch)**2
    
    @property
    def lm_data(self):
        dbdt = self.xyz.dbdt_ch1gt.values
        if "dbdt_inuse_ch1gt" in self.xyz.layer_data:
            dbdt = np.where(self.xyz.dbdt_inuse_ch1gt == 0, np.nan, dbdt)
        tiltcorrection = self.correct_tilt_pitch_for1Dinv
        tiltcorrection = np.tile(tiltcorrection, (dbdt.shape[1], 1)).T
        return - dbdt * self.xyz.model_info.get("scalefactor", 1) * self.gex.Channel1['GateFactor'] * tiltcorrection
    
    @property
    def hm_data(self):
        dbdt = self.xyz.dbdt_ch2gt.values
        if "dbdt_inuse_ch1gt" in self.xyz.layer_data:
            dbdt = np.where(self.xyz.dbdt_inuse_ch2gt == 0, np.nan, dbdt)
        tiltcorrection = self.correct_tilt_pitch_for1Dinv
        tiltcorrection = np.tile(tiltcorrection, (dbdt.shape[1], 1)).T
        return - dbdt * self.xyz.model_info.get("scalefactor", 1) * self.gex.Channel2['GateFactor'] * tiltcorrection

    # NOTE: dbdt_std is a fraction, not an actual standard deviation size!
    @property
    def lm_std(self):
        return self.xyz.dbdt_std_ch1gt.values
    
    @property
    def hm_std(self):
        return self.xyz.dbdt_std_ch2gt.values

    @property
    def data_array_nan(self):
        return np.hstack((self.lm_data, self.hm_data)).flatten()

    @property
    def data_uncert_array(self):
        return np.hstack((self.lm_std, self.hm_std)).flatten()

    uncertainties__floor = 1e-13
    uncertainties__std_data = 0.03
    uncertainties__std_data_override = False
    uncertainties__noise_level_1ms=3e-8
    uncertainties__noise_exponent=-0.5
    @property
    def uncert_array(self):
        times_lm, times_hm = self.times
        n_sounding = self.lm_data.shape[0]
        
        # 1e3 to compensate for noise level being at 1 millisecond
        noise = np.hstack((np.tile((times_lm*1e3)**self.uncertainties__noise_exponent
                                   * (self.uncertainties__noise_level_1ms / self.gex.gex_dict['Channel1']['ApproxDipoleMoment']),
                                   (n_sounding, 1)),
                           np.tile((times_hm*1e3)**self.uncertainties__noise_exponent
                                   * (self.uncertainties__noise_level_1ms / self.gex.gex_dict['Channel2']['ApproxDipoleMoment']),
                                   (n_sounding, 1)))).flatten()

        if not self.uncertainties__std_data_override:
            stds = np.where(self.data_uncert_array_culled < self.uncertainties__std_data,
                            self.uncertainties__std_data,
                            self.data_uncert_array_culled)
            uncertainties = stds * np.abs(self.data_array_nan) + noise
        else:
            uncertainties = self.uncertainties__std_data*np.abs(self.data_array_nan) + noise
        
        return np.where(np.isnan(self.data_array_nan), np.Inf, uncertainties)
    
    @property
    def times_full(self):
        return (np.array(self.gex.gate_times('Channel1')[:,0]),
                np.array(self.gex.gate_times('Channel2')[:,0]))    

    @property
    def times_filter(self):        
        times = self.times_full
        filts = [np.zeros(len(t), dtype=bool) for t in times]
        filts[0][self.gate_filter__start_lm:self.gate_filter__end_lm] = True
        filts[1][self.gate_filter__start_hm:self.gate_filter__end_hm] = True
        return filts
        
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

    
