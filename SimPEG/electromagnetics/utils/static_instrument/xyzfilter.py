import libaarhusxyz
import pandas as pd
import numpy as np
import copy

class FilteredXYZ(libaarhusxyz.XYZ):
    """Represents a filtered EM dataset.

    Filters can be applied both row wise (per sounding) or column wise
    (per layer or gate), or both.

    For datasets that have channels with differing number of dates,
    layerfilter can be a dictionary of boolean arrays, one for each
    key in self.layer_data.
    """

    def __new__(cls, xyz, soundingfilter=None, layerfilter=None):
        self = libaarhusxyz.XYZ.__new__(cls, xyz)
        
        self.xyz = xyz
        self.soundingfilter = soundingfilter
        self.layerfilter = layerfilter

        soundingfilter = self.get_soundingfilter()
        if isinstance(soundingfilter, slice):
            self.flightlines = self.flightlines.iloc[soundingfilter].reset_index(drop=True)
        else:
            self.flightlines = self.flightlines.loc[soundingfilter].reset_index(drop=True)
        for key in self.layer_data.keys():
            if isinstance(soundingfilter, slice):
                soundings = self.layer_data[key].iloc[soundingfilter]
            else:
                soundings = self.layer_data[key].loc[soundingfilter]
            self.layer_data[key] = soundings.iloc[:, self.get_layerfilter(key)].reset_index(drop=True)
            
        return self
            
    def get_soundingfilter(self):
        return self.soundingfilter if self.soundingfilter is not None else slice(None, None, None)
        
    def get_layerfilter(self, layer):
        layerfilter = self.layerfilter
        if isinstance(layerfilter, dict):
            layerfilter = layerfilter.get(layer, None)
        return layerfilter if layerfilter is not None else slice(None, None, None)
            
    def unfilter(self, xyz, soundingfilter=True, layerfilter=True):
        unfilteredxyz = libaarhusxyz.XYZ(xyz)

        if soundingfilter:
            unfilteredxyz.flightlines = copy.deepcopy(self.xyz.flightlines)
            for key in xyz.layer_data.keys():
                old = unfilteredxyz.layer_data[key]
                unfilteredxyz.layer_data[key] = pd.DataFrame(
                    index=self.xyz.flightlines.index,
                    columns=unfilteredxyz.layer_data[key].columns
                ).astype(
                    unfilteredxyz.layer_data[key].dtypes)
                unfilteredxyz.layer_data[key].iloc[self.get_soundingfilter(),:] = old.values
        if layerfilter:
            for key in xyz.layer_data.keys():
                old = unfilteredxyz.layer_data[key]
                unfilteredxyz.layer_data[key] = pd.DataFrame(
                    index=unfilteredxyz.flightlines.index,
                    columns=self.xyz.layer_data[key].columns
                ).astype(
                    self.xyz.layer_data[key].dtypes)
                unfilteredxyz.layer_data[key].iloc[:,self.get_layerfilter(key)] = old.values
        return unfilteredxyz
