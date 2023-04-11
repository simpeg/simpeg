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


def make_2layer(xdist, dtb, layers, res_upper=30, res_lower=300, x=None, y=None):
    gxdist, gz = np.meshgrid(xdist, layers)
    gdtb, dummy = np.meshgrid(dtb, layers)
    
    gdtb = gdtb[:-1,:]
    gzt = gz[:-1,:]
    gzb = gz[1:,:]
        
    bedrockoverlap = np.clip((gdtb - gzb) / (gzt - np.fmin(gzb, gzb[np.isfinite(gzb)].max())), 0, 1)
    res = bedrockoverlap * res_lower + (1 - bedrockoverlap) * res_upper
    
    if x is None:
        x = xdist
    if y is None:
        y = 0 * xdist
    
    flightlines = pd.DataFrame({"xdist": xdist, "x": x, "y": y, "interface_depth": dtb}).assign(line_no=0)
    resistivity = pd.DataFrame(res.T)
    
    xyz = libaarhusxyz.XYZ()

    xyz.flightlines = flightlines
    xyz.layer_data["resistivity"] = resistivity
    xyz.layer_data["dep_top"] = pd.DataFrame(gzt.T)
    xyz.layer_data["dep_bot"] = pd.DataFrame(gzb.T)
    
    return xyz

def add_noise(xyz, rel_uncertainty=0.01):
    dpred = xyz.dbdt_ch1gt.values.flatten()
    noise = rel_uncertainty*np.abs(dpred)*np.random.rand(len(dpred))
    xyz.layer_data["dbdt_ch1gt"] += noise.reshape(xyz.dbdt_ch1gt.shape)

def add_uncertainty_normal(xyz, rel_uncertainty):
    xyz.layer_data["dbdt_std_ch1gt"] = np.abs(rel_uncertainty
                                              * xyz.layer_data["dbdt_ch1gt"]
                                              * np.random.randn(*xyz.layer_data["dbdt_ch1gt"].shape))

def add_uncertainty(xyz, rel_uncertainty):
    xyz.layer_data["dbdt_std_ch1gt"] = rel_uncertainty*np.abs(xyz.dbdt_ch1gt)
