"""
Forward Simulation of 1D Frequency-Domain Data
==============================================





"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
import os
from matplotlib import pyplot as plt
from discretize import TensorMesh

from SimPEG import maps
import SimPEG.electromagnetics.frequency_domain_1d as em1d
from SimPEG.electromagnetics.utils.em1d_utils import plot_layer

plt.rcParams.update({'font.size': 16})
save_file = False

#####################################################################
# Create Survey
# -------------
#
#

h = 30.  # source
source_location = np.array([0., 0., 30.])  # Third entry will be redundant
source_current = 1.
source_radius = 5.
moment_amplitude = 1.

source_receiver_offset = np.array([10., 0., 0.])
receiver_orientation = "z"  # "x", "y" or "z"
field_type = "ppm"  # "secondary", "total" or "ppm"

frequencies = np.array([382, 1822, 7970, 35920, 130100], dtype=float)

# Receiver list
receiver_list = []
receiver_list.append(
    em1d.receivers.PointReceiver(
        source_receiver_offset, frequencies, orientation=receiver_orientation,
        field_type=field_type, component="real", use_source_receiver_offset=True
    )
)
receiver_list.append(
    em1d.receivers.PointReceiver(
        source_receiver_offset, frequencies, orientation=receiver_orientation,
        field_type=field_type, component="imag", use_source_receiver_offset=True
    )
)
    
source_list = [
    em1d.sources.MagneticDipoleSource(
        receiver_list=receiver_list, location=source_location, orientation="z",
        moment_amplitude=moment_amplitude
    )
]

# Survey
survey = em1d.survey.EM1DSurveyFD(source_list)


###############################################
# Defining a 1D Layered Earth Model
# ---------------------------------
#
# Here, we define the layer thicknesses and electrical conductivities for our
# 1D simulation. If we have N layers, we define N electrical conductivity
# values and N-1 layer thicknesses. The lowest layer is assumed to extend to
# infinity.
#

# Physical properties
background_conductivity = 1e-1
layer_conductivity = 1e0

# Layer thicknesses
thicknesses = np.array([20., 40.])
n_layer = len(thicknesses) + 1

# physical property models
model = background_conductivity*np.ones(n_layer)
model[1] = layer_conductivity

# add source heigh to model
model = np.r_[model, h]

# Define a mapping for conductivities
wires = maps.Wires(('sigma', n_layer),('h', 1))
sigma_map = wires.sigma
h_map = wires.h

# Plot conductivity model
plotting_mesh = TensorMesh([np.r_[thicknesses, 40.]])
plotLayer(model[0:-1], plotting_mesh, showlayers=False)

#######################################################################
# Define the Forward Simulation and Predic Data
# ----------------------------------------------
#


# Simulate response for static conductivity
simulation = em1d.simulation.EM1DFMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=sigma_map, hMap=h_map,
)

dpred = simulation.dpred(model)


#######################################################################
# Plotting Results
# -------------------------------------------------
#
#


fig, ax = plt.subplots(1,1, figsize = (7, 7))
ax.loglog(frequencies, np.abs(dpred[0:len(frequencies)]), 'k-o', lw=3, ms=10)
ax.loglog(frequencies, np.abs(dpred[len(frequencies):]), 'k:o', lw=3, ms=10)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|Hs/Hp| (ppm)")
ax.set_title("Magnetic Field as a Function of Frequency")
ax.legend(["Real", "Imaginary"])

if save_file == True:

    noise = 0.05*np.abs(dpred)*np.random.rand(len(dpred))
    dpred += noise
    fname = os.path.dirname(em1d.__file__) + '\\..\\tutorials\\assets\\em1dfm_data.obs'
    np.savetxt(
        fname,
        np.c_[frequencies, dpred[0:len(frequencies)], dpred[len(frequencies):]],
        fmt='%.4e'
    )























