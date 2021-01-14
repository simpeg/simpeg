"""
Single Sounding Forward Simulation of 1D Frequency-Domain Data
==============================================================





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
from SimPEG.electromagnetics import frequency_domain_1d as em1d
from SimPEG.electromagnetics.utils.em1d_utils import plotLayer

plt.rcParams.update({'font.size': 16})
save_file = False

#####################################################################
# Create Survey
# -------------
#
#

source_location = np.array([0., 0., 30.])
source_current = 1.
source_radius = 5.
moment_amplitude = 1.

receiver_location = np.array([10., 0., 30.])
receiver_orientation = "z"  # "x", "y" or "z"
field_type = "ppm"  # "secondary", "total" or "ppm"

frequencies = np.array([382, 1822, 7970, 35920, 130100], dtype=float)

# Receiver list
receiver_list = []
receiver_list.append(
    em1d.receivers.HarmonicPointReceiver(
        receiver_location, frequencies, orientation=receiver_orientation,
        field_type=field_type, component="real"
    )
)
receiver_list.append(
    em1d.receivers.HarmonicPointReceiver(
        receiver_location, frequencies, orientation=receiver_orientation,
        field_type=field_type, component="imag"
    )
)

# Sources
#source_list = [
#    em1d.sources.HarmonicHorizontalLoopSource(
#        receiver_list=receiver_list, location=source_location, a=source_radius,
#        I=source_current
#    )
#]

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

# Define a mapping for conductivities
model_mapping = maps.IdentityMap(nP=n_layer)

# Plot conductivity model
plotting_mesh = TensorMesh([np.r_[thicknesses, 40.]])
plotLayer(model, plotting_mesh, showlayers=False)

#######################################################################
# Define the Forward Simulation and Predict Data
# ----------------------------------------------
#


# Simulate response for static conductivity
simulation = em1d.simulation.EM1DFMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping,
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























