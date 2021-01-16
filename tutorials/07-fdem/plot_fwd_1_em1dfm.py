"""
Forward Simulation for a Single 1D Sounding
===========================================

Here we use the module *SimPEG.electromangetics.frequency_domain_1d* to predict
frequency domain data for a single sounding over a 1D layered Earth.
In this tutorial, we focus on the following:

    - General definition of sources and receivers
    - How to define the survey
    - How to predict total field, secondary field or ppm data
    - The units of the model and resulting data
    - 1D simulation for single FDEM sounding

Our survey geometry consists of a vertical magnetic dipole source
located 30 m above the Earth's surface. The receiver is offset
10 m horizontally from the source.


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
from SimPEG.electromagnetics.utils.em1d_utils import plot_layer

plt.rcParams.update({'font.size': 16})
save_file = False

# sphinx_gallery_thumbnail_number = 2

#####################################################################
# Create Survey
# -------------
#
# Here we demonstrate a general way to define sources and receivers.
# 

# Frequencies being observed
frequencies = np.array([382, 1822, 7970, 35920, 130100], dtype=float)

# Define a list of receivers for each source. In this case we only have
# one source so we will only make one list.
receiver_location = np.array([10., 0., 30.])
receiver_orientation = "z"                   # "x", "y" or "z"
field_type = "ppm"                           # "secondary", "total" or "ppm"

receiver_list = []
receiver_list.append(
    em1d.receivers.PointReceiver(
        receiver_location, frequencies, orientation=receiver_orientation,
        field_type=field_type, component="real"
    )
)
receiver_list.append(
    em1d.receivers.PointReceiver(
        receiver_location, frequencies, orientation=receiver_orientation,
        field_type=field_type, component="imag"
    )
)

# Define the source list.
source_location = np.array([0., 0., 30.])
source_orientation = 'z'                      # "x", "y" or "z"
moment_amplitude = 1.

source_list = [
    em1d.sources.MagneticDipoleSource(
        receiver_list=receiver_list, location=source_location,
        orientation=source_orientation, moment_amplitude=moment_amplitude
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

# physical property model
model = background_conductivity*np.ones(n_layer)
model[1] = layer_conductivity

# Define a mapping for conductivities
model_mapping = maps.IdentityMap(nP=n_layer)

# Plot conductivity model
plotting_mesh = TensorMesh([np.r_[thicknesses, 40.]])
plot_layer(model, plotting_mesh, showlayers=False)

#######################################################################
# Define the Forward Simulation and Predict Data
# ----------------------------------------------
# 
# Here we predict the FDEM sounding data. The simulation requires the user
# define the survey, the layer thicknesses and a mapping from the model
# to the conductivities of the layers.
#

# Define the simulation
simulation = em1d.simulation.EM1DFMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping,
)

# Predict sounding data
dpred = simulation.dpred(model)

# Plot sounding data
fig, ax = plt.subplots(1,1, figsize = (7, 7))
ax.loglog(frequencies, np.abs(dpred[0:len(frequencies)]), 'k-o', lw=3, ms=10)
ax.loglog(frequencies, np.abs(dpred[len(frequencies):]), 'k:o', lw=3, ms=10)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|Hs/Hp| (ppm)")
ax.set_title("Magnetic Field as a Function of Frequency")
ax.legend(["Real", "Imaginary"])

#######################################################################
# Optional: Export Data
# ---------------------
#

if save_file == True:

    dir_path = os.path.dirname(em1d.__file__).split(os.path.sep)[:-4]
    dir_path.extend(["tutorials", "07-fdem", "em1dfm"])
    dir_path = os.path.sep.join(dir_path) + os.path.sep

    noise = 0.05*np.abs(dpred)*np.random.rand(len(dpred))
    dpred += noise
    
    fname = dir_path + 'em1dfm_data.obs'
    np.savetxt(
        fname,
        np.c_[frequencies, dpred[0:len(frequencies)], dpred[len(frequencies):]],
        fmt='%.4e'
    )

