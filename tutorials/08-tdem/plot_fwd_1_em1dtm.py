"""
Forward Simulation for a Single 1D Sounding
===========================================

Here we use the module *SimPEG.electromangetics.time_domain_1d* to predict
the transient response for a single sounding over a 1D layered Earth.
In this tutorial, we focus on the following:

    - Defining receivers, sources and the survey
    - How to predict magnetic field data or its time-derivative
    - The units of the model and resulting data
    - Defining and running the 1D simulation for a single sounding

Our survey geometry consists of a horizontal loop source with a radius of 6 m
located 20 m above the Earth's surface. The receiver is located at the centre
of the loop and measures the vertical component of the response.


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
import SimPEG.electromagnetics.time_domain_1d as em1d
from SimPEG.electromagnetics.utils.em1d_utils import plot_layer

save_file = True
plt.rcParams.update({'font.size': 16})

# sphinx_gallery_thumbnail_number = 2

#####################################################################
# Create Survey
# -------------
#
# Here we demonstrate a general way to define the receivers, sources and survey.
# For this tutorial, we define a single horizontal loop source as well
# a receiver which measures the vertical component of the magnetic flux.
#

source_location = np.array([0., 0., 20.])  
source_orientation = "z"                      # "x", "y" or "z"
source_current = 1.                           # maximum on-time current
source_radius = 6.                            # source loop radius

receiver_location = np.array([0., 0., 20.])
receiver_orientation = "z"                    # "x", "y" or "z"
component = "b"                               # "h", "b", "dhdt" or "dbdt"
field_type = "secondary"                      # "secondary" or "total"
times = np.logspace(-5, -2, 31)               # time channels (s)

# Define receiver list. In our case, we have only a single receiver for each source.
# When simulating the response for multiple component and/or field orientations,
# multiple receiver objects are required.
receiver_list = []
receiver_list.append(
    em1d.receivers.PointReceiver(
        receiver_location, times, orientation=receiver_orientation, component=component
    )
)

# Define source list. In our case, we have only a single source.
# By default, the source assumes a step-off waveform.
source_list = [
    em1d.sources.HorizontalLoopSource(
        receiver_list=receiver_list, location=source_location,
        I=source_current, a=source_radius
    )
]

# Define the survey
survey = em1d.survey.EM1DSurveyTD(source_list)


###############################################
# Defining a 1D Layered Earth Model
# ---------------------------------
#
# Here, we define the layer thicknesses and electrical conductivities for our
# 1D simulation. If we have N layers, we define N electrical conductivity
# values and N-1 layer thicknesses. The lowest layer is assumed to extend to
# infinity. If the Earth is a halfspace, the thicknesses can be defined by
# an empty array, and the physical property values by an array of length 1.
#
# In this case, we have a more conductive layer within a background halfspace.
# This can be defined as a 3 layered Earth model. 
#

# Physical properties
background_conductivity = 1e-1
layer_conductivity = 1e0

# Layer thicknesses
thicknesses = np.array([40., 40.])
n_layer = len(thicknesses) + 1

# physical property models
model = background_conductivity*np.ones(n_layer)
model[1] = layer_conductivity

# Define a mapping for conductivities
model_mapping = maps.IdentityMap(nP=n_layer)

# Plot conductivity model
thicknesses_for_plotting = np.r_[thicknesses, 40.]
mesh_for_plotting = TensorMesh([thicknesses_for_plotting])

fig = plt.figure(figsize=(6, 5))
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
plot_layer(model, mesh_for_plotting, ax=ax, showlayers=False)
plt.gca().invert_yaxis()


#######################################################################
# Define the Forward Simulation and Predict Data
# ----------------------------------------------
#
# Here we define the simulation and predict the 1D TDEM sounding data.
# The simulation requires the user define the survey, the layer thicknesses
# and a mapping from the model to the conductivities of the layers.
# 
# When using the *SimPEG.electromagnetics.time_domain_1d* module,
# predicted data are organized by source, then by receiver, then by time channel.
#

# Define the simulation
simulation = em1d.simulation.EM1DTMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping,
)

# Predict data for a given model
dpred = simulation.dpred(model)

# Plot sounding
fig = plt.figure(figsize = (8, 7))
ax = fig.add_axes([0.15, 0.1, 0.8, 0.85])
ax.loglog(times, dpred, 'k-o')
ax.set_xlabel("Times (s)")
ax.set_ylabel("|B| (T)")
ax.set_title("Magnetic Flux")


##################################################


if save_file == True:

    dir_path = os.path.dirname(em1d.__file__).split(os.path.sep)[:-3]
    dir_path.extend(["tutorials", "08-tdem", "em1dtm"])
    dir_path = os.path.sep.join(dir_path) + os.path.sep

    noise = 0.05*np.abs(dpred)*np.random.rand(len(dpred))
    dpred += noise
    fname = dir_path + 'em1dtm_data.obs'
    np.savetxt(
        fname, np.c_[times, dpred], fmt='%.4e', header='TIME BZ'
    )






















